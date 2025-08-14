#!/usr/bin/env python3
"""
Calcule des statistiques de similarité cosinus entre `anchor_text` et `generated_text`
sur un dataset Hugging Face (ou un fichier local) avec un modèle d'embeddings.

Exemples
--------
python similarite_anchor_generated_stats.py \
  --dataset_name my-org/mon-dataset --split train \
  --anchor_col anchor_text --generated_col generated_text \
  --model intfloat/multilingual-e5-base

python similarite_anchor_generated_stats.py \
  --data_files data.csv --file_format csv \
  --anchor_col source --generated_col generation

Notes
-----
- Par défaut, le script échantillonne tout le split. Utilise --max_rows pour limiter
- Tu peux sauvegarder les scores par ligne avec --output_csv chemin.csv
- Modèles conseillés: intfloat/multilingual-e5-base (multilingue), sentence-transformers/all-MiniLM-L6-v2 (rapide)
"""

import argparse
import math
import os
from datetime import datetime
from typing import Optional, List

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise SystemExit(
        "Il faut installer sentence-transformers: pip install -U sentence-transformers datasets tqdm pandas"
    ) from e


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)

    # Source de données
    src = p.add_argument_group("Données")
    src.add_argument("--dataset_name", type=str, default=None, help="Nom du dataset HF, ex: 'my-org/mon-dataset'")
    src.add_argument("--split", type=str, default="train", help="Split à charger si dataset HF")
    src.add_argument("--data_files", type=str, nargs="+", default=None, help="Fichiers locaux (csv/json/jsonl/parquet)")
    src.add_argument("--file_format", type=str, choices=["csv", "json", "jsonl", "parquet"], default=None,
                    help="Format des fichiers locaux si --data_files est fourni")

    # Colonnes
    p.add_argument("--anchor_col", type=str, default="anchor_text", help="Nom de la colonne anchor")
    p.add_argument("--generated_col", type=str, default="generated_text", help="Nom de la colonne generated")

    # Modèle et perf
    p.add_argument("--model", type=str, default="intfloat/multilingual-e5-base", help="Nom du modèle d'embeddings")
    p.add_argument("--batch_size", type=int, default=64, help="Taille de batch pour l'encodage")
    p.add_argument("--max_rows", type=int, default=None, help="Nombre max de lignes à traiter")
    p.add_argument("--sample_frac", type=float, default=None, help="Fraction aléatoire à échantillonner (0-1)")
    p.add_argument("--seed", type=int, default=42, help="Graine pour l'échantillonnage")

    # Sorties
    p.add_argument("--output_csv", type=str, default=None, help="Chemin pour sauver les scores par ligne")
    p.add_argument("--show_examples", type=int, default=3, help="Nb d'exemples à afficher")

    return p.parse_args()


def load_data(args: argparse.Namespace) -> pd.DataFrame:
    if args.dataset_name is None and args.data_files is None:
        raise SystemExit("Indique --dataset_name ou --data_files")

    if args.dataset_name is not None:
        ds = load_dataset(args.dataset_name, split=args.split)
    else:
        if args.file_format is None:
            raise SystemExit("Précise --file_format pour les fichiers locaux")
        ds = load_dataset(args.file_format, data_files=args.data_files, split="train")

    df = ds.to_pandas()
    return df


def check_columns(df: pd.DataFrame, anchor_col: str, generated_col: str) -> None:
    missing: List[str] = [c for c in [anchor_col, generated_col] if c not in df.columns]
    if missing:
        raise SystemExit(f"Colonnes manquantes: {missing}. Colonnes dispo: {list(df.columns)[:20]} ...")


def maybe_sample(df: pd.DataFrame, max_rows: Optional[int], sample_frac: Optional[float], seed: int) -> pd.DataFrame:
    if sample_frac is not None:
        if not (0 < sample_frac <= 1):
            raise SystemExit("--sample_frac doit être entre 0 et 1")
        df = df.sample(frac=sample_frac, random_state=seed)
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed)
    df = df.reset_index(drop=True)
    return df


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.clip(norm, eps, None)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = normalize(a)
    b_n = normalize(b)
    return np.sum(a_n * b_n, axis=1)


def compute_embeddings(model: SentenceTransformer, texts: List[str], batch_size: int) -> np.ndarray:
    embs: List[np.ndarray] = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encodage", unit="batch"):
        batch = texts[i : i + batch_size]
        e = model.encode(batch, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=False)
        embs.append(e)
    return np.vstack(embs)


def main():
    args = parse_args()

    print("\n=== Chargement des données ===")
    df = load_data(args)
    check_columns(df, args.anchor_col, args.generated_col)
    df = df[[args.anchor_col, args.generated_col]].dropna()
    df = maybe_sample(df, args.max_rows, args.sample_frac, args.seed)
    n = len(df)
    if n == 0:
        raise SystemExit("Aucune ligne à traiter après filtrage")
    print(f"Lignes à traiter: {n}")

    print("\n=== Chargement du modèle d'embeddings ===")
    print(f"Modèle: {args.model}")
    print("trust_remote_code: True")
    model = SentenceTransformer(args.model, trust_remote_code=True)

    anchors = df[args.anchor_col].astype(str).tolist()
    gens = df[args.generated_col].astype(str).tolist()

    print("\n=== Encodage des textes ===")
    A = compute_embeddings(model, anchors, args.batch_size)
    B = compute_embeddings(model, gens, args.batch_size)

    print("\n=== Calcul des similarités ===")
    sims = cosine_sim(A, B)

    # Stats de base
    stats = {
        "n": n,
        "mean": float(np.mean(sims)),
        "median": float(np.median(sims)),
        "std": float(np.std(sims)),
        "min": float(np.min(sims)),
        "p10": float(np.percentile(sims, 10)),
        "p25": float(np.percentile(sims, 25)),
        "p75": float(np.percentile(sims, 75)),
        "p90": float(np.percentile(sims, 90)),
        "max": float(np.max(sims)),
    }

    thresholds = [0.3, 0.5, 0.7, 0.8, 0.9]
    shares = {f">={t}": float(np.mean(sims >= t)) for t in thresholds}

    print("\n=== Résumé ===")
    print(f"Date: {datetime.utcnow().isoformat()}Z")
    if args.dataset_name:
        print(f"Dataset: {args.dataset_name} | split: {args.split}")
    else:
        print(f"Fichiers: {args.data_files} | format: {args.file_format}")
    print(f"Colonnes: anchor='{args.anchor_col}', generated='{args.generated_col}'")
    print(f"Modèle: {args.model}")

    print("\nStatistiques (cosine):")
    for k in ["n", "mean", "median", "std", "min", "p10", "p25", "p75", "p90", "max"]:
        print(f"  {k:>6}: {stats[k]:.6f}" if k != "n" else f"  {k:>6}: {stats[k]}")

    print("\nParts au-dessus des seuils:")
    for k in [f">={t}" for t in thresholds]:
        print(f"  {k:>6}: {shares[k]*100:.2f}%")

    # Exemples extrêmes
    topk = min(args.show_examples, n)
    ord_idx = np.argsort(sims)
    worst_idx = ord_idx[:topk]
    best_idx = ord_idx[-topk:][::-1]

    def show_block(title: str, idxs: np.ndarray):
        print(f"\n=== {title} ({len(idxs)}) ===")
        for i, idx in enumerate(idxs, 1):
            print(f"[{i}] sim={sims[idx]:.4f}")
            print(f"  anchor   : {anchors[idx][:400]}")
            print(f"  generated: {gens[idx][:400]}")

    show_block("Pires paires", worst_idx)
    show_block("Meilleures paires", best_idx)

    if args.output_csv:
        out_path = os.path.abspath(args.output_csv)
        out_df = pd.DataFrame({
            "anchor_text": anchors,
            "generated_text": gens,
            "cosine_similarity": sims,
        })
        out_df.to_csv(out_path, index=False)
        print(f"\nScores sauvegardés: {out_path}")


if __name__ == "__main__":
    main()
