#!/usr/bin/env python3
"""
Calcule des statistiques de similarité cosinus entre `anchor_text` et `generated_text`
sur un dataset Hugging Face (ou un fichier local) avec un modèle d'embeddings.

Fonctions principales
- Stats globales des similarités
- 30 exemples sous un seuil (par défaut 0.6) + compte et %
- Exemples dans un intervalle [0.5, 0.7] + compte et %
- % sous le seuil par modèle (si colonne `model` présente)
- Export CSV optionnel avec colonnes booléennes `below_threshold` et `in_between_range`

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
    p.add_argument("--model_col", type=str, default="model", help="Nom de la colonne modèle (si disponible)")

    # Seuils / affichages
    p.add_argument("--threshold", type=float, default=0.6, help="Seuil pour filtrer les paires faibles")
    p.add_argument("--show_below", type=int, default=30, help="Nb d'exemples à afficher sous le seuil")
    p.add_argument("--range_low", type=float, default=0.5, help="Borne inférieure de l'intervalle à lister")
    p.add_argument("--range_high", type=float, default=0.7, help="Borne supérieure de l'intervalle à lister")
    p.add_argument("--show_between", type=int, default=30, help="Nb d'exemples à afficher dans l'intervalle")
    p.add_argument("--show_examples", type=int, default=3, help="Nb d'exemples extrêmes à afficher")

    # Modèle et perf
    p.add_argument("--model", type=str, default="intfloat/multilingual-e5-base", help="Nom du modèle d'embeddings")
    p.add_argument("--batch_size", type=int, default=64, help="Taille de batch pour l'encodage")
    p.add_argument("--max_rows", type=int, default=None, help="Nombre max de lignes à traiter")
    p.add_argument("--sample_frac", type=float, default=None, help="Fraction aléatoire à échantillonner (0-1)")
    p.add_argument("--seed", type=int, default=42, help="Graine pour l'échantillonnage")

    # Sorties
    p.add_argument("--output_csv", type=str, default=None, help="Chemin pour sauver les scores par ligne")

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

    # Sanity sur intervalle
    if args.range_low > args.range_high:
        args.range_low, args.range_high = args.range_high, args.range_low

    print("=== Chargement des données ===")
    df = load_data(args)
    check_columns(df, args.anchor_col, args.generated_col)
    keep_cols = [args.anchor_col, args.generated_col] + ([args.model_col] if args.model_col in df.columns else [])
    df = df[keep_cols].dropna(subset=[args.anchor_col, args.generated_col])
    df = maybe_sample(df, args.max_rows, args.sample_frac, args.seed)
    n = len(df)
    if n == 0:
        raise SystemExit("Aucune ligne à traiter après filtrage")
    print(f"Lignes à traiter: {n}")

    print("=== Chargement du modèle d'embeddings ===")
    print(f"Modèle: {args.model}")
    print(f"Colonne modèle: {args.model_col if args.model_col in df.columns else 'N/A'}")
    print("trust_remote_code: True")
    print(f"Seuil sous-similarité: {args.threshold} | Exemples à afficher: {args.show_below}")
    print(f"Intervalle à lister: [{args.range_low}, {args.range_high}] | Exemples: {args.show_between}")
    model = SentenceTransformer(args.model, trust_remote_code=True)

    anchors = df[args.anchor_col].astype(str).tolist()
    gens = df[args.generated_col].astype(str).tolist()
    models = df[args.model_col].astype(str).tolist() if args.model_col in df.columns else None

    print("=== Encodage des textes ===")
    A = compute_embeddings(model, anchors, args.batch_size)
    B = compute_embeddings(model, gens, args.batch_size)

    print("=== Calcul des similarités ===")
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

    thresholds = [0.3, 0.6, 0.7, 0.8, 0.9]
    shares = {f">={t}": float(np.mean(sims >= t)) for t in thresholds}

    print("=== Résumé ===")
    print(f"Date: {datetime.utcnow().isoformat()}Z")
    if args.dataset_name:
        print(f"Dataset: {args.dataset_name} | split: {args.split}")
    else:
        print(f"Fichiers: {args.data_files} | format: {args.file_format}")
    print(f"Colonnes: anchor='{args.anchor_col}', generated='{args.generated_col}', model_col='{args.model_col if args.model_col in df.columns else 'N/A'}'")
    print(f"Modèle: {args.model}")

    print("Statistiques (cosine):")
    for k in ["n", "mean", "median", "std", "min", "p10", "p25", "p75", "p90", "max"]:
        print(f"  {k:>6}: {stats[k]:.6f}" if k != "n" else f"  {k:>6}: {stats[k]}")

    print("Parts au-dessus des seuils:")
    for k in [f">={t}" for t in thresholds]:
        print(f"  {k:>6}: {shares[k]*100:.2f}%")

    # Sous seuil
    print(f"=== Paires sous le seuil {args.threshold:.2f} ===")
    below_mask = sims < args.threshold
    nb_below = int(np.sum(below_mask))
    pct_below = nb_below / n
    print(f"Total sous seuil: {nb_below} / {n} ({pct_below*100:.2f}%)")
    idx_below = np.where(below_mask)[0]
    if len(idx_below) > 0:
        idx_sorted = idx_below[np.argsort(sims[idx_below])]
        show_idx = idx_sorted[:min(args.show_below, len(idx_sorted))]
        print(f"=== Exemples sous {args.threshold:.2f} ({len(show_idx)}) ===")
        for i, idx in enumerate(show_idx, 1):
            print(f"[{i}] sim={sims[idx]:.4f}")
            print(f"  anchor   : {anchors[idx][:400]}")
            print(f"  generated: {gens[idx][:400]}")
    else:
        print("Aucune paire sous le seuil.")

    # Intervalle [range_low, range_high]
    print(f"=== Paires dans l'intervalle [{args.range_low:.2f}, {args.range_high:.2f}] ===")
    between_mask = (sims >= args.range_low) & (sims <= args.range_high)
    nb_between = int(np.sum(between_mask))
    pct_between = nb_between / n
    print(f"Total dans l'intervalle: {nb_between} / {n} ({pct_between*100:.2f}%)")
    idx_between = np.where(between_mask)[0]
    if len(idx_between) > 0:
        idx_sorted_b = idx_between[np.argsort(sims[idx_between])]
        show_idx_b = idx_sorted_b[:min(args.show_between, len(idx_sorted_b))]
        print(f"=== Exemples dans l'intervalle ({len(show_idx_b)}) ===")
        for i, idx in enumerate(show_idx_b, 1):
            print(f"[{i}] sim={sims[idx]:.4f}")
            print(f"  anchor   : {anchors[idx][:400]}")
            print(f"  generated: {gens[idx][:400]}")
    else:
        print("Aucune paire dans cet intervalle.")

    # Pourcentage par modèle (sous seuil)
    if models is not None:
        arr_models = np.array(models)
        unique = np.unique(arr_models)
        rows = []
        for m in sorted(unique):
            sel = arr_models == m
            n_m = int(np.sum(sel))
            below_m = int(np.sum(below_mask[sel]))
            pct_m = (below_m / n_m * 100) if n_m > 0 else float('nan')
            mean_m = float(np.mean(sims[sel])) if n_m > 0 else float('nan')
            rows.append((m, n_m, mean_m, below_m, pct_m))
        mdl_df = pd.DataFrame(rows, columns=["model", "n", "mean_sim", "count_below", "pct_below_%"]).sort_values(by="pct_below_%", ascending=False)
        print("=== Pourcentage sous le seuil par modèle ===")
        for _, r in mdl_df.iterrows():
            print(f"  {r['model']}: n={int(r['n'])}, mean_sim={r['mean_sim']:.3f}, sous_seuil={int(r['count_below'])} ({r['pct_below_%']:.2f}%)")
    else:
        print("(Aucune colonne modèle trouvée — passe --model_col si disponible.)")

    # Exemples extrêmes
    topk = min(args.show_examples, n)
    ord_idx = np.argsort(sims)
    worst_idx = ord_idx[:topk]
    best_idx = ord_idx[-topk:][::-1]

    def show_block(title: str, idxs: np.ndarray):
        print(f"=== {title} ({len(idxs)}) ===")
        for i, idx in enumerate(idxs, 1):
            print(f"[{i}] sim={sims[idx]:.4f}")
            print(f"  anchor   : {anchors[idx][:400]}")
            print(f"  generated: {gens[idx][:400]}")

    show_block("Pires paires", worst_idx)
    show_block("Meilleures paires", best_idx)

    if args.output_csv:
        out_path = os.path.abspath(args.output_csv)
        cols = {
            "anchor_text": anchors,
            "generated_text": gens,
            "cosine_similarity": sims,
        }
        if models is not None:
            cols["model"] = models
        # Flags utiles
        cols["below_threshold"] = below_mask.tolist()
        cols["in_between_range"] = between_mask.tolist()
        out_df = pd.DataFrame(cols)
        out_df.to_csv(out_path, index=False)
        print(f"Scores sauvegardés: {out_path}")


if __name__ == "__main__":
    main()
