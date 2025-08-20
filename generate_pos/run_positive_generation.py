#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline parallèle de génération de "positives" à partir d'un dataset Hugging Face,
avec sauvegarde incrémentale sur le Hub tous les N cas traités.

Principales fonctions
- Découpage intelligent par taille via quantiles (q50 long, q25..q50 medium, <q25 short)
- Tirages pondérés selon recettes + styles + word_count
- Appels OpenRouter en parallèle via ThreadPoolExecutor
- Push vers HF Hub par chunk de taille configurable
- Reprise optionnelle en évitant les doublons déjà poussés

Exemple
pip install datasets pyyaml openai
huggingface-cli login  # ou export HF_TOKEN=...
export OPENROUTER_API_KEY="sk-..."

python run_positive_generation.py \
  --dataset matheoqtb/ancre \
  --column positive \
  --yaml positive_strategies.yaml \
  --out-repo matheoqtb/ancre_querry \
  --chunk-size 10000 \
  --concurrency 16 \
  --resume-from-hub
"""
import argparse
import hashlib
import json
import os
import random
import signal
import sys
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

import yaml  # PyYAML
from datasets import Dataset, load_dataset

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# -------------------------------
# Utilitaires
# -------------------------------

def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def percentile_threshold(sorted_vals: List[int], p: float) -> int:
    if not sorted_vals:
        return 0
    idx = int(round((len(sorted_vals) - 1) * p))
    return sorted_vals[idx]

def compute_length_thresholds(texts: List[str]) -> Tuple[int, int]:
    lens = [len((t or '').strip()) for t in texts]
    lens_sorted = sorted(lens)
    q25 = percentile_threshold(lens_sorted, 0.25)
    q50 = percentile_threshold(lens_sorted, 0.50)
    return q25, q50

def categorize_length(length: int, q25: int, q50: int) -> str:
    if length >= q50:
        return "long"
    elif length >= q25:
        return "medium"
    else:
        return "short"

def weighted_choice(items: List[Dict[str, Any]], rng: random.Random, weight_key: str = "poids") -> Dict[str, Any]:
    weights = [max(0, it.get(weight_key, 0)) for it in items]
    total = sum(weights)
    if total <= 0:
        return rng.choice(items)
    r = rng.uniform(0, total)
    s = 0.0
    for it, w in zip(items, weights):
        s += w
        if s >= r:
            return it
    return items[-1]

def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def pick_word_count_if_any(strategy: Dict[str, Any], rng: random.Random) -> Optional[int]:
    rng_spec = strategy.get("word_count_range")
    if isinstance(rng_spec, list) and len(rng_spec) == 2 and all(isinstance(x, int) for x in rng_spec):
        lo, hi = rng_spec
        if lo > hi:
            lo, hi = hi, lo
        return rng.randint(lo, hi)
    return None

def pick_llm_rotation(llm_config: Dict[str, Any], rng: random.Random) -> Tuple[str, Optional[str]]:
    rotation = llm_config.get("rotation", [])
    if not rotation:
        raise ValueError("La configuration YAML ne contient pas llm.rotation")
    choice = rotation[rng.randrange(len(rotation))]
    model = choice.get("model")
    provider = choice.get("provider")
    if not model:
        raise ValueError("Entrée invalide dans llm.rotation: 'model' manquant")
    return model, provider

def format_strategy_prompt(strategy: Dict[str, Any], anchor_text: str, style_instruction: Optional[str], word_count: Optional[int]) -> str:
    """
    Remplacement ciblé des seuls placeholders connus sans utiliser str.format,
    afin d'éviter les KeyError si le prompt YAML contient des accolades non mappées.
    """
    tpl = strategy.get("prompt", "")
    out = tpl.replace("{anchor_text}", anchor_text)
    out = out.replace("{style_instruction}", style_instruction or "")
    out = out.replace("{word_count}", str(word_count) if word_count is not None else "")
    return out

def generate_with_openrouter(
    model: str,
    provider: Optional[str],
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 4000,
    sleep_ms: int = 0,
    x_title: str = 'Positive Generator',
    http_referer: str = 'https://local.test',
) -> Optional[str]:
    system_prompt = system_prompt or (
        "Tu es un expert en manipulation de langage, capable de suivre des instructions de réécriture complexes et nuancées."
    )
    extra_body: Dict[str, Any] = {}
    if provider:
        extra_body["provider"] = {"order": [provider], "allow_fallbacks": False}

    # Throttle avant l'appel pour lisser le débit
    if sleep_ms and sleep_ms > 0:
        time.sleep(sleep_ms / 1000.0)

    try:
        if OpenAI is None:
            raise RuntimeError("La bibliothèque openai n'est pas disponible.")
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            default_headers={"X-Title": x_title, "HTTP-Referer": http_referer},
        )
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )
        choices = getattr(resp, 'choices', None)
        if not choices:
            sys.stderr.write(
                f"[API Error] Empty or missing choices for model={model} provider={provider}\n"
            )
            return None
        content = getattr(choices[0].message, 'content', None)
        if not content:
            sys.stderr.write(
                f"[API Error] Missing content in first choice for model={model} provider={provider}\n"
            )
            return None
        return content.strip()
    except Exception as e:
        # On renvoie None et on log sur stderr, le record contiendra l'erreur
        sys.stderr.write(f"[API Error] {type(e).__name__}: {str(e)[:300]}\n")
        return None

def make_uid(dataset_name: str, split: str, text: str) -> str:
    h = hashlib.sha256()
    h.update((dataset_name or "").encode("utf-8"))
    h.update(b"|")
    h.update((split or "").encode("utf-8"))
    h.update(b"|")
    h.update((text or "").encode("utf-8"))
    return h.hexdigest()

def detect_next_chunk_id(repo_id: str) -> int:
    try:
        ds_dict = load_dataset(repo_id)
        keys = list(ds_dict.keys())
        max_id = 0
        for k in keys:
            cid = None
            if k.startswith("chunk_"):
                try:
                    cid = int(k.split("_")[1])
                except Exception:
                    pass
            elif k.startswith("chunk-"):
                try:
                    cid = int(k.split("-")[1])
                except Exception:
                    pass
            elif k.startswith("chunk."):
                try:
                    cid = int(k.split(".")[1])
                except Exception:
                    pass
            if cid is not None and cid > max_id:
                max_id = cid
        return max_id + 1
    except Exception:
        return 1

def load_existing_uids(repo_id: str) -> set:
    existing = set()
    try:
        ds_dict = load_dataset(repo_id, streaming=True)
        for split in ds_dict.keys():
            for row in ds_dict[split]:
                uid = row.get("uid")
                if isinstance(uid, str):
                    existing.add(uid)
    except Exception as e:
        sys.stderr.write(f"[resume] Lecture UIDs impossible: {e}\n")
    return existing

# -------------------------------
# Pousses et buffers thread-safe
# -------------------------------

buffer_lock = threading.Lock()
buffer_records: List[Dict[str, Any]] = []
_next_chunk_id: Optional[int] = None  # sera défini dans main
push_counter = 0
push_counter_lock = threading.Lock()

def get_push_counter() -> int:
    with push_counter_lock:
        return push_counter

def push_chunk(repo_id: str, records: List[Dict[str, Any]], chunk_id: int, private: Optional[bool] = None) -> None:
    global push_counter
    if not records:
        return
    split_name = f"chunk_{chunk_id:06d}"
    ds = Dataset.from_list(records)
    try:
        if private is None:
            ds.push_to_hub(repo_id, split=split_name)
        else:
            ds.push_to_hub(repo_id, split=split_name, private=private)
        print(f"[push] {len(records)} éléments -> {repo_id}:{split_name}")
        with push_counter_lock:
            push_counter += 1
    except Exception as e:
        sys.stderr.write(f"[push][ERREUR] {e}\n")
        fallback = f"{split_name}.jsonl"
        try:
            with open(fallback, "w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            sys.stderr.write(f"[push] Dump local: {fallback}\n")
        finally:
            with push_counter_lock:
                push_counter += 1

def flush_if_needed(out_repo: str, chunk_size: int, private: Optional[bool]) -> None:
    global buffer_records, _next_chunk_id
    local = None
    with buffer_lock:
        if len(buffer_records) >= chunk_size:
            local = buffer_records
            buffer_records = []
            chunk_id = _next_chunk_id
            _next_chunk_id += 1
        else:
            return
    # push hors du verrou
    push_chunk(out_repo, local, chunk_id, private=private)

def flush_all(out_repo: str, private: Optional[bool]) -> None:
    global buffer_records, _next_chunk_id
    with buffer_lock:
        local = buffer_records
        buffer_records = []
        if not local:
            return
        chunk_id = _next_chunk_id
        _next_chunk_id += 1
    push_chunk(out_repo, local, chunk_id, private=private)

# -------------------------------
# Worker
# -------------------------------

def make_record(
    idx_global: int,
    source_index: int,
    text: str,
    dataset_name: str,
    split: str,
    q25: int,
    q50: int,
    styles: List[Dict[str, Any]],
    strategies: Dict[str, Any],
    recettes: Dict[str, Any],
    llm_cfg: Dict[str, Any],
    max_tokens: int,
    sleep_ms: int,
    seed: Optional[int],
) -> Dict[str, Any]:
    # RNG par élément pour reproductibilité éventuelle
    rng = random.Random(seed + source_index) if seed is not None else random.Random()

    length = len(text.strip())
    category = categorize_length(length, q25, q50)

    recettes_cat = recettes.get(category, [])
    if not recettes_cat:
        return {"error": f"Aucune recette pour {category}", "anchor_text": text, "source_index": source_index}

    chosen_recette = weighted_choice(recettes_cat, rng)
    strat_key = chosen_recette.get("strategie")
    strat = strategies.get(strat_key)
    if not strat:
        return {"error": f"Stratégie absente {strat_key}", "anchor_text": text, "source_index": source_index}

    style_instruction = None
    style_key = None
    if strat.get("requires_style", False):
        if not styles:
            return {"error": "Style requis mais aucun style défini", "anchor_text": text, "source_index": source_index}
        style_obj = rng.choice(styles)
        style_instruction = style_obj.get("instruction", "")
        style_key = style_obj.get("key", None)

    word_count = pick_word_count_if_any(strat, rng)
    prompt = format_strategy_prompt(
        strat, anchor_text=text, style_instruction=style_instruction, word_count=word_count
    )

    model, provider = pick_llm_rotation(llm_cfg, rng)

    generated = generate_with_openrouter(
        model=model,
        provider=provider,
        prompt=prompt,
        temperature=0.7,
        max_tokens=max_tokens,
        sleep_ms=sleep_ms,
    )

    if generated is None:
        return {"error": "api_error_or_rate_limit", "anchor_text": text, "source_index": source_index}

    uid = make_uid(dataset_name, split, text)

    rec: Dict[str, Any] = {
        "uid": uid,
        "source_dataset": dataset_name,
        "source_split": split,
        "source_index": source_index,
        "anchor_text": text,
        "anchor_length": length,
        "category": category,
        "strategy_key": strat_key,
        "style_key": style_key,
        "word_count": word_count,
        "model": model,
        "provider": provider,
        "generated_text": generated,
        "timestamp_utc": iso_now(),
    }

    return rec

# -------------------------------
# Main
# -------------------------------

def main():
    parser = argparse.ArgumentParser(description="Génère des 'positives' en parallèle et push vers HF Hub par chunks")
    parser.add_argument("--dataset", default="matheoqtb/ancre-cleaned")
    parser.add_argument("--split", default=None)
    parser.add_argument("--column", default="positive")
    parser.add_argument("--yaml", default="positive_strategies.yaml")
    parser.add_argument("--out-repo", default="matheoqtb/ancre_querry2")
    parser.add_argument("--chunk-size", type=int, default=10000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sleep-ms", type=int, default=0, help="Pause entre requêtes par thread")
    parser.add_argument("--resume-from-hub", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--concurrency", type=int, default=80, help="Nombre de threads pour les requêtes")
    parser.add_argument("--no-progress", action="store_true", help="Désactive la barre de progression (utile pour logs CI)")
    args = parser.parse_args()

    # RNG global uniquement pour opérations non déterministes générales
    if args.seed is not None:
        random.seed(args.seed)

    # Chargement YAML
    cfg = load_yaml_config(args.yaml)
    styles = cfg.get("styles_pour_positifs", [])
    strategies = cfg.get("strategies", {})
    recettes = cfg.get("recettes_par_categorie", {})
    llm_cfg = cfg.get("llm", {})

    # Dataset source
    ds_builder = load_dataset(args.dataset)
    if args.split is None:
        possible_splits = list(ds_builder.keys())
        split = "train" if "train" in possible_splits else possible_splits[0]
        dataset = ds_builder[split]
    else:
        split = args.split
        dataset = load_dataset(args.dataset, split=split)

    if args.column not in dataset.column_names:
        raise ValueError(f"Colonne '{args.column}' introuvable. Colonnes: {dataset.column_names}")

    raw_texts = dataset[args.column]
    indices = [i for i, t in enumerate(raw_texts) if isinstance(t, str) and t.strip()]
    texts = [raw_texts[i] for i in indices]

    n_total = len(texts) if args.limit is None else min(args.limit, len(texts))
    if n_total == 0:
        print("Aucun texte à traiter", file=sys.stderr)
        return

    # Seuils
    q25, q50 = compute_length_thresholds(texts)

    # Reprise
    existing_uids = set()
    if args.resume_from_hub:
        existing_uids = load_existing_uids(args.out_repo)
        print(f"[resume] UIDs existants: {len(existing_uids)}")

    # Préparation des tâches (filtrage des doublons en amont)
    tasks: List[Tuple[int, int, str]] = []  # (idx_global, source_index, text)
    for i, text in enumerate(texts[:n_total]):
        src_idx = indices[i]
        uid = make_uid(args.dataset, split, text)
        if args.resume_from_hub and uid in existing_uids:
            continue
        tasks.append((i, src_idx, text))

    if not tasks:
        print("[info] Rien à faire après reprise", file=sys.stderr)
        return

    # Init chunk id global
    global _next_chunk_id
    _next_chunk_id = detect_next_chunk_id(args.out_repo)

    # Gestion SIGINT pour flush final
    def handle_sigint(signum, frame):
        sys.stderr.write("\n[signal] Ctrl-C détecté, flush en cours...\n")
        flush_all(args.out_repo, private=args.private)
        sys.exit(1)

    signal.signal(signal.SIGINT, handle_sigint)

    # Exécution parallèle
    from concurrent.futures import ThreadPoolExecutor, as_completed

    max_tokens = 4000  # plafond par défaut
    processed = 0

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        pbar = None
        show_progress = (tqdm is not None) and (not args.no_progress)

        futures = []
        for (idx_global, src_idx, text) in tasks:
            if args.dry_run:
                # Fonction locale pour éviter late-binding sur text/src_idx
                def _dry_record(_text=text, _src_idx=src_idx):
                    length = len(_text.strip())
                    category = categorize_length(length, q25, q50)
                    return {
                        "uid": make_uid(args.dataset, split, _text),
                        "source_dataset": args.dataset,
                        "source_split": split,
                        "source_index": _src_idx,
                        "anchor_text": _text,
                        "anchor_length": length,
                        "category": category,
                        "strategy_key": "dry_run",
                        "style_key": None,
                        "word_count": None,
                        "model": "dry_run_model",
                        "provider": None,
                        "generated_text": f"[DRY-RUN] category={category}",
                        "timestamp_utc": iso_now(),
                    }
                futures.append(ex.submit(_dry_record))
            else:
                futures.append(
                    ex.submit(
                        make_record,
                        idx_global,
                        src_idx,
                        text,
                        args.dataset,
                        split,
                        q25,
                        q50,
                        styles,
                        strategies,
                        recettes,
                        llm_cfg,
                        max_tokens,
                        args.sleep_ms,
                        args.seed,
                    )
                )

        if show_progress:
            pbar = tqdm(total=len(futures), desc="Generating", unit="item")

        for fut in as_completed(futures):
            rec = fut.result()
            if pbar is not None:
                pbar.update(1)
                try:
                    pbar.set_postfix_str(f"buffer={len(buffer_records)} pushes={get_push_counter()}")
                except Exception:
                    pass
            if not isinstance(rec, dict):
                continue
            with buffer_lock:
                buffer_records.append(rec)
                processed += 1
                current_len = len(buffer_records)
            if current_len >= args.chunk_size:
                flush_if_needed(args.out_repo, args.chunk_size, private=args.private)

    # Flush final
    if pbar is not None:
        pbar.close()
    flush_all(args.out_repo, private=args.private)
    print(f"[done] Traités: {processed} | poussés par paquets de {args.chunk_size} | concurrence={args.concurrency}")

if __name__ == "__main__":
    main()
