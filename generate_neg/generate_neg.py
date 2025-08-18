#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Génère 2 "hard negatives" par item (1 depuis generated_text, 1 depuis anchor_text)
avec contrainte de similarité cosinus < seuil, et pousse en chunks vers le Hub.
- Reprise possible (saute les uid déjà dotés de négatifs)
- Flush sur Ctrl-C (rien n'est perdu)
- Progress bar + heartbeat

Exemple :
python generate_hard_negatives.py \
  --in-repo matheoqtb/ancre_querry_cos_filtered_train \
  --out-repo matheoqtb/ancre_querry_cos_filtered_train \
  --yaml hard_negatives.yaml \
  --encoder sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  --sim-max 0.6 \
  --max-tries 4 \
  --chunk-size 5000 \
  --concurrency 8 \
  --resume-from-hub \
  --fallback-on-error
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

import yaml
import numpy as np
from datasets import Dataset, load_dataset

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# sentence-transformers pour embeddings
from sentence_transformers import SentenceTransformer

# -------------------------------
# Utils
# -------------------------------

def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-8
    return float(np.dot(a, b) / denom)

def pick_llm_rotation(llm_config: Dict[str, Any], rng: random.Random) -> Tuple[str, Optional[str]]:
    rotation = llm_config.get("rotation", [])
    if not rotation:
        raise ValueError("YAML: llm.rotation vide")
    choice = rotation[rng.randrange(len(rotation))]
    return choice.get("model"), choice.get("provider")

def build_prompt(template: str, **kwargs) -> str:
    # Remplacement simple {query}, {positive}
    out = template
    for k, v in kwargs.items():
        out = out.replace("{" + k + "}", v)
    return out

def generate_with_openrouter(model: str,
                             provider: Optional[str],
                             prompt: str,
                             system_prompt: Optional[str] = None,
                             temperature: float = 0.7,
                             max_tokens: int = 400,
                             timeout: Optional[float] = 45.0,
                             sleep_ms: int = 0,
                             x_title: str = "HardNegGenerator",
                             http_referer: str = "https://local.test") -> Optional[str]:
    system_prompt = system_prompt or (
        "Tu produis des variantes négatives plausibles en respectant strictement les consignes. "
        "Ne retourne que le texte final, sans guillemets ni notes."
    )
    extra_body = {}
    if provider:
        extra_body["provider"] = {"order": [provider], "allow_fallbacks": False}

    try:
        if OpenAI is None:
            raise RuntimeError("openai>=1.0 requis")
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
            timeout=timeout,
        )
        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)
        try:
            choices = getattr(resp, "choices", None)
            if not choices:
                sys.stderr.write(f"[API Error] Empty choices model={model} provider={provider} resp={getattr(resp, 'model_dump_json', lambda: str(resp))()[:400]}\n")
                return None
            content = getattr(choices[0].message, "content", None)
            if not content:
                sys.stderr.write(f"[API Error] Missing content model={model} provider={provider}\n")
                return None
            return content.strip()
        except Exception as e:
            sys.stderr.write(f"[API Error] Parse failure model={model} provider={provider}: {type(e).__name__}: {str(e)[:300]}\n")
            return None
    except Exception as e:
        sys.stderr.write(f"[API Error] {type(e).__name__}: {str(e)[:300]}\n")
        return None

# -------------------------------
# Push & resume
# -------------------------------

buffer_lock = threading.Lock()
buffer_records: List[Dict[str, Any]] = []
_next_chunk_id = None
push_counter = 0
push_counter_lock = threading.Lock()

def get_push_counter() -> int:
    with push_counter_lock:
        return push_counter

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
            if cid is not None and cid > max_id:
                max_id = cid
        return max_id + 1
    except Exception:
        return 1

def load_done_uids(out_repo: str, require_negatives: bool = True) -> set:
    """
    Si require_negatives=True, ne considère 'fait' que si les champs négatifs existent.
    """
    done = set()
    try:
        ds_dict = load_dataset(out_repo, streaming=True)
        for split in ds_dict.keys():
            for row in ds_dict[split]:
                uid = row.get("uid")
                if not isinstance(uid, str):
                    continue
                if require_negatives:
                    n1 = row.get("negative_from_positive")
                    n2 = row.get("negative_from_anchor")
                    if isinstance(n1, str) and n1.strip() and isinstance(n2, str) and n2.strip():
                        done.add(uid)
                else:
                    done.add(uid)
    except Exception as e:
        sys.stderr.write(f"[resume] Lecture UIDs impossible: {e}\n")
    return done

def push_chunk(repo_id: str, records: List[Dict[str, Any]], chunk_id: int, private: Optional[bool]=None) -> None:
    global push_counter  # <-- UNE SEULE déclaration globale, tout en haut de la fonction
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
        with open(fallback, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        sys.stderr.write(f"[push] Dump local: {fallback}\n")
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

encoder_model: Optional[SentenceTransformer] = None

def make_uid(source_repo: str, split: str, source_index: int, fallback_text: Optional[str] = None) -> str:
    h = hashlib.sha256()
    h.update((source_repo or "").encode("utf-8"))
    h.update(b"|")
    h.update((split or "").encode("utf-8"))
    h.update(b"|")
    h.update(str(source_index).encode("utf-8"))
    if fallback_text:
        h.update(b"|")
        h.update(fallback_text.encode("utf-8"))
    return h.hexdigest()

def gen_one_negative(base_text: str,
                     prompt_templates: List[Dict[str, Any]],
                     llm_cfg: Dict[str, Any],
                     rng: random.Random,
                     retries: int,
                     retry_sleep_ms: int,
                     request_timeout: float,
                     fallback_on_error: bool,
                     max_tokens: int = 300) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Retourne (neg_text, used_prompt_name, used_model_provider) ou (None, ..) si échec.
    """
    if not prompt_templates:
        return None, None, None
    # Choisir un prompt
    prompt_obj = rng.choice(prompt_templates)
    name = prompt_obj.get("name")
    template = prompt_obj.get("prompt", "")
    prompt = build_prompt(template, query=base_text, positive=base_text)

    # Choisir modèle/provider
    model, provider = pick_llm_rotation(llm_cfg, rng)

    attempt = 0
    text = None
    while attempt <= retries and not text:
        attempt += 1
        text = generate_with_openrouter(
            model=model,
            provider=provider,
            prompt=prompt,
            temperature=0.7,
            max_tokens=max_tokens,
            timeout=request_timeout,
        )
        if text:
            break
        if attempt <= retries:
            if fallback_on_error:
                try:
                    model, provider = pick_llm_rotation(llm_cfg, rng)
                except Exception:
                    pass
            if retry_sleep_ms > 0:
                time.sleep(retry_sleep_ms / 1000.0)
    if text:
        return text, name, f"{model}|{provider or ''}"
    return None, name, f"{model}|{provider or ''}"

def process_item(idx_global: int,
                 source_index: int,
                 row: Dict[str, Any],
                 source_repo: str,
                 split: str,
                 prompts_cfg: Dict[str, Any],
                 llm_cfg: Dict[str, Any],
                 sim_max: float,
                 sim_min: float,
                 max_tries: int,
                 retries: int,
                 retry_sleep_ms: int,
                 request_timeout: float,
                 fallback_on_error: bool,
                 seed: Optional[int]) -> Dict[str, Any]:

    rng = random.Random(seed + source_index if seed is not None else None)

    anchor = row.get("anchor_text") or row.get("query") or row.get("question")
    positive = row.get("generated_text") or row.get("positive") or row.get("answer")

    if not isinstance(positive, str) or not positive.strip():
        return {"error": "missing_generated_text", "source_index": source_index}
    if not isinstance(anchor, str) or not anchor.strip():
        anchor = positive  # fallback raisonnable

    uid = row.get("uid") or make_uid(source_repo, split, source_index, fallback_text=anchor[:128])

    # Embedding du positif
    pos_vec = encoder_model.encode([positive], normalize_embeddings=True)[0]

    # 1) Négatif depuis positive_based
    neg1 = None
    name1 = None
    mp1 = None
    tries = 0
    pos = positive
    while tries < max_tries:
        tries += 1
        neg1_try, name_try, mp_try = gen_one_negative(
            base_text=pos,
            prompt_templates=prompts_cfg.get("positive_based", []),
            llm_cfg=llm_cfg,
            rng=rng,
            retries=retries,
            retry_sleep_ms=retry_sleep_ms,
            request_timeout=request_timeout,
            fallback_on_error=fallback_on_error,
            max_tokens=300
        )
        if not neg1_try or not isinstance(neg1_try, str) or not neg1_try.strip():
            continue
        neg_vec = encoder_model.encode([neg1_try], normalize_embeddings=True)[0]
        cs = float(np.dot(pos_vec, neg_vec))  # déjà normalisé
        if (sim_min <= cs) and (cs < sim_max):
            neg1 = neg1_try
            name1 = name_try
            mp1 = mp_try
            cos1 = cs
            break
    if neg1 is None:
        cos1 = None

    # 2) Négatif depuis query_based (anchor)
    neg2 = None
    name2 = None
    mp2 = None
    tries = 0
    while tries < max_tries:
        tries += 1
        neg2_try, name_try, mp_try = gen_one_negative(
            base_text=anchor,
            prompt_templates=prompts_cfg.get("query_based", []),
            llm_cfg=llm_cfg,
            rng=rng,
            retries=retries,
            retry_sleep_ms=retry_sleep_ms,
            request_timeout=request_timeout,
            fallback_on_error=fallback_on_error,
            max_tokens=300
        )
        if not neg2_try or not isinstance(neg2_try, str) or not neg2_try.strip():
            continue
        neg_vec = encoder_model.encode([neg2_try], normalize_embeddings=True)[0]
        cs = float(np.dot(pos_vec, neg_vec))
        if (sim_min <= cs) and (cs < sim_max):
            neg2 = neg2_try
            name2 = name_try
            mp2 = mp_try
            cos2 = cs
            break
    if neg2 is None:
        cos2 = None

    rec = {
        "uid": uid,
        "source_repo": source_repo,
        "source_split": split,
        "source_index": source_index,
        "anchor_text": anchor,
        "generated_text": positive,
        "negative_from_positive": neg1,
        "negative_from_anchor": neg2,
        "neg_from_positive_prompt": name1,
        "neg_from_anchor_prompt": name2,
        "neg_from_positive_model": mp1,
        "neg_from_anchor_model": mp2,
        "cos_pos__neg_from_positive": cos1,
        "cos_pos__neg_from_anchor": cos2,
        "timestamp_utc": iso_now(),
    }
    if neg1 is None:
        rec["error_neg_from_positive"] = "similarity_not_below_threshold_or_api_error"
    if neg2 is None:
        rec["error_neg_from_anchor"] = "similarity_not_below_threshold_or_api_error"
    return rec

# -------------------------------
# Heartbeat
# -------------------------------

heartbeat_stop = threading.Event()

def heartbeat(total_tasks_func, processed_func, buffer_len_func, pushes_func, interval_sec: int):
    while not heartbeat_stop.wait(timeout=interval_sec):
        try:
            total = total_tasks_func()
            processed = processed_func()
            buf = buffer_len_func()
            pushes = pushes_func()
            sys.stderr.write(f"[hb] processed={processed}/{total} buffer={buf} pushes={pushes}\n")
        except Exception:
            pass

# -------------------------------
# Main
# -------------------------------

def main():
    ap = argparse.ArgumentParser(description="Génère 2 hard negatives par item et pousse sur le Hub")
    ap.add_argument("--in-repo", default="matheoqtb/ancre_querry_cos_filtered_train")
    ap.add_argument("--in-split", default=None)
    ap.add_argument("--out-repo", default="matheoqtb/ancre_querry_cos_filtered_train")
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--encoder", default="Lajavaness/bilingual-embedding-large")
    ap.add_argument("--sim-max", type=float, default=0.6, help="Seuil max de similarité (strictement <)")
    ap.add_argument("--sim-min", type=float, default=0.0, help="Seuil min (>=) si voulu, 0 pour désactiver")
    ap.add_argument("--max-tries", type=int, default=4, help="Nb d'essais pour trouver un négatif sous le seuil")
    ap.add_argument("--chunk-size", type=int, default=5000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=80)
    ap.add_argument("--retries", type=int, default=2, help="Retries API par tentative")
    ap.add_argument("--retry-sleep-ms", type=int, default=500)
    ap.add_argument("--request-timeout", type=float, default=45.0)
    ap.add_argument("--fallback-on-error", action="store_true")
    ap.add_argument("--resume-from-hub", action="store_true")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--sleep-ms", type=int, default=0, help="Pause globale après chaque call (ms)")
    ap.add_argument("--heartbeat-sec", type=int, default=30)
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    cfg = load_yaml_config(args.yaml)
    prompts_cfg = cfg.get("prompts", {})
    llm_cfg = cfg.get("llm", {})

    # Dataset source
    ds_builder = load_dataset(args.in_repo)  
    if args.in_split is None:
        possible = list(ds_builder.keys())
        split = "train" if "train" in possible else possible[0]
        dataset = ds_builder[split]
    else:
        split = args.in_split
        dataset = load_dataset(args.in_repo, split=split)

    # Détermine lignes
    n_rows = len(dataset)
    idxs = list(range(n_rows))
    if args.limit is not None:
        idxs = idxs[: args.limit]

    # Encodage
    global encoder_model
    encoder_model = SentenceTransformer(args.encoder)

    # Reprise
    done_uids = set()
    if args.resume_from_hub:
        done_uids = load_done_uids(args.out_repo, require_negatives=True)
        print(f"[resume] UIDs déjà avec négatifs: {len(done_uids)}")

    # Chunk id
    global _next_chunk_id
    _next_chunk_id = detect_next_chunk_id(args.out_repo)

    # SIGINT -> flush
    def handle_sigint(signum, frame):
        sys.stderr.write("\n[signal] Ctrl-C détecté, flush en cours...\n")
        flush_all(args.out_repo, private=args.private)
        heartbeat_stop.set()
        sys.exit(1)

    signal.signal(signal.SIGINT, handle_sigint)

    # Préparer tâches (filtrer déjà traités)
    tasks: List[int] = []
    for i in idxs:
        row = dataset[i]
        uid = row.get("uid")
        if isinstance(uid, str) and uid in done_uids:
            continue
        tasks.append(i)
    total_tasks = len(tasks)
    if total_tasks == 0:
        print("[info] Rien à faire (tout a déjà des négatifs)")
        return

    # Progression
    pbar = None
    show_progress = (tqdm is not None) and (not args.no_progress)

    processed = 0

    # Heartbeat
    hb_thread = None
    if args.heartbeat_sec and args.heartbeat_sec > 0:
        def total_tasks_fn(): return total_tasks
        def processed_fn(): return processed
        def buffer_len_fn():
            with buffer_lock:
                return len(buffer_records)
        def pushes_fn(): return get_push_counter()
        hb_thread = threading.Thread(target=heartbeat, args=(total_tasks_fn, processed_fn, buffer_len_fn, pushes_fn, args.heartbeat_sec), daemon=True)
        hb_thread.start()

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        futures = []
        for i in tasks:
            row = dataset[i]
            futures.append(ex.submit(
                process_item,
                i,  # idx_global
                i,  # source_index
                row,
                args.in_repo,
                split,
                prompts_cfg,
                llm_cfg,
                float(args.sim_max),
                float(args.sim_min),
                int(args.max_tries),
                int(args.retries),
                int(args.retry_sleep_ms),
                float(args.request_timeout),
                bool(args.fallback_on_error),
                args.seed
            ))

        if show_progress:
            pbar = tqdm(total=len(futures), desc="Generating negatives", unit="item")

        for fut in as_completed(futures):
            rec = fut.result()
            with buffer_lock:
                buffer_records.append(rec)
                processed += 1
                current_len = len(buffer_records)
            if pbar is not None:
                pbar.update(1)
                try:
                    pbar.set_postfix_str(f"buffer={current_len} pushes={get_push_counter()}")
                except Exception:
                    pass
            if current_len >= args.chunk_size:
                flush_if_needed(args.out_repo, args.chunk_size, private=args.private)

    # Flush final
    if pbar is not None:
        pbar.close()
    flush_all(args.out_repo, private=args.private)
    if hb_thread is not None:
        heartbeat_stop.set()
        hb_thread.join(timeout=2.0)

    print(f"[done] Traités: {processed} | poussés par paquets de {args.chunk_size} | concurrence={args.concurrency}")

if __name__ == "__main__":
    main()
