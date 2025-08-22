#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation ligne-par-ligne des paires POSITIVES (anchor_text, generated_text).

- Lit un repo HF en entrée (ex: matheoqtb/ancre_querry2)
- Pour chaque ligne, envoie (anchor, candidate) à un LLM "juge"
- Le juge répond STRICTEMENT "OK" ou "NON"
- On pousse SEULEMENT les lignes "OK" dans un NOUVEAU repo (ex: ..._validated)
- Parallélisé, reprenable, push par chunks, barre de progression

Exemple:
export OPENROUTER_API_KEY="sk-..."
huggingface-cli login  # ou export HF_TOKEN=...

python validate_positive_pairs.py \
  --in-repo matheoqtb/ancre_querry2 \
  --out-repo matheoqtb/ancre_querry2_validated \
  --prompts judge_row_prompts_min.yaml \
  --judge-model "google/gemini-2.5-pro" \
  --judge-provider "google-vertex" \
  --chunk-size 5000 \
  --concurrency 16 \
  --resume-from-hub
"""
import argparse
import hashlib
import os
import sys
import time
import json
import random
import signal
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import yaml
from datasets import Dataset, load_dataset

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# =========================
# Utils
# =========================

def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def pair_uid(anchor: str, candidate: str) -> str:
    h = hashlib.sha256()
    h.update((anchor or "").encode("utf-8"))
    h.update(b"||")
    h.update((candidate or "").encode("utf-8"))
    return h.hexdigest()

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def detect_next_chunk_id(repo_id: str) -> int:
    try:
        ds_dict = load_dataset(repo_id)
        max_id = 0
        for k in list(ds_dict.keys()):
            if k.startswith("chunk_"):
                try:
                    cid = int(k.split("_")[1])
                    if cid > max_id:
                        max_id = cid
                except Exception:
                    pass
        return max_id + 1
    except Exception:
        return 1

def load_existing_pair_uids(repo_id: str) -> set:
    existing = set()
    try:
        ds_dict = load_dataset(repo_id, streaming=True)
        for split in ds_dict.keys():
            for row in ds_dict[split]:
                pu = row.get("pair_uid")
                if isinstance(pu, str):
                    existing.add(pu)
    except Exception as e:
        sys.stderr.write(f"[resume] Lecture pair_uids impossible: {e}\n")
    return existing

# =========================
# OpenRouter client
# =========================

_client = None
_client_lock = threading.Lock()

def get_client() -> Any:
    global _client
    if OpenAI is None:
        raise RuntimeError("openai>=1.0 requis")
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                    default_headers={
                        "X-Title": "Positive Pair Validator (OK/NON)",
                        "HTTP-Referer": "https://local.test",
                    },
                )
    return _client

def call_llm_label(
    model: str,
    provider: Optional[str],
    system: str,
    user: str,
    max_tokens: int = 64,
    temperature: float = 0.0,
    timeout: float = 60.0,
    max_retries: int = 5,
    backoff: float = 1.6,
) -> Optional[str]:
    """
    Appel texte simple. On attend que le modèle rende STRICTEMENT 'OK' ou 'NON'.
    Tolère un peu de verbiage : on extrait la 1ère occurrence sur une ligne seule.
    """
    client = get_client()
    extra_body = {}
    if provider:
        extra_body["provider"] = {"order": [provider], "allow_fallbacks": False}

    wanted = {"OK", "NON", "KO", "NO", "PASS", "FAIL"}
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                extra_body=extra_body,
            )
            ch = getattr(resp, "choices", None)
            txt = ""
            if ch and ch[0].message:
                txt = (getattr(ch[0].message, "content", None) or "") or ""
                if not txt.strip():
                    txt = getattr(ch[0].message, "reasoning", "") or ""
            if not txt:
                raise RuntimeError("empty_content")

            # Normalisation stricte
            # On prend la 1ère ligne non vide, upper, strip ponctuation.
            for line in txt.splitlines():
                t = line.strip().upper()
                # enlever guillemets/points éventuels
                t = t.strip(" \t'\"`.,;:!()[]{}")
                if t in wanted:
                    # Mappe vers OK/NON
                    if t in {"OK", "PASS"}:
                        return "OK"
                    else:
                        return "NON"
            # sinon, tente extraction mots isolés
            t = txt.strip().upper()
            for token in wanted:
                if token in t:
                    return "OK" if token in {"OK", "PASS"} else "NON"
            # pas trouvé
            return None

        except Exception as e:
            msg = str(e)
            retryable = any(tok in msg.lower() for tok in ("429", "502", "503", "504", "temporar", "timeout", "bad gateway"))
            if attempt < max_retries - 1 and retryable:
                time.sleep((backoff ** attempt) + random.random() * 0.25)
                continue
            sys.stderr.write(f"[judge][API Error] {type(e).__name__}: {msg[:300]}\n")
            return None

# =========================
# Push buffer
# =========================

buffer_lock = threading.Lock()
buffer_records: List[Dict[str, Any]] = []
_next_chunk_id = None
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
        with open(fallback, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        sys.stderr.write(f"[push] Dump local: {fallback}\n")
        with push_counter_lock:
            push_counter += 1

def flush_if_needed(out_repo: str, chunk_size: int, private: Optional[bool]) -> None:
    global buffer_records, _next_chunk_id
    to_push = None
    with buffer_lock:
        if len(buffer_records) >= chunk_size:
            to_push = buffer_records
            buffer_records = []
            chunk_id = _next_chunk_id
            _next_chunk_id += 1
        else:
            return
    push_chunk(out_repo, to_push, chunk_id, private=private)

def flush_all(out_repo: str, private: Optional[bool]) -> None:
    global buffer_records, _next_chunk_id
    with buffer_lock:
        to_push = buffer_records
        buffer_records = []
        if not to_push:
            return
        chunk_id = _next_chunk_id
        _next_chunk_id += 1
    push_chunk(out_repo, to_push, chunk_id, private=private)

# =========================
# Juge 1 paire
# =========================

def judge_one_pair(
    row: Dict[str, Any],
    prompts: Dict[str, Any],
    judge_model: str,
    judge_provider: Optional[str],
    judge_max_tokens: int,
) -> Tuple[bool, Dict[str, Any]]:
    anchor = row.get("anchor_text") or ""
    candidate = row.get("generated_text") or ""
    if not anchor or not candidate:
        return False, {}

    jcfg = prompts["judge_row"]["positive"]
    sys_prompt = jcfg["system"]
    user_prompt = jcfg["user_template"].format(
        anchor_text=anchor,
        candidate_text=candidate,
        style=row.get("style_key") or "unknown",
        strategy=row.get("strategy_key") or "unknown",
        word_count=str(row.get("word_count")) if row.get("word_count") is not None else "N/A",
    )

    label = call_llm_label(
        model=judge_model,
        provider=judge_provider,
        system=sys_prompt,
        user=user_prompt,
        max_tokens=judge_max_tokens,
        temperature=0.0,
    )
    ok = (label == "OK")

    if not ok:
        return False, {}

    # On garde la ligne d'origine + méta minimale + tag pair_uid + label
    out = {
        "pair_uid": pair_uid(anchor, candidate),
        "judge_label": "OK",
        # champs originaux utiles
        "uid": row.get("uid"),
        "anchor_text": anchor,
        "generated_text": candidate,
        "category": row.get("category"),
        "strategy_key": row.get("strategy_key"),
        "style_key": row.get("style_key"),
        "word_count": row.get("word_count"),
        "model": row.get("model"),
        "provider": row.get("provider"),
        "source_dataset": row.get("source_dataset"),
        "source_split": row.get("source_split"),
        "source_index": row.get("source_index"),
        "anchor_length": row.get("anchor_length"),
        "timestamp_utc": row.get("timestamp_utc"),
        "timestamp_validated_utc": iso_now(),
    }
    return True, out

# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser(description="Valide des paires positives et pousse les 'OK' vers un nouveau repo.")
    ap.add_argument("--in-repo", required=True)
    ap.add_argument("--out-repo", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--judge-model", default="google/gemini-2.5-pro")
    ap.add_argument("--judge-provider", default=None)
    ap.add_argument("--judge-max-tokens", type=int, default=16)
    ap.add_argument("--chunk-size", type=int, default=10000)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--resume-from-hub", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-progress", action="store_true")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    # Prompts
    prompts = load_yaml(args.prompts)

    # Reprise (déjà validés)
    existing_out = set()
    if args.resume_from_hub:
        existing_out = load_existing_pair_uids(args.out_repo)
        print(f"[resume] pairs déjà poussées: {len(existing_out)}")

    # Lecture streaming
    ds_stream = load_dataset(args.in_repo, streaming=True)
    splits = list(ds_stream.keys())

    # Chunk id init
    global _next_chunk_id
    _next_chunk_id = detect_next_chunk_id(args.out_repo)

    # SIGINT => flush
    def on_sigint(signum, frame):
        sys.stderr.write("\n[signal] Ctrl-C: flush en cours...\n")
        flush_all(args.out_repo, private=args.private)
        sys.exit(1)
    signal.signal(signal.SIGINT, on_sigint)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    total_seen = 0
    total_ok = 0
    total_skipped = 0
    total_errors = 0

    show_progress = (tqdm is not None) and (not args.no_progress)
    pbar = tqdm(total=args.limit or 0, desc="Validating", unit="item") if show_progress and args.limit \
           else (tqdm(desc="Validating", unit="item") if show_progress else None)

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        futures = []

        def submit_row(row):
            return ex.submit(
                judge_one_pair,
                row,
                prompts,
                args.judge_model,
                args.judge_provider,
                args.judge_max_tokens,
            )

        for sp in splits:
            for row in ds_stream[sp]:
                if args.limit and total_seen >= args.limit:
                    break

                if not isinstance(row.get("anchor_text"), str) or not isinstance(row.get("generated_text"), str):
                    total_skipped += 1
                    if pbar: pbar.update(1)
                    continue

                pu = pair_uid(row.get("anchor_text",""), row.get("generated_text",""))
                if args.resume_from_hub and pu in existing_out:
                    total_skipped += 1
                    if pbar is not None: pbar.update(1)
                    continue

                futures.append(submit_row(row))
                total_seen += 1

                # vidage progressif pour ne pas saturer la RAM
                if len(futures) >= args.concurrency * 4:
                    for fut in as_completed(futures[:args.concurrency]):
                        try:
                            ok, rec = fut.result()
                        except Exception as e:
                            total_errors += 1
                            sys.stderr.write(f"[judge][ERR] {type(e).__name__}: {str(e)[:200]}\n")
                            if pbar: pbar.update(1)
                            continue
                        if ok:
                            with buffer_lock:
                                buffer_records.append(rec)
                                total_ok += 1
                                if pbar is not None:
                                    pbar.update(1)
                                    try:
                                        pbar.set_postfix_str(f"ok={total_ok} skip={total_skipped} buf={len(buffer_records)} pushes={get_push_counter()}")
                                    except Exception:
                                        pass
                            flush_if_needed(args.out_repo, args.chunk_size, private=args.private)
                        else:
                            total_skipped += 1
                            if pbar is not None: pbar.update(1)
                    futures = futures[args.concurrency:]

            if args.limit and total_seen >= args.limit:
                break

        # Drain final
        from concurrent.futures import as_completed
        for fut in as_completed(futures):
            try:
                ok, rec = fut.result()
            except Exception as e:
                total_errors += 1
                sys.stderr.write(f"[judge][ERR] {type(e).__name__}: {str(e)[:200]}\n")
                if pbar: pbar.update(1)
                continue
            if ok:
                with buffer_lock:
                    buffer_records.append(rec)
                    total_ok += 1
                    if pbar:
                        pbar.update(1)
                        try:
                            pbar.set_postfix_str(f"ok={total_ok} skip={total_skipped} buf={len(buffer_records)} pushes={get_push_counter()}")
                        except Exception:
                            pass
                flush_if_needed(args.out_repo, args.chunk_size, private=args.private)
            else:
                total_skipped += 1
                if pbar: pbar.update(1)

    if pbar:
        pbar.close()
    flush_all(args.out_repo, private=args.private)
    print(f"[done] vus={total_seen} | OK={total_ok} | SKIP={total_skipped} | erreurs={total_errors} | pushes={get_push_counter()}")
    
if __name__ == "__main__":
    main()
