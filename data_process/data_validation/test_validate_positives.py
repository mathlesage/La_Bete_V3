#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test rapide du juge (OK/NON) sur un petit échantillon.

- Ne push rien sur le Hub.
- Lit un repo HF (ex: matheoqtb/ancre_querry2) et prend N lignes valides.
- Appelle le LLM juge avec le prompt YAML (OK/NON strict).
- Affiche un tableau compact + un résumé.

Exemple:
export OPENROUTER_API_KEY="sk-..."
python test_validate_positives.py \
  --in-repo matheoqtb/ancre_querry2 \
  --prompts judge_row_prompts_min.yaml \
  --n 10 \
  --judge-model "google/gemini-2.5-flash-lite" \
  --judge-provider "google-vertex" \
  --concurrency 8
"""
import argparse
import os
import sys
import json
import random
import threading
from typing import Any, Dict, List, Optional, Tuple

import yaml
from datasets import load_dataset

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# =========================
# Prompts & utils
# =========================

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def trunc(s: str, n: int = 140) -> str:
    if not isinstance(s, str):
        return ""
    s = " ".join(s.split())
    return s if len(s) <= n else (s[: n - 1] + "…")


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
                        "X-Title": "Positive Pair QuickTest",
                        "HTTP-Referer": "https://local.test",
                    },
                )
    return _client

def call_llm_label(
    model: str,
    provider: Optional[str],
    system: str,
    user: str,
    max_tokens: int = 32,
    temperature: float = 0.0,
    timeout: float = 45.0,
) -> Optional[str]:
    """
    Attend STRICTEMENT 'OK' ou 'NON'. Normalise aussi PASS/FAIL, KO/NO si ça dérape.
    """
    client = get_client()
    extra_body = {}
    if provider:
        extra_body["provider"] = {"order": [provider], "allow_fallbacks": False}

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            extra_body=extra_body,
        )
    except Exception as e:
        sys.stderr.write(f"[judge][API Error] {type(e).__name__}: {str(e)[:300]}\n")
        return None

    ch = getattr(resp, "choices", None)
    txt = ""
    if ch and ch[0].message:
        txt = (getattr(ch[0].message, "content", None) or "") or ""
        if not txt.strip():
            txt = getattr(ch[0].message, "reasoning", "") or ""

    if not txt:
        return None

    wanted = {"OK", "NON", "KO", "NO", "PASS", "FAIL"}
    # Cherche une ligne propre d'abord
    for line in txt.splitlines():
        t = line.strip().upper().strip(" \t'\"`.,;:!()[]{}")
        if t in wanted:
            return "OK" if t in {"OK", "PASS"} else "NON"
    # Sinon, cherche tokens dans tout le texte
    T = txt.strip().upper()
    for tok in wanted:
        if tok in T:
            return "OK" if tok in {"OK", "PASS"} else "NON"
    return None


# =========================
# Évaluation d'une ligne
# =========================

def judge_row(row: Dict[str, Any], jcfg: Dict[str, Any], judge_model: str, judge_provider: Optional[str]) -> str:
    anchor = row.get("anchor_text") or ""
    cand = row.get("generated_text") or ""
    if not anchor or not cand:
        return "NON"

    sys_prompt = jcfg["system"]
    user_prompt = jcfg["user_template"].format(
        anchor_text=anchor,
        candidate_text=cand,
        style=row.get("style_key") or "unknown",
        strategy=row.get("strategy_key") or "unknown",
        word_count=str(row.get("word_count")) if row.get("word_count") is not None else "N/A",
    )
    label = call_llm_label(
        model=judge_model,
        provider=judge_provider,
        system=sys_prompt,
        user=user_prompt,
    )
    return label or "NON"


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser(description="Test rapide du juge (OK/NON) sur N exemples.")
    ap.add_argument("--in-repo", required=True, help="Repo HF d'entrée (ex: matheoqtb/ancre_querry2)")
    ap.add_argument("--prompts", default="judge_row_prompts_min.yaml", help="Fichier YAML des prompts du juge")
    ap.add_argument("--judge-model", default="google/gemini-2.5-pro")
    ap.add_argument("--judge-provider", default=None)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--sample-mode", choices=["head", "random"], default="head", help="head = premiers N; random = N aléatoires")
    ap.add_argument("--max-preview", type=int, default=120, help="Longueur max d'aperçu imprimé")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    prompts = load_yaml(args.prompts)
    jcfg = prompts["judge_row"]["positive"]

    # Récupération des N lignes
    ds_stream = load_dataset(args.in_repo, streaming=True)
    rows: List[Dict[str, Any]] = []

    if args.sample_mode == "head":
        for sp in ds_stream.keys():
            for r in ds_stream[sp]:
                if isinstance(r.get("anchor_text"), str) and isinstance(r.get("generated_text"), str):
                    rows.append(r)
                    if len(rows) >= args.n:
                        break
            if len(rows) >= args.n:
                break
    else:
        # réservoir sampling aléatoire sur l'ensemble
        k = args.n
        count = 0
        reservoir = []
        for sp in ds_stream.keys():
            for r in ds_stream[sp]:
                if not (isinstance(r.get("anchor_text"), str) and isinstance(r.get("generated_text"), str)):
                    continue
                count += 1
                if len(reservoir) < k:
                    reservoir.append(r)
                else:
                    j = random.randint(1, count)
                    if j <= k:
                        reservoir[j - 1] = r
        rows = reservoir

    if not rows:
        print("[info] Aucun exemple valide trouvé.")
        sys.exit(0)

    # Évalue en parallèle
    from concurrent.futures import ThreadPoolExecutor, as_completed
    show_pbar = (tqdm is not None)
    pbar = tqdm(total=len(rows), desc="Jugement", unit="item") if show_pbar else None

    results: List[Tuple[str, Dict[str, Any]]] = []  # (label, row)

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        futs = [ex.submit(judge_row, r, jcfg, args.judge_model, args.judge_provider) for r in rows]
        for fut, r in zip(as_completed(futs), rows):
            try:
                label = fut.result()
            except Exception as e:
                label = "NON"
                sys.stderr.write(f"[judge][ERR] {type(e).__name__}: {str(e)[:200]}\n")
            results.append((label, r))
            if pbar: pbar.update(1)

    if pbar:
        pbar.close()

    # Affichage
    ok_cnt = 0
    print("\n=== APERÇU DES 10 (ou N) JUGEMENTS ===")
    print("idx | label | strategy | style | anchor_preview | candidate_preview")
    print("-" * 120)
    for i, (label, r) in enumerate(results):
        if label == "OK":
            ok_cnt += 1
        print(f"{i:02d} | {label:3s}  | {str(r.get('strategy_key') or ''):24s} | "
              f"{str(r.get('style_key') or ''):12s} | "
              f"{trunc(r.get('anchor_text',''), args.max_preview)} | "
              f"{trunc(r.get('generated_text',''), args.max_preview)}")

    print("\n--- Résumé ---")
    print(f"Total évalués : {len(results)}")
    print(f"OK            : {ok_cnt}")
    print(f"NON           : {len(results) - ok_cnt}")

    # Optionnel : dump local pour inspection rapide
    preview = [{
        "label": lab,
        "strategy_key": r.get("strategy_key"),
        "style_key": r.get("style_key"),
        "anchor_text": r.get("anchor_text"),
        "generated_text": r.get("generated_text"),
    } for (lab, r) in results]
    with open("test_judge_preview.jsonl", "w", encoding="utf-8") as f:
        for row in preview:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print('Aperçu écrit dans "test_judge_preview.jsonl"')
    

if __name__ == "__main__":
    main()
