#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iterative optimizer for HARD NEGATIVE prompts (query_based or positive_based).

- LLM-only judging: pas de cosinus, pas de filtres lexicaux.
- Le juge rend un verdict GLOBAL: {"batch_label":"PASS|FAIL","note_du_lot":0..1,"notes":[...],"idee_prompt":"..."}.
- En cas d'échec JSON, on réessaie (retries) puis fallback sur un autre modèle de juge.

Exemple:
python iterative_hardneg_optimizer.py \
  --yaml-negs hardneg_generation_prompts.yaml \
  --judge-prompts judge_prompts_hardneg_set.yaml \
  --dataset matheoqtb/ancre \
  --family positive_based \
  --neg-name context_swap \
  --source-col positive \
  --batch-size 50 \
  --max-iters 5 \
  --out-jsonl hardneg_opt_log.jsonl
"""
import argparse
import json
import os
import random
import signal
import sys
import time
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


# -------------------------------
# OpenRouter helpers
# -------------------------------

def make_client(title: str = "IterativeHardNegOpt") -> "OpenAI":
    if OpenAI is None:
        raise RuntimeError("openai>=1.0 required")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY missing")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={"X-Title": title, "HTTP-Referer": "https://local.script"},
    )


def call_llm_text(client: "OpenAI",
                  model: str,
                  provider: Optional[str],
                  system: str,
                  user: str,
                  max_tokens: int = 500,
                  temperature: float = 0.7,
                  timeout: Optional[float] = 60.0) -> Optional[str]:
    extra_body = {}
    if provider:
        extra_body["provider"] = {"order": [provider], "allow_fallbacks": False}
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            extra_body=extra_body,
        )
        ch = getattr(resp, "choices", None)
        if not ch:
            return None
        text = getattr(ch[0].message, "content", None) or ""
        if not text.strip():
            text = getattr(ch[0].message, "reasoning", "") or ""
        return text.strip() or None
    except Exception as e:
        sys.stderr.write(f"[call_llm_text][ERR] {type(e).__name__}: {str(e)[:300]}\n")
        return None


def call_llm_json(client: "OpenAI",
                  model: str,
                  provider: Optional[str],
                  system: str,
                  user: str,
                  max_tokens: int = 900,
                  temperature: float = 0.0,
                  timeout: Optional[float] = 120.0) -> Tuple[Optional[dict], str]:
    """
    Force JSON; best-effort extraction si le modèle ajoute du texte.
    """
    extra_body = {
        "response_format": {"type": "json_object"},
        "experimental": {"force_json": True}
    }
    if provider:
        extra_body["provider"] = {"order": [provider], "allow_fallbacks": False}
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":system},{"role":"user","content":user}],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            extra_body=extra_body,
        )
        text = ""
        ch = getattr(resp, "choices", None)
        if ch and ch[0].message:
            text = getattr(ch[0].message, "content", None) or ""
            if not text.strip():
                text = getattr(ch[0].message, "reasoning", "") or ""
        parsed = None
        if text:
            try:
                s = text.strip()
                first = s.find("{"); last = s.rfind("}")
                candidate = s[first:last+1] if (first != -1 and last != -1 and last > first) else s
                parsed = json.loads(candidate)
            except Exception:
                parsed = None
        return parsed, (text or "")
    except Exception as e:
        sys.stderr.write(f"[call_llm_json][ERR] {type(e).__name__}: {str(e)[:300]}\n")
        return None, ""


# -------------------------------
# Data helpers
# -------------------------------

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sample_bases(repo: str, split: Optional[str], column: str, n: int, seed: Optional[int]) -> List[str]:
    ds_dict = load_dataset(repo)
    if split is None:
        keys = list(ds_dict.keys())
        split = "train" if "train" in keys else keys[0]
    ds = ds_dict[split]
    size = len(ds)
    idxs = list(range(size))
    rnd = random.Random(seed)
    rnd.shuffle(idxs)
    out = []
    for i in idxs[:n]:
        row = ds[i]
        val = row.get(column) or row.get("positive") or row.get("anchor_text") or row.get("generated_text")
        if isinstance(val, str) and val.strip():
            out.append(val.strip())
    return out


# -------------------------------
# Generation
# -------------------------------

def build_pairs_block_hn(pairs: List[Tuple[str, str]], max_chars: int = 200) -> str:
    def trunc(s: str) -> str:
        s = (s or "").replace("\n", " ").strip()
        return (s[:max_chars] + "…") if len(s) > max_chars else s
    lines = []
    for i, (base, neg) in enumerate(pairs):
        lines.append(f"{i:02d} | {trunc(base)} -> {trunc(neg)}")
    return "\n".join(lines)


def generate_negatives(client: "OpenAI",
                       model: str,
                       provider: Optional[str],
                       family: str,
                       neg_prompt_tpl: str,
                       bases: List[str],
                       max_tokens_per_item: int = 400) -> List[Tuple[str, str]]:
    """
    family: 'query_based' uses {query}, 'positive_based' uses {positive}
    Returns list of (base, negative)
    """
    sys_prompt = "Tu suis les instructions et ne retournes QUE le texte demandé, sans métalangage."
    pairs: List[Tuple[str, str]] = []
    pbar = tqdm(total=len(bases), desc=f"Generating HN ({family})", unit="ex") if tqdm else None
    for base in bases:
        user = neg_prompt_tpl.replace("{query}", base) if family == "query_based" else neg_prompt_tpl.replace("{positive}", base)
        txt = call_llm_text(client, model, provider, sys_prompt, user,
                            max_tokens=max_tokens_per_item, temperature=0.7, timeout=60.0)
        pairs.append((base, (txt or "").strip()))
        if pbar: pbar.update(1)
    if pbar: pbar.close()
    return pairs


# -------------------------------
# Judge (style "positif", JSON strict + retries + fallback)
# -------------------------------

def judge_batch_hn(client: "OpenAI",
                   model: str,
                   provider: Optional[str],
                   judge_yaml: Dict[str, Any],
                   pairs: List[Tuple[str, str]],
                   meta: Dict[str, Any],
                   max_tokens: int = 900,
                   retries: int = 2,
                   fallback_model: Optional[str] = "openai/gpt-4o-mini",
                   fallback_provider: Optional[str] = "openai") -> Dict[str, Any]:
    jcfg = judge_yaml["judge_set_hardneg"]
    sys_prompt = jcfg["system"] + "\nNE FOURNIS PAS TON RAISONNEMENT. ÉCRIS UNIQUEMENT LE JSON FINAL DANS TON MESSAGE."
    base_block = jcfg["user_template"].format(
        family=meta.get("family",""),
        neg_prompt=meta.get("neg_prompt",""),
        model=meta.get("model",""),
        provider=meta.get("provider",""),
        n=len(pairs),
        pairs_block=build_pairs_block_hn(pairs, max_chars=200)
    )

    def ask(m, p, u):
        return call_llm_json(client, m, p, sys_prompt, u, max_tokens=max_tokens, temperature=0.0, timeout=180.0)

    parsed, raw = None, ""
    attempt = 0
    while attempt <= retries and not isinstance(parsed, dict):
        attempt += 1
        user = base_block if attempt == 1 else (
            "RENDS STRICTEMENT UN OBJET JSON. PAS DE TEXTE, PAS D’EXPLICATION.\n" + base_block
        )
        parsed, raw = ask(model, provider, user)

    if not isinstance(parsed, dict) and fallback_model:
        attempt = 0
        while attempt <= retries and not isinstance(parsed, dict):
            attempt += 1
            user = base_block if attempt == 1 else (
                "RENDS STRICTEMENT UN OBJET JSON. PAS DE TEXTE, PAS D’EXPLICATION.\n" + base_block
            )
            parsed, raw = ask(fallback_model, fallback_provider, user)

    out = {"batch_label": "FAIL", "note_du_lot": 0.0, "notes": [], "idee_prompt": "", "raw_judge": ""}
    if isinstance(parsed, dict):
        out["batch_label"] = "PASS" if str(parsed.get("batch_label","FAIL")).upper() == "PASS" else "FAIL"
        try:
            out["note_du_lot"] = float(parsed.get("note_du_lot", 0.0) or 0.0)
        except Exception:
            out["note_du_lot"] = 0.0
        notes = parsed.get("notes", [])
        out["notes"] = notes if isinstance(notes, list) else [str(notes)]
        out["idee_prompt"] = parsed.get("idee_prompt") or ""
        return out

    # Dernier recours: JSON fail propre
    out["notes"] = ["parse_error_or_non_json"]
    return out


# -------------------------------
# Prompt improver (JSON strict)
# -------------------------------

PROMPT_IMPROVER_SYSTEM = """Tu es expert en prompts pour générer des HARD NEGATIVES (BASE -> NEGATIVE).
Tu reçois: (1) le prompt négatif actuel, (2) le verdict du juge (JSON), (3) quelques exemples base->neg.
Objectif: renvoyer un NOUVEAU PROMPT.
- Pas de métalangage; consignes opérationnelles.
- N'hésite pas à mettre des exemples et des règles précises et strictes.
- Soit simple direct et efficace.
- Insiste sur: similarité superficielle (lexique/structure) + divergence sémantique franche; pas de paraphrase.
Réponds en JSON strict uniquement: {"title":"...", "prompt":"..."}"""

def _prompt_changed(prev: str, new: str) -> bool:
    prev_t = set((prev or "").lower().split())
    new_t = set((new or "").lower().split())
    if not new or not new.strip(): return False
    if not prev_t: return True
    inter = len(prev_t & new_t)
    union = max(1, len(prev_t | new_t))
    return (inter / union) < 0.7


def build_improver_user(prev_title: str,
                        prev_prompt: str,
                        judge_summary: Dict[str, Any],
                        examples: List[Tuple[str,str]],
                        family: str,
                        neg_name: str) -> str:
    k = min(8, len(examples))
    lines = []
    for i,(base,neg) in enumerate(examples[:k]):
        b1 = (base or "").replace("\n"," ")[:300]
        n1 = (neg or "").replace("\n"," ")[:300]
        lines.append(f"{i:02d}. {b1} -> {n1}")
    judge_json = json.dumps(judge_summary, ensure_ascii=False)
    return f"""FAMILY: {family}
PROMPT_NAME: {neg_name}
TITRE_ACTUEL: {prev_title}
PROMPT_ACTUEL:
{prev_prompt}

VERDICT_JUGE:
{judge_json}

ECHANTILLON (base -> negative):
{chr(10).join(lines)}

Consignes: produire un nouveau prompt plus ciblé pour créer des hard negatives plausibles,
très proches en surface (lexique/structure) mais divergents en sens. Garde le meme titre.
Retourne uniquement le JSON demandé.
"""


def improve_neg_prompt(client: "OpenAI",
                       model: str,
                       provider: Optional[str],
                       prev_title: str,
                       prev_prompt: str,
                       judge_summary: Dict[str, Any],
                       examples: List[Tuple[str,str]],
                       family: str,
                       neg_name: str) -> Tuple[str, str]:
    parsed, _ = call_llm_json(
        client, model, provider,
        PROMPT_IMPROVER_SYSTEM,
        build_improver_user(prev_title, prev_prompt, judge_summary, examples, family, neg_name),
        max_tokens=200000, temperature=0.3, timeout=120.0
    )
    print(parsed)
    title, prompt = prev_title, prev_prompt
    if isinstance(parsed, dict):
        t = parsed.get("title") or prev_title
        p = parsed.get("prompt") or prev_prompt
        if _prompt_changed(prev_prompt, p):
            title, prompt = t, p
    return title, prompt


# -------------------------------
# IO helpers
# -------------------------------

def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# -------------------------------
# Main
# -------------------------------

def main():
    ap = argparse.ArgumentParser(description="Iterative Hard Negative Prompt Optimizer (LLM-only judge)")
    ap.add_argument("--yaml-negs", required=True, help="hardneg_generation_prompts.yaml")
    ap.add_argument("--judge-prompts", required=True, help="judge_prompts_hardneg_set.yaml")
    ap.add_argument("--dataset", default="matheoqtb/ancre")
    ap.add_argument("--split", default=None)
    ap.add_argument("--family", choices=["query_based", "positive_based"], required=True)
    ap.add_argument("--neg-name", required=True, help="name of negative prompt inside the chosen family")
    ap.add_argument("--query-col", default="anchor_text", help="column for query_based family")
    ap.add_argument("--source-col", default="positive", help="column for positive_based family")
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--max-iters", type=int, default=5)
    ap.add_argument("--seed", type=int, default=None)
    # generators
    ap.add_argument("--model", default="google/gemini-2.5-flash-lite")
    ap.add_argument("--provider", default="google-vertex")
    # judge
    ap.add_argument("--judge-model", default="google/gemini-2.5-pro")
    ap.add_argument("--judge-provider", default="google-vertex")
    ap.add_argument("--judge-fallback-model", default="openai/gpt-4o-mini")
    ap.add_argument("--judge-fallback-provider", default="openai")
    ap.add_argument("--out-jsonl", default="hardneg_opt_log.jsonl")
    args = ap.parse_args()

    neg_cfg = load_yaml(args.yaml_negs)
    families = neg_cfg.get("prompts", {})
    if args.family not in families:
        raise SystemExit(f"Family '{args.family}' not in YAML. Options: {list(families.keys())}")

    # pick prompt by name
    def pick_neg_prompt(cfg_family: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
        for d in cfg_family:
            if d.get("name") == name:
                return d
        raise KeyError(f"Negative prompt '{name}' not found in family.")
    neg_def = pick_neg_prompt(families[args.family], args.neg_name)
    neg_title = neg_def.get("name", args.neg_name)
    neg_prompt_tpl = neg_def.get("prompt", "")

    judge_yaml = load_yaml(args.judge_prompts)
    client = make_client("IterativeHardNegOpt")
    rnd = random.Random(args.seed)

    # graceful stop
    stop_flag = {"stop": False}
    def on_sigint(signum, frame):
        sys.stderr.write("\n[signal] Ctrl-C detected, exiting after current step...\n")
        stop_flag["stop"] = True
    signal.signal(signal.SIGINT, on_sigint)

    iteration = 0
    while iteration < args.max_iters and not stop_flag["stop"]:
        iteration += 1

        # 1) Sample bases
        column = args.query_col if args.family == "query_based" else args.source_col
        bases = sample_bases(args.dataset, args.split, column, args.batch_size, seed=rnd.randrange(10**9))

        # 2) Generate negatives
        pairs = generate_negatives(client, args.model, args.provider, args.family, neg_prompt_tpl, bases, max_tokens_per_item=400)

        # 3) Judge (strict JSON + fallback)
        meta = {"family": args.family, "neg_prompt": neg_title, "model": args.model, "provider": args.provider}
        verdict = judge_batch_hn(
            client, args.judge_model, args.judge_provider,
            judge_yaml, pairs, meta,
            max_tokens=900, retries=2,
            fallback_model=args.judge_fallback_model,
            fallback_provider=args.judge_fallback_provider
        )

        # Log + feedback
        append_jsonl(args.out_jsonl, {
            "iteration": iteration,
            "family": args.family,
            "neg_name": neg_title,
            "prompt_template": neg_prompt_tpl,
            "model": args.model,
            "provider": args.provider,
            "verdict": verdict,
            "timestamp": time.time()
        })
        print(json.dumps({
            "iter": iteration,
            "batch_label": verdict["batch_label"],
            "note": verdict["note_du_lot"],
            "notes": verdict["notes"],
            "idee_prompt": verdict.get("idee_prompt","")
        }, ensure_ascii=False))

        if verdict["batch_label"] == "PASS":
            print("[result] Hard negative prompt VALIDÉ ✅")
            break
        if stop_flag["stop"]:
            break

        # 4) Improve prompt
        sample_for_improve = pairs[:8]
        neg_title, neg_prompt_tpl = improve_neg_prompt(
            client, args.judge_model, args.judge_provider,
            neg_title, neg_prompt_tpl, verdict, sample_for_improve,
            args.family, args.neg_name
        )
        print(f"[improve] Nouveau titre: {neg_title}")
        print("[improve] Nouveau prompt:\n" + neg_prompt_tpl + "\n")

    # Final save
    append_jsonl(args.out_jsonl, {
        "final_iteration": iteration,
        "family": args.family,
        "neg_name": args.neg_name,
        "final_title": neg_title,
        "final_prompt_template": neg_prompt_tpl
    })
    print(json.dumps({
        "final_iteration": iteration,
        "final_title": neg_title,
        "final_prompt_template": neg_prompt_tpl
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
