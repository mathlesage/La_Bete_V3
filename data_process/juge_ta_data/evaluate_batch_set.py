#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Juge un lot homogène (≈50 paires) en UNE décision globale (PASS/FAIL) via LLM.
S'adapte aux réponses type OpenRouter/Gemini qui mettent du texte dans `reasoning` / `reasoning_details`
et peuvent finir sur MAX_TOKENS avec `message.content` vide.

- Mode 'positive' : anchor -> generated_text
- Mode 'hard_negative' : generated_text -> negative (from_positive ou from_anchor)

Dépendances:
pip install -U datasets pyyaml "openai>=1.30"
export OPENROUTER_API_KEY="sk-..."
"""
import argparse
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional

import yaml
from datasets import load_dataset

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


def _extract_text_from_resp(resp) -> str:
    """
    Récupère au mieux un texte 'réponse' depuis diverses zones possibles
    (content, reasoning, reasoning_details, dump JSON).
    """
    try:
        ch = getattr(resp, "choices", None)
        if not ch:
            return ""
        msg = getattr(ch[0], "message", None)
        if not msg:
            return ""
        # 1) normal
        content = getattr(msg, "content", None) or ""
        if content and content.strip():
            return content.strip()

        # 2) champ reasoning (modèles "reasoning")
        reasoning = getattr(msg, "reasoning", None)
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning.strip()

        # 3) reasoning_details (liste d’objets avec 'text')
        details = getattr(msg, "reasoning_details", None)
        if isinstance(details, list) and details:
            parts = []
            for d in details:
                t = d.get("text") if isinstance(d, dict) else None
                if isinstance(t, str) and t.strip():
                    parts.append(t.strip())
            if parts:
                return "\n".join(parts)

        # 4) dernier recours : dump JSON complet (au cas où le modèle glisse un JSON quelque part)
        if hasattr(resp, "model_dump_json"):
            return resp.model_dump_json()
        return ""
    except Exception:
        return ""


def call_llm(model: str,
             provider: Optional[str],
             system: str,
             user: str,
             temperature: float = 0.0,
             timeout: Optional[float] = 180.0,
             max_tokens: int = 100000
             ) -> Dict[str, Any]:
    """
    Appelle le LLM et renvoie un dict {"raw": str, "finish_reason": str|None, "native_finish_reason": str|None}
    où 'raw' est le meilleur texte extrait (content/raisonnement/etc.).
    """
    if OpenAI is None:
        raise RuntimeError("openai>=1.0 requis")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY manquant")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={"X-Title": "BatchJudgeSet", "HTTP-Referer": "https://local.test"},
    )
    extra_body = {}
    if provider:
        extra_body["provider"] = {"order": [provider], "allow_fallbacks": False}

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body,
            timeout=timeout,
        )
        raw = _extract_text_from_resp(resp)
        finish_reason = None
        native_finish = None
        try:
            # openrouter renvoie parfois ces champs
            finish_reason = getattr(resp.choices[0], "finish_reason", None)
            native_finish = getattr(resp, "native_finish_reason", None)
        except Exception:
            pass
        return {"raw": raw, "finish_reason": finish_reason, "native_finish_reason": native_finish}
    except Exception as e:
        sys.stderr.write(f"[judge-set][ERR] {type(e).__name__}: {str(e)[:300]}\n")
        return {"raw": "", "finish_reason": None, "native_finish_reason": None}


def parse_set_json(raw: str, finish_reason: Optional[str], native_finish: Optional[str]) -> Dict[str, Any]:
    """
    Attend un JSON:
    {
      "batch_label": "PASS"|"FAIL",
      "Note du lot": 0.0-1.0,
      "notes": [...],
      "exemple des 2 pires paire":[...],
      "flags": {...}
    }
    Tolérant: si pas de JSON exploitable -> FAIL + note.
    """
    default = {
        "batch_label": "FAIL",
        "confidence": 0.0,
        "notes": [],
        "flags": {}
    }
    if not raw or not raw.strip():
        default["notes"] = ["empty_response"]
        return default

    # Cherche un objet JSON dans raw (au cas où le modèle a parlé autour)
    s = raw.strip()
    first = s.find("{")
    last = s.rfind("}")
    candidate = s[first:last+1] if (first != -1 and last != -1 and last > first) else s

    try:
        data = json.loads(candidate)
        label = str(data.get("batch_label", "FAIL")).upper()
        conf = float(data.get("Note du lot", 0.0))
        notes = data.get("notes", [])
        pire_lot = data.get("exemple des 2 pires paire", [])
        flags = data.get("flags", {})
        idee= data.get("idée à rajouter au prompt générateur", "")

        if label not in ("PASS", "FAIL"):
            label = "FAIL"
        if not isinstance(notes, list):
            notes = [str(notes)]
        if not isinstance(flags, dict):
            flags = {}
        return {"batch_label": label, "note": conf, "notes": notes, "flags": flags, "exemple des 2 pires paire": pire_lot, "idée pour prompt générateur": idee}
    except Exception:
        # Si c’est coupé par max tokens, note-le
        notes = ["json_parse_error"]
        if (finish_reason == "length") or (native_finish and "MAX_TOKENS" in str(native_finish)):
            notes.append("max_tokens_exceeded")
        return {**default, "notes": notes}


def truncate(s: Optional[str], max_chars: int = 240) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip().replace("\n", " ")
    return (s[: max_chars] + "…") if len(s) > max_chars else s


def main():
    ap = argparse.ArgumentParser(description="Décision globale sur un lot homogène (PASS/FAIL), compatible réponses reasoning.")
    ap.add_argument("--mode", choices=["positive", "hard_negative"], required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--split", default=None)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--prompts-yaml", required=True)
    ap.add_argument("--judge-model", default="google/gemini-2.5-pro")  # par défaut comme ton exemple
    ap.add_argument("--judge-provider", default=None)
    ap.add_argument("--judge-max-tokens", type=int, default=640)
    ap.add_argument("--max-chars", type=int, default=240)
    ap.add_argument("--seed", type=int, default=None)
    # Filtres pour positives
    ap.add_argument("--filter-model", default=None)
    ap.add_argument("--filter-provider", default=None)
    ap.add_argument("--filter-style", default=None)
    ap.add_argument("--filter-strategy", default=None)
    # Filtres pour hard negatives
    ap.add_argument("--variant", choices=["from_positive", "from_anchor"], default="from_positive")
    ap.add_argument("--filter-neg-prompt", default=None)
    ap.add_argument("--filter-neg-model-substr", default=None)  # colonne combine modèle|provider
    ap.add_argument("--out", default="report_set.json")
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    with open(args.prompts_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    jcfg = cfg.get("judge_set", {})
    if args.mode == "positive":
        sys_prompt = jcfg["positive"]["system"]
        user_tpl = jcfg["positive"]["user_template"]
    else:
        sys_prompt = jcfg["hard_negative"]["system"]
        user_tpl = jcfg["hard_negative"]["user_template"]

    ds_dict = load_dataset(args.repo)
    if args.split is None:
        keys = list(ds_dict.keys())
        split = "train" if "train" in keys else keys[0]
    else:
        split = args.split
    ds = ds_dict[split]
    size = len(ds)

    # Filtrage pour homogénéité
    candidates = []
    for i in range(size):
        row = ds[i]
        if args.mode == "positive":
            if args.filter_model and row.get("model") != args.filter_model:
                continue
            if args.filter_provider and row.get("provider") != args.filter_provider:
                continue
            if args.filter_style and row.get("style_key") != args.filter_style:
                continue
            if args.filter_strategy and row.get("strategy_key") != args.filter_strategy:
                continue
            if not (isinstance(row.get("anchor_text"), str) and isinstance(row.get("generated_text"), str)):
                continue
            candidates.append(i)
        else:
            neg = row.get("negative_from_positive") if args.variant == "from_positive" else row.get("negative_from_anchor")
            neg_prompt = row.get("neg_from_positive_prompt") if args.variant == "from_positive" else row.get("neg_from_anchor_prompt")
            neg_model = row.get("neg_from_positive_model") if args.variant == "from_positive" else row.get("neg_from_anchor_model")
            if args.filter_neg_prompt and neg_prompt != args.filter_neg_prompt:
                continue
            if args.filter_neg_model_substr and (not isinstance(neg_model, str) or args.filter_neg_model_substr not in neg_model):
                continue
            if not (isinstance(row.get("generated_text"), str) and isinstance(neg, str) and neg.strip() and row.get("generated_text").strip()):
                continue
            candidates.append(i)

    if not candidates:
        print(json.dumps({"error": "no_candidates_after_filtering"}, ensure_ascii=False))
        sys.exit(0)

    n = min(args.n, len(candidates))
    random.shuffle(candidates)
    idxs = candidates[:n]

    # Construire bloc + meta
    if args.mode == "positive":
        pairs_lines = []
        model = args.filter_model or ds[idxs[0]].get("model")
        provider = args.filter_provider or ds[idxs[0]].get("provider")
        strategy = args.filter_strategy or ds[idxs[0]].get("strategy_key")
        style = args.filter_style or ds[idxs[0]].get("style_key")
        wc = ds[idxs[0]].get("word_count")
        for j, i in enumerate(idxs):
            row = ds[i]
            anchor = truncate(row.get("anchor_text"), args.max_chars)
            cand = truncate(row.get("generated_text"), args.max_chars)
            # format compact
            pairs_lines.append(f"{j:02d}|{anchor} => {cand}")
        pairs_block = "\n".join(pairs_lines)
        user_prompt = user_tpl.format(
            model=model or "",
            provider=provider or "",
            strategy=strategy or "",
            style=style or "",
            word_count=wc if wc is not None else "",
            n=n,
            pairs_block=pairs_block
        )
    else:
        pairs_lines = []
        neg_prompt = args.filter_neg_prompt or (ds[idxs[0]].get("neg_from_positive_prompt") if args.variant=="from_positive" else ds[idxs[0]].get("neg_from_anchor_prompt"))
        neg_model = args.filter_neg_model_substr or (ds[idxs[0]].get("neg_from_positive_model") if args.variant=="from_positive" else ds[idxs[0]].get("neg_from_anchor_model"))
        for j, i in enumerate(idxs):
            row = ds[i]
            positive = truncate(row.get("generated_text"), args.max_chars)
            negative = truncate(row.get("negative_from_positive") if args.variant=="from_positive" else row.get("negative_from_anchor"), args.max_chars)
            pairs_lines.append(f"{j:02d}|{positive} => {negative}")
        pairs_block = "\n".join(pairs_lines)
        user_prompt = user_tpl.format(
            neg_prompt=neg_prompt or "",
            neg_model=neg_model or "",
            n=n,
            pairs_block=pairs_block
        )

    # Appel LLM (robuste aux réponses "reasoning")
    got = call_llm(
        model=args.judge_model,
        provider=args.judge_provider,
        system=sys_prompt,
        user=user_prompt,
        temperature=0.0,
        timeout=180.0,
        max_tokens=args.judge_max_tokens
    )
    parsed = parse_set_json(got.get("raw", ""), got.get("finish_reason"), got.get("native_finish_reason"))

    # Sauvegarde
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "mode": args.mode,
            "repo": args.repo,
            "split": split,
            "n": n,
            "idxs": idxs,
            "filters": {
                "model": args.filter_model,
                "provider": args.filter_provider,
                "style": args.filter_style,
                "strategy": args.filter_strategy,
                "variant": args.variant,
                "neg_prompt": args.filter_neg_prompt,
                "neg_model_substr": args.filter_neg_model_substr
            },
            "judge": {
                "model": args.judge_model,
                "provider": args.judge_provider,
                "max_tokens": args.judge_max_tokens
            },
            "result": parsed
        }, f, ensure_ascii=False, indent=2)

    print(json.dumps(parsed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
