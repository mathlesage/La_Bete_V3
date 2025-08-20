# -*- coding: utf-8 -*-
"""
Iterative Prompt Optimizer for positive strategies.

Workflow (per strategy):
1) Fix model to google/gemini-2.5-flash-lite (provider google-vertex) unless overridden.
2) Sample a batch of anchors (default 50) from HuggingFace dataset (default matheoqtb/ancre).
3) For a candidate prompt template (from strategies YAML), generate CANDIDATE texts (one per anchor),
   picking style/humeur randomly from styles_pour_positifs and sampling word_count within range if provided.
4) Send the whole batch to a judge (judge_prompts_set.yaml 'positive') to get a GLOBAL verdict and notes.
5) If PASS -> save "validated" prompt version to JSONL and stop (or continue to next strategy if --all-strategies).
6) If FAIL -> ask an LLM to rewrite/improve the prompt (as QUESTIONS) with a TITLE using judge feedback + examples;
   iterate up to --max-iters. Checkpoint after every evaluation so you can stop anytime and resume.

CLI:
pip install -U datasets pyyaml "openai>=1.30" tqdm
export OPENROUTER_API_KEY="sk-..."

Example:
python iterative_prompt_optimizer.py \
  --yaml-strategies positive_strategies.yaml \
  --judge-prompts judge_prompts_set.yaml \
  --dataset matheoqtb/ancre \
  --anchor-column positive \
  --strategy paraphrase_changement_vocabulaire \
  --batch-size 50 \
  --out-jsonl optimized_prompts_log.jsonl \
  --max-iters 5 \
  --seed 42
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

from typing import Any

def make_client(title: str = "IterativePromptOpt") -> Any:
    if OpenAI is None:
        raise RuntimeError("openai>=1.0 required")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY missing")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={"X-Title": title, "HTTP-Referer": "https://local.test"},
    )

def call_llm_json(client: Any,
                  model: str,
                  provider: Optional[str],
                  system: str,
                  user: str,
                  max_tokens: int = 10000,
                  temperature: float = 0.2,
                  timeout: Optional[float] = 120.0) -> Tuple[Optional[dict], str]:
    """
    Force JSON if supported. Returns (parsed_json or None, raw_text)
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
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            extra_body=extra_body,
        )
        # best-effort extraction
        ch = getattr(resp, "choices", None)
        text = ""
        if ch and ch[0].message:
            text = (getattr(ch[0].message, "content", None) or "") or ""
            if not text.strip():
                text = getattr(ch[0].message, "reasoning", "") or ""
            if not text and getattr(ch[0].message, "reasoning_details", None):
                parts = []
                for d in ch[0].message.reasoning_details:
                    t = d.get("text") if isinstance(d, dict) else None
                    if isinstance(t, str) and t.strip():
                        parts.append(t.strip())
                if parts:
                    text = "\n".join(parts)
        parsed = None
        if text:
            try:
                # find first JSON object if the model added extra prose
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

def call_llm_text(client: Any,
                  model: str,
                  provider: Optional[str],
                  system: str,
                  user: str,
                  max_tokens: int = 10000,
                  temperature: float = 0.7,
                  timeout: Optional[float] = 60.0) -> Optional[str]:
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
            timeout=timeout,
            extra_body=extra_body,
        )
        ch = getattr(resp, "choices", None)
        if not ch:
            return None
        text = getattr(ch[0].message, "content", None) or ""
        if not text.strip():
            text = getattr(ch[0].message, "reasoning", "") or ""
        return text.strip() if text else None
    except Exception as e:
        sys.stderr.write(f"[call_llm_text][ERR] {type(e).__name__}: {str(e)[:300]}\n")
        return None

# -------------------------------
# Data helpers
# -------------------------------

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def sample_anchors(repo: str, split: Optional[str], column: str, n: int, seed: Optional[int]) -> List[str]:
    ds_dict = load_dataset(repo)
    if split is None:
        keys = list(ds_dict.keys())
        split = "train" if "train" in keys else keys[0]
    ds = ds_dict[split]
    size = len(ds)
    idxs = list(range(size))
    rnd = random.Random(seed)
    rnd.shuffle(idxs)
    anchors = []
    for i in idxs[:n]:
        row = ds[i]
        val = row.get(column) or row.get("anchor_text")
        if isinstance(val, str) and val.strip():
            anchors.append(val.strip())
    return anchors

# -------------------------------
# Generation for one batch
# -------------------------------

def build_style_cycle(styles: List[Dict[str, Any]], n: int, seed: Optional[int]) -> List[Dict[str, Any]]:
    # cycle through all styles shuffled, repeat until length n
    rnd = random.Random(seed)
    pool = styles.copy()
    rnd.shuffle(pool)
    out = []
    while len(out) < n:
        out.extend(pool)
    return out[:n]

def realize_strategy_prompt(strategy_cfg: Dict[str, Any],
                            anchor_text: str,
                            style_instruction: Optional[str],
                            word_count: Optional[int]) -> str:
    tpl = strategy_cfg.get("prompt", "")
    msg = tpl.replace("{anchor_text}", anchor_text)
    msg = msg.replace("{style_instruction}", style_instruction or "")
    if word_count is not None:
        msg = msg.replace("{word_count}", str(word_count))
    return msg

def pick_word_count(strategy_cfg: Dict[str, Any], rnd: random.Random) -> Optional[int]:
    rng = strategy_cfg.get("word_count_range")
    if isinstance(rng, list) and len(rng) == 2:
        return rnd.randint(int(rng[0]), int(rng[1]))
    return None



def generate_batch_examples(client: Any,
                            model: str,
                            provider: Optional[str],
                            anchors: List[str],
                            styles: List[Dict[str, Any]],
                            strategy_key: str,
                            strategies_cfg: Dict[str, Any],
                            max_tokens_per_item: int = 10000,
                            seed: Optional[int] = None) -> List[Tuple[str, str, str, Optional[int]]]:
    """
    Returns list of tuples: (anchor, candidate, style_key, word_count)
    """
    rnd = random.Random(seed)
    strat = strategies_cfg[strategy_key]
    style_cycle = build_style_cycle(styles, len(anchors), seed)
    results = []
    pbar = tqdm(total=len(anchors), desc=f"Generating ({strategy_key})", unit="ex") if tqdm else None
    for idx, anchor in enumerate(anchors):
        style = style_cycle[idx]
        style_key = style.get("key")
        style_instruction = style.get("instruction", "")
        wc = pick_word_count(strat, rnd)  # <-- tire un word_count aléatoire si range présent
        user_prompt = realize_strategy_prompt(strat, anchor, style_instruction, wc)

        # System prompt: insiste sur l'objectif de longueur si 'wc' est défini
        system_prompt = "Tu suis les instructions et ne retournes QUE le texte demandé, sans métalangage."
        if wc is not None:
            system_prompt += f" Respecte ~{wc} mots (tolérance ±20%)."

        text = call_llm_text(
            client=client,
            model=model,
            provider=provider,
            system=system_prompt,
            user=user_prompt,
            max_tokens=max_tokens_per_item,
            temperature=0.7,
            timeout=60.0
        )
        candidate = (text or "").strip()
        results.append((anchor, candidate, style_key, wc))
        if pbar: pbar.update(1)
    if pbar: pbar.close()
    print(results[:3], "…")  # Show first 3 results for quick check
    return results


# -------------------------------
# Judge
# -------------------------------

def build_pairs_block_positive(pairs: List[Tuple[str, str, str, Optional[int]]], max_chars: int = 400) -> str:
    def trunc(s: str) -> str:
        s = (s or "").replace("\n", " ").strip()
        return (s[:max_chars] + "…") if len(s) > max_chars else s
    lines = []
    for i, (anchor, candidate, style_key, wc) in enumerate(pairs):
        wc_part = f" | wc={wc}" if isinstance(wc, int) else ""
        lines.append(f"{i:02d} | {trunc(anchor)} -> {trunc(candidate)}{wc_part}")
    return "\n".join(lines)


def judge_batch(client: Any,
                model: str,
                provider: Optional[str],
                judge_yaml: Dict[str, Any],
                batch_pairs: List[Tuple[str, str, str, Optional[int]]],
                meta: Dict[str, Any],
                max_tokens: int = 800) -> Dict[str, Any]:
    jcfg = judge_yaml["judge_set"]["positive"]
    # Forcer "JSON ONLY" dans le system prompt du juge
    sys_prompt = jcfg["system"] + "\nNE FOURNIS PAS TON RAISONNEMENT. ÉCRIS UNIQUEMENT LE JSON FINAL DANS TON MESSAGE."
    # Réduire la taille pour éviter les coupures
    pairs_block = build_pairs_block_positive(batch_pairs, max_chars=220)
    user_prompt = jcfg["user_template"].format(
        model=meta.get("model",""),
        provider=meta.get("provider",""),
        strategy=meta.get("strategy",""),
        style=meta.get("style",""),
        word_count=meta.get("word_count",""),
        n=len(batch_pairs),
        pairs_block=pairs_block
    )
    parsed, raw = call_llm_json(
        client=client,
        model=model,
        provider=provider,
        system=sys_prompt,
        user=user_prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        timeout=180.0
    )
    # Normalisation des clés
    out = {"batch_label":"FAIL","note_du_lot":0.0,"notes":[],"idee_prompt":""}
    if isinstance(parsed, dict):
        label = str(parsed.get("batch_label","FAIL")).upper()
        out["batch_label"] = "PASS" if label == "PASS" else "FAIL"
        # "Note du lot" (FR) ou "note_du_lot" (fallback)
        out["note_du_lot"] = float(parsed.get("Note du lot", parsed.get("note_du_lot", 0.0)) or 0.0)
        notes = parsed.get("notes", [])
        out["notes"] = notes if isinstance(notes, list) else [str(notes)]
        # clé FR avec accents + fallback
        out["idee_prompt"] = parsed.get("idée à rajouter au prompt générateur") or parsed.get("idee_prompt") or ""
    else:
        out["notes"] = ["parse_error_or_non_json"]
    out["raw_judge"] = raw
    return out


# -------------------------------
# Prompt improver
# -------------------------------

PROMPT_IMPROVER_SYSTEM = """Tu es un expert en ingénierie de prompts pour générer des réécritures (ANCHOR->CANDIDATE).
Tu reçois: (1) le prompt actuel, (2) le verdict du juge (JSON), (3) un petit échantillon d'exemples.
Ta tâche: retourner un NOUVEAU PROMPT plus efficace, n'hésite pas à être très guidant dans le prompt.
Réponds en JSON strict:
{"title":"...", "prompt":"..."}
"""

def build_improver_user(prev_title: str,
                        prev_prompt: str,
                        judge_summary: Dict[str, Any],
                        examples: List[Tuple[str,str,str,Optional[int]]],
                        strategy_key: str) -> str:
    # include only few examples to stay short
    k = min(20, len(examples))
    lines = []
    for i,(a,c,sty,wc) in enumerate(examples[:k]):
        a1 = (a or "").replace("\n"," ")[:1000]
        c1 = (c or "").replace("\n"," ")[:1000]
        lines.append(f"{i:02d}. {a1} -> {c1}")
    judge_json = json.dumps(judge_summary, ensure_ascii=False)
    return f"""STRATEGIE: {strategy_key}
TITRE_ACTUEL: {prev_title}
PROMPT_ACTUEL:
{prev_prompt}

VERDICT_JUGE:
{judge_json}

ECHANTILLON (anchor -> candidate):
{chr(10).join(lines)}

Consignes: formuler le NOUVEAU prompt n'hésite pas à être très guidant, en gardant le titre actuel.
inclure rappel de style/humeur si utile, limiter le texte au nécessaire.
Retourne uniquement le JSON demandé.
"""

def improve_prompt(client: Any,
                   model: str,
                   provider: Optional[str],
                   prev_title: str,
                   prev_prompt: str,
                   judge_summary: Dict[str, Any],
                   examples: List[Tuple[str,str,str,Optional[int]]],
                   strategy_key: str) -> Tuple[str, str]:
    parsed, raw = call_llm_json(
        client=client,
        model=model,
        provider=provider,
        system=PROMPT_IMPROVER_SYSTEM,
        user=build_improver_user(prev_title, prev_prompt, judge_summary, examples, strategy_key),
        max_tokens=10000,
        temperature=0.3,
        timeout=120.0
    )
    title = prev_title
    prompt = prev_prompt
    if isinstance(parsed, dict):
        title = parsed.get("title", title) or title
        prompt = parsed.get("prompt", prompt) or prompt
        print("Nouveau prompt: ",prompt)
    return title, prompt

# -------------------------------
# Main loop
# -------------------------------

def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser(description="Iterative Prompt Optimizer")
    ap.add_argument("--yaml-strategies", required=True)
    ap.add_argument("--judge-prompts", required=True)
    ap.add_argument("--dataset", default="matheoqtb/ancre")
    ap.add_argument("--anchor-column", default="positive")
    ap.add_argument("--strategy", required=True, help="Key in strategies.* (e.g., paraphrase_changement_vocabulaire)")
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--max-iters", type=int, default=5)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--model", default="google/gemini-2.5-flash-lite")
    ap.add_argument("--provider", default="google-vertex")
    ap.add_argument("--judge-model", default="google/gemini-2.5-pro")
    ap.add_argument("--judge-provider", default=None)
    ap.add_argument("--out-jsonl", default="optimized_prompts_log.jsonl")
    args = ap.parse_args()

    # Load config
    cfg = load_yaml(args.yaml_strategies)
    styles = cfg.get("styles_pour_positifs", [])
    strategies = cfg.get("strategies", {})
    if args.strategy not in strategies:
        raise SystemExit(f"Strategy '{args.strategy}' not found in YAML. Available: {list(strategies.keys())}")
    judge_yaml = load_yaml(args.judge_prompts)

    rnd = random.Random(args.seed)

    # Initial candidate prompt = strategy template itself (as user message)
    strategy_cfg = strategies[args.strategy]
    # We'll prime with the same template; improver will transform into a direct set of questions.
    current_title = args.strategy
    current_prompt_template = strategy_cfg.get("prompt","")

    client = make_client("IterativePromptOpt")

    # graceful stop
    stop_flag = {"stop": False}
    def on_sigint(signum, frame):
        sys.stderr.write("\n[signal] Ctrl-C detected, exiting after current step...\n")
        stop_flag["stop"] = True
    signal.signal(signal.SIGINT, on_sigint)

    iteration = 0
    while iteration < args.max_iters and not stop_flag["stop"]:
        iteration += 1
        # 1) Sample anchors
        anchors = sample_anchors(args.dataset, None, args.anchor_column, args.batch_size, seed=rnd.randrange(10**9))
        # 2) Generate candidates using current_prompt_template
        #    Build a "virtual" strategy config that uses current prompt template
        virt_strategy = dict(strategy_cfg)
        virt_strategy["prompt"] = current_prompt_template
        pairs = generate_batch_examples(
            client=client,
            model=args.model,
            provider=args.provider,
            anchors=anchors,
            styles=styles,
            strategy_key=args.strategy,
            strategies_cfg={args.strategy: virt_strategy},
            max_tokens_per_item=400,
            seed=rnd.randrange(10**9)
        )
        # 3) Judge
# 3) Judge
        rng = strategy_cfg.get("word_count_range")
        range_str = f"{rng[0]}–{rng[1]}" if isinstance(rng, list) and len(rng) == 2 else None
        meta = {
            "model": args.model,
            "provider": args.provider,
            "strategy": args.strategy,
            "style": "mixed",
            "word_count": range_str if range_str else "variable par ligne (voir wc)",
        }

        verdict = judge_batch(
            client=client,
            model=args.judge_model,
            provider=args.judge_provider,
            judge_yaml=judge_yaml,
            batch_pairs=pairs,
            meta=meta,
            max_tokens=10000
        )

        log_rec = {
            "iteration": iteration,
            "strategy": args.strategy,
            "model": args.model,
            "provider": args.provider,
            "current_title": current_title,
            "current_prompt_template": current_prompt_template,
            "verdict": verdict,
            "timestamp": time.time()
        }
        append_jsonl(args.out_jsonl, log_rec)
        print(json.dumps({"iter": iteration, "batch_label": verdict["batch_label"], "note": verdict["note_du_lot"], "notes": verdict["notes"], "idee_prompt": verdict}, ensure_ascii=False))

        if verdict["batch_label"] == "PASS":
            print("[result] Prompt VALIDÉ ✅")
            break

        if stop_flag["stop"]:
            break

        # 4) Improve prompt using judge feedback + few examples
        sample_for_improve = pairs[:6]
        current_title, current_prompt_template = improve_prompt(
            client=client,
            model=args.judge_model,   # can use a stronger model to craft prompts
            provider=args.judge_provider,
            prev_title=current_title,
            prev_prompt=current_prompt_template,
            judge_summary=verdict,
            examples=sample_for_improve,
            strategy_key=args.strategy
        )
        print(f"[improve] Nouveau titre: {current_title}")
        # Loop continues

    # Final save
    final_rec = {
        "final_iteration": iteration,
        "strategy": args.strategy,
        "final_title": current_title,
        "final_prompt_template": current_prompt_template
    }
    append_jsonl(args.out_jsonl, final_rec)
    print(json.dumps({"final_title": current_title, "final_prompt_template": current_prompt_template}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
