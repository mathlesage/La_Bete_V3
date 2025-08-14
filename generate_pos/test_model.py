#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_model.py — Vérifie chaque (model, provider) défini dans positive_strategies.yaml

Usage:
  pip install pyyaml openai
  export OPENROUTER_API_KEY="sk-..."
  python test_model.py --yaml positive_strategies.yaml
Options:
  --prompt "Texte..."            Prompt de test (par défaut: ping simple)
  --max-tokens 32                max_tokens pour le test (faible pour coût réduit)
  --temperature 0.0              temperature pour le test
  --retries 1                    nombre de tentatives sur échec transitoire
  --sleep-ms 0                   délai entre modèles (ms)
  --json-out report.json         enregistre un rapport JSON détaillé
  --fail-fast                    arrête au premier échec
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml  # PyYAML
except Exception as e:
    print("PyYAML est requis: pip install pyyaml", file=sys.stderr)
    raise

try:
    from openai import OpenAI
except Exception as e:
    OpenAI = None

DEFAULT_SYSTEM = "Tu es un assistant minimal chargé de répondre OK si tu reçois ce message."

def load_rotation(yaml_path: str) -> List[Dict[str, Any]]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    llm = data.get("llm", {})
    rotation = llm.get("rotation", [])
    if not rotation:
        raise ValueError("Aucune entrée trouvée dans llm.rotation du YAML")
    return rotation

def call_openrouter(model: str, provider: Optional[str], prompt: str, max_tokens: int, temperature: float) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Retourne (success, output_text, error_message)
    """
    if OpenAI is None:
        return False, None, "La bibliothèque openai n'est pas disponible. Installez-la avec: pip install openai"
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return False, None, "Variable d'environnement OPENROUTER_API_KEY manquante"
    try:
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        extra_body = {}
        if provider:
            extra_body["provider"] = {"order": [provider], "allow_fallbacks": False}
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": DEFAULT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body
        )
        content = (resp.choices[0].message.content or "").strip()
        return True, content, None
    except Exception as e:
        return False, None, str(e)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", default="positive_strategies.yaml")
    parser.add_argument("--prompt", default="Réponds OK si tu reçois ce message")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--sleep-ms", type=int, default=0)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    try:
        rotation = load_rotation(args.yaml)
    except Exception as e:
        print(f"[ERREUR] Lecture YAML: {e}", file=sys.stderr)
        sys.exit(2)

    results: List[Dict[str, Any]] = []
    ok = 0
    ko = 0

    for i, entry in enumerate(rotation, start=1):
        model = entry.get("model")
        provider = entry.get("provider")
        if not model:
            print(f"[{i}/{len(rotation)}] Entrée invalide (model manquant), on saute", file=sys.stderr)
            continue

        attempt = 0
        success = False
        output = None
        error = None
        while attempt <= args.retries and not success:
            attempt += 1
            print(f"[{i}/{len(rotation)}] Test {model} provider={provider or '-'} tentative {attempt}", flush=True)
            success, output, error = call_openrouter(
                model=model,
                provider=provider,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature
            )
            if not success and attempt <= args.retries:
                time.sleep(0.5)

        result = {
            "index": i,
            "model": model,
            "provider": provider,
            "success": success,
            "output_preview": (output[:200] if isinstance(output, str) else None),
            "error": error
        }
        results.append(result)

        if success:
            ok += 1
            print(f"  -> PASS: {result['output_preview']!r}")
        else:
            ko += 1
            print(f"  -> FAIL: {error}", file=sys.stderr)
            if args.fail_fast:
                break

        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

    # Résumé
    print("\n==== RÉSUMÉ ====")
    print(f"Total modèles testés: {len(results)}")
    print(f"Réussites: {ok}")
    print(f"Échecs: {ko}")

    if args.json_out:
        try:
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump({
                    "summary": {"tested": len(results), "ok": ok, "ko": ko},
                    "results": results
                }, f, ensure_ascii=False, indent=2)
            print(f"Rapport JSON écrit dans {args.json_out}")
        except Exception as e:
            print(f"[WARN] Écriture JSON impossible: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
