#!/usr/bin/env python3
"""Brza inferenca nad tekstom / fajlom (zahteva models/<task>/best.pt)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._bootstrap import ensure_project_root

ensure_project_root()

from src.common.config import load_config, resolve_path
from src.common.stdio_utf8 import configure_utf8_stdio
from src.modeling.predict import predict_texts


def main() -> None:
    """CLI ulazna tačka; poziva ``src.modeling.predict.predict_texts``.

    Inferenca nad ``--text`` / ``--file``; ispis labele i verovatnoća (ili JSON).
    """
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description="Inferenca sentiment / sarkazam / multitask.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--task",
        choices=["sentiment", "sarcasm", "multitask"],
        default="sentiment",
        help="Koji checkpoint da učita (trenutno postoji samo sentiment ako si radila smoke test)",
    )
    parser.add_argument("--text", action="append", default=None, help="Tekst (može više puta)")
    parser.add_argument("--file", default=None, help="TXT: jedan tekst po liniji")
    parser.add_argument("--model-dir", default=None, help="Folder sa best.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--json", action="store_true", help="Ispis kao JSON")
    args = parser.parse_args()

    texts: list[str] = []
    if args.text:
        texts.extend(args.text)
    if args.file:
        path = Path(args.file)
        texts.extend(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if not texts:
        raise SystemExit("Dodaj --text \"...\" ili --file putanja.txt")

    config = load_config(args.config)
    results = predict_texts(
        texts,
        config=config,
        task=args.task,
        model_dir=resolve_path(args.model_dir) if args.model_dir else None,
        device_name=args.device,
    )

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    for r in results:
        print("-" * 60)
        print(r["text"])
        if "sentiment" in r:
            print(f"  sentiment: {r['sentiment']}")
            probs = ", ".join(f"{k}={v:.2f}" for k, v in r["sentiment_probs"].items())
            print(f"    probs: {probs}")
        if "sarcasm" in r:
            print(f"  sarcasm:   {r['sarcasm']}")
            probs = ", ".join(f"{k}={v:.2f}" for k, v in r["sarcasm_probs"].items())
            print(f"    probs: {probs}")


if __name__ == "__main__":
    main()
