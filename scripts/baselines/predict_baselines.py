#!/usr/bin/env python3
"""Inferenca sa sačuvanim TF-IDF baseline modelima (.joblib)."""

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

from src.baselines.pipeline import CLASSIFIER_NAMES
from src.baselines.predict import predict_baseline_texts
from src.common.config import load_config, resolve_path
from src.common.stdio_utf8 import configure_utf8_stdio

SENT_NAME = {"1": "positive", "0": "neutral", "-1": "negative"}
SARC_NAME = {"1": "yes", "0": "no"}


def main() -> None:
    """CLI: predikcija sentimenta ili sarkazma baseline modelom."""
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(
        description=(
            "Baseline inferenca (TF-IDF + NB/LR/SVM). "
            "Zahteva models/baselines/<task>/<model>/model.joblib."
        )
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--task",
        choices=["sentiment", "sarcasm"],
        required=True,
    )
    parser.add_argument(
        "--model",
        choices=list(CLASSIFIER_NAMES),
        default="linear_svm",
        help="Klasifikator (default: linear_svm)",
    )
    parser.add_argument("--text", action="append", default=None, help="Tekst (može više puta)")
    parser.add_argument("--file", default=None, help="TXT: jedan tekst po liniji")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Eksplicitna putanja do model.joblib",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Root baselines foldera (default: models/baselines)",
    )
    parser.add_argument("--json", action="store_true", help="Ispis kao JSON")
    parser.add_argument(
        "--show-preprocessed",
        action="store_true",
        help="Prikaži i tekst posle baseline pretprocesiranja",
    )
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
        raise SystemExit('Dodaj --text "..." ili --file putanja.txt')

    config = load_config(args.config)
    results = predict_baseline_texts(
        texts,
        config=config,
        task=args.task,
        model=args.model,
        model_path=resolve_path(args.model_path) if args.model_path else None,
        output_dir=resolve_path(args.output_dir) if args.output_dir else None,
    )

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    name_map = SENT_NAME if args.task == "sentiment" else SARC_NAME
    key = "sentiment" if args.task == "sentiment" else "sarcasm"
    for r in results:
        print("-" * 60)
        print(r["text"])
        if args.show_preprocessed:
            print(f"  [preprocessed] {r['text_preprocessed']}")
        label = r[key]
        pretty = name_map.get(str(label), str(label))
        print(f"  {key}: {label} ({pretty})  [{r['model']}]")
        probs = r.get(f"{key}_probs")
        if probs:
            print("    probs: " + ", ".join(f"{k}={v:.2f}" for k, v in probs.items()))


if __name__ == "__main__":
    main()
