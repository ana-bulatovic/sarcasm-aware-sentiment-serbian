#!/usr/bin/env python3
"""Proba: 10 rečenica kroz sentiment, sarcasm i multitask modele."""

from __future__ import annotations

import argparse
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
from src.modeling.runner import modeling_defaults

# Raznovrsni primeri: iskreni / sarkastični / neutralni
DEMO_TEXTS = [
    "Odličan film, gledala sam ga u jednom dahu!",
    "Katastrofa od filma, dosadan i predugačak.",
    "Bravo majstore, baš si genijalac...",
    "Film traje oko dva sata.",
    "Svaka čast produkciji na ovom promašaju.",
    "Baš mi je prijalo ovo veče u bioskopu.",
    "Naravno da je odvratno — ko bi drugo očekivao?",
    "Da li je neko gledao sinhronizaciju?",
    "Super!!! Još jedan remake koji nam je baš trebao.",
    "Gluma je solidna, priča malo slaba, ali ok je.",
]

SENT_NAME = {"1": "positive", "0": "neutral", "-1": "negative"}
SARC_NAME = {"1": "yes", "0": "no"}


def _fmt_sent(label: str, probs: dict[str, float] | None = None) -> str:
    name = SENT_NAME.get(str(label), str(label))
    if not probs:
        return f"{label} ({name})"
    conf = probs.get(str(label), 0.0)
    return f"{label} ({name}, {conf:.2f})"


def _fmt_sarc(label: str, probs: dict[str, float] | None = None) -> str:
    name = SARC_NAME.get(str(label), str(label))
    if not probs:
        return f"{label} ({name})"
    conf = probs.get(str(label), 0.0)
    return f"{label} ({name}, {conf:.2f})"


def main() -> None:
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(
        description="Inferenca 10 demo rečenica na sentiment / sarcasm / multitask."
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--device", default=None, help="cpu | cuda")
    parser.add_argument(
        "--models-dir",
        default=None,
        help="Root sa sentiment/, sarcasm/, multitask/ (default: models/)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    mcfg = modeling_defaults(config)
    root = resolve_path(args.models_dir) if args.models_dir else resolve_path(mcfg["output_dir"])

    tasks = ("sentiment", "sarcasm", "multitask")
    available: list[str] = []
    for task in tasks:
        ckpt = root / task / "best.pt"
        if ckpt.exists():
            available.append(task)
        else:
            print(f"[skip] Nema checkpointa: {ckpt}")

    if not available:
        raise SystemExit(
            "Nema nijednog best.pt. Prvo istreniraj:\n"
            "  python scripts/modeling/train.py --task all --device cuda"
        )

    results: dict[str, list[dict]] = {}
    for task in available:
        print(f"[predict] {task} ...")
        results[task] = predict_texts(
            DEMO_TEXTS,
            config=config,
            task=task,
            model_dir=root / task,
            device_name=args.device,
        )

    print("\n" + "=" * 100)
    print("DEMO INFERENCA (10 rečenica)")
    print("=" * 100)

    for i, text in enumerate(DEMO_TEXTS):
        print(f"\n{i + 1:2d}. {text}")
        if "sentiment" in results:
            r = results["sentiment"][i]
            print(f"    single sentiment : {_fmt_sent(r['sentiment'], r.get('sentiment_probs'))}")
        if "sarcasm" in results:
            r = results["sarcasm"][i]
            print(f"    single sarcasm   : {_fmt_sarc(r['sarcasm'], r.get('sarcasm_probs'))}")
        if "multitask" in results:
            r = results["multitask"][i]
            print(
                f"    multitask        : sent={_fmt_sent(r['sentiment'], r.get('sentiment_probs'))}"
                f"  |  sarc={_fmt_sarc(r['sarcasm'], r.get('sarcasm_probs'))}"
            )

    print("\nLabele: sentiment 1=pos, 0=neu, -1=neg | sarcasm 1=da, 0=ne")


if __name__ == "__main__":
    main()
