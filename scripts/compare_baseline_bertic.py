#!/usr/bin/env python3
"""Uporedi baseline (TF-IDF) i BERTić inferencu na istim tekstovima."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# OpenMP/MKL: sklearn + torch na Windowsu ponekad crash-uje bez ovoga
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._bootstrap import ensure_project_root

ensure_project_root()

# Torch pre sklearn — izbegava OpenMP/MKL crash na Windowsu pri kombinovanom importu
from src.common.config import load_config, resolve_path
from src.common.stdio_utf8 import configure_utf8_stdio
from src.modeling.predict import predict_texts
from src.modeling.runner import modeling_defaults

from src.baselines.pipeline import CLASSIFIER_NAMES
from src.baselines.predict import list_available_baseline_models, predict_baseline_texts

# Isti stil kao demo_predict — raznovrsni primeri
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

SENT_NAME = {"1": "pos", "0": "neu", "-1": "neg"}
SARC_NAME = {"1": "da", "0": "ne"}


def _fmt(label: str | None, kind: str, probs: dict[str, float] | None = None) -> str:
    """Kratak prikaz labele (+ confidence ako postoji)."""
    if label is None:
        return "—"
    names = SENT_NAME if kind == "sentiment" else SARC_NAME
    pretty = names.get(str(label), str(label))
    if probs and str(label) in probs:
        return f"{label}/{pretty}({probs[str(label)]:.2f})"
    return f"{label}/{pretty}"


def main() -> None:
    """CLI: side-by-side baseline vs BERTić na --text / --file / demo tekstovima."""
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(
        description=(
            "Poređenje baseline (TF-IDF+LR/SVM/NB) i BERTić predikcija. "
            "Baseline: lowercase + ćirilica→latinica. BERTić: bez lowercasing-a."
        )
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--text", action="append", default=None)
    parser.add_argument("--file", default=None, help="TXT: jedan tekst po liniji")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Koristi ugrađene demo rečenice (default ako nema --text/--file)",
    )
    parser.add_argument(
        "--baseline-model",
        choices=list(CLASSIFIER_NAMES),
        default="linear_svm",
        help="Koji baseline klasifikator (default: linear_svm)",
    )
    parser.add_argument(
        "--baselines-dir",
        default=None,
        help="Root sa sentiment|sarcasm/<model>/model.joblib",
    )
    parser.add_argument(
        "--bertic-dir",
        default=None,
        help="Root sa sentiment|sarcasm|multitask/best.pt (default: models/)",
    )
    parser.add_argument("--device", default=None, help="cpu | cuda (samo BERTić)")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Ispis sirovog JSON poređenja",
    )
    args = parser.parse_args()

    texts: list[str] = []
    if args.text:
        texts.extend(args.text)
    if args.file:
        texts.extend(
            line.strip()
            for line in Path(args.file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if args.demo or not texts:
        if not texts:
            texts = list(DEMO_TEXTS)

    config = load_config(args.config)
    bl_cfg = config.get("baselines", {})
    mcfg = modeling_defaults(config)

    baselines_root = (
        resolve_path(args.baselines_dir)
        if args.baselines_dir
        else resolve_path(bl_cfg.get("output_dir", "models/baselines"))
    )
    bertic_root = (
        resolve_path(args.bertic_dir)
        if args.bertic_dir
        else resolve_path(mcfg["output_dir"])
    )

    available_bl = list_available_baseline_models(
        baselines_root,
        models=[args.baseline_model],
    )
    bl_by_task = {task: path for task, model, path in available_bl if model == args.baseline_model}

    bertic_tasks = [
        t
        for t in ("sentiment", "sarcasm", "multitask")
        if (bertic_root / t / "best.pt").exists()
    ]

    if not bl_by_task and not bertic_tasks:
        raise SystemExit(
            "Nema ni baseline .joblib ni BERTić best.pt.\n"
            "  python scripts/baselines/train_baselines.py --task all\n"
            "  python scripts/modeling/prepare_splits.py --csv data/processed/dataset/dataset.csv\n"
            "  python scripts/modeling/train.py --task all"
        )

    print(f"[compare] tekstova: {len(texts)}")
    print(f"[compare] baseline model: {args.baseline_model}")
    print(f"[compare] baseline taskovi: {sorted(bl_by_task) or '—'}")
    print(f"[compare] BERTić taskovi: {bertic_tasks or '—'}")
    print(
        "[compare] pretprocesiranje: "
        "baseline = lowercase+latinica; BERTić = bez lowercasing-a (tekst iz dataseta)"
    )

    baseline_sent: list[dict] | None = None
    baseline_sarc: list[dict] | None = None
    if "sentiment" in bl_by_task:
        print("[compare] baseline sentiment ...")
        baseline_sent = predict_baseline_texts(
            texts,
            config=config,
            task="sentiment",
            model=args.baseline_model,
            output_dir=baselines_root,
        )
    else:
        print(f"[skip] Nema baseline sentiment: {baselines_root / 'sentiment' / args.baseline_model / 'model.joblib'}")

    if "sarcasm" in bl_by_task:
        print("[compare] baseline sarcasm ...")
        baseline_sarc = predict_baseline_texts(
            texts,
            config=config,
            task="sarcasm",
            model=args.baseline_model,
            output_dir=baselines_root,
        )
    else:
        print(f"[skip] Nema baseline sarcasm: {baselines_root / 'sarcasm' / args.baseline_model / 'model.joblib'}")

    bertic: dict[str, list[dict]] = {}
    for task in bertic_tasks:
        print(f"[compare] BERTić {task} ...")
        bertic[task] = predict_texts(
            texts,
            config=config,
            task=task,
            model_dir=bertic_root / task,
            device_name=args.device,
        )

    rows: list[dict] = []
    for i, text in enumerate(texts):
        row: dict = {"text": text, "baseline": {}, "bertic": {}}
        if baseline_sent is not None:
            r = baseline_sent[i]
            row["baseline"]["sentiment"] = r.get("sentiment")
            row["baseline"]["sentiment_probs"] = r.get("sentiment_probs")
            row["baseline"]["text_preprocessed"] = r.get("text_preprocessed")
        if baseline_sarc is not None:
            r = baseline_sarc[i]
            row["baseline"]["sarcasm"] = r.get("sarcasm")
            row["baseline"]["sarcasm_probs"] = r.get("sarcasm_probs")
        for task, preds in bertic.items():
            r = preds[i]
            if task == "multitask":
                row["bertic"]["multitask_sentiment"] = r.get("sentiment")
                row["bertic"]["multitask_sentiment_probs"] = r.get("sentiment_probs")
                row["bertic"]["multitask_sarcasm"] = r.get("sarcasm")
                row["bertic"]["multitask_sarcasm_probs"] = r.get("sarcasm_probs")
            else:
                row["bertic"][task] = r.get(task)
                row["bertic"][f"{task}_probs"] = r.get(f"{task}_probs")
        rows.append(row)

    if args.json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return

    print("\n" + "=" * 110)
    print(f"POREĐENJE  baseline={args.baseline_model}  vs  BERTić")
    print("=" * 110)

    agree_sent = 0
    agree_sarc = 0
    n_sent = 0
    n_sarc = 0

    for i, row in enumerate(rows):
        print(f"\n{i + 1:2d}. {row['text']}")
        bl = row["baseline"]
        bt = row["bertic"]

        bl_s = bl.get("sentiment")
        bt_s = bt.get("sentiment")
        mt_s = bt.get("multitask_sentiment")
        print(
            f"    sentiment | baseline {_fmt(bl_s, 'sentiment', bl.get('sentiment_probs')):22s}"
            f" | BERTić-single {_fmt(bt_s, 'sentiment', bt.get('sentiment_probs')):22s}"
            f" | BERTić-MT {_fmt(mt_s, 'sentiment', bt.get('multitask_sentiment_probs'))}"
        )

        bl_c = bl.get("sarcasm")
        bt_c = bt.get("sarcasm")
        mt_c = bt.get("multitask_sarcasm")
        print(
            f"    sarcasm   | baseline {_fmt(bl_c, 'sarcasm', bl.get('sarcasm_probs')):22s}"
            f" | BERTić-single {_fmt(bt_c, 'sarcasm', bt.get('sarcasm_probs')):22s}"
            f" | BERTić-MT {_fmt(mt_c, 'sarcasm', bt.get('multitask_sarcasm_probs'))}"
        )

        if bl_s is not None and bt_s is not None:
            n_sent += 1
            if str(bl_s) == str(bt_s):
                agree_sent += 1
        if bl_c is not None and bt_c is not None:
            n_sarc += 1
            if str(bl_c) == str(bt_c):
                agree_sarc += 1

    print("\n" + "-" * 110)
    if n_sent:
        print(
            f"Slaganje baseline vs BERTić-single (sentiment): "
            f"{agree_sent}/{n_sent} ({100.0 * agree_sent / n_sent:.1f}%)"
        )
    if n_sarc:
        print(
            f"Slaganje baseline vs BERTić-single (sarcasm):   "
            f"{agree_sarc}/{n_sarc} ({100.0 * agree_sarc / n_sarc:.1f}%)"
        )
    print("Labele: sentiment 1=pos, 0=neu, -1=neg | sarcasm 1=da, 0=ne")


if __name__ == "__main__":
    main()
