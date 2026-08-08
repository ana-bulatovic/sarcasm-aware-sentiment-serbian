#!/usr/bin/env python3
"""Treniraj klasične ML baseline modele (TF-IDF + NB / LR / Linear SVM).

Pipeline: tekst → baseline preprocessing → TF-IDF → klasifikator

Taskovi: sentiment (-1/0/1) i sarcasm (0/1).
CSV: id,source|url,text,tip|topic,sentiment,sarcasm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._bootstrap import ensure_project_root

ensure_project_root()

from src.baselines.pipeline import CLASSIFIER_NAMES
from src.baselines.runner import run_baseline_experiments
from src.common.config import load_config, resolve_path
from src.common.stdio_utf8 import configure_utf8_stdio


def main() -> None:
    """CLI ulazna tačka; poziva ``src.baselines.runner.run_baseline_experiments``.

    Trenira TF-IDF + NB/LR/SVM za sentiment i/ili sarcasm i čuva metrike.
    """
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(
        description=(
            "Baseline eksperimenti: Naive Bayes, Logistic Regression, Linear SVM. "
            "Samo za klasične ML modele (ne BERTić)."
        )
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--csv",
        default=None,
        help="Ulazni CSV (default: paths.dataset_csv iz config-a)",
    )
    parser.add_argument(
        "--task",
        choices=["sentiment", "sarcasm", "all"],
        default="all",
        help="Koji task(ove) trenirati (default: all)",
    )
    parser.add_argument(
        "--model",
        choices=[*CLASSIFIER_NAMES, "all"],
        default="all",
        help="Klasifikator (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Folder za rezultate (default: models/baselines)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=None,
        help="Udeo test skupa za stratified split (default iz config-a)",
    )
    parser.add_argument(
        "--no-save-model",
        action="store_true",
        help="Ne čuvaj .joblib modele (samo metrike i predikcije)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    csv_path = resolve_path(args.csv) if args.csv else None
    output_dir = resolve_path(args.output_dir) if args.output_dir else None

    models = None if args.model == "all" else [args.model]

    run_baseline_experiments(
        config,
        csv_path=csv_path,
        tasks=args.task,
        models=models,
        output_dir=output_dir,
        test_size=args.test_size,
        save_model=False if args.no_save_model else None,
    )


if __name__ == "__main__":
    main()
