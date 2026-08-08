#!/usr/bin/env python3
"""Fine-tune: sentiment | sarcasm | multitask | all."""

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
from src.modeling.runner import run_all_tasks, run_training


def main() -> None:
    """CLI ulazna tačka; poziva ``run_training`` ili ``run_all_tasks``.

    Fine-tune BERTić (sentiment / sarcasm / multitask) prema config i CLI flagovima.
    """
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune BERTić (ili drugog HF encodera): "
            "single-task sentiment, single-task sarkazam, ili multitask."
        )
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--task",
        choices=["sentiment", "sarcasm", "multitask", "all"],
        default="all",
        help="Koji model(e) trenirati (default: all)",
    )
    parser.add_argument(
        "--splits-dir",
        default=None,
        help="Folder sa train/val/test.csv (default iz config-a)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Root za checkpointe (default: models/)",
    )
    parser.add_argument("--model-name", default=None, help="HF model id (npr. classla/bcms-bertic, jerteh/Jerteh-81)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None, help="cpu | cuda | cuda:0 ...")
    parser.add_argument(
        "--list-encoders",
        action="store_true",
        help="Ispisi known_encoders iz configa i izadji",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.list_encoders:
        known = (config.get("modeling") or {}).get("known_encoders") or [
            "classla/bcms-bertic",
            "jerteh/Jerteh-81",
            "jerteh/Jerteh-355",
        ]
        current = (config.get("modeling") or {}).get("model_name", "")
        print("Poznati encoderi:")
        for name in known:
            mark = " (default)" if name == current else ""
            print(f"  - {name}{mark}")
        return
    splits_dir = resolve_path(args.splits_dir) if args.splits_dir else None
    output_dir = resolve_path(args.output_dir) if args.output_dir else None

    kwargs = dict(
        splits_dir=splits_dir,
        output_dir=output_dir,
        model_name=args.model_name,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        device_name=args.device,
    )

    if args.task == "all":
        run_all_tasks(config, **kwargs)
    else:
        run_training(config, task=args.task, **kwargs)


if __name__ == "__main__":
    main()
