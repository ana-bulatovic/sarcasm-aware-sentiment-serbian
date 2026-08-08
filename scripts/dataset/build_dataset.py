#!/usr/bin/env python3
"""CLI: izrada annotation template / finalnog dataseta."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# scripts/<podfolder>/x.py -> project root
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._bootstrap import ensure_project_root

ensure_project_root()

from src.common.config import load_config
from src.common.stdio_utf8 import configure_utf8_stdio
from src.dataset.build import build_annotation_dataset, build_dataset_from_sources


def main() -> None:
    """CLI ulazna tačka za izradu finalnog dataseta.

    Podrazumevano spaja ``processed/sources/*_comments.csv``.
    ``--from-interim`` koristi stari cleaned.jsonl → annotation pipeline.
    """
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(
        description="Kreira finalni dataset CSV (iz source komentara ili interim)."
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--from-interim",
        action="store_true",
        help="Stari tok: interim/cleaned.jsonl → annotation + dataset (id=sr-XXXXX).",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.from_interim:
        build_annotation_dataset(config)
    else:
        build_dataset_from_sources(config)


if __name__ == "__main__":
    main()
