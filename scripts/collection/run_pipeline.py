#!/usr/bin/env python3
"""CLI: pun pipeline (kolekcija + preprocess + dataset + stats)."""

from __future__ import annotations

import argparse

from pathlib import Path
import sys

# scripts/<podfolder>/x.py -> project root
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._bootstrap import ensure_project_root

ensure_project_root()

from src.common.config import load_config
from src.common.stdio_utf8 import configure_utf8_stdio
from src.pipeline import run_full_pipeline


def main() -> None:
    """CLI ulazna tačka; poziva ``src.pipeline.run_full_pipeline``.

    Pokreće kolekciju + preprocess + dataset + stats u jednom prolazu.
    """
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description="Ceo data preparation pipeline.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--sources", nargs="*", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    run_full_pipeline(config, sources=args.sources)


if __name__ == "__main__":
    main()
