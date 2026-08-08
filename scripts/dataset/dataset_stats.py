#!/usr/bin/env python3
"""CLI: statistike dataseta."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._bootstrap import ensure_project_root

ensure_project_root()

from src.common.config import load_config
from src.common.stdio_utf8 import configure_utf8_stdio
from src.dataset.statistics import compute_dataset_statistics, print_statistics


def main() -> None:
    """CLI ulazna tačka; poziva ``compute_dataset_statistics`` / ``print_statistics``.

    Računa i ispisuje brojčane statistike annotation / dataset CSV-a.
    """
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description="Statistike annotation dataseta.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--csv",
        default=None,
        help="Opciona putanja do CSV-a (podrazumevano iz config-a)",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    stats = compute_dataset_statistics(config=config, csv_path=args.csv)
    print_statistics(stats)


if __name__ == "__main__":
    main()
