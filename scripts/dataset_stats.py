#!/usr/bin/env python3
"""CLI: statistike dataseta."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import load_config
from src.common.stdio_utf8 import configure_utf8_stdio
from src.dataset.statistics import compute_dataset_statistics, print_statistics


def main() -> None:
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
