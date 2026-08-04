#!/usr/bin/env python3
"""CLI: izrada annotation template / finalnog dataseta."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import load_config
from src.common.stdio_utf8 import configure_utf8_stdio
from src.dataset.build import build_annotation_dataset


def main() -> None:
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(
        description="Kreira CSV za rucnu anotaciju (sentiment/sarcasm prazni)."
    )
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    build_annotation_dataset(config)


if __name__ == "__main__":
    main()
