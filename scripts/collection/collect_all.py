#!/usr/bin/env python3
"""CLI: prikupljanje podataka iz konfigurisanih izvora."""

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

from src.collection.run_collection import run_collection
from src.common.config import load_config
from src.common.stdio_utf8 import configure_utf8_stdio


def main() -> None:
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description="Prikupljanje sirovih tekstova.")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Putanja do YAML konfiguracije",
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="Podskup izvora (npr. youtube tiktok)",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    run_collection(config, sources=args.sources)


if __name__ == "__main__":
    main()
