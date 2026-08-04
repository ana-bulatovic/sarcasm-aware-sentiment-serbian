#!/usr/bin/env python3
"""CLI: prikupljanje podataka iz konfigurisanih izvora."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
        help="Podskup izvora (npr. senticomments_sr youtube)",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    run_collection(config, sources=args.sources)


if __name__ == "__main__":
    main()
