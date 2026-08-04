#!/usr/bin/env python3
"""Dodaj komentare samo sa NOVIH YouTube videa na postojeci annotation CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.config import load_config
from src.common.stdio_utf8 import configure_utf8_stdio
from src.dataset.append_youtube import append_new_youtube_videos


def main() -> None:
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(
        description=(
            "Append YouTube komentara na annotation_template.csv. "
            "Podrazumevano: samo ID-evi iz youtube_video_ids.txt koji jos nisu skupljeni. "
            "Opciono: --video-id / --url za konkretne nove videe."
        )
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--video-id",
        action="append",
        default=None,
        help="Konkretan video ID ili URL (moze vise puta)",
    )
    parser.add_argument(
        "--url",
        action="append",
        default=None,
        help="Isto kao --video-id (alias)",
    )
    args = parser.parse_args()
    config = load_config(args.config)

    only = None
    if args.video_id or args.url:
        only = list(args.video_id or []) + list(args.url or [])

    append_new_youtube_videos(config, only_video_ids=only)


if __name__ == "__main__":
    main()
