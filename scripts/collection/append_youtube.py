#!/usr/bin/env python3
"""Upis YouTube komentara u poseban CSV (isti format kao annotation_template)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._bootstrap import ensure_project_root

ensure_project_root()

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from src.common.config import load_config
from src.common.stdio_utf8 import configure_utf8_stdio
from src.dataset.append_youtube import append_youtube_fetch


def main() -> None:
    """CLI ulazna tačka; poziva ``src.dataset.append_youtube.append_youtube_fetch``.

    Preuzima YouTube komentare u poseban CSV (ne dira annotation template).
    """
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(
        description=(
            "YouTube komentari -> data/processed/sources/youtube_comments.csv. "
            "Zahteva YOUTUBE_API_KEY. Ne dira annotation_template.csv."
        )
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--tip",
        required=True,
        help="Tema / subject (npr. politika, filmovi, sport)",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Jedan video URL ili ID (inače config/sources/youtube_video_ids.txt)",
    )
    parser.add_argument(
        "--urls-file",
        default=None,
        help="TXT sa video ID/URL (default: config/sources/youtube_video_ids.txt)",
    )
    parser.add_argument("--out", default=None, help="Izlazni CSV")
    parser.add_argument(
        "--max-comments",
        type=int,
        default=0,
        help="Max komentara po videu (0 = config max_comments_per_video)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    append_youtube_fetch(
        config,
        tip=args.tip,
        url=args.url,
        urls_file=args.urls_file,
        out_csv=args.out,
        max_comments=args.max_comments,
    )


if __name__ == "__main__":
    main()
