#!/usr/bin/env python3
"""Dodaj Instagram komentare na annotation CSV (polu-rucno, bez scrapinga / logina)."""

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
from src.dataset.append_instagram import append_instagram_comments


def main() -> None:
    """CLI ulazna tačka; poziva ``src.dataset.append_instagram.append_instagram_comments``.

    Polu-ručno dodaje Instagram komentare (browser + paste, bez scrapinga).
    """
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(
        description=(
            "Otvara Instagram URL u browseru i dopisuje komentare koje TI rucno "
            "kopiras (ToS: nema automatskog logina ni scrapinga)."
        )
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--url", required=True, help="Instagram post/reel URL")
    parser.add_argument(
        "--comments-file",
        default=None,
        help="TXT fajl: jedan komentar po liniji (bez username-a)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Ne otvaraj browser (samo obradi fajl)",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    append_instagram_comments(
        config,
        url=args.url,
        comments_file=args.comments_file,
        open_browser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
