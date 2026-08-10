#!/usr/bin/env python3
"""Upis TikTok komentara u poseban CSV (isti format kao youtube_comments)."""

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
from src.dataset.append_tiktok import append_tiktok_comments, append_tiktok_fetch


def main() -> None:
    """CLI ulazna tačka; poziva fetch (Playwright) ili ručni unos.

    Podrazumevano: otvara Chromium, skida komentare → tiktok_comments.csv.
    """
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(
        description=(
            "TikTok komentari -> data/processed/sources/tiktok_comments.csv. "
            "Podrazumevano: Playwright fetch (headed browser). "
            "Za ručni unos: --manual. Ne dira annotation_template.csv."
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
        help="Jedan video URL (inače config/sources/tiktok_urls.txt)",
    )
    parser.add_argument(
        "--urls-file",
        default=None,
        help="TXT sa URL-ovima (default: config/sources/tiktok_urls.txt)",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Ručni unos (browser + paste), bez Playwright fetch-a",
    )
    parser.add_argument(
        "--comments-file",
        default=None,
        help="TXT komentari (samo sa --manual)",
    )
    parser.add_argument("--out", default=None, help="Izlazni CSV")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="(samo --manual) Ne otvaraj sistemski browser",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Pokreni Chromium bez UI (često ne radi bez sačuvane sesije)",
    )
    parser.add_argument(
        "--max-comments",
        type=int,
        default=0,
        help="Max komentara po videu (0 = bez limita / config)",
    )
    parser.add_argument(
        "--scroll-rounds",
        type=int,
        default=0,
        help="Koliko puta skrolovati panel komentara (0 = config)",
    )
    parser.add_argument(
        "--login-wait",
        type=float,
        default=0.0,
        help="Sekunde čekanja za login umesto Enter prompta (0 = Enter)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.manual or args.comments_file:
        append_tiktok_comments(
            config,
            tip=args.tip,
            url=args.url,
            urls_file=args.urls_file,
            comments_file=args.comments_file,
            open_browser=not args.no_browser,
            out_csv=args.out,
        )
        return

    append_tiktok_fetch(
        config,
        tip=args.tip,
        url=args.url,
        urls_file=args.urls_file,
        out_csv=args.out,
        max_comments=args.max_comments,
        headless=args.headless,
        scroll_rounds=args.scroll_rounds,
        login_wait=args.login_wait,
    )


if __name__ == "__main__":
    main()
