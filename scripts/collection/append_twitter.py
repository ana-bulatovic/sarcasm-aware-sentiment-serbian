#!/usr/bin/env python3
"""Upis Twitter/X komentara u poseban CSV (isti format kao annotation_template)."""

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
from src.dataset.append_twitter import append_twitter_comments, append_twitter_fetch


def main() -> None:
    """CLI ulazna tačka; poziva ``append_twitter_fetch`` ili ``append_twitter_comments``.

    Podrazumevano twikit fetch; sa ``--manual`` / ``--comments-file`` ručni unos.
    """
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(
        description=(
            "Twitter/X replies -> data/processed/sources/twitter_comments.csv. "
            "Podrazumevano: --fetch (twikit). Za ručni unos: --manual."
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
        help="Jedan URL (inače config/sources/twitter_urls.txt)",
    )
    parser.add_argument(
        "--urls-file",
        default=None,
        help="TXT sa URL-ovima (default: config/sources/twitter_urls.txt)",
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Ručni unos (browser + paste), bez twikit fetch-a",
    )
    parser.add_argument(
        "--comments-file",
        default=None,
        help="TXT komentari (samo sa --manual)",
    )
    parser.add_argument("--out", default=None, help="Izlazni CSV")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--username", default=None, help="X username")
    parser.add_argument("--email", default=None, help="X email")
    parser.add_argument("--password", default=None, help="X password")
    parser.add_argument(
        "--cookies-file",
        default=None,
        help="JSON cookies (default: data/external/twitter/session/cookies.json)",
    )
    parser.add_argument(
        "--refresh-session",
        action="store_true",
        help="Ignoriši sačuvane cookies i uloguj se ponovo",
    )
    parser.add_argument(
        "--request-sleep",
        type=float,
        default=2.0,
        help="Pauza između X API poziva (s)",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=5.0,
        help="Pauza između postova (s)",
    )
    parser.add_argument(
        "--max-comments",
        type=int,
        default=0,
        help="Max replies po tweetu (0 = bez limita)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.manual or args.comments_file:
        append_twitter_comments(
            config,
            tip=args.tip,
            url=args.url,
            urls_file=args.urls_file,
            comments_file=args.comments_file,
            open_browser=not args.no_browser,
            out_csv=args.out,
        )
        return

    append_twitter_fetch(
        config,
        tip=args.tip,
        url=args.url,
        urls_file=args.urls_file,
        out_csv=args.out,
        username=args.username,
        email=args.email,
        password=args.password,
        cookies_file=args.cookies_file,
        refresh_session=args.refresh_session,
        request_sleep=args.request_sleep,
        post_sleep=args.sleep,
        max_comments=args.max_comments,
    )


if __name__ == "__main__":
    main()
