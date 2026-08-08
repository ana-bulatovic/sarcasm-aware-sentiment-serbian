"""Polu-rucno dodavanje Reddit komentara (bez scrapinga / neovlascenog API-ja).

Za akademsko istraživanje Reddit preferira Reddit for Researchers (RFR).
Ova skripta ne koristi PRAW scrap ni Data API za research:

1) otvara URL threada u browseru,
2) ti rucno kopiras javne tekstove komentara u TXT (bez username-a),
3) pipeline ocisti, deduplikuje i dopise na annotation_template.csv.

Alternativa: stavi RFR/odobreni eksport u data/external/reddit/export.jsonl
pa pokreni collection pipeline.
"""

from __future__ import annotations

import re
import webbrowser
from pathlib import Path
from typing import Any

from src.common.config import ensure_dir, resolve_path
from src.dataset.append_common import append_texts_to_annotation
from src.dataset.append_tiktok import load_comments_file, read_comments_interactive

# reddit.com/r/sub/comments/ID/...  ili redd.it/ID
_THREAD_RE = re.compile(
    r"(?:reddit\.com/r/[^/]+/comments/([a-z0-9]+))"
    r"|(?:redd\.it/([a-z0-9]+))"
    r"|(?:reddit\.com/comments/([a-z0-9]+))",
    re.IGNORECASE,
)


def extract_reddit_thread_id(url: str) -> str | None:
    """Izvuci thread ID iz reddit.com / redd.it URL-a."""
    url = (url or "").strip()
    if not url:
        return None
    m = _THREAD_RE.search(url)
    if not m:
        return None
    return next(g for g in m.groups() if g)


def append_reddit_comments(
    config: dict[str, Any],
    url: str,
    comments_file: str | Path | None = None,
    *,
    open_browser: bool = True,
) -> list[dict[str, str]]:
    """Polu-ručno dodaj Reddit komentare na annotation CSV (source = pun URL)."""
    url = (url or "").strip()
    if not url:
        raise ValueError("Potreban je Reddit thread URL (--url).")

    thread_id = extract_reddit_thread_id(url)
    if open_browser:
        print(f"[reddit] Otvaram u browseru: {url}")
        webbrowser.open(url)

    texts: list[str] = []
    if comments_file:
        path = resolve_path(comments_file)
        if not path.exists():
            raise FileNotFoundError(f"Fajl sa komentarima ne postoji: {path}")
        texts = load_comments_file(path)
    else:
        reddit_cfg = config.get("collection", {}).get("reddit", {})
        suggested = ensure_dir(
            resolve_path(reddit_cfg.get("comments_dir", "data/external/reddit/"))
        )
        # Lokalna putanja predloženog TXT-a — NIJE šema kolona `tip`
        tip = suggested / "comments_paste.txt"
        print(
            "\nUputstvo (BEZ scrapinga / BEZ neovlascenog API-ja):\n"
            "  1) U browseru otvori Reddit thread i komentare.\n"
            "  2) Rucno kopiraj SAMO tekstove komentara (bez username-a).\n"
            "  3) Sačuvaj u TXT (jedan komentar po liniji), npr:\n"
            f"     {tip}\n"
            "  4) Pokreni ponovo sa --comments-file, ili nalepi ovde sada.\n"
            "\nZa veći akademski eksport: Reddit for Researchers (RFR),\n"
            "  pa data/external/reddit/export.jsonl + collection pipeline.\n"
        )
        choice = input("Nalepi sada u terminal? [y/N]: ").strip().lower()
        if choice in {"y", "yes", "da"}:
            texts = read_comments_interactive()
        else:
            print(
                "OK — sačuvaj komentare u fajl pa pokreni:\n"
                f'  python scripts/collection/append_reddit.py --url "{url}" '
                f'--comments-file "{tip}"'
            )
            return []

    if not texts:
        print("[reddit] Nema komentara za dodavanje.")
        return []

    print(f"[reddit] Ucitano {len(texts)} sirovih komentara.")
    return append_texts_to_annotation(
        config,
        texts,
        source=url,
        metadata={
            "url": url,
            "thread_id": thread_id or "",
            "platform": "reddit",
        },
    )
