"""Polu-rucno dodavanje Instagram komentara (bez logina i bez scrapinga).

Instagram/Meta ToS zabranjuje automatsko skidanje komentara i zaobilaženje
autentifikacije. Zvaničan put za app-ove je Meta Graph API (sa odobrenjem).

Ova skripta (isto kao TikTok u projektu):
1) otvara URL objave u browseru,
2) ti rucno kopiras javne tekstove komentara u TXT (bez username-a),
3) pipeline ocisti, deduplikuje i dopise na annotation_template.csv.
"""

from __future__ import annotations

import re
import webbrowser
from pathlib import Path
from typing import Any

from src.common.config import ensure_dir, resolve_path
from src.dataset.append_common import append_texts_to_annotation
from src.dataset.append_tiktok import load_comments_file, read_comments_interactive

# /p/CODE/  /reel/CODE/  /tv/CODE/
_SHORTCODE_RE = re.compile(
    r"(?:instagram\.com)/(?:p|reel|tv)/([A-Za-z0-9_-]+)",
    re.IGNORECASE,
)


def extract_instagram_shortcode(url: str) -> str | None:
    """Izvuci shortcode iz /p/, /reel/ ili /tv/ Instagram URL-a."""
    url = (url or "").strip()
    if not url:
        return None
    m = _SHORTCODE_RE.search(url)
    return m.group(1) if m else None


def append_instagram_comments(
    config: dict[str, Any],
    url: str,
    comments_file: str | Path | None = None,
    *,
    open_browser: bool = True,
) -> list[dict[str, str]]:
    """Polu-ručno dodaj Instagram komentare na annotation CSV (source = pun URL)."""
    url = (url or "").strip()
    if not url:
        raise ValueError("Potreban je Instagram URL (--url).")

    shortcode = extract_instagram_shortcode(url)
    if open_browser:
        print(f"[instagram] Otvaram u browseru: {url}")
        webbrowser.open(url)

    texts: list[str] = []
    if comments_file:
        path = resolve_path(comments_file)
        if not path.exists():
            raise FileNotFoundError(f"Fajl sa komentarima ne postoji: {path}")
        texts = load_comments_file(path)
    else:
        ig_cfg = config.get("collection", {}).get("instagram", {})
        suggested = ensure_dir(
            resolve_path(ig_cfg.get("comments_dir", "data/external/instagram/"))
        )
        # Lokalna putanja predloženog TXT-a — NIJE šema kolona `tip`
        tip = suggested / "comments_paste.txt"
        print(
            "\nUputstvo (BEZ scrapinga / BEZ automatskog logina):\n"
            "  1) U browseru otvori komentare na Instagram objavi (uloguj se rucno ako treba).\n"
            "  2) Rucno kopiraj SAMO tekstove komentara (bez username-a).\n"
            "  3) Sačuvaj u TXT (jedan komentar po liniji), npr:\n"
            f"     {tip}\n"
            "  4) Pokreni ponovo sa --comments-file, ili nalepi ovde sada.\n"
        )
        choice = input("Nalepi sada u terminal? [y/N]: ").strip().lower()
        if choice in {"y", "yes", "da"}:
            texts = read_comments_interactive()
        else:
            print(
                "OK — sačuvaj komentare u fajl pa pokreni:\n"
                f'  python scripts/collection/append_instagram.py --url "{url}" '
                f'--comments-file "{tip}"'
            )
            return []

    if not texts:
        print("[instagram] Nema komentara za dodavanje.")
        return []

    print(f"[instagram] Ucitano {len(texts)} sirovih komentara.")
    return append_texts_to_annotation(
        config,
        texts,
        source=url,
        metadata={
            "url": url,
            "shortcode": shortcode or "",
            "platform": "instagram",
        },
    )
