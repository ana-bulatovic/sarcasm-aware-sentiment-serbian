"""Polu-rucno dodavanje TikTok komentara (bez browser scrapinga).

TikTok zabranjuje scraping i zahteva Research API za akademski pristup.
Ova skripta:
1) otvara URL u tvom browseru,
2) ti rucno kopiras javne komentare u TXT (jedan po liniji) ili nalepis u terminal,
3) pipeline ocisti, deduplikuje i dopise na annotation_template.csv.
"""

from __future__ import annotations

import re
import webbrowser
from pathlib import Path
from typing import Any

from src.common.config import ensure_dir, resolve_path
from src.dataset.append_common import append_texts_to_annotation

_VIDEO_ID_RE = re.compile(
    r"(?:video|photo)/(\d+)|[?&]share_item_id=(\d+)|tiktok\.com/.*/(\d{10,})"
)


def extract_tiktok_video_id(url: str) -> str | None:
    url = (url or "").strip()
    if not url:
        return None
    m = _VIDEO_ID_RE.search(url)
    if not m:
        return None
    return next(g for g in m.groups() if g)


def load_comments_file(path: Path) -> list[str]:
    """TXT: jedan komentar po liniji. Prazne linije i # komentari se ignorisu.

    Alternativno: blokovi odvojeni linijom '---'.
    """
    raw = path.read_text(encoding="utf-8")
    if "\n---\n" in raw or raw.startswith("---\n"):
        parts = re.split(r"\n---\n", raw)
        return [p.strip() for p in parts if p.strip() and not p.strip().startswith("#")]
    texts: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        texts.append(line)
    return texts


def read_comments_interactive() -> list[str]:
    print(
        "Nalepi komentare (jedan po liniji). Zavrsi sa praznim redom pa Enter,\n"
        "ili napisi END u novom redu."
    )
    lines: list[str] = []
    empty_streak = 0
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip().upper() == "END":
            break
        if not line.strip():
            empty_streak += 1
            if empty_streak >= 1 and lines:
                break
            continue
        empty_streak = 0
        lines.append(line.strip())
    return lines


def append_tiktok_comments(
    config: dict[str, Any],
    url: str,
    comments_file: str | Path | None = None,
    *,
    open_browser: bool = True,
) -> list[dict[str, str]]:
    url = (url or "").strip()
    if not url:
        raise ValueError("Potreban je TikTok URL (--url).")

    video_id = extract_tiktok_video_id(url)
    if open_browser:
        print(f"[tiktok] Otvaram u browseru: {url}")
        webbrowser.open(url)

    texts: list[str] = []
    if comments_file:
        path = resolve_path(comments_file)
        if not path.exists():
            raise FileNotFoundError(f"Fajl sa komentarima ne postoji: {path}")
        texts = load_comments_file(path)
    else:
        # Predlozi fajl gde korisnik moze da snimi komentare
        suggested = ensure_dir(resolve_path(config.get("collection", {}).get("tiktok", {}).get("comments_dir", "data/external/tiktok/")))
        tip = suggested / "comments_paste.txt"
        print(
            "\nUputstvo:\n"
            "  1) U browseru otvori komentare na TikTok videu.\n"
            "  2) Rucno kopiraj SAMO tekstove komentara (bez username-a).\n"
            "  3) Sačuvaj ih u TXT (jedan komentar po liniji), npr:\n"
            f"     {tip}\n"
            "  4) Pokreni ponovo sa --comments-file, ili nalepi ovde sada.\n"
        )
        choice = input("Nalepi sada u terminal? [y/N]: ").strip().lower()
        if choice in {"y", "yes", "da"}:
            texts = read_comments_interactive()
        else:
            print(
                "OK — sačuvaj komentare u fajl pa pokreni:\n"
                f'  python scripts/append_tiktok.py --url "{url}" '
                f'--comments-file "{tip}"'
            )
            return []

    if not texts:
        print("[tiktok] Nema komentara za dodavanje.")
        return []

    print(f"[tiktok] Ucitano {len(texts)} sirovih komentara.")
    return append_texts_to_annotation(
        config,
        texts,
        source=url,
        metadata={"url": url, "video_id": video_id or "", "platform": "tiktok"},
    )
