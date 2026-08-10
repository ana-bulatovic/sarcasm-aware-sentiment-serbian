"""Prikupljanje TikTok komentara u POSEBAN CSV (ne u annotation_template).

Podrazumevano: automatski fetch preko Playwright (headed browser + intercept
``/api/comment/list/``). Alternativa: ``--manual`` (paste / comments-file).

Kolone: id, source, text, tip, sentiment, sarcasm
  - source = uvek alias "tiktok" (ne pun video URL)
  - tip    = tema (--tip), šema kolona tip
"""

from __future__ import annotations

import re
import webbrowser
from pathlib import Path
from typing import Any

import pandas as pd

from src.collection.base import text_fingerprint
from src.common.config import ensure_dir, resolve_path
from src.common.io_utils import save_csv
from src.common.language import is_likely_serbian
from src.common.schema import FINAL_COLUMNS, DatasetRecord
from src.dataset.append_common import next_annotation_id
from src.preprocessing.clean import preprocess_text
from src.preprocessing.deduplicate import normalize_for_dedup

_ID_PREFIX = "tt"

_VIDEO_ID_RE = re.compile(
    r"(?:video|photo)/(\d+)|[?&]share_item_id=(\d+)|tiktok\.com/.*/(\d{10,})"
)


def extract_tiktok_video_id(url: str) -> str | None:
    """Izvuci video/photo ID iz TikTok URL-a."""
    url = (url or "").strip()
    if not url:
        return None
    m = _VIDEO_ID_RE.search(url)
    if not m:
        return None
    return next(g for g in m.groups() if g)


def load_tiktok_urls(path: Path) -> list[str]:
    """TXT: jedan URL po liniji; # komentari se ignorišu."""
    if not path.exists():
        return []
    urls: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls


def load_comments_file(path: Path) -> list[str]:
    """TXT: jedan komentar po liniji. Prazne linije i # komentari se ignorišu.

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
    """Učitaj komentare sa stdin (jedan po liniji; prazan red ili END završava)."""
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


def _next_tiktok_id(existing_ids: list[str]) -> int:
    """Sledeći broj za tt-XXXXX (fallback na next_annotation_id ako nema tt-)."""
    max_n = 0
    pat = re.compile(rf"^{_ID_PREFIX}-(\d+)$", re.I)
    for raw in existing_ids:
        m = pat.match(str(raw).strip())
        if m:
            max_n = max(max_n, int(m.group(1)))
    if max_n == 0:
        return next_annotation_id(existing_ids)
    return max_n + 1


def _load_existing(out_path: Path) -> list[dict[str, str]]:
    """Učitaj postojeći tiktok_comments CSV ili vrati praznu listu."""
    if not out_path.exists():
        return []
    return (
        pd.read_csv(
            out_path,
            encoding="utf-8-sig",
            dtype=str,
            engine="python",
            on_bad_lines="warn",
        )
        .fillna("")
        .to_dict(orient="records")
    )


def _append_texts(
    config: dict[str, Any],
    texts: list[str],
    *,
    tip: str,
    out_path: Path,
    url_note: str = "",
) -> list[dict[str, str]]:
    """Očisti/dedup tekstove i dopiši u out_path; tip = šema kolona (tema)."""
    prep_cfg = config.get("preprocessing", {})
    lang_cfg = config.get("language", {})
    min_len = int(lang_cfg.get("min_text_length", 15))
    max_len = int(lang_cfg.get("max_text_length", 2000))

    existing = _load_existing(out_path)
    existing_texts = {normalize_for_dedup(str(r.get("text", ""))) for r in existing}
    existing_fps = {text_fingerprint(str(r.get("text", ""))) for r in existing}
    next_id = _next_tiktok_id([str(r.get("id", "")) for r in existing])

    appended: list[dict[str, str]] = []
    for raw in texts:
        cleaned = preprocess_text(raw, prep_cfg)
        if len(cleaned) < min_len or len(cleaned) > max_len:
            continue
        if not is_likely_serbian(cleaned, lang_cfg):
            continue
        key = normalize_for_dedup(cleaned)
        fp = text_fingerprint(cleaned)
        if key in existing_texts or fp in existing_fps:
            continue
        existing_texts.add(key)
        existing_fps.add(fp)
        appended.append(
            DatasetRecord(
                id=f"{_ID_PREFIX}-{next_id:05d}",
                source="tiktok",
                text=cleaned,
                tip=tip,
                sentiment="",
                sarcasm="",
            ).to_dict()
        )
        next_id += 1

    if not appended:
        print("[tiktok] Nema novih tekstova posle čišćenja/dedupa.")
        return []

    combined = existing + appended
    save_csv(combined, out_path, columns=FINAL_COLUMNS)
    print(
        f"[tiktok] +{len(appended)} (tip={tip}) -> ukupno {len(combined)} | {out_path}"
    )
    if url_note:
        print(f"[tiktok] video: {url_note}")
    return appended


def append_tiktok_fetch(
    config: dict[str, Any],
    *,
    tip: str,
    url: str | None = None,
    urls_file: str | Path | None = None,
    out_csv: str | Path | None = None,
    max_comments: int = 0,
    headless: bool = False,
    scroll_rounds: int = 0,
    login_wait: float = 0.0,
) -> list[dict[str, str]]:
    """Automatski skini komentare (Playwright) i upiši u tiktok_comments.csv."""
    tip = (tip or "").strip() or "ostalo"
    tt_cfg = config.get("collection", {}).get("tiktok", {})
    default_out = tt_cfg.get("output_csv", "data/processed/sources/tiktok_comments.csv")
    default_urls = tt_cfg.get("urls_file", "config/sources/tiktok_urls.txt")
    out_path = resolve_path(out_csv or default_out)
    ensure_dir(out_path.parent)

    urls: list[str] = []
    if url and str(url).strip():
        urls = [str(url).strip()]
    else:
        urls_path = resolve_path(urls_file or default_urls)
        urls = load_tiktok_urls(urls_path)
        print(f"[tiktok] Učitano {len(urls)} URL-ova iz {urls_path}")

    if not urls:
        print("[tiktok] Nema URL-ova za fetch.")
        return []

    user_data = resolve_path(
        tt_cfg.get("user_data_dir", "data/external/tiktok/session/browser")
    )
    rounds = (
        int(scroll_rounds)
        if scroll_rounds > 0
        else int(tt_cfg.get("scroll_rounds", 25))
    )
    per_video = (
        int(max_comments)
        if max_comments > 0
        else int(tt_cfg.get("max_comments_per_video", 0))
    )

    from src.collection.tiktok_fetch import fetch_comments_sync

    by_id = fetch_comments_sync(
        urls,
        user_data_dir=user_data,
        headless=headless,
        max_comments=per_video,
        scroll_rounds=rounds,
        scroll_pause_s=float(tt_cfg.get("scroll_pause_seconds", 1.2)),
        login_wait_s=float(login_wait),
        post_sleep_s=float(tt_cfg.get("post_sleep_seconds", 3.0)),
    )

    all_appended: list[dict[str, str]] = []
    for vid, texts in by_id.items():
        if not texts:
            print(f"[tiktok] Nema komentara za video={vid}")
            continue
        note = f"https://www.tiktok.com/video/{vid}"
        got = _append_texts(
            config, texts, tip=tip, out_path=out_path, url_note=note
        )
        all_appended.extend(got)

    print(f"[tiktok] Ukupno dodato u ovoj sesiji: {len(all_appended)}")
    return all_appended


def append_tiktok_comments(
    config: dict[str, Any],
    *,
    tip: str,
    url: str | None = None,
    urls_file: str | Path | None = None,
    comments_file: str | Path | None = None,
    open_browser: bool = True,
    out_csv: str | Path | None = None,
) -> list[dict[str, str]]:
    """Ručni / polu-ručni unos komentara u tiktok_comments.csv (bez mrežnog skidanja)."""
    tip = (tip or "").strip() or "ostalo"
    tt_cfg = config.get("collection", {}).get("tiktok", {})
    default_out = tt_cfg.get("output_csv", "data/processed/sources/tiktok_comments.csv")
    default_urls = tt_cfg.get("urls_file", "config/sources/tiktok_urls.txt")
    out_path = resolve_path(out_csv or default_out)
    ensure_dir(out_path.parent)

    urls: list[str] = []
    if url and str(url).strip():
        urls = [str(url).strip()]
    else:
        urls_path = resolve_path(urls_file or default_urls)
        urls = load_tiktok_urls(urls_path)
        if urls:
            print(f"[tiktok] Učitano {len(urls)} URL-ova iz {urls_path}")
        elif not comments_file:
            print(
                f"[tiktok] Nema URL-ova. Dodaj ih u {urls_path} "
                "ili prosledi --url / --comments-file."
            )
            return []

    if comments_file:
        path = resolve_path(comments_file)
        if not path.exists():
            raise FileNotFoundError(f"Fajl sa komentarima ne postoji: {path}")
        if open_browser and urls:
            print(f"[tiktok] Otvaram u browseru: {urls[0]}")
            webbrowser.open(urls[0])
        texts = load_comments_file(path)
        if not texts:
            print("[tiktok] Nema komentara u fajlu.")
            return []
        return _append_texts(
            config,
            texts,
            tip=tip,
            out_path=out_path,
            url_note=urls[0] if urls else "",
        )

    if not urls:
        print(
            "\nNema URL liste. Nalepi komentare sada (jedan po liniji, END za kraj).\n"
            f"  tip={tip}, source=tiktok\n"
        )
        texts = read_comments_interactive()
        if not texts:
            return []
        return _append_texts(config, texts, tip=tip, out_path=out_path)

    all_appended: list[dict[str, str]] = []
    for i, u in enumerate(urls, start=1):
        print(f"\n=== [{i}/{len(urls)}] {u} ===")
        if open_browser:
            print("[tiktok] Otvaram u browseru...")
            webbrowser.open(u)
        print(
            "Uputstvo:\n"
            "  1) U browseru otvori komentare na TikTok videu "
            "(uloguj se rucno ako traži).\n"
            "  2) Kopiraj SAMO tekstove komentara (bez username-a).\n"
            "  3) Nalepi ovde (jedan po liniji). Zavrsi sa praznim redom ili END.\n"
            "  4) Napisi SKIP da preskocis ovaj URL.\n"
        )
        first = input("Nalepi sada? [Y/skip/n]: ").strip().lower()
        if first in {"skip", "s"}:
            print("[tiktok] Preskočeno.")
            continue
        if first in {"n", "no"}:
            print("[tiktok] Prekid liste URL-ova.")
            break
        texts = read_comments_interactive()
        if not texts:
            print("[tiktok] Prazan unos — sledeći URL.")
            continue
        got = _append_texts(
            config, texts, tip=tip, out_path=out_path, url_note=u
        )
        all_appended.extend(got)

    print(f"[tiktok] Ukupno dodato u ovoj sesiji: {len(all_appended)}")
    return all_appended
