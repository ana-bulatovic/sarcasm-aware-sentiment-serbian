"""Prikupljanje Twitter/X komentara u POSEBAN CSV (ne u annotation_template).

Podrazumevano: automatski fetch replies preko twikit (--fetch).
Alternativa: --manual (browser + paste / comments-file).

Twitter NIJE u COLLECTOR_REGISTRY — ovaj modul + twitter_fetch su odvojeni put.

Kolone: id, source, text, tip, sentiment, sarcasm
  - source = uvek alias "twitter" (ne pun status URL)
  - tip    = tema (--tip), šema kolona tip
"""

from __future__ import annotations

import re
import webbrowser
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.config import ensure_dir, resolve_path
from src.common.io_utils import save_csv
from src.common.schema import FINAL_COLUMNS, DatasetRecord
from src.dataset.append_common import next_annotation_id
from src.dataset.append_tiktok import load_comments_file, read_comments_interactive
from src.preprocessing.clean import preprocess_text

_ID_PREFIX = "tw"


def load_twitter_urls(path: Path) -> list[str]:
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


def _next_twitter_id(existing_ids: list[str]) -> int:
    """Sledeći broj za tw-XXXXX (fallback na next_annotation_id ako nema tw-)."""
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
    """Učitaj postojeći twitter_comments CSV ili vrati praznu listu."""
    if not out_path.exists():
        return []
    # engine=python + on_bad_lines: ručno uređen CSV sa zarezima u textu
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
    existing = _load_existing(out_path)
    existing_texts = {str(r.get("text", "")).strip() for r in existing}
    next_id = _next_twitter_id([str(r.get("id", "")) for r in existing])

    appended: list[dict[str, str]] = []
    for raw in texts:
        cleaned = preprocess_text(raw, prep_cfg)
        if len(cleaned) < 8:
            continue
        if cleaned in existing_texts:
            continue
        existing_texts.add(cleaned)
        appended.append(
            DatasetRecord(
                id=f"{_ID_PREFIX}-{next_id:05d}",
                source="twitter",
                text=cleaned,
                tip=tip,
                sentiment="",
                sarcasm="",
            ).to_dict()
        )
        next_id += 1

    if not appended:
        print("[twitter] Nema novih tekstova posle čišćenja/dedupa.")
        return []

    combined = existing + appended
    save_csv(combined, out_path, columns=FINAL_COLUMNS)
    print(
        f"[twitter] +{len(appended)} (tip={tip}) -> ukupno {len(combined)} | {out_path}"
    )
    if url_note:
        print(f"[twitter] URL: {url_note}")
    return appended


def append_twitter_fetch(
    config: dict[str, Any],
    *,
    tip: str,
    url: str | None = None,
    urls_file: str | Path | None = None,
    out_csv: str | Path | None = None,
    username: str | None = None,
    email: str | None = None,
    password: str | None = None,
    cookies_file: str | Path | None = None,
    refresh_session: bool = False,
    request_sleep: float = 2.0,
    post_sleep: float = 5.0,
    max_comments: int = 0,
) -> list[dict[str, str]]:
    """Automatski skini replies (twikit) i upiši u twitter_comments.csv.

    tip: šema kolona tip (tema). Ne ide kroz COLLECTOR_REGISTRY.
    """
    tip = (tip or "").strip() or "ostalo"
    tw_cfg = config.get("collection", {}).get("twitter", {})
    default_out = tw_cfg.get("output_csv", "data/processed/sources/twitter_comments.csv")
    default_urls = tw_cfg.get("urls_file", "config/sources/twitter_urls.txt")
    out_path = resolve_path(out_csv or default_out)
    ensure_dir(out_path.parent)

    urls: list[str] = []
    if url and str(url).strip():
        urls = [str(url).strip()]
    else:
        urls_path = resolve_path(urls_file or default_urls)
        urls = load_twitter_urls(urls_path)
        print(f"[twitter] Učitano {len(urls)} URL-ova iz {urls_path}")

    if not urls:
        print("[twitter] Nema URL-ova za fetch.")
        return []

    cookies_path = resolve_path(
        cookies_file
        or tw_cfg.get("cookies_file", "data/external/twitter/session/cookies.json")
    )

    from src.collection.twitter_fetch import fetch_replies_sync

    by_id = fetch_replies_sync(
        urls,
        cookies_file=cookies_path,
        username=username,
        email=email,
        password=password,
        refresh_session=refresh_session,
        request_sleep=request_sleep,
        post_sleep=post_sleep,
        max_comments=max_comments,
    )

    all_appended: list[dict[str, str]] = []
    for tid, texts in by_id.items():
        if not texts:
            continue
        note = f"https://x.com/i/status/{tid}"
        got = _append_texts(
            config, texts, tip=tip, out_path=out_path, url_note=note
        )
        all_appended.extend(got)

    print(f"[twitter] Ukupno dodato u ovoj sesiji: {len(all_appended)}")
    return all_appended


def append_twitter_comments(
    config: dict[str, Any],
    *,
    tip: str,
    url: str | None = None,
    urls_file: str | Path | None = None,
    comments_file: str | Path | None = None,
    open_browser: bool = True,
    out_csv: str | Path | None = None,
) -> list[dict[str, str]]:
    """Ručni / polu-ručni unos replies u twitter_comments.csv (bez mrežnog skidanja)."""
    tip = (tip or "").strip() or "ostalo"
    tw_cfg = config.get("collection", {}).get("twitter", {})
    default_out = tw_cfg.get("output_csv", "data/processed/sources/twitter_comments.csv")
    default_urls = tw_cfg.get("urls_file", "config/sources/twitter_urls.txt")
    out_path = resolve_path(out_csv or default_out)
    ensure_dir(out_path.parent)

    urls: list[str] = []
    if url and str(url).strip():
        urls = [str(url).strip()]
    else:
        urls_path = resolve_path(urls_file or default_urls)
        urls = load_twitter_urls(urls_path)
        if urls:
            print(f"[twitter] Učitano {len(urls)} URL-ova iz {urls_path}")
        elif not comments_file:
            print(
                f"[twitter] Nema URL-ova. Dodaj ih u {urls_path} "
                "ili prosledi --url / --comments-file."
            )
            return []

    if comments_file:
        path = resolve_path(comments_file)
        if not path.exists():
            raise FileNotFoundError(f"Fajl sa komentarima ne postoji: {path}")
        if open_browser and urls:
            print(f"[twitter] Otvaram u browseru: {urls[0]}")
            webbrowser.open(urls[0])
        texts = load_comments_file(path)
        if not texts:
            print("[twitter] Nema komentara u fajlu.")
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
            f"  tip={tip}, source=twitter\n"
        )
        texts = read_comments_interactive()
        if not texts:
            return []
        return _append_texts(config, texts, tip=tip, out_path=out_path)

    all_appended: list[dict[str, str]] = []
    for i, u in enumerate(urls, start=1):
        print(f"\n=== [{i}/{len(urls)}] {u} ===")
        if open_browser:
            print("[twitter] Otvaram u browseru...")
            webbrowser.open(u)
        print(
            "Uputstvo:\n"
            "  1) U browseru otvori odgovore (uloguj se rucno na X ako traži).\n"
            "  2) Kopiraj SAMO tekstove (bez @username).\n"
            "  3) Nalepi ovde (jedan po liniji). Zavrsi sa praznim redom ili END.\n"
            "  4) Napisi SKIP da preskocis ovaj URL.\n"
        )
        first = input("Nalepi sada? [Y/skip/n]: ").strip().lower()
        if first in {"skip", "s"}:
            print("[twitter] Preskočeno.")
            continue
        if first in {"n", "no"}:
            print("[twitter] Prekid liste URL-ova.")
            break
        texts = read_comments_interactive()
        if not texts:
            print("[twitter] Prazan unos — sledeći URL.")
            continue
        got = _append_texts(
            config, texts, tip=tip, out_path=out_path, url_note=u
        )
        all_appended.extend(got)

    print(f"[twitter] Ukupno dodato u ovoj sesiji: {len(all_appended)}")
    return all_appended
