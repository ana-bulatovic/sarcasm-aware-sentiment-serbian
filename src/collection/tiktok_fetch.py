"""Automatsko skidanje TikTok komentara preko Playwright browsera.

Otvara headed Chromium, korisnik se po potrebi uloguje jednom (sesija se čuva).
Hvata odgovore ``/api/comment/list/`` dok skroluje panel komentara.
Ne čuva username autora — samo tekst komentara.

Nije deo COLLECTOR_REGISTRY; koristi se preko append_tiktok / CLI skripti.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

VIDEO_URL_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.|vm\.)?tiktok\.com/.+?(?:video|photo)/(\d+)",
    re.IGNORECASE,
)

_COMMENT_API_MARKERS = (
    "/api/comment/list/",
    "/api/comment/list/reply/",
)


def extract_video_id(url: str) -> str | None:
    """Izvuci aweme/video ID iz TikTok URL-a."""
    url = (url or "").strip()
    if not url:
        return None
    m = VIDEO_URL_PATTERN.search(url)
    if m:
        return m.group(1)
    # fallback: dugačak numerički ID negde u putanji
    m2 = re.search(r"/(\d{10,})", url)
    return m2.group(1) if m2 else None


def normalize_video_url(url: str) -> str:
    """Vrati čist watch URL (bez tracking query-ja ako je moguće)."""
    url = (url or "").strip()
    vid = extract_video_id(url)
    if not vid:
        return url
    # zadrži @user ako postoji
    m = re.search(r"(tiktok\.com/@[^/]+)/(?:video|photo)/\d+", url, re.I)
    if m:
        return f"https://www.{m.group(1)}/video/{vid}"
    return f"https://www.tiktok.com/video/{vid}"


def _texts_from_comment_payload(data: Any) -> list[str]:
    """Izvuci tekstove iz JSON odgovora comment/list API-ja."""
    texts: list[str] = []
    if not isinstance(data, dict):
        return texts

    comments = data.get("comments") or data.get("comment_list") or []
    if not isinstance(comments, list):
        return texts

    for item in comments:
        if not isinstance(item, dict):
            continue
        text = item.get("text") or item.get("share_info", {}).get("desc") or ""
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
        # ugnježđeni replies (ako API vrati)
        for reply in item.get("reply_comment") or item.get("reply_list") or []:
            if not isinstance(reply, dict):
                continue
            rtext = reply.get("text") or ""
            if isinstance(rtext, str) and rtext.strip():
                texts.append(rtext.strip())
    return texts


def _is_comment_api(url: str) -> bool:
    """Da li URL odgovara TikTok comment list endpointu."""
    path = urlparse(url).path or ""
    return any(marker in path for marker in _COMMENT_API_MARKERS)


async def _scroll_comments(page: Any, rounds: int, pause_s: float) -> None:
    """Skroluj panel komentara da pokrene lazy-load API pozive."""
    selectors = [
        '[data-e2e="comment-list"]',
        '[class*="CommentList"]',
        '[class*="DivCommentListContainer"]',
        'div[class*="comment-main"]',
    ]
    panel = None
    for sel in selectors:
        loc = page.locator(sel).first
        try:
            if await loc.count() > 0 and await loc.is_visible():
                panel = loc
                break
        except Exception:
            continue

    for _ in range(max(1, rounds)):
        try:
            if panel is not None:
                await panel.evaluate(
                    "(el) => { el.scrollTop = el.scrollHeight; }"
                )
            else:
                await page.mouse.wheel(0, 2400)
        except Exception:
            try:
                await page.mouse.wheel(0, 2400)
            except Exception:
                pass
        await page.wait_for_timeout(int(pause_s * 1000))


async def fetch_comments_for_url(
    page: Any,
    url: str,
    *,
    max_comments: int = 0,
    scroll_rounds: int = 25,
    scroll_pause_s: float = 1.2,
    settle_s: float = 3.0,
) -> list[str]:
    """Otvori video, skroluj komentare, vrati jedinstvene tekstove."""
    collected: list[str] = []
    seen: set[str] = set()
    state = {"has_more": True, "active": True}

    async def on_response(response: Any) -> None:
        if not state["active"]:
            return
        try:
            if response.status != 200:
                return
            if not _is_comment_api(response.url):
                return
            data = await response.json()
        except Exception:
            return

        for text in _texts_from_comment_payload(data):
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            collected.append(text)

        if isinstance(data, dict) and "has_more" in data:
            state["has_more"] = bool(data.get("has_more"))

    page.on("response", on_response)
    try:
        print(f"[tiktok] Učitavam: {url}")
        await page.goto(url, wait_until="domcontentloaded", timeout=90_000)
        await page.wait_for_timeout(int(settle_s * 1000))

        # Probaj da otvoriš / fokusiraš komentare (desktop UI)
        for sel in (
            '[data-e2e="comment-icon"]',
            'button[data-e2e="browse-comment-icon"]',
            '[data-e2e="comment-level-1"]',
        ):
            try:
                btn = page.locator(sel).first
                if await btn.count() > 0 and await btn.is_visible():
                    await btn.click(timeout=2000)
                    await page.wait_for_timeout(800)
                    break
            except Exception:
                continue

        stagnant = 0
        last_count = 0
        for i in range(max(1, scroll_rounds)):
            if max_comments > 0 and len(collected) >= max_comments:
                break
            await _scroll_comments(page, rounds=1, pause_s=scroll_pause_s)
            if len(collected) == last_count:
                stagnant += 1
            else:
                stagnant = 0
                last_count = len(collected)
                print(f"[tiktok] Skupljeno komentara: {len(collected)} (scroll {i + 1})")
            if stagnant >= 4 and not state["has_more"]:
                break
            if stagnant >= 8:
                break

        # kratko čekanje na poslednje XHR-ove
        await page.wait_for_timeout(1500)
    finally:
        state["active"] = False

    if max_comments > 0:
        return collected[:max_comments]
    return collected


async def fetch_comments_async(
    urls: list[str],
    *,
    user_data_dir: Path,
    headless: bool = False,
    max_comments: int = 0,
    scroll_rounds: int = 25,
    scroll_pause_s: float = 1.2,
    login_wait_s: float = 0.0,
    post_sleep_s: float = 3.0,
) -> dict[str, list[str]]:
    """Za svaki URL skini komentare; vraća mapu video_id -> tekstovi."""
    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise ImportError(
            "Nedostaje playwright. Instaliraj:\n"
            "  pip install playwright\n"
            "  playwright install chromium"
        ) from exc

    user_data_dir = Path(user_data_dir)
    user_data_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, list[str]] = {}
    async with async_playwright() as p:
        context = await p.chromium.launch_persistent_context(
            user_data_dir=str(user_data_dir),
            headless=headless,
            viewport={"width": 1280, "height": 900},
            locale="sr-RS",
            args=["--disable-blink-features=AutomationControlled"],
        )
        page = context.pages[0] if context.pages else await context.new_page()

        # Prvi put: otvori TikTok da korisnik može da se uloguje
        print(
            "[tiktok] Otvaram browser (Chromium).\n"
            "  Ako TikTok traži login / CAPTCHA — uradi to u prozoru,\n"
            "  pa se vrati u terminal."
        )
        await page.goto("https://www.tiktok.com/", wait_until="domcontentloaded")
        if login_wait_s > 0:
            print(f"[tiktok] Čekam {int(login_wait_s)}s za login...")
            await page.wait_for_timeout(int(login_wait_s * 1000))
        else:
            try:
                input(
                    "[tiktok] Kad si spreman/na (ulogovan ako treba), pritisni Enter..."
                )
            except EOFError:
                await page.wait_for_timeout(8000)

        for i, raw_url in enumerate(urls, start=1):
            url = normalize_video_url(raw_url)
            vid = extract_video_id(url) or f"url{i}"
            print(f"\n=== [{i}/{len(urls)}] {url} ===")
            try:
                texts = await fetch_comments_for_url(
                    page,
                    url,
                    max_comments=max_comments,
                    scroll_rounds=scroll_rounds,
                    scroll_pause_s=scroll_pause_s,
                )
            except Exception as exc:
                print(f"[tiktok] Greška za {url}: {type(exc).__name__}: {exc}")
                texts = []
            results[vid] = texts
            print(f"[tiktok] video={vid}: {len(texts)} sirovih komentara")
            if i < len(urls) and post_sleep_s > 0:
                time.sleep(post_sleep_s)

        await context.close()

    return results


def fetch_comments_sync(
    urls: list[str],
    *,
    user_data_dir: Path,
    headless: bool = False,
    max_comments: int = 0,
    scroll_rounds: int = 25,
    scroll_pause_s: float = 1.2,
    login_wait_s: float = 0.0,
    post_sleep_s: float = 3.0,
) -> dict[str, list[str]]:
    """Sinhroni omotač oko ``fetch_comments_async``."""
    import asyncio

    return asyncio.run(
        fetch_comments_async(
            urls,
            user_data_dir=user_data_dir,
            headless=headless,
            max_comments=max_comments,
            scroll_rounds=scroll_rounds,
            scroll_pause_s=scroll_pause_s,
            login_wait_s=login_wait_s,
            post_sleep_s=post_sleep_s,
        )
    )


def dump_debug_payload(path: Path, payload: Any) -> None:
    """Opcioni debug dump JSON-a (za razvoj selektora)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
