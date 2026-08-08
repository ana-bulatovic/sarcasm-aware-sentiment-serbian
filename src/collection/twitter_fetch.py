"""Automatsko skidanje X/Twitter replies preko twikit (neslužbeni klijent).

Zahteva ulogovan nalog (cookies ili X_USERNAME / X_PASSWORD).
Ne čuva username autora — samo tekst odgovora.

Nije deo COLLECTOR_REGISTRY; koristi se preko append_twitter / CLI skripti.
"""

from __future__ import annotations

import asyncio
import getpass
import json
import os
import re
from pathlib import Path
from typing import Any

TWEET_URL_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.|mobile\.)?(?:twitter|x)\.com/[^/]+/status(?:es)?/(\d+)",
    re.IGNORECASE,
)


def _patch_twikit_json() -> None:
    """Ublažava KeyError-e kada X GraphQL odgovori nemaju očekivane ključeve."""

    class SafeDict(dict):
        """Dict koji za nedostajuće ključeve vraća prazan SafeDict umesto KeyError.

        Namerno je falsy (``bool`` → False) da twikit uslovi tipa ``if data.get("x")``
        ne padnu kada GraphQL polje ne postoji.
        """

        def __getitem__(self, key):  # type: ignore[no-untyped-def]
            try:
                val = super().__getitem__(key)
            except KeyError:
                return SafeDict()
            if isinstance(val, dict) and not isinstance(val, SafeDict):
                return SafeDict(val)
            return val

        def get(self, key, default=None):  # type: ignore[no-untyped-def]
            """Kao dict.get, ali ugnježdene dict-ove pretvara u SafeDict."""
            if key in self:
                val = self[key]
                if isinstance(val, dict) and not isinstance(val, SafeDict):
                    return SafeDict(val)
                return val
            return SafeDict() if default is None else default

        def __str__(self) -> str:
            return ""

        def __bool__(self) -> bool:
            return False

        def __iter__(self):  # type: ignore[no-untyped-def]
            return iter([])

    orig_loads = json.loads

    def patched_loads(s, *args, **kwargs):  # type: ignore[no-untyped-def]
        """json.loads omotač: svi objekti postaju SafeDict."""
        if "object_hook" not in kwargs:
            kwargs["object_hook"] = lambda d: SafeDict(d)
        res = orig_loads(s, *args, **kwargs)
        if isinstance(res, dict) and not isinstance(res, SafeDict):
            return SafeDict(res)
        return res

    json.loads = patched_loads  # type: ignore[assignment]


def _patch_twikit_client_transaction() -> None:
    """X menja webpack format; twikit često ne nađe KEY_BYTE → nema self.key.

    Vidi: https://github.com/d60/twikit/issues/408
    """
    try:
        import twikit.x_client_transaction.transaction as tx_mod
    except ImportError:
        return

    # Stari format: "ondemand.s":"hash"
    old_file_regex = re.compile(
        r"""['\"]ondemand\.s['\"]\s*:\s*['\"]([\w]+)['\"]""",
        flags=(re.VERBOSE | re.MULTILINE),
    )
    # Novi format: ,123:"ondemand.s" + hash na istom indeksu
    new_file_regex = re.compile(
        r""",(\d+)\s*:\s*["']ondemand\.s["']""",
        flags=(re.VERBOSE | re.MULTILINE),
    )
    # Chunk name map: 123:"ondemand.s" (bez zareza ispred)
    chunk_name_regex = re.compile(
        r"""["']?(\d+)["']?\s*:\s*["']ondemand\.s["']""",
        flags=(re.VERBOSE | re.MULTILINE),
    )
    hash_pattern = r""",?\s*{}\s*:\s*["']([0-9a-fA-F]+)["']"""
    indices_regex = re.compile(r"(\(\w{1,2}\[(\d{1,2})\],\s*16\))+")

    def _orig_init(self):  # type: ignore[no-untyped-def]
        self.home_page_response = None
        self.key = None
        self.key_bytes = None
        self.animation_key = None

    tx_mod.ClientTransaction.__init__ = _orig_init  # type: ignore[method-assign]

    _orig_ct_init = tx_mod.ClientTransaction.init

    async def _patched_init(self, session, headers):  # type: ignore[no-untyped-def]
        try:
            return await _orig_ct_init(self, session, headers)
        except Exception:
            # Ako get_indices padne posle setovanja home_page_response,
            # sledeći request misli da je init gotov → 'no attribute key'.
            self.home_page_response = None
            self.key = None
            raise

    tx_mod.ClientTransaction.init = _patched_init  # type: ignore[method-assign]

    async def _patched_get_indices(self, home_page_response, session, headers):  # type: ignore[no-untyped-def]
        key_byte_indices: list[str] = []
        response = self.validate_response(home_page_response) or self.home_page_response
        response_str = str(response)
        filename: str | None = None

        m_old = old_file_regex.search(response_str)
        if m_old:
            filename = m_old.group(1)

        if not filename:
            m_new = new_file_regex.search(response_str) or chunk_name_regex.search(
                response_str
            )
            if m_new:
                idx = m_new.group(1)
                hm = re.compile(hash_pattern.format(idx)).search(response_str)
                # hash map često ima više pogodaka; uzmi prvi koji nije "ondemand"
                if not hm:
                    # pokušaj sve hash-eve uz taj indeks u loose formi
                    for cand in re.finditer(
                        rf"""["']?{idx}["']?\s*:\s*["']([0-9a-fA-F]{{8,}})["']""",
                        response_str,
                    ):
                        if cand.group(1).lower() != "ondemand":
                            hm = cand
                            break
                if hm:
                    filename = hm.group(1)

        if not filename:
            raise Exception(
                "Couldn't get KEY_BYTE indices (ondemand.s hash not found). "
                "X je možda promenio frontend — ažuriraj twikit ili zakrpu."
            )

        on_demand_file_url = (
            "https://abs.twimg.com/responsive-web/client-web/"
            f"ondemand.s.{filename}a.js"
        )
        on_demand_file_response = await session.request(
            method="GET", url=on_demand_file_url, headers=headers
        )
        for item in indices_regex.finditer(str(on_demand_file_response.text)):
            key_byte_indices.append(item.group(2))

        if not key_byte_indices:
            raise Exception("Couldn't get KEY_BYTE indices")

        key_byte_indices_int = list(map(int, key_byte_indices))
        return key_byte_indices_int[0], key_byte_indices_int[1:]

    tx_mod.ClientTransaction.get_indices = _patched_get_indices  # type: ignore[method-assign]

    # Bezbedniji pristup self.key (ako init delimično padne)
    _orig = tx_mod.ClientTransaction.generate_transaction_id

    def _safe_generate(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        if not getattr(self, "key", None) and getattr(self, "home_page_response", None) is not None:
            try:
                self.key = self.get_key(response=self.home_page_response)
                self.key_bytes = self.get_key_bytes(key=self.key)
                if getattr(self, "DEFAULT_KEY_BYTES_INDICES", None):
                    self.animation_key = self.get_animation_key(
                        key_bytes=self.key_bytes,
                        response=self.home_page_response,
                    )
            except Exception:
                pass
        return _orig(self, *args, **kwargs)

    tx_mod.ClientTransaction.generate_transaction_id = _safe_generate  # type: ignore[method-assign]


_patch_twikit_json()

try:
    from twikit import Client
    from twikit.errors import TwitterException
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Nedostaje twikit. Instaliraj: pip install twikit"
    ) from exc

_patch_twikit_client_transaction()

def extract_tweet_id(value: str) -> str:
    """Izvuci tweet ID iz x.com/twitter URL-a ili vrati čist numerički ID."""
    value = value.strip()
    match = TWEET_URL_PATTERN.search(value)
    if match:
        return match.group(1)
    if re.fullmatch(r"\d+", value):
        return value
    raise ValueError(f"Ne mogu da parsujem X/Twitter URL ili ID: {value!r}")


def unique_tweet_ids(urls: list[str]) -> list[str]:
    """Jedinstveni tweet ID-evi iz liste URL-ova / ID-eva (redosled očuvan)."""
    seen: set[str] = set()
    out: list[str] = []
    for item in urls:
        tid = extract_tweet_id(item)
        if tid not in seen:
            seen.add(tid)
            out.append(tid)
    return out


async def create_client(
    *,
    username: str | None = None,
    email: str | None = None,
    password: str | None = None,
    cookies_file: Path,
    refresh_session: bool = False,
) -> Client:
    """Uloguj twikit Client (cookies pa login); cookies se čuvaju u cookies_file."""
    client = Client("en-US")
    cookies_file.parent.mkdir(parents=True, exist_ok=True)

    if not refresh_session and cookies_file.exists():
        try:
            with cookies_file.open("r", encoding="utf-8") as f:
                cookie_data = json.load(f)

            if isinstance(cookie_data, list):
                print(
                    f"[twitter] Browser cookie format u {cookies_file.name} — konvertujem..."
                )
                twikit_cookies = {
                    c["name"]: c["value"]
                    for c in cookie_data
                    if isinstance(c, dict) and c.get("name") and c.get("value")
                }
                client.set_cookies(twikit_cookies)
                client.save_cookies(str(cookies_file))
            elif isinstance(cookie_data, dict):
                client.load_cookies(str(cookies_file))

            try:
                await client.user()
                print(f"[twitter] Session OK: {cookies_file}")
            except Exception as verify_exc:
                msg = str(verify_exc)
                if "Cloudflare" in msg or "Attention Required" in msg or "403" in msg:
                    print(
                        "[twitter] Cloudflare blocked the session check.\n"
                        "  Re-export cookies while logged in on x.com "
                        "(solve any CAPTCHA first), save cookies.json, retry.\n"
                        f"  File: {cookies_file}"
                    )
                else:
                    print(f"[twitter] Cookies loaded (verify skipped): {cookies_file}")
            return client
        except Exception as exc:
            print(f"[twitter] Saved session invalid ({type(exc).__name__}), re-login...")

    username = (
        username
        or os.environ.get("X_USERNAME")
        or os.environ.get("TWITTER_USERNAME")
    )
    email = email or os.environ.get("X_EMAIL") or os.environ.get("TWITTER_EMAIL")
    password = (
        password
        or os.environ.get("X_PASSWORD")
        or os.environ.get("TWITTER_PASSWORD")
    )

    if not username:
        raise ValueError(
            "X login je obavezan.\n"
            "Opcija A — cookies:\n"
            "  1) Uloguj se na x.com u browseru\n"
            "  2) Izvezi cookies (npr. Cookie-Editor) kao JSON\n"
            f"  3) Sačuvaj u: {cookies_file.resolve()}\n"
            "Opcija B — .env:\n"
            "  X_USERNAME=...\n"
            "  X_PASSWORD=...\n"
            "  (opciono X_EMAIL=...)"
        )

    if not password:
        password = getpass.getpass(f"X password for @{username}: ")

    try:
        await client.login(
            auth_info_1=username,
            auth_info_2=email or username,
            password=password,
            cookies_file=str(cookies_file),
        )
    except TwitterException as exc:
        raise ValueError(
            f"X login greška: {exc}\n"
            "Ako traži 2FA/verifikaciju: uloguj se u browseru, izvezi cookies "
            f"u {cookies_file}."
        ) from exc

    client.save_cookies(str(cookies_file))
    print(f"[twitter] Ulogovano @{username}, cookies: {cookies_file}")
    return client


async def call_with_retry(coro_fn, *, max_attempts: int = 4, base_sleep: float = 5.0):
    """Pozovi async fabriku coro_fn sa backoff-om na prolazne TwitterException greške."""
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return await coro_fn()
        except TwitterException as exc:
            last_exc = exc
            msg = str(exc)
            if getattr(exc, "status", None) == 422 or "GRAPHQL_VALIDATION_FAILED" in msg:
                raise
            if attempt + 1 >= max_attempts:
                break
            wait = base_sleep * (attempt + 1)
            print(f"  [twitter] API greška ({exc}), čekam {wait:.0f}s...")
            await asyncio.sleep(wait)
    assert last_exc is not None
    raise last_exc


async def fetch_reply_texts(
    client: Client,
    tweet_id: str,
    *,
    request_sleep: float = 2.0,
    max_comments: int = 0,
) -> list[str]:
    """Vrati listu tekstova replies (bez autora); 0 = bez limita."""

    async def _get_tweet():
        return await client.get_tweet_by_id(tweet_id)

    tweet = await call_with_retry(_get_tweet)
    total_expected = getattr(tweet, "reply_count", None)
    if total_expected:
        print(f"  Očekivano replies: ~{total_expected}")

    texts: list[str] = []
    seen_ids: set[str] = set()
    replies = getattr(tweet, "replies", None)
    page = 0

    while replies:
        page += 1
        batch = list(replies)
        added = 0
        for reply in batch:
            reply_id = str(reply.id)
            if reply_id in seen_ids:
                continue
            text = (
                getattr(reply, "full_text", None)
                or getattr(reply, "text", None)
                or ""
            ).strip()
            if not text:
                continue
            seen_ids.add(reply_id)
            texts.append(text)
            added += 1
            if max_comments and len(texts) >= max_comments:
                print(f"  Stranica {page}: +{added} (limit {max_comments})")
                return texts
        print(f"  Stranica {page}: +{added} (ukupno {len(texts)})")
        if not batch:
            break
        if request_sleep > 0:
            await asyncio.sleep(request_sleep)

        async def _next():
            return await replies.next()

        try:
            replies = await call_with_retry(_next)
        except Exception as exc:
            if getattr(exc, "status", None) == 422 or "GRAPHQL_VALIDATION_FAILED" in str(
                exc
            ):
                print("  [twitter] Kursor 422 — Search API fallback...")
            else:
                print(f"  [twitter] Paginacija prekinuta: {exc}")
            break

    if max_comments and len(texts) >= max_comments:
        return texts

    try:
        search_results = await client.search_tweet(
            f"conversation_id:{tweet_id}", "Latest"
        )
        search_page = 0
        while search_results:
            search_batch = list(search_results)
            if not search_batch:
                break
            search_page += 1
            added = 0
            for reply in search_batch:
                reply_id = str(reply.id)
                if reply_id in seen_ids or reply_id == tweet_id:
                    continue
                text = (
                    getattr(reply, "full_text", None)
                    or getattr(reply, "text", None)
                    or ""
                ).strip()
                if not text:
                    continue
                seen_ids.add(reply_id)
                texts.append(text)
                added += 1
                if max_comments and len(texts) >= max_comments:
                    print(f"  Search {search_page}: +{added} (limit)")
                    return texts
            if added:
                print(f"  Search {search_page}: +{added} (ukupno {len(texts)})")
            if request_sleep > 0:
                await asyncio.sleep(request_sleep)

            async def _next_search():
                return await search_results.next()

            try:
                search_results = await call_with_retry(_next_search)
            except Exception:
                break
    except Exception as search_exc:
        print(f"  [twitter] Search kraj: {search_exc}")

    return texts


async def fetch_all_urls(
    urls: list[str],
    *,
    cookies_file: Path,
    username: str | None = None,
    email: str | None = None,
    password: str | None = None,
    refresh_session: bool = False,
    request_sleep: float = 2.0,
    post_sleep: float = 5.0,
    max_comments: int = 0,
) -> dict[str, list[str]]:
    """Za svaki URL/ID: tweet_id → lista tekstova replies."""
    ids = unique_tweet_ids(urls)
    if not ids:
        return {}

    client = await create_client(
        username=username,
        email=email,
        password=password,
        cookies_file=cookies_file,
        refresh_session=refresh_session,
    )

    result: dict[str, list[str]] = {}
    for i, tid in enumerate(ids):
        print(f"\n=== [{i + 1}/{len(ids)}] https://x.com/i/status/{tid} ===")
        try:
            result[tid] = await fetch_reply_texts(
                client,
                tid,
                request_sleep=request_sleep,
                max_comments=max_comments,
            )
            print(f"  Skinuto: {len(result[tid])} replies")
        except Exception as exc:
            print(f"  Greška za {tid}: {exc}")
            result[tid] = []
        if post_sleep > 0 and i + 1 < len(ids):
            await asyncio.sleep(post_sleep)
    return result


def fetch_replies_sync(urls: list[str], **kwargs: Any) -> dict[str, list[str]]:
    """Sinhroni omotač oko fetch_all_urls (asyncio.run)."""
    return asyncio.run(fetch_all_urls(urls, **kwargs))
