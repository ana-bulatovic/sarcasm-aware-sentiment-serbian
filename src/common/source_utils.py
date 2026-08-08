"""Pomocne funkcije za source = pun URL i platformu (youtube/tiktok/...)."""

from __future__ import annotations


def youtube_watch_url(video_id: str) -> str:
    """Napravi kanonski YouTube watch URL od video ID-a."""
    return f"https://www.youtube.com/watch?v={video_id}"


def platform_from_source(source: str) -> str:
    """Mapira vrednost kolone ``source`` (URL ili stari alias) na platformu.

    Primeri: ``"youtube"`` → ``youtube``;
    ``"https://www.youtube.com/watch?v=..."`` → ``youtube``.
    """
    s = (source or "").strip().lower()
    if not s:
        return "unknown"
    if s in {"youtube", "twitter", "tiktok", "instagram", "reddit", "reviews"}:
        return s
    if "youtube.com" in s or "youtu.be" in s:
        return "youtube"
    if "twitter.com" in s or "x.com" in s:
        return "twitter"
    if "tiktok.com" in s:
        return "tiktok"
    if "instagram.com" in s:
        return "instagram"
    if "reddit.com" in s:
        return "reddit"
    return s


PLATFORM_ORDER = ("youtube", "twitter", "tiktok", "instagram", "reddit", "reviews")


def platform_sort_key(source: str) -> tuple[int, str]:
    """Ključ za stabilno sortiranje redova po platformi, pa po ``source`` stringu."""
    platform = platform_from_source(source)
    try:
        return PLATFORM_ORDER.index(platform), source
    except ValueError:
        return len(PLATFORM_ORDER), source
