"""Deduplikacija tekstova."""

from __future__ import annotations

import hashlib
from typing import Any


def normalize_for_dedup(text: str) -> str:
    """Normalizacija samo za poređenje duplikata (ne menja sačuvani tekst)."""
    return " ".join((text or "").split()).casefold()


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_for_dedup(text).encode("utf-8")).hexdigest()


def deduplicate_records(records: list[dict[str, Any]], text_key: str = "text") -> list[dict[str, Any]]:
    """Ukloni tačne duplikate po normalizovanom tekstu (zadrži prvi)."""
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for rec in records:
        key = text_hash(str(rec.get(text_key, "")))
        if key in seen:
            continue
        seen.add(key)
        unique.append(rec)
    return unique
