"""Deduplikacija tekstova po SHA-256 hešu normalizovanog sadržaja.

Razlika u odnosu na kolekciju: ``collection.base.text_fingerprint`` služi
za fingerprint pri sakupljanju; ovde ``text_hash`` / ``normalize_for_dedup``
služe isključivo za uklanjanje duplikata u preprocessing pipeline-u.
"""

from __future__ import annotations

import hashlib
from typing import Any


def normalize_for_dedup(text: str) -> str:
    """Normalizacija samo za poređenje duplikata (ne menja sačuvani tekst).

    Sažima beline i radi ``casefold``; rezultat se ne upisuje u dataset.
    """
    return " ".join((text or "").split()).casefold()


def text_hash(text: str) -> str:
    """SHA-256 heš normalizovanog teksta za deduplikaciju.

    Za fingerprint u kolekciji vidi ``src.collection.base.text_fingerprint``.
    """
    return hashlib.sha256(normalize_for_dedup(text).encode("utf-8")).hexdigest()


def deduplicate_records(records: list[dict[str, Any]], text_key: str = "text") -> list[dict[str, Any]]:
    """Ukloni tačne duplikate po ``text_hash`` (zadrži prvi pojavak)."""
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for rec in records:
        key = text_hash(str(rec.get(text_key, "")))
        if key in seen:
            continue
        seen.add(key)
        unique.append(rec)
    return unique
