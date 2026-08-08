"""Podešavanje stdout/stderr na UTF-8 (Windows konzola)."""

from __future__ import annotations

import sys


def configure_utf8_stdio() -> None:
    """Podesi stdout/stderr na UTF-8 (izbegava greške sa ćirilicom na Windows konzoli)."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
