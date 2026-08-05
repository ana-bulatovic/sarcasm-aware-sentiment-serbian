"""Pronađi root projekta i stavi ga na sys.path (radi iz bilo kog scripts/ podfoldera)."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_project_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in [here, *here.parents]:
        if (candidate / "config" / "config.yaml").is_file():
            root = candidate
            break
    else:
        raise RuntimeError("Nije pronađen root projekta (config/config.yaml).")

    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root
