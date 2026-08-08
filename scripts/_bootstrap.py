"""Bootstrap: pronađi root projekta i stavi ga na ``sys.path``.

Koristi se iz bilo kog ``scripts/`` podfoldera da bi ``import src...`` radio
bez obzira odakle je skripta pokrenuta.
"""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_project_root() -> Path:
    """Pronađi root (folder sa ``config/config.yaml``), dodaj ga na ``sys.path``.

    Traži unazad od ``scripts/`` preko parent direktorijuma. Ako root nije
    pronađen, baca ``RuntimeError``.

    Returns:
        Apsolutna putanja do root-a projekta.
    """
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
