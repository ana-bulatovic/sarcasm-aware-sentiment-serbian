"""Mapiranje originalnih SentiComments.SR labela na našu šemu.

Ulaz: npr. ``+1``, ``-M``, ``+NS``, sa opcionim sufiksom ``s`` (sarkazam).
Izlaz: ``(sentiment, sarcasm)`` gde je sentiment ``\"1\"|\"0\"|\"-1\"``, sarcasm ``\"1\"|\"0\"``.
"""

from __future__ import annotations

# Batanovic et al.: +1/-1, +M/-M, +NS/-NS, opciono sufiks 's' za sarkazam.
# +NS/-NS = iskazi bez eksplicitnog sentimenta, ali sa nagibom; kod nas -> 0 (neutral).


def map_senticomments_label(original: str) -> tuple[str, str]:
    """Vrati (sentiment, sarcasm) u nasim vrednostima.

    sentiment: 1 | 0 | -1
    sarcasm: 1 | 0
    """
    label = (original or "").strip()
    sarcasm = "1" if label.endswith("s") and len(label) > 1 else "0"
    core = label[:-1] if sarcasm == "1" else label

    if core in {"+1", "+M"}:
        return "1", sarcasm
    if core in {"-1", "-M"}:
        return "-1", sarcasm
    if core in {"+NS", "-NS"}:
        return "0", sarcasm

    # Nepoznata labela — ostavi prazno za rucnu proveru
    return "", ""
