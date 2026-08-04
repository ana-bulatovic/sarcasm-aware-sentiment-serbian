"""Mapiranje originalnih SentiComments.SR labela na nasu semu."""

from __future__ import annotations

# Batanovic et al.: +1/-1, +M/-M, +NS/-NS, opciono sufiks 's' za sarkazam.
# +NS/-NS = iskazi bez eksplicitnog sentimenta, ali sa nagibom; kod nas -> neutral.


def map_senticomments_label(original: str) -> tuple[str, str]:
    """Vrati (sentiment, sarcasm) u nasim vrednostima.

    sentiment: positive | neutral | negative
    sarcasm: yes | no
    """
    label = (original or "").strip()
    sarcasm = "yes" if label.endswith("s") and len(label) > 1 else "no"
    core = label[:-1] if sarcasm == "yes" else label

    if core in {"+1", "+M"}:
        return "positive", sarcasm
    if core in {"-1", "-M"}:
        return "negative", sarcasm
    if core in {"+NS", "-NS"}:
        return "neutral", sarcasm

    # Nepoznata labela — ostavi prazno za rucnu proveru
    return "", ""
