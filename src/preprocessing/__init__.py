"""Preprocesiranje tekstova — mapa modula:

- ``clean`` — lagano čišćenje (``preprocess_text``) za dataset / BERTić.
- ``baseline`` — agresivnije čišćenje (``clean_text``) **samo** za TF-IDF baseline-e.
- ``deduplicate`` — hash / deduplikacija po normalizovanom tekstu.
- ``pipeline`` — raw → interim (čišćenje + jezik + dedup).

``preprocess_text`` / ``run_preprocessing`` — lagano čišćenje za dataset
(kolekcija → interim); koristi se i kao ulaz za BERTić.
``clean_text`` / baseline modul — agresivnije pretprocesiranje **samo** za
klasične ML baseline modele (TF-IDF + LR/SVM/…). BERTić to NE koristi.
"""

from src.preprocessing.baseline import (
    clean_text,
    clean_text_from_config,
    lemmatize_text,
    normalize_script,
)
from src.preprocessing.clean import preprocess_text
from src.preprocessing.pipeline import run_preprocessing

__all__ = [
    "preprocess_text",
    "run_preprocessing",
    "clean_text",
    "clean_text_from_config",
    "normalize_script",
    "lemmatize_text",
]
