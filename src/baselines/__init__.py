"""Klasični ML baseline eksperimenti (TF-IDF + NB / LR / Linear SVM).

Mapa modula:
- ``data`` — učitavanje CSV-a, filtriranje labela, baseline pretprocesiranje.
- ``pipeline`` — TF-IDF vektorizer + klasifikatori.
- ``metrics`` — accuracy / macro-F1 / confusion matrix.
- ``runner`` — train/test split, fit, čuvanje rezultata.

Nije namenjeno BERTić fine-tune pipeline-u.
"""

from __future__ import annotations

from src.baselines.runner import run_baseline_experiments

__all__ = ["run_baseline_experiments"]
