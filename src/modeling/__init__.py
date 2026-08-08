"""Fine-tune sentiment / sarkazam / multitask modela (BERTić i slični).

Mapa modula:
- ``labels`` — mapiranje string labela ↔ ID.
- ``data`` — ``CommentDataset``, DataLoader, učitavanje splitova.
- ``balancing`` — class / sample weights iz train splita.
- ``models`` — single-task HF klasifikator i ``MultiTaskModel``.
- ``metrics`` — accuracy, macro-F1, podskup sarcasm=yes.
- ``train_loop`` — trening / evaluacija; selection score za multitask.
- ``runner`` — orkestracija treninga i evaluacije.
- ``predict`` — inferenca nad tekstovima.
"""

from __future__ import annotations

from src.modeling.runner import run_all_tasks, run_evaluation, run_training

__all__ = ["run_training", "run_all_tasks", "run_evaluation"]
