"""Fine-tune sentiment / sarkazam / multitask modela."""

from __future__ import annotations

from src.modeling.runner import run_all_tasks, run_evaluation, run_training

__all__ = ["run_training", "run_all_tasks", "run_evaluation"]
