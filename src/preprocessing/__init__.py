"""Preprocesiranje tekstova za anotaciju."""

from src.preprocessing.clean import preprocess_text
from src.preprocessing.pipeline import run_preprocessing

__all__ = ["preprocess_text", "run_preprocessing"]
