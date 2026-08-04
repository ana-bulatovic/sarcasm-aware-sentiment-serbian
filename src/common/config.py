"""Učitavanje konfiguracije i putanja projekta."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Korijen repozitorijuma: .../sarcasm-aware-sentiment-serbian
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_project_root() -> Path:
    return PROJECT_ROOT


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Učitaj YAML konfiguraciju i .env fajl."""
    load_dotenv(PROJECT_ROOT / ".env")

    if config_path is None:
        config_path = os.getenv("CONFIG_PATH", "config/config.yaml")

    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"Config fajl nije pronađen: {path}")

    with path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Config mora biti YAML mapa (dict).")

    return config


def resolve_path(relative_or_absolute: str | Path) -> Path:
    """Relativne putanje se vezuju za korijen projekta."""
    path = Path(relative_or_absolute)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
