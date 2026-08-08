#!/usr/bin/env python3
"""Izvoz validno anotiranih primera + stratifikovani train/val/test split.

Tok: učitaj annotation CSV → filtriraj validne labele → strata=sentiment|sarcasm
→ ``_stratified_split`` → labeled/train/val/test.csv + split_meta.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._bootstrap import ensure_project_root

ensure_project_root()

from src.common.config import load_config, resolve_path
from src.common.schema import SARCASM_VALUES, SENTIMENT_VALUES
from src.common.stdio_utf8 import configure_utf8_stdio

VALID_SENT = set(SENTIMENT_VALUES)
VALID_SARC = set(SARCASM_VALUES)


def _stratified_split(
    df: pd.DataFrame,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratifikovani split po koloni ``strata`` (bez sklearn zavisnosti).

    Za svaku stratum grupu meša redove (``seed``), pa deli na train/val/test
    prema ratio-ima. Posebni slučajevi: n=1 → samo train; n=2 → train+test.
    Garantuje bar 1 primer u testu kad je n≥3 (po mogućnosti).

    Returns:
        (train_df, val_df, test_df) — svaki ponovo izmešan istim seed-om.
    """
    train_parts: list[pd.DataFrame] = []
    val_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    for _, group in df.groupby("strata", sort=False):
        g = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(g)
        if n == 1:
            train_parts.append(g)
            continue
        if n == 2:
            train_parts.append(g.iloc[:1])
            test_parts.append(g.iloc[1:])
            continue

        n_train = max(1, int(round(n * train_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)
        n_test = n - n_train - n_val
        if n_test < 1:
            n_test = 1
            n_train = n - n_val - n_test

        train_parts.append(g.iloc[:n_train])
        val_parts.append(g.iloc[n_train : n_train + n_val])
        test_parts.append(g.iloc[n_train + n_val :])

    def _cat(parts: list[pd.DataFrame]) -> pd.DataFrame:
        """Spoji delove split-a i izmešaj; prazna lista → prazan DataFrame."""
        if not parts:
            return df.iloc[0:0].copy()
        out = pd.concat(parts, ignore_index=True)
        return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return _cat(train_parts), _cat(val_parts), _cat(test_parts)


def main() -> None:
    """CLI: filtrira validne labele i pravi stratifikovani train/val/test.

    Čita annotation CSV (ili ``--csv``), odbacuje nevalidne/prazne redove,
    poziva ``_stratified_split``, upisuje CSV-ove i ``split_meta.json``.
    """
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(
        description=(
            "Filtrira validne labele iz annotation CSV i pravi "
            "train/val/test split (stratifikovano po sentiment+sarcasm)."
        )
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--csv",
        default=None,
        help="Ulazni CSV (podrazumevano annotation_csv iz config-a)",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Izlazni folder (podrazumevano data/processed/splits)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    csv_path = Path(args.csv) if args.csv else resolve_path(config["paths"]["annotation_csv"])
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else resolve_path(config["paths"].get("splits_dir", "data/processed/splits"))
    )
    seed = int(config.get("random_seed", 42))

    df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str).fillna("")
    if "text" not in df.columns:
        raise SystemExit(f"{csv_path} mora imati kolonu 'text'")
    if "sentiment" not in df.columns or "sarcasm" not in df.columns:
        raise SystemExit(f"{csv_path} mora imati kolone 'sentiment' i 'sarcasm'")

    # id / source (URL) nisu obavezni za trening — popuni praznim ako fale
    if "id" not in df.columns:
        df["id"] = [f"row-{i:05d}" for i in range(1, len(df) + 1)]
    else:
        empty_id = df["id"].astype(str).str.strip() == ""
        if empty_id.any():
            df.loc[empty_id, "id"] = [
                f"row-{i:05d}" for i in range(1, int(empty_id.sum()) + 1)
            ]
    if "source" not in df.columns:
        df["source"] = ""
    if "tip" not in df.columns:
        df["tip"] = ""

    df["sentiment"] = df["sentiment"].str.strip().str.lower()
    df["sarcasm"] = df["sarcasm"].str.strip().str.lower()
    # Prazan tekst ne ulazi u split
    has_text = df["text"].astype(str).str.strip() != ""

    valid = has_text & df["sentiment"].isin(VALID_SENT) & df["sarcasm"].isin(VALID_SARC)
    labeled = df.loc[valid].copy()
    skipped = int((~valid).sum())

    if labeled.empty:
        raise SystemExit(f"Nema validno anotiranih redova u {csv_path}")

    labeled["strata"] = labeled["sentiment"] + "|" + labeled["sarcasm"]
    train_df, val_df, test_df = _stratified_split(
        labeled, seed=seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio
    )

    drop_cols = [c for c in ("strata",) if c in labeled.columns]
    for part in (train_df, val_df, test_df, labeled):
        part.drop(columns=drop_cols, inplace=True, errors="ignore")

    out_dir.mkdir(parents=True, exist_ok=True)
    labeled_path = out_dir / "labeled.csv"
    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    test_path = out_dir / "test.csv"
    meta_path = out_dir / "split_meta.json"

    labeled.to_csv(labeled_path, index=False, encoding="utf-8-sig")
    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    val_df.to_csv(val_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    meta = {
        "source_csv": str(csv_path),
        "seed": seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "total_rows": int(len(df)),
        "valid_labeled": int(len(labeled)),
        "skipped_invalid": skipped,
        "train": int(len(train_df)),
        "val": int(len(val_df)),
        "test": int(len(test_df)),
        "sentiment": labeled["sentiment"].value_counts().to_dict(),
        "sarcasm": labeled["sarcasm"].value_counts().to_dict(),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[split] ulaz: {csv_path}")
    print(f"[split] validno: {len(labeled)} (preskočeno nevalidnih: {skipped})")
    print(f"[split] train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    print(f"[split] sačuvano u: {out_dir}")
    print(f"[split] meta: {meta_path}")


if __name__ == "__main__":
    main()
