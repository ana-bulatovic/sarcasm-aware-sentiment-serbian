"""Dataset i DataLoader pomoćne funkcije."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from src.modeling.labels import SARCASM_LABEL2ID, SENTIMENT_LABEL2ID

TaskName = Literal["sentiment", "sarcasm", "multitask"]


class CommentDataset(Dataset):
    """Tokenizovani komentari sa jednom ili obe labele."""

    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        task: TaskName,
        max_length: int = 128,
    ) -> None:
        self.texts = frame["text"].astype(str).tolist()
        self.ids = frame["id"].astype(str).tolist() if "id" in frame.columns else [str(i) for i in range(len(frame))]
        self.sarcasm_raw = (
            frame["sarcasm"].astype(str).str.strip().str.lower().tolist()
            if "sarcasm" in frame.columns
            else [""] * len(frame)
        )
        self.task = task
        self.tokenizer = tokenizer
        self.max_length = max_length

        if task in ("sentiment", "multitask"):
            self.sentiment_labels = [
                SENTIMENT_LABEL2ID[x]
                for x in frame["sentiment"].astype(str).str.strip().str.lower()
            ]
        else:
            self.sentiment_labels = [-1] * len(frame)

        if task in ("sarcasm", "multitask"):
            self.sarcasm_labels = [
                SARCASM_LABEL2ID[x]
                for x in frame["sarcasm"].astype(str).str.strip().str.lower()
            ]
        else:
            self.sarcasm_labels = [-1] * len(frame)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        item: dict[str, Any] = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "example_id": self.ids[idx],
            "sarcasm_raw": self.sarcasm_raw[idx],
        }
        if self.task in ("sentiment", "multitask"):
            item["sentiment_labels"] = self.sentiment_labels[idx]
        if self.task in ("sarcasm", "multitask"):
            item["sarcasm_labels"] = self.sarcasm_labels[idx]
        if self.task == "sentiment":
            item["labels"] = self.sentiment_labels[idx]
        elif self.task == "sarcasm":
            item["labels"] = self.sarcasm_labels[idx]
        return item


def _collate(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = []
    attention_mask = []
    for x in batch:
        pad_len = max_len - len(x["input_ids"])
        input_ids.append(x["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append(x["attention_mask"] + [0] * pad_len)

    out: dict[str, Any] = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "example_id": [x["example_id"] for x in batch],
        "sarcasm_raw": [x["sarcasm_raw"] for x in batch],
    }
    if "labels" in batch[0]:
        out["labels"] = torch.tensor([x["labels"] for x in batch], dtype=torch.long)
    if "sentiment_labels" in batch[0]:
        out["sentiment_labels"] = torch.tensor(
            [x["sentiment_labels"] for x in batch], dtype=torch.long
        )
    if "sarcasm_labels" in batch[0]:
        out["sarcasm_labels"] = torch.tensor(
            [x["sarcasm_labels"] for x in batch], dtype=torch.long
        )
    return out


def load_split_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Nedostaje split fajl: {path}. Pokreni scripts/modeling/prepare_splits.py"
        )
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    required = {"text", "sentiment", "sarcasm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} nema kolone: {sorted(missing)}")
    df["sentiment"] = df["sentiment"].str.strip().str.lower()
    df["sarcasm"] = df["sarcasm"].str.strip().str.lower()
    mask = df["sentiment"].isin(SENTIMENT_LABEL2ID) & df["sarcasm"].isin(SARCASM_LABEL2ID)
    return df.loc[mask].reset_index(drop=True)


def load_splits(splits_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "train": load_split_csv(splits_dir / "train.csv"),
        "val": load_split_csv(splits_dir / "val.csv"),
        "test": load_split_csv(splits_dir / "test.csv"),
    }


def make_loader(
    frame: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    task: TaskName,
    batch_size: int,
    max_length: int,
    shuffle: bool,
    seed: int = 42,
) -> DataLoader:
    dataset = CommentDataset(frame, tokenizer, task=task, max_length=max_length)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id or 0

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: _collate(batch, pad_token_id=pad_id),
        generator=generator if shuffle else None,
    )
