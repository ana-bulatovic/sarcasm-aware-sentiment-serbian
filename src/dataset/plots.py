"""Grafikoni i tabele o datasetu za poglavlje o podacima u master radu."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.common.schema import SARCASM_VALUES, SENTIMENT_VALUES, normalize_label

SENT_ORDER = ["1", "0", "-1"]
SARC_ORDER = ["1", "0"]
SENT_NAMES = {"1": "Pozitivno", "0": "Neutralno", "-1": "Negativno"}
SARC_NAMES = {"1": "Sarkazam", "0": "Nije sarkazam"}
SOURCE_NAMES = {
    "youtube": "YouTube",
    "twitter": "Twitter/X",
    "tiktok": "TikTok",
    "instagram": "Instagram",
    "reddit": "Reddit",
}
SENT_COLORS = {"1": "#2ca02c", "0": "#7f7f7f", "-1": "#d62728"}
SARC_COLORS = {"1": "#e67e22", "0": "#2980b9"}
SCRIPT_COLORS = {
    "latinica": "#3498db",
    "ćirilica": "#9b59b6",
    "mešano": "#f39c12",
    "ostalo": "#95a5a6",
}

_CYRILLIC = re.compile(r"[\u0400-\u04FF]")
_LATIN = re.compile(r"[A-Za-zÀ-ž]")


def _pretty_source(value: str) -> str:
    key = str(value or "").strip().lower()
    return SOURCE_NAMES.get(key, value or "nepoznato")


# Retke teme se na grafikonima prikazuju zajedno kao „Ostalo“.
TIP_GROUP_INTO_OSTALO = frozenset({"fakultet", "tenis", "hrana"})


def _pretty_tip(value: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "(nije navedeno)"
    if text in TIP_GROUP_INTO_OSTALO:
        text = "ostalo"
    return text.capitalize()


def _script_type(text: str) -> str:
    has_cyr = bool(_CYRILLIC.search(text))
    has_lat = bool(_LATIN.search(text))
    if has_cyr and has_lat:
        return "mešano"
    if has_cyr:
        return "ćirilica"
    if has_lat:
        return "latinica"
    return "ostalo"


def _setup_style() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.unicode_minus": False,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
        }
    )
    for candidate in ("Segoe UI", "DejaVu Sans", "Arial"):
        try:
            plt.rcParams["font.family"] = candidate
            break
        except Exception:
            continue


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _annotate_bars(ax: plt.Axes, values: list[int] | np.ndarray, *, as_pct: bool = False) -> None:
    ymax = 0.0
    for bar, val in zip(ax.patches, values):
        height = bar.get_height()
        ymax = max(ymax, height)
        label = f"{val:.1f}%" if as_pct else f"{int(val)}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )
    if ymax > 0:
        ax.set_ylim(0, ymax * 1.14)


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["text"] = out.get("text", "").astype(str).fillna("").str.strip()
    out["source"] = out.get("source", "").astype(str).fillna("").map(lambda x: x.strip().lower())
    out["tip"] = out.get("tip", "").astype(str).fillna("").str.strip()
    out["sentiment"] = out.get("sentiment", "").map(normalize_label)
    out["sarcasm"] = out.get("sarcasm", "").map(normalize_label)
    out["n_chars"] = out["text"].str.len()
    out["n_words"] = out["text"].map(lambda t: len(str(t).split()))
    out["script"] = out["text"].map(_script_type)
    out["source_label"] = out["source"].map(_pretty_source)
    out["tip_label"] = out["tip"].map(_pretty_tip)
    out["labeled"] = out["sentiment"].isin(SENTIMENT_VALUES) & out["sarcasm"].isin(SARCASM_VALUES)
    return out


def _bar_counts(
    counts: pd.Series,
    *,
    title: str,
    xlabel: str,
    color: str | list[str],
    rotate: int = 0,
) -> plt.Figure:
    labels = [str(x) for x in counts.index]
    values = counts.to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.bar(labels, values, color=color, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Broj komentara")
    if rotate:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=rotate, ha="right")
    _annotate_bars(ax, values)
    fig.tight_layout()
    return fig


def _bar_and_pie(
    counts: pd.Series,
    *,
    title: str,
    xlabel: str,
    colors: list[str],
) -> plt.Figure:
    labels = [str(x) for x in counts.index]
    values = counts.to_numpy(dtype=int)
    total = int(values.sum()) or 1
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))

    axes[0].bar(labels, values, color=colors, edgecolor="white")
    axes[0].set_title(title)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Broj komentara")
    _annotate_bars(axes[0], values)

    pie_labels = [f"{lab}\n{val} ({100 * val / total:.1f}%)" for lab, val in zip(labels, values)]
    wedges, _ = axes[1].pie(
        values,
        colors=colors,
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.2},
    )
    axes[1].legend(wedges, pie_labels, loc="center left", bbox_to_anchor=(0.92, 0.5), frameon=False)
    axes[1].set_title("Udeo klasa")
    fig.tight_layout()
    return fig


def _stacked_pair(
    ct: pd.DataFrame,
    *,
    title_counts: str,
    title_share: str,
    colors: list[str],
    xlabel: str,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))
    ct.plot(kind="bar", stacked=True, ax=axes[0], color=colors, edgecolor="white", width=0.78)
    axes[0].set_title(title_counts)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Broj komentara")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].legend(frameon=False, title="")

    share = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0) * 100
    share.plot(kind="bar", stacked=True, ax=axes[1], color=colors, edgecolor="white", width=0.78)
    axes[1].set_title(title_share)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("Udeo (%)")
    axes[1].set_ylim(0, 100)
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend(frameon=False, title="")
    fig.tight_layout()
    return fig


def _combo_heatmap(labeled: pd.DataFrame) -> plt.Figure:
    ct = pd.crosstab(labeled["sentiment"], labeled["sarcasm"])
    ct = ct.reindex(index=SENT_ORDER, columns=SARC_ORDER, fill_value=0)
    display = ct.copy()
    display.index = [SENT_NAMES[i] for i in display.index]
    display.columns = [SARC_NAMES[c] for c in display.columns]
    total = int(ct.to_numpy().sum()) or 1
    annot = display.copy().astype(str)
    for i in display.index:
        for c in display.columns:
            n = int(display.loc[i, c])
            annot.loc[i, c] = f"{n}\n({100 * n / total:.1f}%)"

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    sns.heatmap(
        display,
        annot=annot,
        fmt="",
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "Broj komentara"},
        linewidths=0.6,
        linecolor="white",
    )
    ax.set_xlabel("Sarkazam")
    ax.set_ylabel("Sentiment")
    ax.set_title("Zajednička raspodela sentimenta i sarkazma")
    fig.tight_layout()
    return fig


def _combo_grouped(labeled: pd.DataFrame) -> plt.Figure:
    ct = pd.crosstab(labeled["sentiment"], labeled["sarcasm"])
    ct = ct.reindex(index=SENT_ORDER, columns=SARC_ORDER, fill_value=0)
    ct = ct.rename(index=SENT_NAMES, columns=SARC_NAMES)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ct.plot(kind="bar", ax=ax, color=[SARC_COLORS["1"], SARC_COLORS["0"]], edgecolor="white", width=0.78)
    ax.set_title("Kombinacije sentiment × sarkazam")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Broj komentara")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="", frameon=False)
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", fontsize=9, padding=2)
    ax.set_ylim(0, max(int(ct.to_numpy().max()), 1) * 1.16)
    fig.tight_layout()
    return fig


def _share_bar(series_pct: pd.Series, *, title: str, xlabel: str, color: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.bar(series_pct.index.astype(str), series_pct.to_numpy(), color=color, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Udeo sarkazma (%)")
    ax.set_ylim(0, 100)
    _annotate_bars(ax, series_pct.to_numpy(), as_pct=True)
    fig.tight_layout()
    return fig


def _hist(series: pd.Series, *, title: str, xlabel: str, color: str, bins: int = 40) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.hist(series, bins=bins, color=color, edgecolor="white")
    ax.axvline(series.median(), color="#c0392b", linestyle="--", linewidth=1.4, label=f"Medijana = {series.median():.0f}")
    ax.axvline(series.mean(), color="#1a5276", linestyle=":", linewidth=1.4, label=f"Prosek = {series.mean():.1f}")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Broj komentara")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _box(df: pd.DataFrame, x: str, y: str, *, title: str, xlabel: str, ylabel: str, order: list[str], palette: dict[str, str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    sns.boxplot(
        data=df,
        x=x,
        y=y,
        hue=x,
        order=order,
        palette=[palette[k] for k in order],
        legend=False,
        ax=ax,
        fliersize=3,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig


def _length_stats(series: pd.Series) -> dict[str, float]:
    return {
        "min": float(series.min()) if len(series) else 0.0,
        "p25": float(series.quantile(0.25)) if len(series) else 0.0,
        "median": float(series.median()) if len(series) else 0.0,
        "mean": float(series.mean()) if len(series) else 0.0,
        "p75": float(series.quantile(0.75)) if len(series) else 0.0,
        "p90": float(series.quantile(0.90)) if len(series) else 0.0,
        "p95": float(series.quantile(0.95)) if len(series) else 0.0,
        "max": float(series.max()) if len(series) else 0.0,
        "std": float(series.std(ddof=0)) if len(series) else 0.0,
    }


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def generate_dataset_figures(
    csv_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Napravi PNG grafikone i CSV tabele o datasetu. Vrati sažetak brojeva."""
    _setup_style()
    df_raw = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str).fillna("")
    df = _prepare(df_raw)
    labeled = df[df["labeled"]].copy()
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tabele"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    n_total = int(len(df))
    n_labeled = int(len(labeled))
    n_unlabeled = n_total - n_labeled

    source_counts = df["source_label"].value_counts()
    tip_counts = df["tip_label"].value_counts()
    sent_counts = (
        labeled["sentiment"].value_counts().reindex(SENT_ORDER, fill_value=0).rename(index=SENT_NAMES)
        if n_labeled
        else pd.Series(dtype=int)
    )
    sarc_counts = (
        labeled["sarcasm"].value_counts().reindex(SARC_ORDER, fill_value=0).rename(index=SARC_NAMES)
        if n_labeled
        else pd.Series(dtype=int)
    )

    saved: list[str] = []

    def remember(path: Path) -> None:
        saved.append(str(path.relative_to(out_dir)).replace("\\", "/"))

    remember(
        _save(
            _bar_counts(
                source_counts,
                title="Broj komentara po izvoru",
                xlabel="Izvor",
                color=["#c0392b", "#2c3e50", "#111111", "#8e44ad"][: len(source_counts)],
            ),
            figures_dir / "01_izvori.png",
        )
    )
    remember(
        _save(
            _bar_counts(
                tip_counts,
                title="Broj komentara po temi",
                xlabel="Tema",
                color="#6c5ce7",
                rotate=15,
            ),
            figures_dir / "02_teme.png",
        )
    )

    if n_labeled:
        remember(
            _save(
                _bar_and_pie(
                    sent_counts,
                    title="Raspodela sentimenta",
                    xlabel="Sentiment",
                    colors=[SENT_COLORS[k] for k in SENT_ORDER],
                ),
                figures_dir / "03_sentiment.png",
            )
        )
        remember(
            _save(
                _bar_and_pie(
                    sarc_counts,
                    title="Raspodela sarkazma",
                    xlabel="Sarkazam",
                    colors=[SARC_COLORS[k] for k in SARC_ORDER],
                ),
                figures_dir / "04_sarkazam.png",
            )
        )
        remember(_save(_combo_heatmap(labeled), figures_dir / "05_kombinacije_heatmap.png"))
        remember(_save(_combo_grouped(labeled), figures_dir / "06_kombinacije_grouped.png"))

        sent_by_src = pd.crosstab(labeled["source_label"], labeled["sentiment"])
        sent_by_src = sent_by_src.reindex(columns=SENT_ORDER, fill_value=0).rename(columns=SENT_NAMES)
        remember(
            _save(
                _stacked_pair(
                    sent_by_src,
                    title_counts="Sentiment po izvoru",
                    title_share="Udeo sentimenta po izvoru",
                    colors=[SENT_COLORS[k] for k in SENT_ORDER],
                    xlabel="Izvor",
                ),
                figures_dir / "07_sentiment_po_izvoru.png",
            )
        )

        sarc_by_src = pd.crosstab(labeled["source_label"], labeled["sarcasm"])
        sarc_by_src = sarc_by_src.reindex(columns=SARC_ORDER, fill_value=0).rename(columns=SARC_NAMES)
        remember(
            _save(
                _stacked_pair(
                    sarc_by_src,
                    title_counts="Sarkazam po izvoru",
                    title_share="Udeo sarkazma po izvoru",
                    colors=[SARC_COLORS[k] for k in SARC_ORDER],
                    xlabel="Izvor",
                ),
                figures_dir / "08_sarkazam_po_izvoru.png",
            )
        )

        sent_by_tip = pd.crosstab(labeled["tip_label"], labeled["sentiment"])
        sent_by_tip = sent_by_tip.reindex(columns=SENT_ORDER, fill_value=0).rename(columns=SENT_NAMES)
        remember(
            _save(
                _stacked_pair(
                    sent_by_tip,
                    title_counts="Sentiment po temi",
                    title_share="Udeo sentimenta po temi",
                    colors=[SENT_COLORS[k] for k in SENT_ORDER],
                    xlabel="Tema",
                ),
                figures_dir / "09_sentiment_po_temi.png",
            )
        )

        sarc_by_tip = pd.crosstab(labeled["tip_label"], labeled["sarcasm"])
        sarc_by_tip = sarc_by_tip.reindex(columns=SARC_ORDER, fill_value=0).rename(columns=SARC_NAMES)
        remember(
            _save(
                _stacked_pair(
                    sarc_by_tip,
                    title_counts="Sarkazam po temi",
                    title_share="Udeo sarkazma po temi",
                    colors=[SARC_COLORS[k] for k in SARC_ORDER],
                    xlabel="Tema",
                ),
                figures_dir / "10_sarkazam_po_temi.png",
            )
        )

        sarc_share_sent = (
            labeled.groupby("sentiment")["sarcasm"]
            .apply(lambda s: 100.0 * (s == "1").mean())
            .reindex(SENT_ORDER)
            .rename(index=SENT_NAMES)
        )
        remember(
            _save(
                _share_bar(
                    sarc_share_sent,
                    title="Udeo sarkastičnih komentara unutar svake klase sentimenta",
                    xlabel="Sentiment",
                    color="#e67e22",
                ),
                figures_dir / "11_udeo_sarkazma_po_sentimentu.png",
            )
        )

        sarc_share_tip = (
            labeled.groupby("tip_label")["sarcasm"].apply(lambda s: 100.0 * (s == "1").mean()).sort_values(ascending=False)
        )
        remember(
            _save(
                _share_bar(
                    sarc_share_tip,
                    title="Udeo sarkazma po temi",
                    xlabel="Tema",
                    color="#d35400",
                ),
                figures_dir / "12_udeo_sarkazma_po_temi.png",
            )
        )

        labeled_box = labeled.copy()
        labeled_box["sentiment_ime"] = labeled_box["sentiment"].map(SENT_NAMES)
        labeled_box["sarcasm_ime"] = labeled_box["sarcasm"].map(SARC_NAMES)
        remember(
            _save(
                _box(
                    labeled_box,
                    "sentiment_ime",
                    "n_chars",
                    title="Dužina teksta (karakteri) po sentimentu",
                    xlabel="Sentiment",
                    ylabel="Broj karaktera",
                    order=[SENT_NAMES[k] for k in SENT_ORDER],
                    palette={SENT_NAMES[k]: SENT_COLORS[k] for k in SENT_ORDER},
                ),
                figures_dir / "15_duzina_po_sentimentu.png",
            )
        )
        remember(
            _save(
                _box(
                    labeled_box,
                    "sarcasm_ime",
                    "n_chars",
                    title="Dužina teksta (karakteri) po sarkazmu",
                    xlabel="Sarkazam",
                    ylabel="Broj karaktera",
                    order=[SARC_NAMES[k] for k in SARC_ORDER],
                    palette={SARC_NAMES[k]: SARC_COLORS[k] for k in SARC_ORDER},
                ),
                figures_dir / "16_duzina_po_sarkazmu.png",
            )
        )

    remember(
        _save(
            _hist(df["n_chars"], title="Raspodela dužine komentara (karakteri)", xlabel="Broj karaktera", color="#2980b9"),
            figures_dir / "13_duzina_karaktera.png",
        )
    )
    remember(
        _save(
            _hist(df["n_words"], title="Raspodela dužine komentara (reči)", xlabel="Broj reči", color="#16a085", bins=30),
            figures_dir / "14_duzina_reci.png",
        )
    )

    script_counts = df["script"].value_counts().reindex(["latinica", "ćirilica", "mešano", "ostalo"], fill_value=0)
    script_counts = script_counts[script_counts > 0]
    remember(
        _save(
            _bar_and_pie(
                script_counts,
                title="Pismo komentara",
                xlabel="Pismo",
                colors=[SCRIPT_COLORS.get(k, "#95a5a6") for k in script_counts.index],
            ),
            figures_dir / "17_pismo.png",
        )
    )

    if n_unlabeled:
        cov = pd.Series({"Potpuno anotirano": n_labeled, "Nedostaje labela": n_unlabeled})
        remember(
            _save(
                _bar_and_pie(
                    cov,
                    title="Pokriće anotacije",
                    xlabel="",
                    colors=["#27ae60", "#e74c3c"],
                ),
                figures_dir / "18_anotacija.png",
            )
        )

    combo_rows = []
    if n_labeled:
        ct = pd.crosstab(labeled["sentiment"], labeled["sarcasm"]).reindex(
            index=SENT_ORDER, columns=SARC_ORDER, fill_value=0
        )
        for s in SENT_ORDER:
            for c in SARC_ORDER:
                n = int(ct.loc[s, c])
                combo_rows.append(
                    {
                        "sentiment": s,
                        "sentiment_ime": SENT_NAMES[s],
                        "sarcasm": c,
                        "sarcasm_ime": SARC_NAMES[c],
                        "n": n,
                        "udeo_%": round(100.0 * n / n_labeled, 2) if n_labeled else 0.0,
                    }
                )

    source_table = (
        df["source_label"]
        .value_counts()
        .rename_axis("izvor")
        .reset_index(name="n")
        .assign(**{"udeo_%": lambda x: (100.0 * x["n"] / n_total).round(2)})
    )
    tip_table = (
        df["tip_label"]
        .value_counts()
        .rename_axis("tema")
        .reset_index(name="n")
        .assign(**{"udeo_%": lambda x: (100.0 * x["n"] / n_total).round(2)})
    )
    sent_table = (
        sent_counts.rename_axis("sentiment")
        .reset_index(name="n")
        .assign(**{"udeo_%": lambda x: (100.0 * x["n"] / n_labeled).round(2) if n_labeled else 0.0})
        if n_labeled
        else pd.DataFrame(columns=["sentiment", "n", "udeo_%"])
    )
    sarc_table = (
        sarc_counts.rename_axis("sarkazam")
        .reset_index(name="n")
        .assign(**{"udeo_%": lambda x: (100.0 * x["n"] / n_labeled).round(2) if n_labeled else 0.0})
        if n_labeled
        else pd.DataFrame(columns=["sarkazam", "n", "udeo_%"])
    )
    length_rows = [
        {"mera": "karakteri", **_length_stats(df["n_chars"])},
        {"mera": "reci", **_length_stats(df["n_words"])},
    ]
    if n_labeled:
        for key, name in SENT_NAMES.items():
            length_rows.append(
                {"mera": f"karakteri | sentiment={name}", **_length_stats(labeled.loc[labeled["sentiment"] == key, "n_chars"])}
            )
        for key, name in SARC_NAMES.items():
            length_rows.append(
                {"mera": f"karakteri | {name.lower()}", **_length_stats(labeled.loc[labeled["sarcasm"] == key, "n_chars"])}
            )

    _write_table(source_table, tables_dir / "po_izvoru.csv")
    _write_table(tip_table, tables_dir / "po_temi.csv")
    _write_table(sent_table, tables_dir / "sentiment.csv")
    _write_table(sarc_table, tables_dir / "sarkazam.csv")
    _write_table(pd.DataFrame(combo_rows), tables_dir / "kombinacije.csv")
    _write_table(pd.DataFrame(length_rows), tables_dir / "duzina_teksta.csv")
    _write_table(
        df["script"]
        .value_counts()
        .rename_axis("pismo")
        .reset_index(name="n")
        .assign(**{"udeo_%": lambda x: (100.0 * x["n"] / n_total).round(2)}),
        tables_dir / "pismo.csv",
    )
    if n_labeled:
        _write_table(
            pd.crosstab(labeled["source_label"], labeled["sentiment"], margins=True)
            .rename(columns=SENT_NAMES)
            .reset_index(),
            tables_dir / "sentiment_po_izvoru.csv",
        )
        _write_table(
            pd.crosstab(labeled["source_label"], labeled["sarcasm"], margins=True)
            .rename(columns=SARC_NAMES)
            .reset_index(),
            tables_dir / "sarkazam_po_izvoru.csv",
        )
        _write_table(
            pd.crosstab(labeled["tip_label"], labeled["sentiment"], margins=True)
            .rename(columns=SENT_NAMES)
            .reset_index(),
            tables_dir / "sentiment_po_temi.csv",
        )
        _write_table(
            pd.crosstab(labeled["tip_label"], labeled["sarcasm"], margins=True)
            .rename(columns=SARC_NAMES)
            .reset_index(),
            tables_dir / "sarkazam_po_temi.csv",
        )

    n_exact_dup = int(df["text"].duplicated().sum())
    summary: dict[str, Any] = {
        "csv": str(csv_path),
        "n_ukupno": n_total,
        "n_anotirano": n_labeled,
        "n_neanotirano": n_unlabeled,
        "n_tacni_duplikati_teksta": n_exact_dup,
        "po_izvoru": source_table.set_index("izvor")["n"].to_dict(),
        "po_temi": tip_table.set_index("tema")["n"].to_dict(),
        "sentiment": sent_table.set_index("sentiment")["n"].to_dict() if len(sent_table) else {},
        "sarkazam": sarc_table.set_index("sarkazam")["n"].to_dict() if len(sarc_table) else {},
        "kombinacije": combo_rows,
        "duzina": {
            "karakteri": _length_stats(df["n_chars"]),
            "reci": _length_stats(df["n_words"]),
        },
        "pismo": df["script"].value_counts().to_dict(),
        "figures": saved,
        "figures_dir": str(figures_dir),
        "tables_dir": str(tables_dir),
    }
    (out_dir / "sazetak.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return summary
