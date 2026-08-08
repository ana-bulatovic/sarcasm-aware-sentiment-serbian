#!/usr/bin/env python3
"""Generiši HTML izveštaj: podela podataka, labele, teme, metrike, confusion matrices."""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts._bootstrap import ensure_project_root

ensure_project_root()

from src.common.config import load_config, resolve_path
from src.common.stdio_utf8 import configure_utf8_stdio

SENT_LABELS = {"1": "pozitivno", "0": "neutralno", "-1": "negativno"}
SARC_LABELS = {"1": "sarkazam", "0": "nije sarkazam"}
MODEL_NAMES = {
    "naive_bayes": "Naive Bayes",
    "logistic_regression": "Logistic Regression",
    "linear_svm": "Linear SVM",
}

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["axes.unicode_minus"] = False


def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _save_and_b64(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="png", bbox_inches="tight", facecolor="white")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _bar_counts(
    counts: dict[str, int],
    *,
    title: str,
    xlabel: str,
    color: str = "#3b82f6",
    rename: dict[str, str] | None = None,
) -> plt.Figure:
    items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    labels = [rename.get(k, k) if rename else k for k, _ in items]
    values = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(labels, values, color=color, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Broj komentara")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(val),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, max(values) * 1.15 if values else 1)
    fig.tight_layout()
    return fig


def _stacked_split(
    split_frames: dict[str, pd.DataFrame],
    col: str,
    *,
    title: str,
    label_map: dict[str, str],
) -> plt.Figure:
    order = ["train", "val", "test"]
    classes = list(label_map.keys())
    data = {c: [] for c in classes}
    for split in order:
        df = split_frames.get(split)
        if df is None or col not in df.columns:
            for c in classes:
                data[c].append(0)
            continue
        vc = df[col].astype(str).str.strip().value_counts()
        for c in classes:
            data[c].append(int(vc.get(c, 0)))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(order))
    bottom = np.zeros(len(order))
    colors = {"1": "#22c55e", "0": "#94a3b8", "-1": "#ef4444"}
    for c in classes:
        vals = np.array(data[c], dtype=float)
        ax.bar(x, vals, bottom=bottom, label=label_map[c], color=colors.get(c, "#64748b"))
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in order])
    ax.set_ylabel("Broj primera")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def _confusion_heatmap(
    matrix: list[list[int]],
    labels: list[str],
    *,
    title: str,
    label_map: dict[str, str] | None = None,
) -> plt.Figure:
    display = [label_map.get(l, l) if label_map else l for l in labels]
    arr = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    sns.heatmap(
        arr,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        xticklabels=display,
        yticklabels=display,
        ax=ax,
        cbar=True,
        square=True,
    )
    ax.set_xlabel("Predikcija")
    ax.set_ylabel("Stvarna klasa")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def _metrics_bar(rows: list[dict[str, Any]], *, title: str) -> plt.Figure:
    names = [r["name"] for r in rows]
    acc = [r["accuracy"] for r in rows]
    f1 = [r["macro_f1"] for r in rows]
    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width / 2, acc, width, label="Accuracy", color="#3b82f6")
    ax.bar(x + width / 2, f1, width, label="Macro-F1", color="#f59e0b")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Skor")
    ax.set_title(title)
    ax.legend()
    for i, (a, f) in enumerate(zip(acc, f1)):
        ax.text(i - width / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(i + width / 2, f + 0.02, f"{f:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    return fig


def _load_baseline_metrics(baselines_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"sentiment": {}, "sarcasm": {}}
    for task in ("sentiment", "sarcasm"):
        for model in MODEL_NAMES:
            path = baselines_dir / task / model / "metrics.json"
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            out[task][model] = payload.get("metrics", payload)
    return out


def _img_tag(b64: str, alt: str) -> str:
    return f'<img alt="{alt}" src="data:image/png;base64,{b64}" style="max-width:100%;height:auto;border:1px solid #e2e8f0;border-radius:8px;margin:8px 0;" />'


def generate_report(
    *,
    config: dict[str, Any],
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    dataset_csv = resolve_path(config["paths"]["dataset_csv"])
    splits_dir = resolve_path(config["paths"].get("splits_dir", "data/processed/splits"))
    baselines_dir = resolve_path(config.get("baselines", {}).get("output_dir", "models/baselines"))
    models_dir = resolve_path(config.get("modeling", {}).get("output_dir", "models"))

    df = pd.read_csv(dataset_csv, encoding="utf-8-sig", dtype=str).fillna("")
    # valid tip / labels for charts
    tip_counts = df["tip"].astype(str).str.strip().replace("", "<prazno>").value_counts().to_dict()
    source_counts = df["source"].astype(str).str.strip().value_counts().to_dict()
    sent_counts = {
        k: int(v)
        for k, v in df["sentiment"].astype(str).str.strip().value_counts().items()
        if k in SENT_LABELS
    }
    sarc_counts = {
        k: int(v)
        for k, v in df["sarcasm"].astype(str).str.strip().value_counts().items()
        if k in SARC_LABELS
    }
    invalid_sarc = int((~df["sarcasm"].astype(str).str.strip().isin(list(SARC_LABELS) + [""])).sum())
    # empty counted separately
    empty_sarc = int((df["sarcasm"].astype(str).str.strip() == "").sum())
    weird_sarc = int(
        (~df["sarcasm"].astype(str).str.strip().isin(list(SARC_LABELS))).sum()
    )

    split_frames: dict[str, pd.DataFrame] = {}
    for name in ("train", "val", "test", "labeled"):
        p = splits_dir / f"{name}.csv"
        if p.exists():
            split_frames[name] = pd.read_csv(p, encoding="utf-8-sig", dtype=str).fillna("")

    split_meta_path = splits_dir / "split_meta.json"
    split_meta = (
        json.loads(split_meta_path.read_text(encoding="utf-8"))
        if split_meta_path.exists()
        else {}
    )

    images: dict[str, str] = {}
    images["source"] = _save_and_b64(
        _bar_counts(source_counts, title="Broj komentara po izvoru", xlabel="Izvor", color="#0ea5e9"),
        figures_dir / "01_izvori.png",
    )
    images["tip"] = _save_and_b64(
        _bar_counts(tip_counts, title="Broj komentara po temi (tip)", xlabel="Tema", color="#8b5cf6"),
        figures_dir / "02_teme.png",
    )
    images["sentiment"] = _save_and_b64(
        _bar_counts(
            sent_counts,
            title="Raspodela sentiment labela (dataset)",
            xlabel="Sentiment",
            color="#22c55e",
            rename=SENT_LABELS,
        ),
        figures_dir / "03_sentiment_labels.png",
    )
    images["sarcasm"] = _save_and_b64(
        _bar_counts(
            sarc_counts,
            title="Raspodela sarkazam labela (dataset)",
            xlabel="Sarkazam",
            color="#f59e0b",
            rename=SARC_LABELS,
        ),
        figures_dir / "04_sarcasm_labels.png",
    )

    if {"train", "val", "test"} <= set(split_frames):
        images["split_sent"] = _save_and_b64(
            _stacked_split(
                split_frames,
                "sentiment",
                title="BERTić split: sentiment po train/val/test",
                label_map=SENT_LABELS,
            ),
            figures_dir / "05_split_sentiment.png",
        )
        images["split_sarc"] = _save_and_b64(
            _stacked_split(
                split_frames,
                "sarcasm",
                title="BERTić split: sarkazam po train/val/test",
                label_map=SARC_LABELS,
            ),
            figures_dir / "06_split_sarcasm.png",
        )

    baseline_metrics = _load_baseline_metrics(baselines_dir)
    bl_rows_sent = []
    bl_rows_sarc = []
    for model, pretty in MODEL_NAMES.items():
        if model in baseline_metrics["sentiment"]:
            m = baseline_metrics["sentiment"][model]
            bl_rows_sent.append(
                {
                    "name": pretty,
                    "accuracy": float(m["accuracy"]),
                    "macro_f1": float(m["macro_f1"]),
                    "n": int(m.get("n", 0)),
                }
            )
        if model in baseline_metrics["sarcasm"]:
            m = baseline_metrics["sarcasm"][model]
            bl_rows_sarc.append(
                {
                    "name": pretty,
                    "accuracy": float(m["accuracy"]),
                    "macro_f1": float(m["macro_f1"]),
                    "n": int(m.get("n", 0)),
                }
            )

    if bl_rows_sent:
        images["bl_sent_metrics"] = _save_and_b64(
            _metrics_bar(bl_rows_sent, title="Baseline metrike — sentiment (test)"),
            figures_dir / "07_baseline_sentiment_metrics.png",
        )
    if bl_rows_sarc:
        images["bl_sarc_metrics"] = _save_and_b64(
            _metrics_bar(bl_rows_sarc, title="Baseline metrike — sarkazam (test)"),
            figures_dir / "08_baseline_sarcasm_metrics.png",
        )

    # confusion matrices
    cm_html_parts: list[str] = []
    for task, label_map in (("sentiment", SENT_LABELS), ("sarcasm", SARC_LABELS)):
        for model, pretty in MODEL_NAMES.items():
            m = baseline_metrics.get(task, {}).get(model)
            if not m or "confusion_matrix" not in m:
                continue
            cm = m["confusion_matrix"]
            fig = _confusion_heatmap(
                cm["matrix"],
                cm["labels"],
                title=f"Baseline CM — {task} / {pretty}",
                label_map=label_map,
            )
            key = f"cm_{task}_{model}"
            images[key] = _save_and_b64(fig, figures_dir / f"cm_{task}_{model}.png")
            cm_html_parts.append(
                f"<h4>{pretty} ({task})</h4>" + _img_tag(images[key], key)
            )

    # BERTić
    bertic_sections: list[str] = []
    bertic_compare_rows: list[dict[str, Any]] = []
    for task in ("sentiment", "sarcasm", "multitask"):
        task_dir = models_dir / task
        test_path = task_dir / "test_metrics.json"
        metrics_path = task_dir / "metrics.json"
        if not test_path.exists() and not metrics_path.exists():
            bertic_sections.append(
                f"<p><b>{task}</b>: nema checkpointa / metrika "
                f"(<code>models/{task}/</code>).</p>"
            )
            continue
        payload = json.loads(
            (test_path if test_path.exists() else metrics_path).read_text(encoding="utf-8")
        )
        # metrics.json wraps test under test_metrics
        if "test_metrics" in payload and "overall" not in payload:
            payload = payload["test_metrics"]

        if task == "multitask":
            sent = payload.get("sentiment", {}).get("overall", {})
            sarc = payload.get("sarcasm", {}).get("overall", {})
            bertic_sections.append(
                "<h4>Multitask</h4>"
                f"<ul><li>Sentiment acc={sent.get('accuracy', '—'):.3f}, "
                f"macro-F1={sent.get('macro_f1', 0):.3f}</li>"
                f"<li>Sarkazam acc={sarc.get('accuracy', '—'):.3f}, "
                f"macro-F1={sarc.get('macro_f1', 0):.3f}</li></ul>"
            )
            continue

        overall = payload.get("overall", payload)
        acc = float(overall.get("accuracy", 0))
        f1 = float(overall.get("macro_f1", 0))
        n = int(overall.get("n", payload.get("n", 0) or 0))
        bertic_compare_rows.append({"name": f"BERTić {task}", "accuracy": acc, "macro_f1": f1, "n": n})

        # try CM from per_class support isn't enough — use history? 
        # BERTić metrics don't always store raw CM; build from per_class if possible no.
        # Check metrics.json val_metrics for confusion if present
        full = {}
        if metrics_path.exists():
            full = json.loads(metrics_path.read_text(encoding="utf-8"))
        best_ep = full.get("best_epoch", "?")
        hist_n = len(full.get("history") or [])

        note = ""
        if hist_n <= 1:
            note = (
                "<p style='color:#b45309'><b>Napomena:</b> izgleda kao kratak / smoke trening "
                f"(history={hist_n} epoha, best_epoch={best_ep}). "
                "Rezultati nisu reprezentativni za finalni model.</p>"
            )

        bertic_sections.append(
            f"<h4>BERTić — {task}</h4>"
            f"{note}"
            f"<ul><li>Accuracy: <b>{acc:.3f}</b></li>"
            f"<li>Macro-F1: <b>{f1:.3f}</b></li>"
            f"<li>n (test): {n}</li>"
            f"<li>best_epoch: {best_ep}</li></ul>"
        )

        # optional: on_sarcasm_yes for sentiment
        sy = payload.get("on_sarcasm_yes")
        if sy:
            bertic_sections.append(
                "<p>Sentiment na podskupu <code>sarcasm=1</code>: "
                f"acc={sy.get('accuracy', 0):.3f}, macro-F1={sy.get('macro_f1', 0):.3f}, "
                f"n={sy.get('n', '—')}</p>"
            )

        # confusion from classification report isn't matrix — skip unless in file
        cm_candidate = overall.get("confusion_matrix") or payload.get("confusion_matrix")
        if cm_candidate and "matrix" in cm_candidate:
            label_map = SENT_LABELS if task == "sentiment" else SARC_LABELS
            fig = _confusion_heatmap(
                cm_candidate["matrix"],
                cm_candidate["labels"],
                title=f"BERTić CM — {task}",
                label_map=label_map,
            )
            key = f"cm_bertic_{task}"
            images[key] = _save_and_b64(fig, figures_dir / f"cm_bertic_{task}.png")
            bertic_sections.append(_img_tag(images[key], key))

    if bertic_compare_rows:
        # combine with best baseline for chart
        compare_rows = list(bertic_compare_rows)
        if bl_rows_sent:
            best = max(bl_rows_sent, key=lambda r: r["macro_f1"])
            compare_rows.append(
                {"name": f"Baseline best sent ({best['name']})", **{k: best[k] for k in ("accuracy", "macro_f1", "n")}}
            )
        if bl_rows_sarc:
            best = max(bl_rows_sarc, key=lambda r: r["macro_f1"])
            compare_rows.append(
                {"name": f"Baseline best sarc ({best['name']})", **{k: best[k] for k in ("accuracy", "macro_f1", "n")}}
            )
        images["compare"] = _save_and_b64(
            _metrics_bar(compare_rows, title="Poređenje modela (Accuracy / Macro-F1)"),
            figures_dir / "09_compare_models.png",
        )

    # HTML
    split_table_rows = ""
    for split in ("train", "val", "test"):
        if split not in split_frames:
            continue
        d = split_frames[split]
        split_table_rows += (
            f"<tr><td>{split}</td><td>{len(d)}</td>"
            f"<td>{int((d['sentiment']=='1').sum())}/{int((d['sentiment']=='0').sum())}/{int((d['sentiment']=='-1').sum())}</td>"
            f"<td>{int((d['sarcasm']=='1').sum())}/{int((d['sarcasm']=='0').sum())}</td></tr>"
        )

    bl_table = ""
    for task, rows in (("sentiment", bl_rows_sent), ("sarcasm", bl_rows_sarc)):
        for r in rows:
            bl_table += (
                f"<tr><td>{task}</td><td>{r['name']}</td>"
                f"<td>{r['accuracy']:.3f}</td><td>{r['macro_f1']:.3f}</td>"
                f"<td>{r['n']}</td></tr>"
            )

    html = f"""<!DOCTYPE html>
<html lang="sr">
<head>
  <meta charset="utf-8" />
  <title>Izveštaj — dataset, baseline i BERTić</title>
  <style>
    body {{ font-family: "Segoe UI", system-ui, sans-serif; margin: 32px auto; max-width: 980px;
           color: #0f172a; line-height: 1.45; padding: 0 16px; }}
    h1,h2,h3 {{ margin-top: 1.6em; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 14px; }}
    th, td {{ border: 1px solid #e2e8f0; padding: 8px 10px; text-align: left; }}
    th {{ background: #f8fafc; }}
    code {{ background: #f1f5f9; padding: 1px 5px; border-radius: 4px; }}
    .note {{ background: #fffbeb; border: 1px solid #fcd34d; padding: 12px 14px; border-radius: 8px; }}
    .path {{ color: #475569; font-size: 13px; }}
  </style>
</head>
<body>
  <h1>Izveštaj: podaci, baseline i BERTić</h1>
  <p class="path">Dataset: <code>{dataset_csv}</code><br/>
  Generisano iz projekta sarcasm-aware-sentiment-serbian.</p>

  <div class="note">
    <b>Gde su sirove metrike u projektu</b>
    <ul>
      <li>Baseline: <code>models/baselines/summary.json</code> i
          <code>models/baselines/&lt;task&gt;/&lt;model&gt;/metrics.json</code>,
          <code>confusion_matrix.csv</code></li>
      <li>BERTić: <code>models/&lt;task&gt;/test_metrics.json</code>,
          <code>metrics.json</code>, <code>run_meta.json</code></li>
      <li>Split: <code>data/processed/splits/</code> (+ <code>split_meta.json</code>)</li>
    </ul>
  </div>

  <h2>1. Dataset — veličina i izvori</h2>
  <p>Ukupno redova u <code>dataset.csv</code>: <b>{len(df)}</b></p>
  {_img_tag(images['source'], 'izvori')}
  {_img_tag(images['tip'], 'teme')}

  <h2>2. Raspodela labela</h2>
  {_img_tag(images['sentiment'], 'sentiment')}
  {_img_tag(images['sarcasm'], 'sarcasm')}
  <p>Nevalidne / čudne <code>sarcasm</code> vrednosti (nije 0/1): <b>{weird_sarc}</b>
     (uključujući prazne: {empty_sarc}).</p>

  <h2>3. Podela podataka (BERTić train/val/test)</h2>
  <p>Izvor split-a: <code>{split_meta.get('source_csv', splits_dir)}</code><br/>
  Ratio: train={split_meta.get('train_ratio', 0.7)}, val={split_meta.get('val_ratio', 0.15)}
  (ostatak test).</p>
  <table>
    <tr><th>Split</th><th>n</th><th>Sentiment 1/0/-1</th><th>Sarkazam 1/0</th></tr>
    {split_table_rows or '<tr><td colspan="4">Nema split fajlova</td></tr>'}
  </table>
  {(_img_tag(images['split_sent'], 'split sentiment') if 'split_sent' in images else '')}
  {(_img_tag(images['split_sarc'], 'split sarcasm') if 'split_sarc' in images else '')}
  <p class="note">Baseline koristi <b>sopstveni</b> 80/20 split po tasku (vidi
  <code>models/baselines/&lt;task&gt;/_split/</code>), nije isti kao BERTić split.</p>

  <h2>4. Baseline metrike</h2>
  <table>
    <tr><th>Task</th><th>Model</th><th>Accuracy</th><th>Macro-F1</th><th>n test</th></tr>
    {bl_table or '<tr><td colspan="5">Nema baseline metrika</td></tr>'}
  </table>
  {(_img_tag(images['bl_sent_metrics'], 'bl sent') if 'bl_sent_metrics' in images else '')}
  {(_img_tag(images['bl_sarc_metrics'], 'bl sarc') if 'bl_sarc_metrics' in images else '')}

  <h2>5. Baseline — matrice zbunjenosti</h2>
  {''.join(cm_html_parts) if cm_html_parts else '<p>Nema confusion matrix fajlova.</p>'}

  <h2>6. BERTić</h2>
  {''.join(bertic_sections)}
  {(_img_tag(images['compare'], 'compare') if 'compare' in images else '')}

  <h2>7. Zaključak (kratko)</h2>
  <ul>
    <li>Baseline za <b>sarkazam</b> je trenutno znatno jači od postojećeg BERTić sentiment smoke-run-a.</li>
    <li>BERTić <b>sarcasm</b> / <b>multitask</b> checkpointi nedostaju ili nisu kompletirani — treba pravi trening (6+ epoha) na svežem splitu od <code>dataset.csv</code>.</li>
    <li>Grafike su sačuvane i u folderu <code>{figures_dir.as_posix()}</code>.</li>
  </ul>
</body>
</html>
"""

    html_path = out_dir / "izvestaj.html"
    html_path.write_text(html, encoding="utf-8")

    meta = {
        "dataset_rows": len(df),
        "source_counts": source_counts,
        "tip_counts": tip_counts,
        "sentiment_counts": sent_counts,
        "sarcasm_counts": sarc_counts,
        "split_meta": split_meta,
        "baseline": baseline_metrics,
        "html": str(html_path),
        "figures_dir": str(figures_dir),
    }
    (out_dir / "report_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return html_path


def main() -> None:
    configure_utf8_stdio()
    parser = argparse.ArgumentParser(description="Generiši HTML izveštaj sa grafikama.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--out-dir",
        default="reports/izvestaj",
        help="Izlazni folder (HTML + figures/)",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    out = resolve_path(args.out_dir)
    path = generate_report(config=config, out_dir=out)
    print(f"[report] HTML → {path}")
    print(f"[report] figures → {out / 'figures'}")


if __name__ == "__main__":
    main()
