# Šta pokrenuti na GPU računaru (dataset 2000 + izveštaj)

Sve iz **root foldera projekta**, sa aktiviranim `.venv`.
PowerShell: `.\.venv\Scripts\Activate.ps1`

Ako si već trenirala na starom splitu (~1550), **ponovi split + trening**.
Izveštaj čita `dataset.csv` + `splits/` + `models/` — stari checkpointi ne odgovaraju novih 2000.

---

## 0) Ažuriraj kod i dataset

```powershell
git pull
python -c "import pandas as pd; df=pd.read_csv('data/processed/dataset/dataset.csv', encoding='utf-8-sig'); print('dataset.csv:', len(df))"
```

Mora ispisati **2000**. Ako nije, `git pull` nije uzeo poslednji commit.

---

## 1) Novi train/val/test split (obavezno)

Bez `--csv` skripta čita `annotation_template.csv` (~1411), ne 2000.

```powershell
python scripts/modeling/prepare_splits.py --csv data/processed/dataset/dataset.csv
```

Provera — `validno` treba da bude ~2000:

```powershell
python -c "import json; m=json.load(open('data/processed/splits/split_meta.json', encoding='utf-8')); print(m['total_rows'], 'redova,', m['valid_labeled'], 'validnih; train/val/test=', m['train'], m['val'], m['test'])"
```

---

## 2) Statistike i grafikoni dataseta (za poglavlje o podacima)

```powershell
python scripts/dataset/dataset_stats.py --csv data/processed/dataset/dataset.csv
python scripts/dataset/plot_dataset_stats.py --csv data/processed/dataset/dataset.csv --out-dir reports/dataset_statistike
```

Izlaz: `reports/dataset_statistike/` (`sazetak.json`, `figures/`, `tabele/`).

---

## 3) Baseline (TF-IDF + NB / LR / SVM)

Čita **direktno** `dataset.csv` (svoj 80/20 split, ne BERTić split).

```powershell
python scripts/baselines/train_baselines.py --task all --csv data/processed/dataset/dataset.csv
```

Izlaz (nije u gitu): `models/baselines/` — `summary.json`, `metrics.json`, matrice, `.joblib`.

---

## 4) Encoder (BERTić) — sentiment, sarkazam, multitask

Koristi split iz koraka 1.

```powershell
python scripts/modeling/train.py --task all --device cuda
```

Ako stane VRAM: dodaj `--batch-size 4`. Više epoha (preporuka za rad): `--epochs 6`.

Izlaz (nije u gitu):

```
models/sentiment/     best.pt, metrics.json, test_metrics.json, run_meta.json
models/sarcasm/       ...
models/multitask/     ...
models/comparison.json
```

---

## 5) HTML izveštaj (metrike + slike za rad)

Mora posle koraka 1–4, da uvuče nove splitove i nove `models/**/test_metrics.json`.

```powershell
python scripts/reports/generate_report.py
```

Izlaz (ovo **jeste** za git):

- `reports/izvestaj/izvestaj.html` — otvori u browseru
- `reports/izvestaj/report_meta.json` — brojevi za tabele u Wordu
- `reports/izvestaj/figures/` — PNG za poglavlja

---

## 6) Šta ide na git, šta ne

```powershell
git add data/processed/splits/ reports/dataset_statistike/ reports/izvestaj/
git status
```

Zatim commit + push (splitovi + izveštaj).

**Ne šalji** `models/` (gitignore). Ako ti treba inferenca na laptopu, prekopiraj folder `models/` ručno (USB/disk).

---

## Brza provera pre commita

| Provera | Očekivano |
|---------|-----------|
| `dataset.csv` | 2000 |
| `split_meta.json` → `valid_labeled` | ~2000, ne 1531 |
| `reports/izvestaj/report_meta.json` → `split_meta.total_rows` | 2000, ne 1546 |
| `izvestaj.html` sekcija BERTić | ima brojeve, ne „nema checkpointa“ |
