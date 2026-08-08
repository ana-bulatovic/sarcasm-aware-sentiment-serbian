# Sarcasm-aware sentiment analysis for Serbian

Pipeline za prikupljanje, preprocesiranje i pripremu do **2000** prirodnih srpskih tekstova za ručnu anotaciju **sentimenta** i **sarkazma** (master rad).

Dataset je **samostalan** (bez SentiComments.SR): YouTube + (opciono) TikTok polu-ručno + ostali dozvoljeni izvori.

## Struktura projekta

```
config/
  config.yaml
  sources/              # twitter_urls.txt, youtube_video_ids.txt
src/
  common/               # konfiguracija, šema, I/O, jezik
  collection/           # kolektori po izvoru
  preprocessing/        # čišćenje i deduplikacija
  dataset/              # annotation CSV + statistike
  baselines/            # TF-IDF + Naive Bayes / LR / Linear SVM
  modeling/             # fine-tune (single-task + multitask)
  pipeline.py           # end-to-end orkestracija
scripts/
  collection/           # skupljanje / append / pun pipeline
  preprocessing/        # raw -> interim; text_preprocessing.py (baseline ML)
  baselines/            # TF-IDF + NB/LR/SVM eksperimenti
  dataset/              # build CSV, stats, draft anotacija
  modeling/             # split, train, evaluate (BERTić)
models/                 # checkpointi (gitignored)
data/
  external/             # eksporte / ručno nalepljeni komentari
  raw/                  # sirovi podaci po izvoru
  interim/              # očišćeni zapisi
  processed/
    sources/            # youtube_comments.csv, twitter_comments.csv
    annotation/         # annotation_template.csv
    dataset/            # dataset.csv, stats
    splits/             # train/val/test
    scratch/            # privremeni fajlovi
docs/
```

## Brzi start

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# unesite YOUTUBE_API_KEY u .env
```

### YouTube (API)

1. Video ID-evi u `config/sources/youtube_video_ids.txt`
2. Skidanje / rebuild:

```bash
python scripts/collection/run_pipeline.py --sources youtube
```

3. Append u **poseban** CSV (kao Twitter):

```bash
# URL/ID-eve u config/sources/youtube_video_ids.txt, ili --url
python scripts/collection/append_youtube.py --tip filmovi
python scripts/collection/append_youtube.py --tip politika --url "https://www.youtube.com/watch?v=NOVI_ID"
```

Upisuje u `data/processed/sources/youtube_comments.csv` (`source=youtube`, `tip`=tema). Ne dira `annotation_template.csv`.

### TikTok (polu-ručno, bez scrapinga)

TikTok **zabranjuje scraping**. Zvaničan akademski put: [Research API](https://developers.tiktok.com/products/research-api).

```bash
python scripts/collection/append_tiktok.py --url "https://www.tiktok.com/@nesto/video/123" --comments-file data/external/tiktok/comments_paste.txt
```

### Twitter / X (polu-ručno → poseban CSV)

```bash
# 1) URL-ove u config/sources/twitter_urls.txt
# 2) pip install twikit
# 3) Login: cookies JSON -> data/external/twitter/session/cookies.json
#    (ili X_USERNAME / X_PASSWORD u .env)
python scripts/collection/append_twitter.py --tip politika
# Ručno (bez fetch-a): dodaj --manual
```

Upisuje u `data/processed/sources/twitter_comments.csv` (`source=twitter`, `tip`=tema). Ne dira `annotation_template.csv`.

### Reddit (polu-ručno)

```bash
python scripts/collection/append_reddit.py --url "https://www.reddit.com/r/serbia/comments/...." --comments-file data/external/reddit/comments_paste.txt
```

Dobri subredditi za srpski: npr. `r/serbia`, `r/askserbia` (javni threadovi).  
Veći akademski eksport: Reddit for Researchers → `data/external/reddit/export.jsonl`.

### Instagram (polu-ručno, bez scrapinga)

Isto kao TikTok — **nema** automatskog logina ni scrapinga (Instagram ToS).

```bash
python scripts/collection/append_instagram.py --url "https://www.instagram.com/p/SHORTCODE/" --comments-file data/external/instagram/comments_paste.txt
```

### Statistike i split

```bash
python scripts/dataset/dataset_stats.py --csv data/processed/annotation/annotation_template.csv
python scripts/modeling/prepare_splits.py
```

### Trening modela

Posle anotacije:

```bash
# --- Klasični ML baseline (TF-IDF) ---
python scripts/baselines/train_baselines.py --task all
# ili: --task sentiment | --task sarcasm
#      --model naive_bayes | logistic_regression | linear_svm

# --- BERTić fine-tune ---
python scripts/modeling/prepare_splits.py
python scripts/modeling/train.py --task all

# ili pojedinačno
python scripts/modeling/train.py --task sentiment
python scripts/modeling/train.py --task sarcasm
python scripts/modeling/train.py --task multitask

# evaluacija sačuvanog checkpointa
python scripts/modeling/evaluate.py --task sentiment --split test
```

**Baseline** rezultati: `models/baselines/<task>/<model>/` (`metrics.json`, `confusion_matrix.csv`, `predictions.csv`, `run_meta.json`, `model.joblib`) + `models/baselines/summary.json`.

Balansiranje klasa (baseline + BERTić): `training.use_class_weights` u `config.yaml`  
- `true` → ponderisani loss / `class_weight='balanced'`  
- `false` → običan loss  
Korišćene težine se čuvaju u `run_meta.json` (`class_weights_used`).

Encoder (BERTić): `classla/bcms-bertic` (podešava se u `config/config.yaml` → `modeling`).  
Rezultati: `models/<task>/best.pt`, `metrics.json`, `test_metrics.json`, `run_meta.json`, plus `models/comparison.json` za `--task all`.  
Metrike: accuracy, macro-F1, i posebno na podskupu `sarcasm=1`.

## Pretprocesiranje po modelu

Postoje **dva** nivoa čišćenja — ne mešati ih.

| Model | Šta se radi sa tekstom | Modul / config |
|-------|------------------------|----------------|
| **BERTić** (`classla/bcms-bertic`) | Samo lagano čišćenje iz kolekcije: HTML, `@mention`, URL → `[URL]`, emoji (po configu), beline. **Bez** lowercase, **bez** ćirilica→latinica, **bez** lematizacije. Tokenizator radi na prirodnom tekstu. | `src/preprocessing/clean.py` → `preprocessing:` u `config.yaml` |
| **Klasični ML baseline** (TF-IDF + LR/SVM/NB) | Dodatni pipeline: uklanjanje URL-ova, Unicode NFC, sažimanje belina; opciono emoji, ćirilica→latinica, lowercase, lematizacija. Zatim TF-IDF → klasifikator. Balans: `training.use_class_weights` → `class_weight='balanced'` (LR/SVM) ili `sample_weight` (NB). | `src/baselines/` + `baseline_preprocessing:` / `baselines:` / `training:` |

```bash
# Primer: baseline tekstovi (ćirilica→latinica + lowercase, iz configa)
python scripts/preprocessing/text_preprocessing.py --use-config \
  --csv data/processed/dataset/dataset.csv \
  --out data/processed/scratch/dataset_baseline.csv

# Jedan tekst / ručne opcije
python scripts/preprocessing/text_preprocessing.py --text "Види https://x.com тест" \
  --cyrillic-to-latin --lowercase --remove-emojis
```

U kodu: `from src.preprocessing.baseline import clean_text, normalize_script, lemmatize_text`.

## Konfiguracija

[`config/config.yaml`](config/config.yaml) — limiti, putanje, jezik, izvori.

## Izvori

| Izvor | Pristup |
|-------|---------|
| YouTube | Data API v3 (`append_youtube.py` → `processed/sources/youtube_comments.csv`) |
| TikTok | polu-ručni unos (`append_tiktok.py`) / Research API ako odobre |
| Instagram | polu-ručni unos (`append_instagram.py`) / Meta Graph API ako odobre |
| Twitter/X | twikit fetch (`append_twitter.py` → `processed/sources/twitter_comments.csv`) |
| Reddit | polu-ručni (`append_reddit.py`) / RFR eksport |
| Reviews | lokalni CSV/JSONL/TXT |

Detalji: [`docs/collection_ethics.md`](docs/collection_ethics.md).

## Labele (ručno)

- `tip`: npr. `filmovi`
- `sentiment`: `1` | `0` | `-1`
- `sarcasm`: `1` | `0`

Vidi: [`docs/data_dictionary.md`](docs/data_dictionary.md), [`docs/annotation_guidelines.md`](docs/annotation_guidelines.md).
