# Sarcasm-aware sentiment analysis for Serbian

Pipeline za prikupljanje, preprocesiranje i pripremu do **2000** prirodnih srpskih tekstova za ručnu anotaciju **sentimenta** i **sarkazma** (master rad).

Dataset je **samostalan** (bez SentiComments.SR): YouTube + (opciono) TikTok polu-ručno + ostali dozvoljeni izvori.

## Struktura projekta

```
config/                 # config.yaml, liste video ID-eva
src/
  common/               # konfiguracija, šema, I/O, jezik
  collection/           # kolektori po izvoru
  preprocessing/        # čišćenje i deduplikacija
  dataset/              # annotation CSV + statistike
  modeling/             # fine-tune (single-task + multitask)
  pipeline.py           # end-to-end orkestracija
scripts/
  collection/           # skupljanje / append / pun pipeline
  preprocessing/        # raw -> interim
  dataset/              # build CSV, stats, draft anotacija
  modeling/             # split, train, evaluate
models/                 # checkpointi (gitignored)
data/
  external/             # eksporte / ručno nalepljeni komentari
  raw/                  # sirovi podaci po izvoru
  interim/              # očišćeni zapisi
  processed/            # annotation_template.csv, splits/
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

1. Video ID-evi u `config/youtube_video_ids.txt`
2. Skidanje / rebuild:

```bash
python scripts/collection/run_pipeline.py --sources youtube
```

3. Append **samo novih** videa:

```bash
python scripts/collection/append_youtube.py
# ili
python scripts/collection/append_youtube.py --url "https://www.youtube.com/watch?v=NOVI_ID"
```

### TikTok (polu-ručno, bez scrapinga)

TikTok **zabranjuje scraping**. Zvaničan akademski put: [Research API](https://developers.tiktok.com/products/research-api).

```bash
python scripts/collection/append_tiktok.py --url "https://www.tiktok.com/@nesto/video/123" --comments-file data/external/tiktok/comments_paste.txt
```

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
python scripts/dataset/dataset_stats.py --csv data/processed/annotation_template.csv
python scripts/modeling/prepare_splits.py
```

### Trening modela

Posle anotacije i splita:

```bash
python scripts/modeling/prepare_splits.py

# sva tri modela (sentiment, sarcasm, multitask)
python scripts/modeling/train.py --task all

# ili pojedinačno
python scripts/modeling/train.py --task sentiment
python scripts/modeling/train.py --task sarcasm
python scripts/modeling/train.py --task multitask

# evaluacija sačuvanog checkpointa
python scripts/modeling/evaluate.py --task sentiment --split test
```

Encoder: `classla/bcms-bertic` (podešava se u `config/config.yaml` → `modeling`).  
Rezultati: `models/<task>/best.pt`, `metrics.json`, `test_metrics.json`, plus `models/comparison.json` za `--task all`.  
Metrike: accuracy, macro-F1, i posebno na podskupu `sarcasm=1`.

## Konfiguracija

[`config/config.yaml`](config/config.yaml) — limiti, putanje, jezik, izvori.

## Izvori

| Izvor | Pristup |
|-------|---------|
| YouTube | Data API v3 |
| TikTok | polu-ručni unos (`append_tiktok.py`) / Research API ako odobre |
| Instagram | polu-ručni unos (`append_instagram.py`) / Meta Graph API ako odobre |
| Reddit | polu-ručni (`append_reddit.py`) / RFR eksport |
| Reviews | lokalni CSV/JSONL/TXT |

Detalji: [`docs/collection_ethics.md`](docs/collection_ethics.md).

## Labele (ručno)

- `tip`: npr. `filmovi`
- `sentiment`: `1` | `0` | `-1`
- `sarcasm`: `1` | `0`

Vidi: [`docs/data_dictionary.md`](docs/data_dictionary.md), [`docs/annotation_guidelines.md`](docs/annotation_guidelines.md).
