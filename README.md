# Sarcasm-aware sentiment analysis for Serbian

Pipeline za prikupljanje, preprocesiranje i pripremu do **2000** prirodnih srpskih tekstova za ručnu anotaciju **sentimenta** i **sarkazma** (master rad).

Dataset je **samostalan** (bez SentiComments.SR): YouTube + (opciono) TikTok polu-ručno + ostali dozvoljeni izvori.

## Struktura projekta

```
config/                 # config.yaml, liste video ID-eva
src/
  common/               # konfiguracija, šema, I/O, jezik
  collection/           # posebni kolektori po izvoru
  preprocessing/        # čišćenje i deduplikacija
  dataset/              # annotation CSV + statistike
  pipeline.py           # end-to-end orkestracija
data/
  external/             # eksporte / rucno nalepjeni komentari
  raw/                  # sirovi podaci po izvoru
  interim/              # očišćeni zapisi
  processed/            # annotation_template.csv, dataset.csv
docs/
scripts/
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
python scripts/run_pipeline.py --sources youtube
```

3. Append **samo novih** videa:

```bash
python scripts/append_youtube.py
# ili
python scripts/append_youtube.py --url "https://www.youtube.com/watch?v=NOVI_ID"
```

### TikTok (polu-ručno, bez scrapinga)

TikTok **zabranjuje scraping**. Zvaničan akademski put: [Research API](https://developers.tiktok.com/products/research-api).

U projektu:

```bash
python scripts/append_tiktok.py --url "https://www.tiktok.com/@nesto/video/123" --comments-file data/external/tiktok/comments_paste.txt
```

1. Skripta otvori URL u browseru.
2. Ručno kopiraj **samo tekstove** komentara u TXT (jedan po liniji, bez username-a).
3. Skripta očisti, deduplikuje i dopiše na `annotation_template.csv` (`source=tiktok`).

### Statistike

```bash
python scripts/dataset_stats.py
```

## Konfiguracija

[`config/config.yaml`](config/config.yaml) — limiti, putanje, jezik, izvori.

## Izvori

| Izvor | Pristup |
|-------|---------|
| YouTube | Data API v3 |
| TikTok | polu-ručni unos (`append_tiktok.py`) / Research API ako odobre |
| Reddit | samo odobreni RFR eksport |
| Reviews | lokalni CSV/JSONL/TXT |

Detalji: [`docs/collection_ethics.md`](docs/collection_ethics.md).

## Labele (ručno)

- `sentiment`: `positive` | `neutral` | `negative`
- `sarcasm`: `yes` | `no`

Vidi: [`docs/data_dictionary.md`](docs/data_dictionary.md), [`docs/annotation_guidelines.md`](docs/annotation_guidelines.md).
