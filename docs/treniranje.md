# Trening posle prikupljanja i anotacije

Da — projekat ima **klasične ML baseline** modele (TF-IDF + Naive Bayes / Logistic Regression / Linear SVM) i **BERTić** fine-tune. Ovaj vodič pokriva redosled koraka od anotiranog CSV-a do rezultata.

Za **detaljan** opis pretprocesiranja i koraka u kodu vidi [pipeline_trening_detaljno.md](pipeline_trening_detaljno.md).  
Za GPU okruženje (PyTorch + CUDA) vidi [treniranje_gpu.md](treniranje_gpu.md).  
Za pravila anotacije vidi [annotation_guidelines.md](annotation_guidelines.md).

---

## 0. Preduslovi

1. Aktiviran venv i instaliran `requirements.txt`.
2. Anotirani komentari u `data/processed/dataset/dataset.csv` sa kolonama:

| Kolona | Vrednosti |
|--------|-----------|
| `id` | redni broj (npr. `1`, `2`, …) |
| `source` | `youtube` / `twitter` / `instagram` |
| `text` | tekst komentara |
| `tip` | domen (filmovi, politika, …) |
| `sentiment` | `1` / `0` / `-1` |
| `sarcasm` | `1` / `0` |

3. U trening ulaze **samo potpuno anotirani** redovi (oba polja validna). Prazna ili čudna polja (npr. `tip=0`, pogrešan `sarcasm`) se preskaču ili kvare metrike — proveri statistike pre treninga.

```bash
python scripts/dataset/dataset_stats.py --csv data/processed/dataset/dataset.csv
```

Ako si dodavala komentare u `data/processed/sources/*.csv`, prvo obnovi dataset:

```bash
python scripts/dataset/build_dataset.py
```

---

## 1. Brzi pregled: šta trenirati

| Tip | Šta radi | Ulaz | Komanda |
|-----|----------|------|---------|
| **Baseline** | TF-IDF + NB / LR / Linear SVM | direktno `dataset.csv` (sam pravi train/test) | `scripts/baselines/train_baselines.py` |
| **BERTić** | fine-tune `classla/bcms-bertic` | `train/val/test` split | `prepare_splits.py` pa `train.py` |
| **Jerteh** | fine-tune `jerteh/Jerteh-81` (ili `-355`) | isti split | `train.py --model-name jerteh/Jerteh-81 --output-dir models/jerteh81` |

Preporuka za master rad: prvo **baseline** (brzo, CPU), zatim **BERTić** (bolji rezultati, bolje na GPU).

---

## 2. Baseline trening (klasični ML)

Baseline **ne zahteva** `prepare_splits.py` — sam radi stratifikovani train/test split (default `test_size: 0.2` u `config.yaml`).

Tekst prolazi kroz `baseline_preprocessing` (URL-ovi, Unicode, ćirilica→latinica, lowercase, …), zatim TF-IDF.

### Sve odjednom (oba taska × 3 modela)

```bash
python scripts/baselines/train_baselines.py --task all
```

### Samo jedan task / model

```bash
python scripts/baselines/train_baselines.py --task sentiment
python scripts/baselines/train_baselines.py --task sarcasm
python scripts/baselines/train_baselines.py --task sentiment --model linear_svm
```

Dozvoljeni `--model`: `naive_bayes` | `logistic_regression` | `linear_svm` | `all`.

### Korisne opcije

```bash
# drugi CSV
python scripts/baselines/train_baselines.py --csv data/processed/dataset/dataset.csv

# samo metrike, bez čuvanja .joblib
python scripts/baselines/train_baselines.py --task all --no-save-model

# drugi folder za rezultate
python scripts/baselines/train_baselines.py --output-dir models/baselines_run2
```

### Gde su rezultati

```
models/baselines/
  summary.json
  sentiment/
    naive_bayes/      # metrics.json, confusion_matrix.csv, predictions.csv, run_meta.json, model.joblib
    logistic_regression/
    linear_svm/
  sarcasm/
    ...
```

Glavne metrike: accuracy, macro-F1 (i detalji po klasama u `metrics.json`).

**HTML izveštaj** (grafike: izvori, teme, labele, split, metrike, confusion matrices):

```bash
python scripts/reports/generate_report.py
# → reports/izvestaj/izvestaj.html + reports/izvestaj/figures/
```

Balansiranje klasa: `training.use_class_weights` u `config/config.yaml` (`true` / `false`).

### Inferenca (baseline)

Zahteva sačuvan `model.joblib` (treniraj **bez** `--no-save-model`).

```bash
python scripts/baselines/predict_baselines.py --task sentiment --model linear_svm \
  --text "Odličan film!" --show-preprocessed

python scripts/baselines/predict_baselines.py --task sarcasm --model logistic_regression \
  --text "Bravo majstore, baš si genijalac..."
```

---

## 3. BERTić trening (transformer)

### Korak A — train / val / test split

Baseline koristi svoj split; BERTić koristi fiksne fajlove u `data/processed/splits/`.

**Bitno:** podrazumevani ulaz skripte je `annotation_template.csv`. Pošto su anotacije u `dataset.csv`, prosledi ga eksplicitno:

```bash
python scripts/modeling/prepare_splits.py --csv data/processed/dataset/dataset.csv
```

Default razmere: **70% train / 15% val / 15% test**, stratifikovano po kombinaciji `sentiment|sarcasm`.

Izlaz:

```
data/processed/splits/
  labeled.csv
  train.csv
  val.csv
  test.csv
  split_meta.json
```

### Korak B — fine-tune

```bash
# sva tri režima: sentiment, sarcasm, multitask
python scripts/modeling/train.py --task all

# ili pojedinačno
python scripts/modeling/train.py --task sentiment
python scripts/modeling/train.py --task sarcasm
python scripts/modeling/train.py --task multitask
```

### Alternativni encoder: Jerteh (srpski RoBERTa)

Podrazumevano je `classla/bcms-bertic` (BERTić). Jerteh modeli se uključuju preko `--model-name` i **odvojenog** `--output-dir` da se checkpointi ne pregaze:

| HF ID | Napomena |
|-------|----------|
| `jerteh/Jerteh-81` | ~81M, preporuka za probu / laptop |
| `jerteh/Jerteh-355` | ~355M, više VRAM-a |

```bash
python scripts/modeling/prepare_splits.py --csv data/processed/dataset/dataset.csv

python scripts/modeling/train.py --task all \
  --model-name jerteh/Jerteh-81 \
  --output-dir models/jerteh81 \
  --epochs 6 \
  --device cuda

# samo sarkazam
python scripts/modeling/train.py --task sarcasm \
  --model-name jerteh/Jerteh-81 \
  --output-dir models/jerteh81 \
  --epochs 6
```

Evaluacija / inferenca:

```bash
python scripts/modeling/evaluate.py --task sarcasm --model-dir models/jerteh81/sarcasm
python scripts/modeling/predict.py --task sarcasm --model-dir models/jerteh81/sarcasm \
  --text "Bravo majstore, baš si genijalac..."
```

Lista poznatih encoder-a je i u `config/config.yaml` → `modeling.known_encoders`.

### Korisne opcije

```bash
python scripts/modeling/train.py --task sentiment --epochs 4 --batch-size 8 --device cuda
python scripts/modeling/train.py --task all --device cpu
python scripts/modeling/train.py --task sarcasm --model-name jerteh/Jerteh-81 --output-dir models/jerteh81
```

Hiperparametri (model, LR, epohe, …) su u `config/config.yaml` → sekcija `modeling`.

### Korak C — evaluacija

```bash
python scripts/modeling/evaluate.py --task sentiment --split test
python scripts/modeling/evaluate.py --task sarcasm --split test
python scripts/modeling/evaluate.py --task multitask --split test
```

### Gde su rezultati

```
models/
  sentiment/     # best.pt, metrics.json, test_metrics.json, run_meta.json
  sarcasm/
  multitask/
  comparison.json   # ako si trenirala --task all
```

Metrike: accuracy, macro-F1, i posebno na podskupu gde je `sarcasm=1`.

### Inferenca (BERTić)

```bash
python scripts/modeling/predict.py --task sentiment --text "Odličan film!"
python scripts/modeling/predict.py --task sarcasm --text "Bravo majstore..."
python scripts/modeling/predict.py --task multitask --text "Svaka čast na ovom promašaju."
python scripts/modeling/demo_predict.py
```

**Bez lowercasing-a:** BERTić dobija tekst kakav je u `dataset.csv` (samo lagano čišćenje pri kolekciji). Lowercase i ćirilica→latinica su **samo** za baseline.

---

## 4. Poređenje baseline vs BERTić

```bash
# demo rečenice (default)
python scripts/compare_baseline_bertic.py --demo

# svoj tekst / fajl
python scripts/compare_baseline_bertic.py --text "Baš si genijalac..."
python scripts/compare_baseline_bertic.py --file moji_tekstovi.txt --baseline-model logistic_regression
```

Ispisuje side-by-side sentiment/sarcasm za baseline i BERTić (single + multitask ako postoji), plus procenat slaganja.

---

## 5. Preporučeni tok (checklist)

Posle što si skupila i anotirala podatke:

```bash
# 1) Obnovi dataset iz source CSV-ova (ako treba)
python scripts/dataset/build_dataset.py

# 2) Proveri anotacije
python scripts/dataset/dataset_stats.py --csv data/processed/dataset/dataset.csv

# 3) Baseline (brzo)
python scripts/baselines/train_baselines.py --task all

# 4) Split za BERTić
python scripts/modeling/prepare_splits.py --csv data/processed/dataset/dataset.csv

# 5) BERTić
python scripts/modeling/train.py --task all

# 6) Evaluacija
python scripts/modeling/evaluate.py --task sentiment --split test
python scripts/modeling/evaluate.py --task sarcasm --split test
python scripts/modeling/evaluate.py --task multitask --split test
```

---

## 6. Česte greške

| Problem | Šta uraditi |
|---------|-------------|
| `Nema validno anotiranih redova` | Proveri da `sentiment` ∈ {`-1`,`0`,`1`} i `sarcasm` ∈ {`0`,`1`}, bez praznina |
| Baseline / split preskače dosta redova | `dataset_stats.py` → `unlabeled` / `other`; ispravi labele u CSV-u |
| BERTić spor na laptopu | Koristi GPU ([treniranje_gpu.md](treniranje_gpu.md)) ili smanji `--batch-size` / `--epochs` |
| Rezultati se „ne menjaju“ posle nove anotacije | Ponovo `build_dataset.py` (ako treba), pa `prepare_splits.py`, pa trening |
| Mešaš baseline i BERTić pretprocesiranje | Baseline ima dodatno čišćenje (uključujući lowercase); BERTić **ne** radi lowercase |
| `Nema sačuvanog baseline modela` | Treniraj bez `--no-save-model`: `python scripts/baselines/train_baselines.py --task all` |

---

## 7. Povezani fajlovi

| Fajl / folder | Uloga |
|---------------|--------|
| `config/config.yaml` | putanje, `baselines:`, `modeling:`, `training.use_class_weights` |
| `scripts/baselines/train_baselines.py` | baseline trening |
| `scripts/baselines/predict_baselines.py` | baseline inferenca |
| `scripts/compare_baseline_bertic.py` | poređenje baseline vs BERTić |
| `scripts/modeling/prepare_splits.py` | train/val/test |
| `scripts/modeling/train.py` | BERTić trening |
| `scripts/modeling/evaluate.py` | evaluacija checkpointa |
| `scripts/modeling/predict.py` | BERTić inferenca |
| `src/baselines/` | implementacija baseline pipeline-a |
| `src/modeling/` | implementacija BERTić treninga |
