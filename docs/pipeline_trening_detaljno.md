# Detaljan opis trening pipeline-a

Ovaj dokument objašnjava **šta se dešava u kodu** tokom treninga: nivoe pretprocesiranja, korake baseline i BERTić pipeline-a, feature-e, loss, balansiranje i metrike.

Za kratke komande vidi [treniranje.md](treniranje.md).  
Za GPU setup vidi [treniranje_gpu.md](treniranje_gpu.md).

---

## 1. Velika slika

Postoje **dva nezavisna treninga** nad istim anotiranim CSV-om (`data/processed/dataset/dataset.csv`):

```
                    ┌─────────────────────────────────────┐
                    │  dataset.csv (anotirani komentari)   │
                    │  id, source, text, tip, sentiment,    │
                    │  sarcasm                            │
                    └──────────────┬──────────────────────┘
                                   │
           ┌───────────────────────┴───────────────────────┐
           ▼                                               ▼
   ┌───────────────────┐                         ┌────────────────────┐
   │  BASELINE (sklearn)│                         │  BERTić (PyTorch)  │
   │  TF-IDF + NB/LR/SVM│                         │  classla/bcms-bertic│
   └─────────┬─────────┘                         └─────────┬──────────┘
             │                                             │
             │  + agresivniji clean                        │  tekst ostaje
             │    (latinica, lowercase)                    │  kakav je u CSV
             │                                             │
             ▼                                             ▼
   models/baselines/…                            models/<task>/best.pt
```

Važno: **ne mešati** dva pretprocesiranja. Tekst u `dataset.csv` je već lagano očišćen pri kolekciji; baseline na treningu radi **još jedan** sloj čišćenja; BERTić **ne**.

---

## 2. Dva nivoa pretprocesiranja

### 2.1 Nivo A — lagano čišćenje (kolekcija / dataset / BERTić)

**Modul:** `src/preprocessing/clean.py` → `preprocess_text`  
**Config:** `preprocessing:` u `config/config.yaml`  
**Kada:** pri append-u komentara (YouTube/Twitter/…) i u datasetu koji BERTić čita.

| Korak | Default | Efekat |
|-------|---------|--------|
| `strip_html` | `true` | uklanja HTML tagove, dekoduje entitete |
| `remove_emojis` | `true` | skida emoji |
| `remove_mentions` | `true` | skida `@username` |
| `replace_urls_with` | `"[URL]"` | URL → token `[URL]` (ne briše potpuno) |
| `normalize_whitespace` | `true` | sažima beline, jedan red |
| `remove_punctuation` | `false` | **zabranjeno** uključiti (važno za sarkazam) |
| `remove_stopwords` | `false` | zabranjeno |
| `transliterate` | `false` | latinica i ćirilica ostaju odvojeno |

**Šta se namerno čuva:** interpunkcija (`!`, `...`, `?!`), velika/mala slova, originalno pismo, sarkastični obrasci.

Primer:

```
Ulaz:  "@n1srbija Super 😂 https://x.com/a ..."
Izlaz: "Super [URL] ..."
```

### 2.2 Nivo B — agresivnije čišćenje (samo TF-IDF baseline)

**Modul:** `src/preprocessing/baseline.py` → `clean_text` / `clean_text_from_config`  
**Config:** `baseline_preprocessing:`  
**Kada:** unutar `load_baseline_frame` pre TF-IDF (ako je `baselines.apply_preprocessing: true`).

Redosled operacija:

1. Unicode NFC normalizacija  
2. potpuno uklanjanje URL-ova (razmak, **ne** `[URL]`)  
3. opciono emoji  
4. ćirilica → latinica (mapa digrafa `љ→lj`, `њ→nj`, …)  
5. lowercase  
6. opciono lematizacija (`simplemma`, jezik `hbs`)  
7. sažimanje belina  

Trenutni default u configu:

```yaml
baseline_preprocessing:
  remove_urls: true
  collapse_whitespace: true
  normalize_unicode: true
  remove_emojis: false      # emoji su već skinuti u nivou A
  cyrillic_to_latin: true
  lowercase: true
  lemmatize: false
```

Primer:

```
Ulaz (iz dataset.csv): "Види [URL] То је супер!!!"
Posle baseline clean:  "vidi to je super!!!"
```

Originalni tekst se čuva u koloni `text_raw` u memoriji tokom baseline run-a; u model ide `text` posle čišćenja.

---

## 3. Labele (zajednički ugovor)

| Polje | Dozvoljene vrednosti | Značenje |
|-------|----------------------|----------|
| `sentiment` | `1`, `0`, `-1` | pozitivno / neutralno / negativno |
| `sarcasm` | `1`, `0` | da / ne |

Za BERTić string → ID (`src/modeling/labels.py`):

- sentiment: `1→0`, `0→1`, `-1→2` (redosled iz `SENTIMENT_VALUES`)
- sarcasm: `1→0`, `0→1`

Nevalidni ili prazni redovi **ne ulaze** u trening (filtriraju se u baseline loaderu / `prepare_splits.py`).

---

## 4. Baseline pipeline — korak po korak

**Ulaz:** `python scripts/baselines/train_baselines.py --task all`  
**Orkestracija:** `src/baselines/runner.py` → `run_baseline_experiments`

### Korak 1 — učitavanje

1. Čita CSV (`baselines.csv` ili `paths.dataset_csv`).
2. Normalizuje kolone (`topic`→`tip`, `url`→`source` ako treba).
3. Za svaki task filtrira validne labele:
   - sentiment task: validan `sentiment` + neprazan `text`
   - sarcasm task: validan `sarcasm` + neprazan `text`
4. Ako je `apply_preprocessing: true`, na svaki tekst primeni **nivo B**.
5. Odbaci redove koji posle čišćenja ostanu prazni.

### Korak 2 — train / test split

- Stratifikovani `train_test_split` (sklearn), `test_size=0.2`, `random_seed=42`.
- Stratifikacija po labeli taska; ako neka klasa ima premali broj primera, fallback bez stratifikacije.
- Split se čuva u `models/baselines/<task>/_split/{train,test}.csv` radi reprodukcije.

**Napomena:** baseline **ne koristi** `data/processed/splits/` (to je za BERTić). Svaki task ima svoj 80/20 split.

### Korak 3 — TF-IDF feature-i

Sklearn `Pipeline`: `TfidfVectorizer` → klasifikator.

Parametri (`baselines.tfidf`):

| Parametar | Default | Značenje |
|-----------|---------|----------|
| `max_features` | 20000 | max veličina vokabulara |
| `ngram_range` | `[1, 2]` | unigrami + bigrami |
| `min_df` | 2 | ignoriši termine u < 2 dokumenta |
| `max_df` | 0.95 | ignoriši termine u > 95% dokumenata |
| `sublinear_tf` | `true` | `1 + log(tf)` umesto sirovog tf |

Vektorizer uči vokabular **samo na train** skupu (`fit`), pa transformiše test.

### Korak 4 — klasifikatori

Tri modela (isti TF-IDF, različiti `clf`):

| Ime | Algoritam | Ključni hiperparametri |
|-----|-----------|------------------------|
| `naive_bayes` | `MultinomialNB` | `alpha=1.0` |
| `logistic_regression` | `LogisticRegression` | `C=1.0`, `max_iter=2000`, solver `lbfgs` |
| `linear_svm` | `LinearSVC` | `C=1.0`, `max_iter=5000` |

### Korak 5 — balansiranje klasa

Flag: `training.use_class_weights` (default `true`).

| Model | Kako se balansira |
|-------|-------------------|
| LR / Linear SVM | `class_weight='balanced'` (sklearn automatski težine) |
| Naive Bayes | nema `class_weight` → `sample_weight` po klasi pri `fit` |

Ako je flag `false`, svi treniraju bez ponderisanja.

### Korak 6 — predikcija i metrike

Na test skupu:

- accuracy  
- precision / recall / F1 **macro**  
- F1 **weighted**  
- confusion matrix  
- `classification_report` po klasama  

Artefakti po modelu:

```
models/baselines/<task>/<model>/
  metrics.json
  confusion_matrix.csv
  predictions.csv      # id, text, sentiment, sarcasm, y_true, y_pred
  run_meta.json        # use_class_weights, težine, n_train/n_test
  model.joblib         # ceo Pipeline (ako save_model=true)
```

Plus `models/baselines/summary.json` sa pregledom svih run-ova.

---

## 5. BERTić pipeline — korak po korak

### 5.1 Priprema split-ova

**Komanda:**

```bash
python scripts/modeling/prepare_splits.py --csv data/processed/dataset/dataset.csv
```

**Šta radi** (`scripts/modeling/prepare_splits.py`):

1. Učitaj CSV.  
2. Zadrži samo redove gde su **oba** `sentiment` i `sarcasm` validna i `text` nije prazan.  
3. Napravi stratum: `sentiment + "|" + sarcasm` (6 mogućih kombinacija).  
4. Po svakom stratumu izmešaj (`seed=42`) i podeli:
   - **70%** train  
   - **15%** val  
   - **15%** test  
5. Upisi `labeled.csv`, `train.csv`, `val.csv`, `test.csv`, `split_meta.json`.

Za male grupe: 1 primer → samo train; 2 primera → train+test; inače garantuje bar 1 u testu kad je moguće.

**Pretprocesiranje ovde:** nema dodatnog — tekst ostaje onaj iz `dataset.csv` (nivo A).

### 5.2 Učitavanje i tokenizacija

**Orkestracija:** `src/modeling/runner.py` → `run_training`  
**Dataset:** `src/modeling/data.py` → `CommentDataset`

1. Učitaj train/val/test.  
2. Izračunaj class weights i (opciono) sample weights sa **train** skupa.  
3. Učitaj Hugging Face tokenizator za `classla/bcms-bertic`.  
4. Za svaki primer:
   - tekst ide **direktno** u tokenizator (bez baseline clean-a),
   - `truncation=True`, `max_length=128` (config),
   - padding se radi u collate funkciji po batch-u.

### 5.3 Balansiranje (BERTić)

Dva mehanizma (oba default ON):

| Flag | Efekat |
|------|--------|
| `training.use_class_weights` / `modeling.use_class_weights` | ponderisani `CrossEntropyLoss` po frekvenciji klasa na trainu |
| `modeling.use_weighted_sampler` | `WeightedRandomSampler` po kombinaciji `sentiment|sarcasm` (samo train loader) |

Težine se loguju u `run_meta.json` i `class_balance.json`.

### 5.4 Arhitekture modela

**Single-task** (`sentiment` ili `sarcasm`):

- `AutoModelForSequenceClassification` na BERTić encoderu  
- 3 klase (sentiment) ili 2 (sarcasm)  
- loss: CrossEntropy (opciono weighted)

**Multitask:**

- zajednički `AutoModel` encoder  
- dropout  
- dve linearne glave: sentiment (3) + sarcasm (2)  
- ukupan loss = `w_s * CE_sentiment + w_c * CE_sarcasm`  
  (default `w_s = w_c = 1.0`, config `modeling.multitask`)

### 5.5 Trening petlja (`src/modeling/train_loop.py`)

Po epohi:

1. `model.train()`  
2. Za svaki batch: forward → loss → backward → **gradient clipping** (`max_grad_norm=1.0`) → AdamW step → LR scheduler step  
3. Evaluacija na **val**  
4. Selekcija najboljeg modela po val score-u (macro-F1 / kombinovani score za multitask)  
5. Čuvanje `best.pt` kad score poraste  

Optimizer / raspored:

| Parametar | Default |
|-----------|---------|
| Optimizer | AdamW |
| `learning_rate` | `2e-5` |
| `weight_decay` | `0.01` |
| `num_epochs` | `4` |
| `warmup_ratio` | `0.1` (linear warmup pa decay) |
| `batch_size` | `8` (train) / `16` (eval) |
| `device` | CUDA ako postoji, inače CPU |

Posle treninga: evaluacija **best** checkpointa na testu → `test_metrics.json`.

### 5.6 Metrike (BERTić)

- overall: accuracy, macro-F1, …  
- posebno **na podskupu gde je `sarcasm=1`** (važno za master temu)  
- multitask: odvojeno za sentiment i sarcasm glavu  

Artefakti:

```
models/<task>/
  best.pt
  metrics.json / test_metrics.json
  run_meta.json
  class_balance.json
  tokenizer fajlovi (sačuvani uz run)
models/comparison.json   # ako --task all
```

---

## 6. Uporedni pregled

| Aspekt | Baseline | BERTić |
|--------|----------|--------|
| Ulazni tekst | dataset + **nivo B** clean (**uključujući lowercase**) | dataset (**samo nivo A**, **bez lowercase**) |
| Feature-i | ručni TF-IDF (1–2 grama) | kontekstualni embeddingi tokena |
| Split | interno 80/20 po tasku | fiksni 70/15/15 u `splits/` |
| Val skup | nema (samo train/test) | da — biranje `best.pt` |
| Modeli | NB, LR, Linear SVM | single-task ×2 + multitask |
| Balans | class_weight / sample_weight | weighted CE + WeightedRandomSampler |
| Inferenca | `scripts/baselines/predict_baselines.py` | `scripts/modeling/predict.py` |
| Poređenje | `scripts/compare_baseline_bertic.py` | isto |
| Uređaj | CPU dovoljan | GPU preporučen |
| Tipično trajanje | minute | desetine minuta–sati (zavisno od hardvera) |

---

## 7. Šta se dešava pre treninga (prikupljanje)

Radi kompletnosti — put od sirovog komentara do reda u datasetu:

1. **Kolekcija** (`append_youtube` / `append_twitter` / `append_instagram`) skida ili unosi tekst.  
2. **Nivo A** čišćenje + filter dužine/jezika + deduplikacija.  
3. Upis u `data/processed/sources/<platform>_comments.csv` sa `tip`/`topic` i (ručno) labelama.  
4. `python scripts/dataset/build_dataset.py` spaja sva tri CSV-a → `dataset.csv` sa ID `1…N`.  
5. Anotacija `sentiment` / `sarcasm` (ako nije urađena tokom unosa).  
6. Tek onda trening (sekcije 4 i 5).

---

## 8. Seed i reproduktivnost

- Globalni `random_seed: 42` u configu.  
- Baseline: isti seed za split i (gde postoji) `random_state` klasifikatora.  
- BERTić: `set_seed` za `random` / `numpy` / `torch` (+ CUDA).  
- Split fajlovi i `_split/` CSV-ovi omogućavaju ponavljanje evaluacije na istom testu.

Potpuna bit-identičnost na GPU nije uvek garantovana (nedeterminizam CUDA), ali rezultati treba da budu bliski.

---

## 9. Gde je šta u kodu

| Deo | Put |
|-----|-----|
| Lagano čišćenje | `src/preprocessing/clean.py` |
| Baseline čišćenje | `src/preprocessing/baseline.py` |
| Baseline load + filter | `src/baselines/data.py` |
| TF-IDF + klasifikatori | `src/baselines/pipeline.py` |
| Baseline orkestracija | `src/baselines/runner.py` |
| Baseline metrike | `src/baselines/metrics.py` |
| Split | `scripts/modeling/prepare_splits.py` |
| Dataset / DataLoader | `src/modeling/data.py` |
| Label mape | `src/modeling/labels.py` |
| Modeli | `src/modeling/models.py` |
| Trening petlja | `src/modeling/train_loop.py` |
| BERTić orkestracija | `src/modeling/runner.py` |
| Class / sample weights | `src/modeling/balancing.py` |
| Config | `config/config.yaml` |
