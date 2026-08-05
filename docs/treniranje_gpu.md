# Pokretanje treninga na GPU mašini

Kratak vodič: šta prebaciti, kako podesiti okruženje i koje komande pokrenuti.

---

## 1. Šta prebaciti sa laptopa

Ceo projekat (kod + config + anotirani CSV). Anotacije (`annotation_template.csv`) **jesu** u gitu — posle `git push` / `git pull` stignu na drugi računar.

Ne moraš: `.env` (YouTube ključ nije potreban za trening), `models/` (checkpointi su lokalni / veliki).

---

## 2. Okruženje (jednom)

U root folderu projekta:

```bash
python -m venv .venv
```

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Linux:**
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### PyTorch sa CUDA (važno)

`requirements.txt` često stavi CPU torch. Na GPU mašini instaliraj CUDA build, npr. za CUDA 12.1:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Za drugu CUDA verziju vidi: https://pytorch.org/get-started/locally/

### Provera GPU-a

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NEMA GPU')"
```

Mora: `cuda: True` i ime kartice.

---

## 3. Split + trening

Uvek iz root foldera projekta, sa aktiviranim `.venv`:

```bash
# 1) stratifikovani train/val/test iz annotation_template.csv
python scripts/modeling/prepare_splits.py

# 2) sva tri modela: sentiment, sarcasm, multitask
python scripts/modeling/train.py --task all --device cuda
```

### Korisne varijante

```bash
# više epoha / veći batch (ako stane u VRAM)
python scripts/modeling/train.py --task all --device cuda --epochs 4 --batch-size 16

# samo jedan task
python scripts/modeling/train.py --task sentiment --device cuda
python scripts/modeling/train.py --task sarcasm --device cuda
python scripts/modeling/train.py --task multitask --device cuda
```

Defaulti (epohe, LR, model) su u `config/config.yaml` → sekcija `modeling`  
(encoder: `classla/bcms-bertic`, podrazumevano 4 epohe).

---

## 4. Gde su rezultati

```
models/
  sentiment/     best.pt, metrics.json, test_metrics.json
  sarcasm/       ...
  multitask/     ...
  comparison.json   ← samo posle --task all
```

Evaluacija ponovo:

```bash
python scripts/modeling/evaluate.py --task sentiment --split test
python scripts/modeling/evaluate.py --task multitask --split test
```

---

## 5. Inferenca (proba)

```bash
python scripts/modeling/predict.py --task sentiment --text "Odličan film, svaka čast!"
python scripts/modeling/predict.py --task multitask --text "Bravo majstore, baš si genijalac..."
```

Više tekstova:

```bash
python scripts/modeling/predict.py --task multitask --text "Prvi komentar" --text "Drugi komentar"
```

Ili iz fajla (jedan tekst po liniji):

```bash
python scripts/modeling/predict.py --task multitask --file moji_tekstovi.txt
```

---

## 6. Redosled kad anotiraš na laptopu, treniraš na GPU

1. Završi / proširi anotaciju na laptopu (`annotation_template.csv`).
2. `git add` + `commit` + `push` anotiranog CSV-a, pa na GPU mašini `git pull`.
3. Na GPU: `prepare_splits.py` pa `train.py --task all --device cuda`.
4. Po želji prebaci nazad ceo folder `models/` na laptop za inferencu / pisanje rada (modeli nisu u gitu).

---

## 7. Česte greške

| Problem | Šta uraditi |
|---------|-------------|
| `cuda: False` | Instaliraj CUDA torch (korak 2), proveri NVIDIA drajver (`nvidia-smi`) |
| `Nedostaje split fajl` | Prvo `prepare_splits.py` |
| `Nema validno anotiranih` | Proveri CSV: `sentiment` ∈ positive/neutral/negative, `sarcasm` ∈ yes/no |
| OOM (nestalo VRAM) | Smanji `--batch-size` (npr. 8 ili 4) |
| Sporo / loši skorovi | Malo podataka ili premalo epoha — prvo ~1500–2000 anotiranih, pa 4+ epohe |
