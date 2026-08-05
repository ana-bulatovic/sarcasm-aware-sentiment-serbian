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

### PyTorch + CUDA (usklađivanje verzija)

`pip install -r requirements.txt` **ne** instalira torch — na GPU ga stavljaš ručno da se poklopi sa drajverom.

**1) Proveri šta drajver podržava:**
```bash
nvidia-smi
```
Gore desno piše npr. `CUDA Version: 12.4` — to je **maksimum** koji drajver dozvoljava (ne moraš imati isti CUDA toolkit instaliran).

**2) Obriši stari / pogrešan torch:**
```bash
pip uninstall -y torch torchvision torchaudio
```

**3) Instaliraj odgovarajući wheel** (biraj ≤ verziji iz `nvidia-smi`):

| nvidia-smi CUDA | Komanda |
|-----------------|--------|
| 12.1+ / **12.2** | `pip install "torch>=2.6" --index-url https://download.pytorch.org/whl/cu121` |
| 12.4+ | `pip install "torch>=2.6" --index-url https://download.pytorch.org/whl/cu124` |
| 12.6+ | `pip install "torch>=2.6" --index-url https://download.pytorch.org/whl/cu126` |
| 11.8+ | `pip install "torch>=2.6" --index-url https://download.pytorch.org/whl/cu118` |

Ako nisi sigurna, za novije kartice najčešće radi **cu124** ili **cu121**.

Alternativa (conda):
```bash
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
```

Zvanični birač: https://pytorch.org/get-started/locally/

**4) Provera:**
```bash
python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
```

Mora: `cuda: True`. Ako je `False`, torch je i dalje CPU build ili drajver/CUDA wheel ne odgovaraju — ponovi korake 1–3.

**Česte greške:**
- `pip install torch` sa PyPI → često **CPU** verzija
- torch `cu121` na drajveru koji javlja samo CUDA 11.x → ne radi
- mešanje conda torch + pip torch u istom env-u → ukloni jedno
- `ValueError: ... upgrade torch to at least v2.6` → noviji `transformers` zahteva torch≥2.6; reinstaliraj sa `"torch>=2.6"` i odgovarajućim `cuXXX` index-url

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
| `Nema validno anotiranih` | Proveri CSV: `sentiment` ∈ 1/0/-1, `sarcasm` ∈ 1/0 |
| OOM (nestalo VRAM) | Smanji `--batch-size` (npr. 8 ili 4) |
| Sporo / loši skorovi | Malo podataka ili premalo epoha — prvo ~1500–2000 anotiranih, pa 4+ epohe |
