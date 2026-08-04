# Data dictionary — annotation dataset

Fajl: `data/processed/annotation_template.csv` (isto i `dataset.csv`)

Kodiranje: **UTF-8 with BOM** (`utf-8-sig`) radi lakog otvaranja u Excel-u.

| Kolona | Tip | Obavezno | Opis | Dozvoljene vrednosti |
|--------|-----|----------|------|----------------------|
| `id` | string | da | Stabilni identifikator uzorka u ovom datasetu | Format `sr-00001`, `sr-00002`, … |
| `source` | string | da | Pun URL snimka/videa sa kog je tekst uzet | npr. `https://www.youtube.com/watch?v=...` ili TikTok URL |
| `text` | string | da | Očišćen srpski tekst (latinica ili ćirilica) | Slobodan tekst; interpunkcija sačuvana |
| `sentiment` | string | ručno / predpopunjeno | Polaritet iskaza | `positive`, `neutral`, `negative` (prazno dok nije anotirano; za `senticomments_sr` se mapira iz originalnih labela) |
| `sarcasm` | string | ručno / predpopunjeno | Da li je tekst sarkastičan | `yes`, `no` (za `senticomments_sr`: sufiks `s` → `yes`) |

## Napomene

- Kolone `sentiment` i `sarcasm` se **ručno** popunjavaju (YouTube/TikTok/ostalo).
- Redosled redova: prvo `youtube`, zatim `tiktok`, zatim ostali izvori (pri rebuild-u).
- Ne koristite druge vrednosti (npr. `pozitivno`, `1`, `true`) — samo navedene engleske oznake radi konzistentnosti u kodu.
- Latinica i ćirilica se **ne** mešaju transliteracijom; ostaju kakve jesu u izvornom tekstu.
- `source` označava kanal prikupljanja, ne autora. Lični identifikatori (username, email) se ne čuvaju.
- Latinica i ćirilica se **ne** mešaju transliteracijom; ostaju kakve jesu u izvornom tekstu.
- `source` označava kanal prikupljanja, ne autora. Lični identifikatori (username, email) se ne čuvaju.

## Šest kombinacija labela

| sentiment | sarcasm | Značenje (kratko) |
|-----------|---------|-------------------|
| positive | no | Iskreno pozitivan stav |
| positive | yes | Pozitivan površinski ton, sarkazam (često zapravo negativna namera) |
| neutral | no | Neutralan, bez sarkazma |
| neutral | yes | Neutralan sadržaj izrečen sarkastično |
| negative | no | Iskreno negativan stav |
| negative | yes | Negativan ton sa sarkazmom / sarkastična kritika |

Detaljna pravila: [annotation_guidelines.md](annotation_guidelines.md).
