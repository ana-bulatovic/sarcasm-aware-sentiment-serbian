# Data dictionary — annotation dataset

Fajl: `data/processed/annotation_template.csv` (isto i `dataset.csv`)

Kodiranje: **UTF-8 with BOM** (`utf-8-sig`) radi lakog otvaranja u Excel-u.

| Kolona | Tip | Obavezno | Opis | Dozvoljene vrednosti |
|--------|-----|----------|------|----------------------|
| `id` | string | ne* | Identifikator uzorka | npr. `sr-00001`; ako fali, split/train generišu privremeni |
| `source` | string | ne* | URL / izvor | može biti prazan; nije potreban za trening |
| `text` | string | da | Očišćen srpski tekst | Interpunkcija sačuvana |
| `tip` | string | preporučeno | Domen / tip sadržaja | npr. `filmovi` |
| `sentiment` | string/int | da (za trening) | Polaritet (preneseni stav) | `1` / `0` / `-1` |
| `sarcasm` | string/int | da (za trening) | Da li je tekst sarkastičan | `1` / `0` |

\* Za trening su dovoljni `text` + `sentiment` + `sarcasm`.

## Napomene

- Kolone `sentiment` i `sarcasm` se **ručno** popunjavaju.
- Ne koristite stare oznake (`positive`/`yes`) — samo `1` / `0` / `-1`.
- Latinica i ćirilica se ne transliterišu.
- `source` je kanal prikupljanja, ne autor (bez username-a).

## Šest kombinacija labela

| sentiment | sarcasm | Značenje (kratko) |
|-----------|---------|-------------------|
| 1 | 0 | Iskreno pozitivan stav |
| 1 | 1 | Namenjeni pozitivni stav izražen sarkastično |
| 0 | 0 | Neutralan, bez sarkazma |
| 0 | 1 | Neutralan sadržaj izrečen sarkastično |
| -1 | 0 | Iskreno negativan stav |
| -1 | 1 | Negativan stav sa sarkazmom (npr. „bravo majstore“) |

Detaljna pravila: [annotation_guidelines.md](annotation_guidelines.md).
