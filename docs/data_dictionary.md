# Data dictionary — annotation dataset

Fajl: `data/processed/annotation/annotation_template.csv` (isto i `data/processed/dataset/dataset.csv`)

Kodiranje: **UTF-8 with BOM** (`utf-8-sig`) radi lakog otvaranja u Excel-u.

| Kolona | Tip | Obavezno | Opis | Dozvoljene vrednosti |
|--------|-----|----------|------|----------------------|
| `id` | string | ne* | Identifikator uzorka | npr. `sr-00001`; ako fali, split/train generišu privremeni |
| `source` | string | preporučeno | **Platforma** (ne tema) | `youtube`, `twitter`, `reddit`, … (ili URL ako tako vodiš provenijenciju) |
| `text` | string | da | Očišćen srpski tekst | Interpunkcija sačuvana |
| `tip` | string | preporučeno | **Tema / subject** (domen) | npr. `filmovi`, `serije`, `politika`, `sport`, `ostalo` |
| `sentiment` | string/int | da (za trening) | Polaritet (preneseni stav) | `1` / `0` / `-1` |
| `sarcasm` | string/int | da (za trening) | Da li je tekst sarkastičan | `1` / `0` |

\* Za trening su dovoljni `text` + `sentiment` + `sarcasm`.  
`source` = gde je tekst sa (platforma); `tip` = o čemu je (tema). To **ne ulazi** u BERTić ulaz osim ako eksplicitno ne dodaš kao feature.

## Napomene

- Kolone `sentiment` i `sarcasm` se **ručno** popunjavaju.
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
