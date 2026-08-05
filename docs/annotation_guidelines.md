# Uputstvo za ručnu anotaciju sentimenta i sarkazma

Ovaj dokument definiše kako anotirati kolone `sentiment` i `sarcasm` u fajlu `data/processed/annotation_template.csv`.

## Dozvoljene vrednosti

- **sentiment:** `1` (positive) | `0` (neutral) | `-1` (negative)
- **sarcasm:** `1` (yes) | `0` (no)
- **tip:** npr. `filmovi` (domen sadržaja)

Ne ostavljajte druge oznake. Ako ste u potpunoj nedoumici, ostavite polje prazno i označite red u posebnom listu „za diskusiju“ (npr. komentar u Sheets-u).

---

## Opšti principi

1. **Anotirajte namerni / preneseni stav**, ne samo površinske pozitivne/negativne reči.
2. **Sarkazam** = govornik kaže nešto što (u kontekstu) znači suprotno ili podsmešljivo, često sa ciljem kritike ili ironičnog komentara.
3. **Sentiment** u ovom datasetu odnosi se na **evaluativni polaritet poruke kako je namenjena** (šta govornik zaista izražava prema predmetu), a `sarcasm` posebno belži da li je sredstvo izražavanja sarkastično.
4. Interpunkcija (`!`, `...`, `?!`), navodnici i „pretjerani“ komplimenti često signaliziraju sarkazam — obratite pažnju.
5. Latinica i ćirilica se tretiraju jednako.

### Preporučeni redosled

1. Pročitajte tekst.
2. Odlučite da li ima **sarkazma** (`1`/`0`).
3. Odlučite **sentiment** poruke (`1`/`0`/`-1`).
4. Proverite da li kombinacija ima smisla (vidi teške slučajeve ispod).

---

## Sentiment

| Labela | Kada koristiti |
|--------|----------------|
| `1` | Pohvala, zadovoljstvo, odobravanje, pozitivna ocena |
| `0` | Činjenično, opisno, pitanje bez jasnog stava, mešovito bez dominante |
| `-1` | Kritika, nezadovoljstvo, odbacivanje, negativna ocena |

---

## Sarkazam

| Labela | Kada koristiti |
|--------|----------------|
| `1` | Postoji jasan sarkastični / podsmešljivi prenos značenja |
| `0` | Doslovno ili neutrano figurativno izražavanje bez sarkazma |

**Ironija vs sarkazam:** Ironija je širi pojam (suprotnost između rečenog i mišljenog). Sarkazam je tipično **oštriji, podsmešljiv, usmeren** (često prema osobi, delu, pojavi). U ovom projektu:
- jasno podsmešljive / „ubodne“ ironije → `sarcasm = 1`
- blaga stilska ironija bez jasnog uboda može biti granična — ako niste sigurni, preferirajte konzervativno `0` i zabeležite primer za kasniju kalibraciju

---

## Teški slučajevi (primeri na srpskom)

### 1) Pozitivan sentiment izražen sarkastično

Često: površinski kompliment, a namera je kritika.  
**Preporuka:** `sentiment = -1` (prava evaluacija), `sarcasm = 1`.  
Ako anotaciona šema zahteva „površinski polaritet“, to ovde **nije** slučaj — pratimo **preneseni** stav.

| Tekst | sentiment | sarcasm |
|-------|-----------|---------|
| Baš ste genijalni što ste ovo pustili u bioskope. | -1 | 1 |
| Bravo, još jedan „remek-delo“ od 2 sata dosade. | -1 | 1 |
| Одлично, баш ми је требао још један наставак који ништа не доноси. | -1 | 1 |

### 2) Negativan sentiment izražen sarkastično

Eksplicitna negativnost + sarkazam / podsmeh.

| Tekst | sentiment | sarcasm |
|-------|-----------|---------|
| Naravno da je odvratno — ko bi drugo očekivao od ovog režisera? | -1 | 1 |
| Šta ćemo, katastrofa kao i uvek, zar ne? | -1 | 1 |

Bez sarkazma:

| Tekst | sentiment | sarcasm |
|-------|-----------|---------|
| Film mi se nije dopao, spor je i dosadan. | -1 | 0 |

### 3) Neutralne sarkastične izjave

Sarkazam postoji, ali nema jasne pohvale ni kritike prema objektu (ili je stav nejasan / usmeren na situaciju).

| Tekst | sentiment | sarcasm |
|-------|-----------|---------|
| Ah da, opet „najbolja večer ikada“ — kao i svaki utorak. | 0 | 1 |
| Jeste, baš smo mi ovde stručnjaci za sve. | 0 | 1 |

Ako je meta jasno negativno ocenjena, pomerite ka `-1` + `1`.

### 4) Retorička pitanja

Retoričko pitanje **nije automatski** sarkazam.

| Tekst | sentiment | sarcasm | Objašnjenje |
|-------|-----------|---------|-------------|
| Zašto uopšte snimaju ovakve filmove? | -1 | 0 | Kritika bez sarkazma |
| Zašto da ne volimo ovo remek-delo od nula zvezda? | -1 | 1 | Retorika + sarkazam |
| Da li je neko gledao hrvatsku sinhronizaciju? | 0 | 0 | Informativno pitanje |

### 5) Ironija naspram sarkazma

| Tekst | Predlog | Napomena |
|-------|---------|----------|
| Lepo vreme za šetnju, a napolju pljusak. | 0 / 0 ili 1 | Blaga situaciona ironija; `1` samo ako zvuči podsmešljivo |
| Svaka čast produkciji na ovom promašaju. | -1 / 1 | Klasičan sarkazam |

### 6) Ambiguous / granični slučajevi

| Tekst | Predlog | Napomena |
|-------|---------|----------|
| Ma dobro je... | 0 / 0 | Nejasno; bez jačeg signala ne forsirajte sarkazam |
| „Dobro“ je. | -1 / 1 | Navodnici često signaliziraju distancu / sarkazam |
| Super!!! | 1 / 0 | Pretjerani uzvičnici sami po sebi nisu dovoljni |
| Super!!! (posle opisa katastrofe u threadu) | -1 / 1 | Kontext pomaže; ako kontekst nije u tekstu, budite oprezni |

Ako tekst **sam po sebi** ne nosi dovoljno signala, birajte konzervativnije labele (`sarcasm = 0`) ili ostavite prazno za kasniju konsultaciju.

---

## Šta ne raditi

- Ne anotirati na osnovu autora, izvora ili popularnosti.
- Ne „popravljati“ pravopis u koloni `text`.
- Ne izmišljati labele van dogovorenog skupa.
- Ne automatski tretirati emoji ostatke / URL tokene kao sentiment.

---

## Kontrola kvaliteta (preporuka)

- Na početku anotirajte 50–100 primera i sa mentorom uskladite granične slučajeve.
- Povremeno ponovo anotirajte isti mali uzorak radi provere konzistentnosti.
- Vodite kratku listu spornih primera.

Za opis kolona: [data_dictionary.md](data_dictionary.md).
