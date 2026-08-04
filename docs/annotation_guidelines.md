# Uputstvo za ručnu anotaciju sentimenta i sarkazma

Ovaj dokument definiše kako anotirati kolone `sentiment` i `sarcasm` u fajlu `data/processed/annotation_template.csv`.

## Dozvoljene vrednosti

- **sentiment:** `positive` | `neutral` | `negative`
- **sarcasm:** `yes` | `no`

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
2. Odlučite da li ima **sarkazma** (`yes`/`no`).
3. Odlučite **sentiment** poruke (`positive`/`neutral`/`negative`).
4. Proverite da li kombinacija ima smisla (vidi teške slučajeve ispod).

---

## Sentiment

| Labela | Kada koristiti |
|--------|----------------|
| `positive` | Pohvala, zadovoljstvo, odobravanje, pozitivna ocena |
| `neutral` | Činjenično, opisno, pitanje bez jasnog stava, mešovito bez dominante |
| `negative` | Kritika, nezadovoljstvo, odbacivanje, negativna ocena |

---

## Sarkazam

| Labela | Kada koristiti |
|--------|----------------|
| `yes` | Postoji jasan sarkastični / podsmešljivi prenos značenja |
| `no` | Doslovno ili neutrano figurativno izražavanje bez sarkazma |

**Ironija vs sarkazam:** Ironija je širi pojam (suprotnost između rečenog i mišljenog). Sarkazam je tipično **oštriji, podsmešljiv, usmeren** (često prema osobi, delu, pojavi). U ovom projektu:
- jasno podsmešljive / „ubodne“ ironije → `sarcasm = yes`
- blaga stilska ironija bez jasnog uboda može biti granična — ako niste sigurni, preferirajte konzervativno `no` i zabeležite primer za kasniju kalibraciju

---

## Teški slučajevi (primeri na srpskom)

### 1) Pozitivan sentiment izražen sarkastično

Često: površinski kompliment, a namera je kritika.  
**Preporuka:** `sentiment = negative` (prava evaluacija), `sarcasm = yes`.  
Ako anotaciona šema zahteva „površinski polaritet“, to ovde **nije** slučaj — pratimo **preneseni** stav.

| Tekst | sentiment | sarcasm |
|-------|-----------|---------|
| Baš ste genijalni što ste ovo pustili u bioskope. | negative | yes |
| Bravo, još jedan „remek-delo“ od 2 sata dosade. | negative | yes |
| Одлично, баш ми је требао још један наставак који ништа не доноси. | negative | yes |

### 2) Negativan sentiment izražen sarkastično

Eksplicitna negativnost + sarkazam / podsmех.

| Tekst | sentiment | sarcasm |
|-------|-----------|---------|
| Naravno da je odvratno — ko bi drugo očekivao od ovog režisera? | negative | yes |
| Šta ćemo, katastrofa kao i uvek, zar ne? | negative | yes |

Bez sarkazma:

| Tekst | sentiment | sarcasm |
|-------|-----------|---------|
| Film mi se nije dopao, spor je i dosadan. | negative | no |

### 3) Neutralne sarkastične izjave

Sarkazam postoji, ali nema jasne pohvale ni kritike prema objektu (ili je stav nejasan / usmeren na situaciju).

| Tekst | sentiment | sarcasm |
|-------|-----------|---------|
| Ah da, opet „najbolja večer ikada“ — kao i svaki utorak. | neutral | yes |
| Jeste, baš smo mi ovde stručnjaci za sve. | neutral | yes |

Ako je meta jasno negativno ocenjena, pomerite ka `negative` + `yes`.

### 4) Retorička pitanja

Retoričko pitanje **nije automatski** sarkazam.

| Tekst | sentiment | sarcasm | Objašnjenje |
|-------|-----------|---------|-------------|
| Zašto uopšte snimaju ovakve filmove? | negative | no | Kritika bez sarkazma |
| Zašto da ne volimo ovo remek-delo od nula zvezda? | negative | yes | Retorika + sarkazam |
| Da li je neko gledao hrvatsku sinhronizaciju? | neutral | no | Informativno pitanje |

### 5) Ironija naspram sarkazma

| Tekst | Predlog | Napomena |
|-------|---------|----------|
| Lepo vreme za šetnju, a napolju pljusak. | neutral / no ili yes | Blaga situaciona ironija; `yes` samo ako zvuči podsmešljivo |
| Svaka čast produkciji na ovom promašaju. | negative / yes | Klasičan sarkazam |

### 6) Ambiguous / granični slučajevi

| Tekst | Predlog | Napomena |
|-------|---------|----------|
| Ma dobro je... | neutral / no | Nejasno; bez jačeg signala ne forsiraјte sarkazam |
| „Dobro“ je. | negative / yes | Navodnici često signaliziraju distancu / sarkazam |
| Super!!! | positive / no | Pretjerani uzvičnici sami po sebi nisu dovoljni |
| Super!!! (posle opisa katastrofe u threadu) | negative / yes | Kontext pomaže; ako kontekst nije u tekstu, budite oprezni |

Ako tekst **sam po sebi** ne nosi dovoljno signala, birajte konzervativnije labele (`sarcasm = no`) ili ostavite prazno za kasniju konsultaciju.

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
