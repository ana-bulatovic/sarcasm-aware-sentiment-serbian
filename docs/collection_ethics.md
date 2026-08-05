# Etičko i pravno prikupljanje podataka

Ovaj projekat prikuplja **javno dostupne** tekstove za **nekomercijalno akademsko** istraživanje (master rad). Cilj je reproducibilnost uz poštovanje API-ja, licenci i privatnosti.

## Principi

1. Koristiti **zvanične API-je** ili **eksplicitno licencirane korpuse**.
2. **Ne** zaobilaziti autentifikaciju, CAPTCHA, paywall ili rate-limit.
3. **Ne** scrapovati HTML kada postoji zabranjujući ToS ili kada je scraping jedini način zaobilaženja kontrole pristupa.
4. Čuvati samo tekst potreban za istraživanje; **ne** čuvati username, email, profile URL, IP, itd.
5. Čuvati **sirove** podatke odvojeno od procesiranih (`data/raw/`).
6. Poštovati licence izvornih korpusa (atribuiranje, non-commercial, share-alike gde važi).

## Izvori u ovom pipeline-u

### YouTube

- Pristup: **YouTube Data API v3** (`commentThreads.list`) uz API key.
- Samo javni komentari; bez OAuth-a za čitanje.
- Poštovati kvote i [YouTube API Terms of Service](https://developers.google.com/youtube/terms/api-services-terms-of-service).
- Ne čuvati ime autora / channel ID autora.

### TikTok

- **Automatski browser scraping nije podržan** (krši TikTok ToS i često zahteva zaobilaženje zaštita).
- Akademski pristup: [TikTok Research Tools / Research API](https://developers.tiktok.com/products/research-api) uz prijavu i odobrenje.
- U ovom projektu: polu-ručni unos preko `scripts/collection/append_tiktok.py` (korisnik otvara URL i kopira javne tekstove komentara u TXT).

### Reddit

- Za akademsko istraživanje Reddit navodi da je zvaničan put **Reddit for Researchers (RFR)**.
- Pipeline **ne** implementira scraping ni neovlašćeni API pristup.
- `RedditExportCollector` učitava eksport koji istraživač dobije **odobrenim** putem.

### Javne recenzije (`reviews`)

- Nema ugrađenog web scrapera.
- Korisnik dodaje lokalne CSV/JSONL/TXT fajlove dobijene u skladu sa uslovima izvora.

### SentiComments.SR

- **Nije deo ovog samostalnog dataseta** (isključen iz pipeline-a i annotation fajla).
- Modul u kodu može ostati kao referenca, ali se ne koristi.

## Privatnost (GDPR / dobra praksa)

- Minimalni podaci: `text` + `source` (+ neobavezni ne-PII ID stavke).
- Izbegavati tekstove koji sadrže očigledne lične podatke (adrese, brojeve telefona); po potrebi ih ručno isključiti pre objave dataseta.
- Ako dataset kasnije bude javno deljen, proveriti licence svih izvora i anonimizaciju.

## Sintetički tekstovi

U ovoj fazi **nisu dozvoljeni** automatski generisani sintetički tekstovi. Dataset treba da bude od prirodno nastalih srpskih tekstova.
