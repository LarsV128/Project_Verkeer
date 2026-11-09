# Mission statement – Verkeersongevallen in België, 2017-2024

> Bijgewerkt op 2026-11-09. Alle details en getallen staan uitgeschreven in [`aanames.txt`](aanames.txt).

## Missie
Verkeersongevallen in België (2017-2024) verkennen en begrijpen: waar, wanneer en met welke afloop ze gebeuren.
We zoeken de plaatsen waar ongevallen zich opstapelen ("zwarte punten") en onderzoeken later hoe die zich per jaar
ontwikkelen. Elke bewering in dit project moet met echte cijfers uit de data kunnen worden onderbouwd en zonder AI
kunnen worden uitgelegd.

## Werkprincipes
- **De ruwe data is heilig.** Niets in `data/raw/` wordt ooit gewijzigd; een hook (`.claude/hooks/protect_raw.py`) blokkeert dat.
- **Opgeschoonde data staat in `data/`**, met de naam `<bronnaam>_clean.<ext>`. De cleaning is reproduceerbaar via `clean_data.py` (data/raw → data). De app voert dit script zelf uit als de parquet ontbreekt of verouderd is; de parquet staat niet in git.
- **We werken samen:** eerst vragen en afspreken, dan pas wijzigen (zie `CLAUDE.md`).
- **Aannames zijn expliciet.** Elke keuze die een uitkomst kan veranderen, staat in `aanames.txt` en in het beslissingslogboek.
- **Eerst controleren, dan vertrouwen:** cijfers rekenen we uit, en onverwachte resultaten (zoals de foute coördinaten) onderzoeken we.

## Keuzes die we al maakten

### Data
| Onderwerp | Keuze |
|---|---|
| Bron | `data/raw/OPENDATA_MAP_2017-2024.txt` (289.532 rijen, scheidingsteken `\|`, decimaal komma t/m 2022 en punt vanaf 2023) |
| Eenheid | Eén rij = één ongeval. Er is niet ontdubbeld. |
| Brussel | Heeft geen provincie; we vullen "Geen provincie (Brussel)" in. Het voorvoegsel "Provincie " laten we weg. |
| Types (`class_accident`) | 1 Met doden · 2 Dodelijk gewonden · 3 Met zwaargewonden · 4 Met lichtgewonden. Labels beginnen met een hoofdletter. |
| Groepen | licht = code 4 · zwaar = code 3 · doden = code 1 én 2 samen (*voorlopig*) |
| Taal | We gebruiken de Nederlandse tekstkolommen (`TX_*_NL`). |

### Coördinaten
- Bron is **Lambert72** (EPSG:31370, meters); voor de kaart rekenen we om naar breedte- en lengtegraad (WGS84, `pyproj`).
- Status per rij (`coord_status`): **ok** 253.649 · **hersteld** 2 · **ontbreekt** 35.881 · **onmogelijk** 0.
- **De 2 onmogelijke waarden zijn hersteld.** In de bron staan ze in wetenschappelijke notatie (Excel-afronding):
  Knokke-Heist (`7,30077E+15`) en Herentals (`1,8085E+16`). We zetten de komma op de plek in België die het dichtst bij het
  midden van de gemeente ligt. Beide punten liggen op een weg (visuele controle); Herentals blijft het zwakste vermoeden.
  De originele waarden staan in `x_bron` en `y_bron`.
- **Dekking verschilt per jaar:** 81-85% bruikbare coördinaten t/m 2021, 94-97% vanaf 2022. Een stijging in 2022-2024
  is dus deels een meeteffect.

### Puntenscore (Vlaamse telling, sinds 2026-11-09)
- Bron: [Vlaanderen, gevaarlijke punten](https://www.vlaanderen.be/mobiliteitsprofessionals/verkeersveiligheid/gevaarlijke-punten). Volledige samenvatting in `aanames.txt` onderdeel 4.
- Elk ongeval krijgt punten volgens zijn type: licht 1 · zwaar 3 · doden 5 (instelbaar met het tandwiel ⚙️).
- **Afwijking van Vlaanderen:** zij tellen per *slachtoffer*; onze bron (45 variabelen, gecontroleerd) heeft geen slachtoffers, enkel het zwaarste type per *ongeval*. Wij tellen dus per ongeval.
- De Vlaamse **verhogingsfactor 1,7** (voetgangers, fietsers, bromfietsers) nemen we **niet** over: de bron toont enkel wie *betrokken* was, niet wie het slachtoffer was. Dat zou een benadering zijn.
- Gebruikt voor de methode "Puntenscore" (standaard) bij zwarte punten en voor de ranglijst van gemeenten.

### Zwarte punten
- Een zwart punt is een plek waar **binnen een gekozen afstand genoeg (ernstige) ongevallen** gebeurden: standaard **score ≥ 15 én ≥ 3 ongevallen in een periode van 3 jaar** (Vlaamse regel). In het tabblad kies je die periode met pijltjes (2017-2019 … 2022-2024).
- **Buren tellen in plaats van een raster:** rastercellen snijden clusters op de celrand door en missen 13-17% van de ongevallen die een cirkel wel vindt.
- Werkwijze: het ongeval met de hoogste som vormt een cluster met zijn buren, die worden weggenomen, en dat herhaalt zich. Zo overlappen punten niet.
- **Afstand:** standaard 50 m (instelbaar 25-200 m), ongeveer één kruispunt.
- **Twee methodes:** *Puntenscore* (standaard, vanaf 15 punten en minstens 3 ongevallen) en *Aantal ongevallen* (standaard minstens 3).
- Resultaat standaard (50 m, heel België): 2022-2024 → 215 zwarte punten; 2021-2023 → 180.
- **Kaart:** elk zwart punt is een even grote cirkel op het gemiddelde van zijn ongevallen; score en aantallen staan in de tabel.
- **Extra voorwaarden (optioneel, in het tandwiel):** een plek met enkel lichtgewonden telt niet mee; minimum voor licht, zwaar en doden; minimum zwaar en doden combineren met *en* of *of*.
- **Twee weergaven:** *Vaste locaties, aantallen per jaar* (voor evolutie van dezelfde plek) en *Elk jaar apart*.

### App (Streamlit)
- Bestand `app.py`, starten met `streamlit run app.py` (we draaien op poort 1234). Geen zijbalk.
- In **Zwarte punten** staan alle keuzes (weergave, methode, afstand, punten, drempel, extra voorwaarden) in één menu ⚙️; bovenaan enkel de filters.
- Tabbladen **Overzicht**, **Zwarte punten** en **Foute coördinaten**, elk met eigen filters bovenaan (één gewest, één provincie, één gemeente, jaren en types) en een overzicht per type ongeval.
- Onder elke tabel, kaart en grafiek staat **"weergegeven / totaal geregistreerd"** in het gekozen gebied (bv. 253.651 / 289.532 = 87,6 % voor kaarten in heel België); onderaan staat de **bron** (Statbel).
- **Overzicht** bevat ook een ranglijst van gemeenten op puntenscore, genummerd over heel België: bij een gewest of provincie alle gemeenten, bij een gemeente de top 5, de onderste 5 en 5 erboven en eronder.

## Open punten
- [x] De **officiële definitie van "zwart punt"** opzoeken (2026-11-09: Vlaamse puntentelling overgenomen, zie boven).
- [ ] Een **afstand of segmentdefinitie** van Vlaanderen zoeken (de pagina noemt er geen) en **slachtoffers per ongeval** zoeken (andere Statbel-dataset?) zodat we per slachtoffer kunnen tellen (en eventueel de factor 1,7 kunnen toepassen).
- [ ] De drempel **15 punten** komt uit de telling per slachtoffer; onderzoeken wat een passende drempel is bij onze telling per ongeval.
- [ ] Het verschil tussen "Met doden" (code 1) en "Dodelijk gewonden" (code 2) navragen; nu samengevoegd.
- [ ] **Eerlijk vergelijken over de jaren:** rekening houden met de verschillende dekking van de coördinaten.
- [x] De hook `protect_raw.py` verfijnen (2026-11-09): lezen uit `data/raw/` mag nu (`open(pad)`, `open(pad, "r")`, `shutil.copy` met raw als bron); schrijven, verwijderen, verplaatsen en kopiëren *naar* raw blijft geblokkeerd, ook via `Path(...).unlink()`.
- [ ] De keuzes in het beslissingslogboek zetten (`/timtim:logboek`).
