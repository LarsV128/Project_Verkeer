# Mission statement – Verkeersongevallen in België, 2017-2024

> Bijgewerkt op 2026-10-27. Alle details en getallen staan uitgeschreven in [`aanames.txt`](aanames.txt).

## Missie
Verkeersongevallen in België (2017-2024) verkennen en begrijpen: waar, wanneer en met welke afloop ze gebeuren.
We zoeken de plaatsen waar ongevallen zich opstapelen ("zwarte punten") en onderzoeken later hoe die zich per jaar
ontwikkelen. Elke bewering in dit project moet met echte cijfers uit de data kunnen worden onderbouwd en zonder AI
kunnen worden uitgelegd.

## Werkprincipes
- **De ruwe data is heilig.** Niets in `data/raw/` wordt ooit gewijzigd; een hook (`.claude/hooks/protect_raw.py`) blokkeert dat.
- **Opgeschoonde data staat in `data/`**, met de naam `<bronnaam>_clean.<ext>`. De cleaning is reproduceerbaar via `clean_data.py` (data/raw → data).
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

### Puntenscore
- Elk ongeval krijgt punten: licht 1 · zwaar 3 · doden 5 (instelbaar met het tandwiel ⚙️).
- Gebruikt voor de methode "Puntenscore" bij zwarte punten en voor de ranglijst van gemeenten.

### Zwarte punten (voorlopige definitie)
- Een zwart punt is een plek waar **binnen een gekozen afstand genoeg ongevallen** gebeurden.
- **Buren tellen in plaats van een raster:** rastercellen snijden clusters op de celrand door en missen 13-17% van de ongevallen die een cirkel wel vindt.
- Werkwijze: het ongeval met de hoogste som vormt een cluster met zijn buren, die worden weggenomen, en dat herhaalt zich. Zo overlappen punten niet.
- **Afstand:** standaard 50 m (instelbaar 25-200 m), ongeveer één kruispunt.
- **Twee methodes:** *Aantal ongevallen* (standaard minstens 5 totaal of 3 per jaar) en *Puntenscore* (standaard vanaf 7 punten).
- **Extra voorwaarden (optioneel, in het tandwiel):** een plek met enkel lichtgewonden telt niet mee; minimum voor licht, zwaar en doden; minimum zwaar en doden combineren met *en* of *of*.
- **Twee weergaven:** *Vaste locaties, aantallen per jaar* (voor evolutie van dezelfde plek) en *Elk jaar apart*.

### App (Streamlit)
- Bestand `app.py`, starten met `streamlit run app.py` (we draaien op poort 1234). Geen zijbalk.
- Tabbladen **Overzicht**, **Zwarte punten** en **Foute coördinaten**, elk met eigen filters bovenaan (één gewest, één provincie, één gemeente, jaren en types) en een overzicht per type ongeval.
- **Overzicht** bevat ook een ranglijst van gemeenten op puntenscore, genummerd over heel België: bij een gewest of provincie alle gemeenten, bij een gemeente de top 5, de onderste 5 en 5 erboven en eronder.

## Open punten
- [ ] De **officiële definitie van "zwart punt"** opzoeken en onze definitie daarop afstemmen.
- [ ] Het verschil tussen "Met doden" (code 1) en "Dodelijk gewonden" (code 2) navragen; nu samengevoegd.
- [ ] **Eerlijk vergelijken over de jaren:** rekening houden met de verschillende dekking van de coördinaten.
- [ ] De hook `protect_raw.py` verfijnen: hij blokkeert nu ook het lezen met `open("data/raw/...")`.
- [ ] Eventueel een correctie voor inwonersaantal bij de ranglijst (extern bestand nodig) en de evolutie van zwarte punten onderzoeken.
- [ ] De keuzes in het beslissingslogboek zetten (`/timtim:logboek`).
