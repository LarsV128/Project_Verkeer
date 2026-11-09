# Project Verkeer

## Werkwijze
- We werken samen: **stel eerst vragen en wacht op antwoord voor je iets doet** (bestanden maken of wijzigen, commando's uitvoeren, app starten). Alleen lezen en uitleggen mag zonder te vragen.
- Bied bij een nieuwe taak kort een voorstel aan (wat, waarom, welke keuzes) en laat de gebruiker kiezen.
- **Commits:** maak nooit zelf een commit. Vraag altijd eerst of er gecommit mag worden en stel het commit-bericht voor. Houd dat bericht kort (één regel) en wacht op akkoord of aanpassing.
- Houd antwoorden kort: geen lange uitleg of opsommingen als de gebruiker er niet om vraagt.

## Data-afspraken
- `data/raw/` bevat de originele data (OPENDATA_MAP_2017-2024, VAR_LIST_ACC_OPENDATA_MAP). **Wijzig, hernoem of verwijder nooit een bestand in `data/raw/`.** Een hook (`.claude/hooks/protect_raw.py`) blokkeert dit. De inhoud staat niet in git (te groot); enkel de lege map via `.gitkeep`.
- Opgeschoonde data gaat in `data/`, nooit terug in `data/raw/`.
- Naamgeving: `<bronnaam>_clean.<ext>`, bv. `data/OPENDATA_MAP_2017-2024_clean.csv`. Tussenstappen krijgen een beschrijvend achtervoegsel (`_clean_geo`, ...). Geen `final2`, `nieuw`, `kopie`.
- Houd de cleaning reproduceerbaar: een script of notebook leest uit `data/raw/` en schrijft naar `data/`.
- De huidige opgeschoonde data is `data/OPENDATA_MAP_2017-2024_clean.parquet`, gemaakt met `python clean_data.py`. De app voert dit script zelf uit als de parquet ontbreekt of ouder is dan `data/raw/` of het script; de parquet staat niet in git. Bij elke wijziging in de cleaning: script opnieuw uitvoeren en `aanames.txt` bijwerken.

## Bestanden in dit project
- `app.py`: Streamlit-app (`streamlit run app.py --server.port 1234`), (de app heet "Ongevallen"), met de tabbladen Kaart, Evolutie (lijngrafieken per periode van 3 jaar) en Data View (met onderaan de foute coördinaten). Geen zijbalk; filters staan bovenaan elk tabblad, alle andere instellingen van Kaart in het tandwiel.
- `clean_data.py`: data/raw → data, inclusief omrekening van de coördinaten en het herstel van de 2 onmogelijke waarden.
- `aanames.txt`: alle keuzes en instellingen uitgeschreven (coördinaten, puntenscore, definitie van een zwart punt, ranglijst). **Houd dit bestand bij elke wijziging in de logica actueel.**
- `MISSION.md`: missie, werkprincipes, genomen keuzes en open punten. Houd dit ook actueel.

## Werken aan de definitie van een zwart punt
- De definitie is voorlopig (zie `aanames.txt` onderdeel 5). Wijzig drempels of werkwijze nooit stil: leg de wijziging vooraf voor en noteer ze daarna in `aanames.txt` en `MISSION.md`.
