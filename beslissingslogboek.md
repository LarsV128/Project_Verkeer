# Beslissingslogboek — Verkeer

De kolom **Waarom** vul je zelf in, in je eigen woorden: dat is wat je op het no-AI gesprek moet kunnen uitleggen.

| # | Datum | Beslissing | Voorstel van de AI | Wat ik deed | Waarom | Alternatief dat ik overwoog |
|---|---|---|---|---|---|---|
| 1 | 2026-11-09 | Score per zwart punt: 5 / 3 / 1 punten per ongeval | Vlaamse telling overnemen: per slachtoffer (5 / 3 / 1) met factor 1,7 voor voetganger, fiets en bromfiets | aangepast: punten per ongeval volgens het zwaarste type (`class_accident`); factor 1,7 verworpen, want de bron heeft geen slachtoffers (VAR_LIST: 45 variabelen, geen enkele over slachtoffers) | _(vul zelf in)_ | Per slachtoffer tellen en factor 1,7 toepassen op de betrokken weggebruikers (benadering, verworpen); eerst slachtoffer-data zoeken |
| 2 | 2026-11-09 | Standaard zwart punt: score ≥ 15 én ≥ 3 ongevallen | Beide regels uit de Vlaamse pagina overnemen als standaard (en periode van 3 jaar met pijltjes) | aanvaard (let op: 15 komt uit een telling per slachtoffer; bij ons per ongeval betekent het iets anders) | _(vul zelf in)_ | Terug naar de eigen drempel 7 (met min. 3 ongevallen) |
| 3 | 2026-11-09 | Alle zwarte punten als even grote cirkels op de kaart | Cirkelgrootte laten meegroeien met de score (straal 20 m × wortel(score)) en een hover met de score via pydeck | verworpen: elke cirkel even groot (60 m), een locatie-aanduiding; score in tabel (kleurschaal komt nog) | _(vul zelf in)_ | Grootte volgens score (zoals in de eerste versie: 15 m × score, gaf grote overlappende cirkels bij drempel 15) |
| 4 | 2026-11-09 | Coördinaten omrekenen naar kaartpunten en 2 onmogelijke waarden herstellen | Lambert72 omrekenen met pyproj; komma verschuiven bij de 2 afgeknotte waarden naar de plek die het dichtst bij het midden van de gemeente ligt | aanvaard (beide hersteld; visueel gecontroleerd op een weg) | _(vul zelf in)_ | De 2 rijen wegnemen als "onbekend" (Herentals is het zwakste vermoeden) |
| 5 | 2026-11-09 | Vaste kleurschaal voor de score van een zwart punt (blauw < 20, lichtblauw 20-24, geel 25-29, oranje 30-49, rood ≥ 50), overal dezelfde, met legende; kleur is altijd de puntenscore | Cirkelgrootte of hover met de score (pydeck) | aangepast: kleur i.p.v. grootte of hover; klassengrenzen gekozen op basis van de verdeling van de scores (mediaan 18, 90 % onder 26) | _(vul zelf in)_ | Doorlopende kleurschaal tot een vast maximum; geen kleur en enkel de tabel |
| 6 | 2026-11-09 | Indeling app: tabbladen Zwarte punten (eerst) en Data View; foute coördinaten onderaan Data View; data viewer uitklapbaar bovenaan; geen overzicht per type in Zwarte punten | Eigen tabblad Foute coördinaten en ruwe rijen (eerste 1000) onderaan | aangepast op jouw vraag | _(vul zelf in)_ | Tabbladen houden zoals ze waren |
| 7 | 2026-11-09 | Tabblad Kaart: geen type-filter (alle types tellen mee via de punten), boven de kaart enkel een tabel met aantal zwarte punten (België, gewesten, provincies); app heet Zwarte Punten | Type-filter en overzicht per type behouden; aantallen als losse cijfers boven de kaart | aangepast op jouw vraag; zwarte punten worden nu eerst voor heel België gezocht en daarna per gebied gefilterd | _(vul zelf in)_ | Zoeken enkel binnen het gekozen gebied (kan op grenzen een ander resultaat geven) |

## Uitleg: hoe de coördinaten werken

Alles staat in `clean_data.py` (cleaning) en `app.py` (kaart en zwarte punten). Details en cijfers: `aanames.txt` onderdeel 3.

**1. Wat staat in de bron?** Per ongeval `MS_X_COORD` en `MS_Y_COORD` in **Lambert72** (Belgian Lambert 72, EPSG:31370), in **meters**.
Tot 2022 met decimale komma (`149732,99`), vanaf 2023 met punt; de cleaning zet beide om naar een getal.

**2. Status per rij (`coord_status`).**
- `ontbreekt`: X of Y is leeg (35.881 rijen, 12,4 %). Die ongevallen blijven in de tellingen maar staan niet op de kaart en tellen niet mee bij zwarte punten.
- `onmogelijk`: buiten België (X buiten 14.000-300.000 of Y buiten 20.000-250.000).
- `ok`: aanwezig en binnen België (253.649 rijen).
- `hersteld`: een onmogelijke waarde die we konden herstellen (2 rijen).

**3. Herstel van de 2 onmogelijke waarden.** In de bron staan ze in wetenschappelijke notatie (waarschijnlijk een Excel-afronding):
Knokke-Heist `7,30077E+15 / 2,26233E+14`, Herentals `1,8085E+16 / 2,004E+15`. Alleen de eerste cijfers bleven over.
De functie `herstel()` zet de komma op elke mogelijke plek (waarde = cijfers × 10^3 … 10^6), houdt enkel de plekken die in België vallen,
en kiest degene die het dichtst bij de **mediaan** van de andere ongevallen van dezelfde gemeente ligt (zelfde NIS-code).
Resultaat: Knokke-Heist X = 73.007,7 / Y = 226.233,0 en Herentals X = 180.850,0 / Y = 200.400,0.
Dit is een **vermoeden**; de originele waarden bewaren we in `x_bron` en `y_bron`.

**4. Omrekenen voor de kaart.** Een kaart werkt met breedte- en lengtegraad (WGS84, EPSG:4326). `pyproj` rekent om:
`Transformer.from_crs(31370, 4326, always_xy=True)`. Controle: het midden van Brussel komt uit op 50,85 N / 4,36 O.

**5. Afstanden.** Voor het samenvoegen van ongevallen tot een zwart punt rekenen we in **Lambert72 (meters)**, niet in graden: `cKDTree.query_ball_point` zoekt alle ongevallen binnen de gekozen afstand (standaard 50 m).

**6. Waar staat een zwart punt?** Op het **gemiddelde van de lat/lon** van de ongevallen die erin zitten (`bouw_spots` in `app.py`). De cirkel duidt dus een gemiddelde plek aan, niet een exact kruispunt.
