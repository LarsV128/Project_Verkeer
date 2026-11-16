"""Verwerking van de zwarte punten: scores, clusters en tellingen per periode.

Dit bestand bevat enkel berekeningen (geen Streamlit); `app.py` toont de resultaten.
De instelling van een zwart punt staat hier bovenaan. Wijzig ze nooit stil.
"""
import heapq

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# ---------- 1. Instelling ----------

# class_accident_code -> groep voor de scores en zwarte punten
LICHT, ZWAAR, DODEN = [4], [3], [1, 2]

# Punten per ONGEVAL (niet per slachtoffer: de bron heeft geen slachtoffers), volgens het zwaarste type.
PUNT_LICHT, PUNT_ZWAAR, PUNT_DODEN = 1, 3, 5
AFSTAND = 50  # meter: ongevallen binnen deze afstand worden samengenomen tot één zwart punt
SCORE_DREMPEL = 15  # een zwart punt heeft minstens zoveel punten (dus ook minstens 3 ongevallen: 3 x 5 = 15)
PERIODE_JAREN = 3  # Vlaanderen rekent gevaarlijke punten op 3 jaar ongevallendata
BRUIKBAAR = ["ok", "hersteld"]  # coord_status waarvan de coördinaten bruikbaar zijn

# De clusterfunctie rekent enkel met gehele getallen (exact, geen afrondingsfouten). Kies je ooit andere punten
# (bv. 1,7 x 5), dan moet zoek_clusters opnieuw worden aangepast.
if not all(isinstance(w, int) for w in (PUNT_LICHT, PUNT_ZWAAR, PUNT_DODEN, SCORE_DREMPEL)):
    raise TypeError("PUNT_LICHT, PUNT_ZWAAR, PUNT_DODEN en SCORE_DREMPEL moeten gehele getallen zijn.")


# ---------- 2. Score per ongeval ----------


def punten_per_ongeval(p):
    """Score van elk ongeval als geheel getal: de punten van zijn type (1 / 3 / 5, zoals bij de Vlaamse telling voor
    gevaarlijke punten)."""
    code = p["class_accident_code"].astype(int)
    return np.select([code.isin(DODEN), code.isin(ZWAAR)], [PUNT_DODEN, PUNT_ZWAAR], default=PUNT_LICHT).astype(np.int64)


# ---------- 3. Welke ongevallen doen mee ----------


def zonder_standaard(p):
    """Laat de verdachte standaardcoördinaten weg (bv. stadhuizen, zie clean_data.py): ze tellen niet mee voor de
    zwarte punten, wel voor alle andere tellingen."""
    return p[~p["verdacht_coord"]]


def is_bruikbaar(p):
    """Per ongeval True als de coördinaat bruikbaar is: status ok/hersteld én geen standaardcoördinaat."""
    return p["coord_status"].isin(BRUIKBAAR) & ~p["verdacht_coord"]


def bruikbare_ongevallen(p):
    """De ongevallen waarmee we zwarte punten zoeken: bruikbare coördinaten en geen standaardcoördinaten."""
    return p[is_bruikbaar(p)]


# ---------- 4. Clusters zoeken ----------


def zoek_clusters(x, y, gewicht, straal, drempel):
    """Geeft per ongeval een clusternummer (-1 = geen zwart punt).

    Voor elk ongeval tellen we het gewicht (een geheel getal) van alle ongevallen binnen `straal` meter (met
    gewicht 1 voor iedereen is dat gewoon het aantal). Het ongeval met de hoogste som vormt een cluster met
    die buren; die worden weggenomen en we herhalen dat zolang een som minstens `drempel` is.
    Zo overlappen clusters niet.
    """
    xy = np.column_stack([x, y])
    buren = cKDTree(xy).query_ball_point(xy, straal)  # inclusief het ongeval zelf
    tel = np.array([gewicht[b].sum() for b in buren], dtype=np.int64)
    weg = np.zeros(len(xy), dtype=bool)
    cluster = np.full(len(xy), -1)

    wachtrij = [(-t, i) for i, t in enumerate(tel) if t >= drempel]
    heapq.heapify(wachtrij)
    nummer = 0
    while wachtrij:
        neg, i = heapq.heappop(wachtrij)
        if weg[i]:
            continue
        if -neg != tel[i]:  # verouderde score: opnieuw inschuiven met de huidige
            if tel[i] >= drempel:
                heapq.heappush(wachtrij, (-tel[i], i))
            continue
        leden = [j for j in buren[i] if not weg[j]]
        cluster[leden] = nummer
        nummer += 1
        weg[leden] = True
        for j in leden:
            for k in buren[j]:
                if not weg[k]:
                    tel[k] -= gewicht[j]
    return cluster


def bouw_spots(punten, cluster):
    """Telt per cluster de ongevallen per type, de puntenscore en de gemiddelde plaats.

    De score is de puntenscore (5 / 3 / 1 per ongeval); de kleur op de kaart volgt die score.
    Geeft (ongevallen die in een cluster zitten, tabel met één rij per cluster)."""
    leden = punten.assign(cluster=cluster, punt=punten_per_ongeval(punten))
    leden = leden[leden["cluster"] >= 0]
    code = leden["class_accident_code"].astype(int)
    leden = leden.assign(licht=code.isin(LICHT), zwaar=code.isin(ZWAAR), doden=code.isin(DODEN))
    spots = leden.groupby("cluster").agg(
        totaal=("cluster", "size"),
        score=("punt", "sum"),
        licht=("licht", "sum"),
        zwaar=("zwaar", "sum"),
        doden=("doden", "sum"),
        lat=("lat", "mean"),
        lon=("lon", "mean"),
        gemeente=("gemeente", lambda s: s.value_counts().index[0]),
        provincie=("provincie", lambda s: s.value_counts().index[0]),
        gewest=("gewest", lambda s: s.value_counts().index[0]),
    )
    return leden, spots


def zoek_spots(p, zoek=zoek_clusters):
    """Zwarte punten (vaste instelling) over één reeks ongevallen `p`, gezocht in heel België.

    `p` moet al gefilterd zijn (periode, bruikbare coördinaten, zie `bruikbare_ongevallen`) en een nieuwe index hebben.
    `zoek` is de functie die de clusters zoekt: de app geeft hier een versie met cache mee.
    Geeft (ongevallen in een zwart punt, tabel met één rij per zwart punt)."""
    cluster = zoek(
        p["x_lambert72"].to_numpy(), p["y_lambert72"].to_numpy(), punten_per_ongeval(p), AFSTAND, SCORE_DREMPEL
    )
    return bouw_spots(p, cluster)


def in_gebied(spots, gewest, provincie, gemeente):
    """Houdt enkel de zwarte punten over in het gekozen gewest, de gekozen provincie en gemeente.

    Een punt hoort bij het gebied waar de meeste van zijn ongevallen liggen. Een filter dat niet gekozen is, is `None`."""
    if gewest is not None:
        spots = spots[spots["gewest"] == gewest]
    if provincie is not None:
        spots = spots[spots["provincie"] == provincie]
    if gemeente is not None:
        spots = spots[spots["gemeente"] == gemeente]
    return spots


# ---------- 5. Evolutie: elke periode van 3 jaar ----------


def periodes_van(jaar_min, jaar_max):
    """[(eindjaar, beginjaar), ...]: 2017-2019, 2018-2020, ..., tot het laatste jaar."""
    return [(eind, eind - PERIODE_JAREN + 1) for eind in range(jaar_min + PERIODE_JAREN - 1, jaar_max + 1)]


def spots_per_periode(df, periodes, zoek=zoek_clusters):
    """Zoekt de zwarte punten in heel België voor elke periode (zelfde instelling als op de kaart).

    Geeft per periode {"spots": tabel, "met_coord": aantal gebruikte ongevallen, "alle": alle ongevallen van die periode}."""
    resultaat = []
    for eind, begin in periodes:
        in_periode = df["jaar"].between(begin, eind)
        p = bruikbare_ongevallen(df[in_periode]).reset_index(drop=True)
        resultaat.append({"spots": zoek_spots(p, zoek)[1], "met_coord": len(p), "alle": int(in_periode.sum())})
    return resultaat


def aantal_per_periode(per_periode, kolom=None, waarde=None):
    """Aantal zwarte punten per periode, voor heel België of voor één gewest of provincie (`kolom` == `waarde`)."""
    return [len(x["spots"]) if kolom is None else int((x["spots"][kolom] == waarde).sum()) for x in per_periode]


def dekking_per_periode(df, periodes, gewest=None):
    """% ongevallen met bruikbare coördinaten (zie is_bruikbaar) per periode: eerst per jaar, dan het gemiddelde van de
    jaren (elk jaar telt even zwaar mee). Met `gewest` enkel de ongevallen van dat gewest (bv. "Vlaams Gewest")."""
    if gewest is not None:
        df = df[df["gewest"] == gewest]
    procent_per_jaar = is_bruikbaar(df).groupby(df["jaar"]).mean() * 100
    return [procent_per_jaar.loc[begin:eind].mean() for eind, begin in periodes]
