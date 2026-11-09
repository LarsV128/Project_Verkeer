"""Streamlit-app om verkeersongevallen (2017-2024) te bekijken.

Starten:  streamlit run app.py
"""
import heapq
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import streamlit as st
from pyproj import Transformer
from scipy.spatial import cKDTree

DATA = Path(__file__).parent / "data" / "OPENDATA_MAP_2017-2024_clean.parquet"

# class_accident_code -> groep voor de scores en zwarte punten
LICHT, ZWAAR, DODEN = [4], [3], [1, 2]

st.set_page_config(page_title="Zwarte Punten", layout="wide", initial_sidebar_state="collapsed")
st.title("Zwarte Punten")


@st.cache_data
def laad_data(gewijzigd):
    """`gewijzigd` is enkel een sleutel: verandert het bestand, dan wordt opnieuw ingelezen."""
    return pd.read_parquet(DATA)


@st.cache_data(show_spinner="Zwarte punten zoeken...")
def zoek_clusters(x, y, gewicht, straal, drempel):
    """Geeft per ongeval een clusternummer (-1 = geen zwart punt).

    Voor elk ongeval tellen we het gewicht van alle ongevallen binnen `straal` meter (met gewicht 1
    voor iedereen is dat gewoon het aantal). Het ongeval met de hoogste som vormt een cluster met
    die buren; die worden weggenomen en we herhalen dat zolang een som minstens `drempel` is.
    Zo overlappen clusters niet.
    """
    xy = np.column_stack([x, y])
    buren = cKDTree(xy).query_ball_point(xy, straal)  # inclusief het ongeval zelf
    tel = np.array([gewicht[b].sum() for b in buren], dtype=float)
    weg = np.zeros(len(xy), dtype=bool)
    cluster = np.full(len(xy), -1)

    drempel -= 1e-9  # gewichten als 1,7 of 5,1 geven afrondingsfoutjes: 15 mag niet als 14,999999999 tellen
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


import clean_data

# Opgeschoonde data ontbreekt of is ouder dan de ruwe data of het cleaningscript? Dan eerst opnieuw maken.
if not DATA.exists() or DATA.stat().st_mtime < max(
    clean_data.RAW.stat().st_mtime if clean_data.RAW.exists() else 0,
    Path(clean_data.__file__).stat().st_mtime,
):
    if not clean_data.RAW.exists():
        st.error(f"Ruwe data niet gevonden. Plaats OPENDATA_MAP_2017-2024.txt in `{clean_data.RAW.parent}`.")
        st.stop()
    with st.spinner("Data opschonen uit data/raw (enkele minuten, enkel de eerste keer)..."):
        clean_data.main()

df = laad_data(DATA.stat().st_mtime)

# volgorde van de code: 1 = ernstigst
TYPES = list(
    df[["class_accident_code", "class_accident"]].drop_duplicates().sort_values("class_accident_code")["class_accident"]
)
JAAR_MIN, JAAR_MAX = int(df["jaar"].min()), int(df["jaar"].max())


def punten_instellingen():
    """Punten per type, ingesteld via het tandwiel in het tabblad Zwarte punten.

    Standaard 1 / 3 / 5 zoals bij de Vlaamse telling voor gevaarlijke punten, maar per ONGEVAL (volgens het zwaarste
    type) en niet per slachtoffer: de bron heeft geen slachtoffers."""
    s = st.session_state
    return SimpleNamespace(licht=s.get("p_licht", 1), zwaar=s.get("p_zwaar", 3), doden=s.get("p_doden", 5))


def punten_per_ongeval(p):
    """Score van elk ongeval: de punten van zijn type."""
    inst = punten_instellingen()
    code = p["class_accident_code"].astype(int)
    return np.select([code.isin(DODEN), code.isin(ZWAAR)], [inst.doden, inst.zwaar], default=inst.licht).astype(float)


PERIODE_JAREN = 3  # Vlaanderen rekent gevaarlijke punten op 3 jaar ongevallendata


def schuif_periode(key, stap):
    """Knop-actie: de periode van 3 jaar één jaar opschuiven (binnen de jaren die we hebben)."""
    st.session_state[key] = min(max(st.session_state[key] + stap, JAAR_MIN), JAAR_MAX - PERIODE_JAREN + 1)


def toon_filters(prefix, periode_pijltjes=False, types_filter=True):
    """Filters bovenaan een tabblad: één gewest, één provincie en/of één gemeente, jaren en types.

    Met `periode_pijltjes` kies je geen vrije jaren maar een periode van 3 jaar (2017-2019, 2018-2020, ...) die je
    met pijltjes opschuift; standaard de laatste 3 jaar.
    Zonder `types_filter` is er geen keuzelijst voor het type ongeval: alle types tellen mee (via de punten).
    Geeft een object met de gekozen waarden en `sel`, de ongevallen die aan alles voldoen."""
    k1, k2, k3, k4 = st.columns(4)
    gewest = k1.selectbox("Gewest", ["Alle gewesten"] + sorted(df["gewest"].unique()), key=f"{prefix}_gewest")
    sel = df if gewest == "Alle gewesten" else df[df["gewest"] == gewest]

    provincie = k2.selectbox("Provincie", ["Alle provincies"] + sorted(sel["provincie"].unique()), key=f"{prefix}_prov")
    sel = sel if provincie == "Alle provincies" else sel[sel["provincie"] == provincie]

    gemeente = k3.selectbox("Gemeente", ["Alle gemeenten"] + sorted(sel["gemeente"].unique()), key=f"{prefix}_gem")
    sel = sel if gemeente == "Alle gemeenten" else sel[sel["gemeente"] == gemeente]

    if periode_pijltjes:
        sleutel = f"{prefix}_start"
        st.session_state.setdefault(sleutel, JAAR_MAX - PERIODE_JAREN + 1)
        start = st.session_state[sleutel]
        k4.markdown("Periode")
        links, midden, rechts = k4.columns([1, 2, 1])
        links.button("◀", key=f"{prefix}_vorige", on_click=schuif_periode, args=(sleutel, -1), disabled=start <= JAAR_MIN)
        midden.markdown(f"**{start}-{start + PERIODE_JAREN - 1}**")
        rechts.button(
            "▶", key=f"{prefix}_volgende", on_click=schuif_periode, args=(sleutel, 1),
            disabled=start >= JAAR_MAX - PERIODE_JAREN + 1,
        )
        jaren = (start, start + PERIODE_JAREN - 1)
    else:
        jaren = k4.slider("Jaren", JAAR_MIN, JAAR_MAX, (JAAR_MIN, JAAR_MAX), key=f"{prefix}_jaren")
    if types_filter:
        types = st.multiselect("Type ongeval (class_accident)", TYPES, default=TYPES, key=f"{prefix}_types")
    else:
        types = TYPES

    sel = sel[sel["jaar"].between(*jaren) & sel["class_accident"].isin(types)]
    return SimpleNamespace(sel=sel, gewest=gewest, provincie=provincie, gemeente=gemeente, jaren=jaren, types=types)


def nl(n):
    """Getal met punt als duizendtalscheiding: 12345 -> '12.345'."""
    return f"{int(n):,}".replace(",", ".")


def plaats_label(filt):
    """Het kleinste gebied dat in de filters gekozen is: gemeente, provincie, gewest of heel België."""
    if filt.gemeente != "Alle gemeenten":
        return filt.gemeente
    if filt.provincie != "Alle provincies":
        return f"provincie {filt.provincie}"
    if filt.gewest != "Alle gewesten":
        return filt.gewest
    return "België"


def toon_dekking(weergegeven, totaal, plaats, uitleg=""):
    """Toont de verhouding weergegeven / totaal geregistreerde ongevallen onder een tabel, kaart of grafiek.

    `totaal` = alle geregistreerde ongevallen in `plaats` (binnen de gekozen jaren en types); `weergegeven` = hoeveel
    daarvan in de tabel of op de kaart zitten. Ongevallen zonder bruikbare coördinaten verwerken we niet op de kaart."""
    pct = f"{weergegeven / totaal * 100:.1f}".replace(".", ",") if totaal else "–"
    st.caption(f"Weergegeven / totaal geregistreerd in {plaats}: **{nl(weergegeven)} / {nl(totaal)} ({pct} %)**. {uitleg}")


def toon_type_overzicht(sel, plaats):
    """Aantal ongevallen per type (class_accident) voor de selectie."""
    st.subheader("Overzicht per type ongeval")
    aantal = sel["class_accident"].value_counts().reindex(TYPES, fill_value=0)
    tabel = pd.DataFrame({"type": TYPES, "aantal": aantal.to_numpy()})
    totaal = int(tabel["aantal"].sum())
    tabel["aandeel (%)"] = (tabel["aantal"] / totaal * 100).round(1) if totaal else 0.0
    tabel.loc[len(tabel)] = ["Totaal", totaal, 100.0 if totaal else 0.0]
    st.dataframe(tabel, width="stretch", hide_index=True)
    toon_dekking(len(sel), len(sel), plaats, "Alle ongevallen staan in deze tabel, ook die zonder coördinaten.")


def gemeente_ranglijst(jaren, types):
    """Alle gemeenten van België, gerangschikt op puntenscore (positie 1 = hoogste score)."""
    basis = df[df["jaar"].between(*jaren) & df["class_accident"].isin(types)]
    code = basis["class_accident_code"].astype(int)
    basis = basis.assign(
        score=punten_per_ongeval(basis), licht=code.isin(LICHT), zwaar=code.isin(ZWAAR), doden=code.isin(DODEN)
    )
    g = (
        basis.groupby(["gemeente", "provincie", "gewest"], observed=True)
        .agg(score=("score", "sum"), totaal=("score", "size"), licht=("licht", "sum"), zwaar=("zwaar", "sum"), doden=("doden", "sum"))
        .reset_index()
        .assign(score=lambda t: t["score"].round(1))
        .sort_values(["score", "totaal", "gemeente"], ascending=[False, False, True])
        .reset_index(drop=True)
    )
    g.insert(0, "positie", g.index + 1)
    return g


def toon_ranglijst(filt):
    """Ranglijst van gemeenten, met nummering over heel België.

    Gewest of provincie gekozen: alle gemeenten daarin. Gemeente gekozen: de hoogste en laagste
    scores en de gemeenten rondom (max. 5 erboven en 5 eronder)."""
    g = gemeente_ranglijst(filt.jaren, filt.types)
    inst = punten_instellingen()
    st.subheader("Gemeenten gerangschikt op puntenscore")
    st.caption(
        f"Score = {inst.licht} × licht + {inst.zwaar} × zwaar + {inst.doden} × doden, per ongeval (niet per slachtoffer; "
        f"instellen via ⚙️ in het tabblad Zwarte punten). Nummering over alle {len(g)} gemeenten in België; "
        "positie 1 heeft de hoogste score."
    )

    def toon(tabel, titel):
        st.markdown(f"**{titel}**")
        if filt.gemeente != "Alle gemeenten":
            stijl = tabel.style.apply(
                lambda r: ["background-color: rgba(255, 190, 0, 0.35)" if r["gemeente"] == filt.gemeente else "" for _ in r],
                axis=1,
            )
            st.dataframe(stijl, width="stretch", hide_index=True)
        else:
            st.dataframe(tabel, width="stretch", hide_index=True)

    if filt.gemeente != "Alle gemeenten":
        rij = g.index[g["gemeente"] == filt.gemeente]
        if len(rij) == 0:
            st.info(f"{filt.gemeente} heeft geen ongevallen met deze jaren en types, dus geen positie.")
            return
        i = int(rij[0])
        st.metric(f"Positie van {filt.gemeente}", f"{i + 1} van {len(g)}")
        getoond = sorted(set(range(5)) | set(range(max(i - 5, 0), i + 6)) | set(range(max(len(g) - 5, 0), len(g))))
        toon_dekking(
            g.loc[[x for x in getoond if x < len(g)], "totaal"].sum(),
            g["totaal"].sum(),
            "België",
            "De ranglijst gebruikt alle ongevallen, ook die zonder coördinaten; hier zie je enkel de gemeenten uit de drie tabellen.",
        )
        toon(g.head(5), "Hoogste scores (top 5)")
        toon(g.iloc[max(i - 5, 0) : i + 6], f"Rondom {filt.gemeente} (5 erboven, 5 eronder)")
        toon(g.tail(5), "Laagste scores (onderste 5)")
        return

    if filt.provincie != "Alle provincies":
        g, titel = g[g["provincie"] == filt.provincie], f"Alle gemeenten in provincie {filt.provincie}"
    elif filt.gewest != "Alle gewesten":
        g, titel = g[g["gewest"] == filt.gewest], f"Alle gemeenten in {filt.gewest}"
    else:
        titel = "Alle gemeenten in België"
    toon_dekking(
        g["totaal"].sum(),
        g["totaal"].sum(),
        plaats_label(filt),
        "Alle ongevallen van deze gemeenten tellen mee, ook die zonder coördinaten.",
    )
    toon(g, titel)


tab_punten, tab_data = st.tabs(["Kaart", "Data View"])

# ---------- Tab Data View (met onderaan de foute coördinaten, zie verderop) ----------
with tab_data:
    f1 = toon_filters("overzicht")
    sel = f1.sel
    plaats1 = plaats_label(f1)
    if sel.empty:
        st.warning("Geen ongevallen voor deze selectie. Pas de filters aan.")
    else:
        with st.expander("🔎 Data viewer: alle ongevallen van de selectie"):
            st.caption(
                "Klik op een kolomkop om te sorteren, gebruik het zoekicoon rechtsboven de tabel om te zoeken en het "
                "pijltje om de tabel schermvullend te zien."
            )
            kolommen_kiezen = st.multiselect("Kolommen", list(sel.columns), default=list(sel.columns), key="viewer_kolommen")
            st.dataframe(
                sel[kolommen_kiezen],
                width="stretch",
                height=450,
                hide_index=True,
                column_config={c: st.column_config.NumberColumn(format="%.1f") for c in ("x_lambert72", "y_lambert72")}
                | {c: st.column_config.NumberColumn(format="%.5e") for c in ("x_bron", "y_bron")},
            )
            toon_dekking(len(sel), len(sel), plaats1, "Alle rijen van de selectie staan in deze tabel.")

        st.metric("Aantal ongevallen in selectie", nl(len(sel)))
        toon_type_overzicht(sel, plaats1)

        st.subheader("Per jaar, per type")
        st.bar_chart(sel.groupby(["jaar", "class_accident"], observed=True).size().unstack(fill_value=0))
        toon_dekking(len(sel), len(sel), plaats1, "Alle ongevallen staan in deze grafiek, ook die zonder coördinaten.")

    toon_ranglijst(f1)

# ---------- Tab 2: zwarte punten ----------
MODUS_VAST = "Vaste locaties, aantallen per jaar"
MODUS_JAAR = "Elk jaar apart"
METHODE_AANTAL = "Aantal ongevallen"
METHODE_SCORE = "Puntenscore"


def bouw_spots(punten, cluster):
    """Telt per cluster de ongevallen per type, de puntenscore en de gemiddelde plaats.

    De score is altijd de puntenscore (5 / 3 / 1 per ongeval), ook als de zwarte punten op aantal gezocht zijn,
    zodat de kleur op de kaart altijd hetzelfde betekent.
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
    spots["score"] = spots["score"].round(1)
    return leden, spots


CIRKEL_STRAAL = 60  # meter; elk zwart punt is even groot, de cirkel duidt enkel de locatie aan

# Vaste kleurschaal voor de score van een zwart punt, overal dezelfde (elk tabblad, weergave, periode en gebied):
# (ondergrens, kleur, label). De grenzen volgen de verdeling van de scores: de meeste punten liggen tussen 15 en 25.
KLEURKLASSEN = [
    (0, "#2c7bb6", "minder dan 20"),
    (20, "#abd9e9", "20 tot 24"),
    (25, "#ffdf6e", "25 tot 29"),
    (30, "#fc8d59", "30 tot 49"),
    (50, "#d73027", "50 of meer"),
]


def score_kleur(score):
    """Hexkleur van een score volgens de vaste klassen in KLEURKLASSEN."""
    kleur = KLEURKLASSEN[0][1]
    for ondergrens, k, _ in KLEURKLASSEN:
        if score >= ondergrens:
            kleur = k
    return kleur


def toon_legende():
    """Legende onder de kaart: dezelfde klassen en kleuren als de cirkels."""
    delen = [f"<span style='color:{k}; font-size:1.2em'>●</span> {label}" for _, k, label in KLEURKLASSEN]
    st.markdown(
        "<small><b>Score van het zwarte punt (punten):</b> " + " &nbsp;·&nbsp; ".join(delen) + "</small>",
        unsafe_allow_html=True,
    )


def in_gebied(spots, filt):
    """Houdt enkel de zwarte punten over in het gekozen gewest, de gekozen provincie en gemeente.

    Een punt hoort bij het gebied waar de meeste van zijn ongevallen liggen."""
    if filt.gewest != "Alle gewesten":
        spots = spots[spots["gewest"] == filt.gewest]
    if filt.provincie != "Alle provincies":
        spots = spots[spots["provincie"] == filt.provincie]
    if filt.gemeente != "Alle gemeenten":
        spots = spots[spots["gemeente"] == filt.gemeente]
    return spots


def toon_telling(spots, alle_jaren, per_jaar_modus):
    """Tabellen met het aantal zwarte punten in de gekozen periode: per gewest en per provincie, elk met een rij
    Totaal (= heel België, dat zo impliciet vermeld wordt).

    Telt altijd heel België, ook als je hieronder op een gebied filtert. Bij 'Elk jaar apart' één kolom per jaar."""
    if spots.empty:
        spots = pd.DataFrame({"gewest": [], "provincie": [], "jaar": []})

    def tel(deel):
        if per_jaar_modus:
            per_jaar = deel.groupby("jaar").size().reindex(alle_jaren, fill_value=0)
            return {str(j): int(per_jaar[j]) for j in alle_jaren}
        return {"Zwarte punten": len(deel)}

    groepen = [
        ("Gewesten", [(g, spots[spots["gewest"] == g]) for g in sorted(df["gewest"].unique())]),
        ("Provincies", [(p, spots[spots["provincie"] == p]) for p in sorted(df["provincie"].unique())]),
    ]

    def vet_totaal(rij):
        """Opmaak: de rij 'Totaal' (= heel België) in vet, met een lijn erboven."""
        if rij["Gebied"] == "Totaal":
            return ["font-weight: bold; border-top: 2px solid currentColor"] * len(rij)
        return [""] * len(rij)

    # Twee afgebakende blokken naast elkaar, elk met een eigen kader en titel; onderaan de rij Totaal (= België)
    for kolom, (titel, delen) in zip(st.columns([1, 1.3]), groepen):
        with kolom.container(border=True):
            st.markdown(f"**{titel}**")
            rijen = [{"Gebied": naam, **tel(deel)} for naam, deel in delen] + [{"Gebied": "Totaal", **tel(spots)}]
            st.table(pd.DataFrame(rijen).style.hide(axis="index").apply(vet_totaal, axis=1))


def toon_kaart_en_tabel(spots, kolommen, dek, onder_kaart=None):
    """`dek` = (weergegeven, totaal, plaats): ongevallen met coördinaten, alle ongevallen, en het gebied.
    `onder_kaart` is een functie die direct onder de kaart iets toont (hier: de telling per gebied).

    Alle cirkels zijn even groot (ze duiden de locatie aan); de kleur toont de score volgens vaste klassen."""
    kaart = spots[["lat", "lon"]].assign(kleur=spots["score"].map(score_kleur))
    st.map(kaart, latitude="lat", longitude="lon", color="kleur", size=CIRKEL_STRAAL)
    toon_legende()
    toon_dekking(*dek, "De kaart gebruikt enkel ongevallen met bruikbare coördinaten.")
    if onder_kaart:
        onder_kaart()
    st.dataframe(spots.reset_index(drop=True)[kolommen], width="stretch", hide_index=True)
    toon_dekking(*dek, "De zwarte punten zijn gezocht onder ongevallen met bruikbare coördinaten.")


with tab_punten:
    f2 = toon_filters("punten", periode_pijltjes=True, types_filter=False)
    sel, jaren = f2.sel, f2.jaren

    # Alle keuzes zitten in dit ene menu; bovenaan het tabblad staan enkel de filters.
    with st.popover("⚙️ Instellingen"):
        st.markdown("**Weergave en zoeken**")
        modus = st.radio("Weergave", [MODUS_VAST, MODUS_JAAR], horizontal=True)
        methode = st.radio("Wanneer is het een zwart punt?", [METHODE_AANTAL, METHODE_SCORE], index=1, horizontal=True)
        per_jaar_modus = modus == MODUS_JAAR
        score_modus = methode == METHODE_SCORE
        eenheid = "per jaar" if per_jaar_modus else "totaal"
        if per_jaar_modus:
            st.caption(
                "Elk jaar wordt apart onderzocht. Een plek kan in het ene jaar een zwart punt zijn en in het andere niet; "
                "punten van verschillende jaren liggen niet op exact dezelfde plek."
            )
        else:
            st.caption(
                "Eén keer zoeken over alle gekozen jaren samen. Elke plek blijft dezelfde locatie, zodat je ze per jaar "
                "kan vergelijken."
            )
        straal = st.select_slider(
            "Afstand (m)",
            options=[25, 50, 75, 100, 150, 200],
            value=50,
            help="Ongevallen binnen deze afstand worden samengenomen tot één zwart punt (op het gemiddelde van hun plaats).",
        )
        min_totaal = st.number_input(
            f"Min. ongevallen ({eenheid})",
            min_value=2,
            value=3,
            step=1,
            key="min_jaar" if per_jaar_modus else "min_vast",
            help="Bij 'Aantal ongevallen': de drempel zelf. Bij 'Puntenscore': een extra voorwaarde naast de score "
            "(Vlaanderen: minstens 3 ongevallen).",
        )

        st.divider()
        st.markdown("**Puntenscore**")
        st.caption(
            "Elk ongeval telt mee met de punten van zijn type. Gebruikt bij de methode 'Puntenscore' en voor de "
            "ranglijst van gemeenten in het tabblad Data View."
        )
        st.number_input("Punten per ongeval met lichtgewonden", min_value=0, value=1, step=1, key="p_licht")
        st.number_input("Punten per ongeval met zwaargewonden", min_value=0, value=3, step=1, key="p_zwaar")
        st.number_input("Punten per ongeval met doden (incl. dodelijk gewonden)", min_value=0, value=5, step=1, key="p_doden")
        score_drempel = st.number_input(f"Zwart punt vanaf (punten, {eenheid})", min_value=1, value=15, step=1)

        st.divider()
        st.markdown("**Extra voorwaarden** (voor beide methodes)")
        st.caption("Ze worden toegepast nadat de zwarte punten gevonden zijn.")
        enkel_licht_telt_niet = st.checkbox(
            "Een plek met enkel lichtgewonden telt niet mee",
            value=False,
            help="Een zwart punt heeft dan minstens 1 ongeval met zwaargewonden of doden.",
        )
        min_licht = st.number_input(f"Min. met lichtgewonden ({eenheid})", min_value=0, value=0, step=1)
        min_zwaar = st.number_input(f"Min. met zwaargewonden ({eenheid})", min_value=0, value=0, step=1)
        min_doden = st.number_input(f"Min. met doden, incl. dodelijk gewonden ({eenheid})", min_value=0, value=0, step=1)
        ernst_combi = st.radio(
            "Minimum zwaargewonden en doden:",
            ["en", "of"],
            horizontal=True,
            help="'en': beide minimums moeten gehaald worden. 'of': één van beide volstaat. "
            "Een minimum van 0 telt niet mee.",
        )

    drempel =score_drempel if score_modus else min_totaal
    maat = "score" if score_modus else "totaal"  # waarop de tabellen gesorteerd worden
    sorteer = [maat, "doden", "zwaar"]

    def gewichten(p):
        """Gewicht per ongeval: 1 bij de methode 'Aantal ongevallen', anders de punten van het type."""
        return punten_per_ongeval(p) if score_modus else np.ones(len(p))

    def voldoet(spots):
        """Past de extra voorwaarden toe op de gevonden zwarte punten."""
        ok = spots["licht"] >= min_licht
        if score_modus:  # bij 'Aantal ongevallen' is dit minimum al de zoekdrempel zelf
            ok = ok & (spots["totaal"] >= min_totaal)
        if enkel_licht_telt_niet:
            ok = ok & ((spots["zwaar"] + spots["doden"]) >= 1)
        ernst = []
        if min_zwaar > 0:
            ernst.append((spots["zwaar"] >= min_zwaar).to_numpy())
        if min_doden > 0:
            ernst.append((spots["doden"] >= min_doden).to_numpy())
        if ernst:
            ok = ok & (np.logical_or.reduce(ernst) if ernst_combi == "of" else np.logical_and.reduce(ernst))
        return spots[ok]

    basis_kolommen = ["gemeente", "score", "totaal", "licht", "zwaar", "doden"]

    plaats2 = plaats_label(f2)
    alle_jaren = list(range(jaren[0], jaren[1] + 1))
    bruikbaar = ["ok", "hersteld"]

    # Zwarte punten zoeken in heel België (alle types ongevallen, via de punten); de filters hierboven bepalen
    # daarna enkel welke punten je op de kaart en in de lijst ziet. Zo klopt de tabel hieronder altijd voor België.
    punten_nl = df[df["jaar"].between(*jaren) & df["coord_status"].isin(bruikbaar)].reset_index(drop=True)
    punten = sel[sel["coord_status"].isin(bruikbaar)]  # enkel het gekozen gebied; voor de verhouding weergegeven/totaal
    dek = (len(punten), len(sel), plaats2)

    if not per_jaar_modus:
        w = gewichten(punten_nl)
        cluster = zoek_clusters(punten_nl["x_lambert72"].to_numpy(), punten_nl["y_lambert72"].to_numpy(), w, straal, drempel)
        leden, spots_nl = bouw_spots(punten_nl, cluster)
        per_jaar = (
            leden.groupby(["cluster", "jaar"]).size().unstack(fill_value=0).reindex(columns=alle_jaren, fill_value=0)
        )
        per_jaar.columns = [str(j) for j in per_jaar.columns]
        spots_nl = voldoet(spots_nl).join(per_jaar)
    else:
        delen = []
        for jaar in alle_jaren:
            pj = punten_nl[punten_nl["jaar"] == jaar].reset_index(drop=True)
            if pj.empty:
                continue
            w = gewichten(pj)
            cluster = zoek_clusters(pj["x_lambert72"].to_numpy(), pj["y_lambert72"].to_numpy(), w, straal, drempel)
            _, spots_jaar = bouw_spots(pj, cluster)
            delen.append(voldoet(spots_jaar).assign(jaar=jaar))
        spots_nl = pd.concat(delen, ignore_index=True) if delen else pd.DataFrame()

    def telling_onder_kaart():
        """De tabel met het aantal zwarte punten (altijd heel België) onder de kaart."""
        st.subheader(f"Aantal zwarte punten, periode {jaren[0]}-{jaren[1]}")
        toon_telling(spots_nl, alle_jaren, per_jaar_modus)
        toon_dekking(
            len(punten_nl),
            int(df["jaar"].between(*jaren).sum()),
            "België",
            "Zwarte punten zijn gezocht onder ongevallen met bruikbare coördinaten; deze tabel telt heel België.",
        )

    spots = in_gebied(spots_nl, f2)
    if not per_jaar_modus:
        spots = spots.sort_values(sorteer, ascending=False)

    if not per_jaar_modus:
        if spots.empty:
            st.info("Geen zwarte punten met deze drempels. Verlaag een drempel of vergroot de afstand.")
            telling_onder_kaart()
        else:
            toon_kaart_en_tabel(
                spots, basis_kolommen + list(per_jaar.columns) + ["lat", "lon"], dek, onder_kaart=telling_onder_kaart
            )

    else:
        if spots.empty:
            st.info("Geen zwarte punten met deze drempels. Verlaag een drempel of vergroot de afstand.")
            telling_onder_kaart()
        else:
            samenvatting = (
                spots.groupby("jaar")
                .agg(zwarte_punten=("totaal", "size"), ongevallen_in_punten=("totaal", "sum"), doden=("doden", "sum"))
                .reindex(alle_jaren, fill_value=0)
            )
            st.subheader("Aantal zwarte punten per jaar")
            st.bar_chart(samenvatting["zwarte_punten"])
            st.dataframe(samenvatting, width="stretch")
            toon_dekking(*dek, "Let op: de dekking verschilt per jaar (zie aanames.txt 3.4), dit is het totaal over de gekozen jaren.")

            jaar_keuze = st.select_slider("Toon de zwarte punten van", options=alle_jaren, value=alle_jaren[-1])
            van_jaar = spots[spots["jaar"] == jaar_keuze].sort_values(sorteer, ascending=False)
            if van_jaar.empty:
                st.info(f"Geen zwarte punten in {jaar_keuze}.")
                telling_onder_kaart()
            else:
                dek_jaar = (int((punten["jaar"] == jaar_keuze).sum()), int((sel["jaar"] == jaar_keuze).sum()), plaats2)
                toon_kaart_en_tabel(van_jaar, basis_kolommen + ["lat", "lon"], dek_jaar, onder_kaart=telling_onder_kaart)

    # Uitleg en gekozen instellingen staan onder de kaart, zodat boven de kaart enkel de tabel staat.
    with st.expander("ℹ️ Uitleg en gekozen instellingen"):
        st.markdown(
            "Een **zwart punt** is hier een plek waar binnen de gekozen afstand genoeg (ernstige) ongevallen gebeurden. "
            "We lijken op de puntentelling van Vlaanderen voor "
            "[gevaarlijke punten](https://www.vlaanderen.be/mobiliteitsprofessionals/verkeersveiligheid/gevaarlijke-punten) "
            "(3 jaar, score van minstens 15, minstens 3 ongevallen), maar tellen **per ongeval** en niet per slachtoffer: "
            "de bron bevat geen slachtoffers. Alle types ongevallen tellen mee via hun punten. "
            "*De afstand dient enkel om ongevallen tot één punt samen te voegen.*"
        )
        inst = punten_instellingen()
        st.caption(f"Weergave: {modus} · afstand {straal} m · methode: {methode} (aanpassen via ⚙️ Instellingen).")
        if score_modus:
            st.caption(
                f"Score = {inst.licht} × licht + {inst.zwaar} × zwaar + {inst.doden} × doden, per ongeval (niet per "
                f"slachtoffer). Zwart punt vanaf **{score_drempel} punten** en minstens **{min_totaal} ongevallen** ({eenheid})."
            )
        voorwaarden = []
        if enkel_licht_telt_niet:
            voorwaarden.append("minstens 1 ongeval met zwaargewonden of doden")
        if min_licht:
            voorwaarden.append(f"min. {min_licht} met lichtgewonden")
        ernst_delen = ([f"min. {min_zwaar} met zwaargewonden"] if min_zwaar else []) + (
            [f"min. {min_doden} met doden"] if min_doden else []
        )
        if ernst_delen:
            voorwaarden.append(f" {ernst_combi} ".join(ernst_delen))
        if voorwaarden:
            st.caption(f"Extra voorwaarden ({eenheid}): " + "; ".join(voorwaarden) + ".")
        st.caption(
            "De filters gewest, provincie en gemeente bepalen welke zwarte punten op de kaart en in de lijst staan; een "
            "punt hoort bij het gebied waar de meeste van zijn ongevallen liggen. De tabel bovenaan telt altijd heel België."
        )

# ---------- Tab Data View, deel 2: foute coördinaten (onderaan hetzelfde tabblad) ----------
with tab_data:
    st.divider()
    st.header("Foute coördinaten")
    st.markdown(
        "Coördinaten staan in **Lambert72** (meters). Staat een waarde buiten België, dan is ze onmogelijk; "
        "lege waarden ontbreken gewoon. De filters bovenaan dit tabblad gelden voor de tabel per type; de lijst en de "
        "kaarten met herstelde coördinaten daaronder tonen altijd alle ongevallen."
    )

    st.subheader("Status van de coördinaten per type ongeval")
    status = pd.crosstab(f1.sel["class_accident"], f1.sel["coord_status"]).reindex(TYPES, fill_value=0)
    status = status.reindex(columns=["ok", "hersteld", "ontbreekt", "onmogelijk"], fill_value=0)
    status.loc["Totaal"] = status.sum()
    status["% bruikbaar"] = ((status["ok"] + status["hersteld"]) / status.sum(axis=1).replace(0, np.nan) * 100).round(1)
    status = status.rename(columns={"ok": "bruikbaar"}).reset_index().rename(columns={"class_accident": "type"})
    st.dataframe(status, width="stretch", hide_index=True)
    toon_dekking(
        int(f1.sel["coord_status"].isin(["ok", "hersteld"]).sum()),
        len(f1.sel),
        plaats1,
        "Dit is de verhouding van ongevallen mét bruikbare coördinaten (kolommen bruikbaar + hersteld) tot alle ongevallen "
        "in de tabel, zoals ook in de kolom % bruikbaar.",
    )

    fout = df[df["coord_status"].isin(["hersteld", "onmogelijk"])]
    st.subheader("Onmogelijke coördinaten in de bron (hersteld)")
    st.dataframe(
        fout.drop(columns=["lat", "lon"]),
        width="stretch",
        hide_index=True,
        column_config={
            c: st.column_config.NumberColumn(format="%.1f") for c in ("x_lambert72", "y_lambert72")
        }
        | {c: st.column_config.NumberColumn(format="%.5e") for c in ("x_bron", "y_bron")},
    )
    toon_dekking(len(fout), len(fout), "België", "Alle ongevallen met een onmogelijke coördinaat in de bron staan in deze tabel.")

    st.subheader("Waar staan ze nu?")
    st.markdown(
        "In de bron staan deze waarden in wetenschappelijke notatie (bv. `7,30077E+15`), waarschijnlijk een "
        "afronding in Excel: enkel de eerste cijfers bleven over. In `clean_data.py` zetten we de komma op de plek "
        "waar het punt in België valt én het dichtst bij het midden van de gemeente ligt; de ongevallen krijgen "
        "de status *hersteld* en de originele waarden staan in `x_bron` en `y_bron`. "
        "**Dit blijft een vermoeden, maar beide punten liggen volgens jouw controle op een weg.**  \n"
        "🔴 gebruikte plaats · 🔵 midden van de gemeente (mediaan van de andere ongevallen) · "
        "⚪ andere ongevallen in die gemeente"
    )

    geldig = df[df["coord_status"] == "ok"]
    naar_gps = Transformer.from_crs(31370, 4326, always_xy=True)
    kolommen = st.columns(max(len(fout), 1))
    for kolom, (_, rij) in zip(kolommen, fout.iterrows()):
        with kolom:
            st.markdown(f"**{rij['gemeente']}** ({rij['jaar']}-{int(rij['maand']):02d})")
            if rij["coord_status"] != "hersteld":
                st.warning("Deze coördinaat kon niet hersteld worden en staat niet op de kaart.")
                continue
            st.caption(f"Bron: X = {rij['x_bron']:.5e}, Y = {rij['y_bron']:.5e}")

            gem = geldig[geldig["nis"] == rij["nis"]]
            mx, my = gem["x_lambert72"].median(), gem["y_lambert72"].median()
            cx, cy = rij["x_lambert72"], rij["y_lambert72"]

            afstand_midden = np.hypot(cx - mx, cy - my)
            afstand_dichtste = np.hypot(gem["x_lambert72"] - cx, gem["y_lambert72"] - cy).min()
            buurt = int((np.hypot(gem["x_lambert72"] - cx, gem["y_lambert72"] - cy) <= 500).sum())
            in_kader = gem["x_lambert72"].min() <= cx <= gem["x_lambert72"].max() and (
                gem["y_lambert72"].min() <= cy <= gem["y_lambert72"].max()
            )

            punten_kaart = pd.concat(
                [
                    gem[["lat", "lon"]].assign(kleur="#bbbbbb", grootte=15),
                    pd.DataFrame(
                        [naar_gps.transform(mx, my)[::-1], naar_gps.transform(cx, cy)[::-1]], columns=["lat", "lon"]
                    ).assign(kleur=["#1f77b4", "#d62728"], grootte=[60, 60]),
                ],
                ignore_index=True,
            )
            st.map(punten_kaart, latitude="lat", longitude="lon", color="kleur", size="grootte")
            van_gemeente = df[df["nis"] == rij["nis"]]
            toon_dekking(
                int(van_gemeente["coord_status"].isin(["ok", "hersteld"]).sum()),
                len(van_gemeente),
                rij["gemeente"],
                "Alle jaren en types van deze gemeente; de kaart toont enkel ongevallen met bruikbare coördinaten.",
            )

            st.write(
                f"Vermoedelijke coördinaat: X = {cx:,.0f}, Y = {cy:,.0f} (Lambert72)\n\n"
                f"- Afstand tot het midden van de gemeente: **{afstand_midden / 1000:.1f} km**\n"
                f"- Afstand tot het dichtstbijzijnde andere ongeval in {rij['gemeente']}: "
                f"**{afstand_dichtste:,.0f} m**\n"
                f"- Ongevallen in {rij['gemeente']} binnen 500 m van dit punt: **{buurt}**\n"
                f"- Binnen het gebied van alle andere ongevallen van de gemeente: **{'ja' if in_kader else 'nee'}**"
            )
            if not (in_kader and afstand_dichtste <= 1000):
                st.warning("Past niet goed. Behandel dit punt als onbekend.")
            elif buurt < 10:
                st.info("Aannemelijk, maar minder zeker: het punt ligt aan de rand van de gemeente, waar weinig ongevallen zijn.")
            else:
                st.success("Sterk: het punt ligt midden tussen de andere ongevallen van deze gemeente.")

# ---------- Bron (onder alle tabbladen) ----------
st.divider()
st.caption(
    "**Bron:** Statbel, verkeersongevallen (open data), 2017-2024: "
    "[OPENDATA_MAP_2017-2024.xlsx](https://statbel.fgov.be/sites/default/files/files/opendata/Accident_XY/OPENDATA_MAP_2017-2024.xlsx). "
    "Opgeschoond en bewerkt in dit project: coördinaten omgerekend naar breedte- en lengtegraad en 2 onmogelijke "
    "waarden hersteld (zie onderaan het tabblad Data View). Ongevallen zonder bruikbare coördinaten staan niet op de kaart."
)
