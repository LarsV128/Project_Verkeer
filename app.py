"""Streamlit-app om verkeersongevallen (2017-2024) te bekijken.

Starten:  streamlit run app.py
"""
import heapq
from pathlib import Path
from types import SimpleNamespace

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from pyproj import Transformer
from scipy.spatial import cKDTree

try:
    import pydeck as pdk  # kaart met hover en klik; staat in .venv en requirements.txt, niet in syntra-ds
except ImportError:
    pdk = None  # zonder pydeck valt de kaart terug op de gewone st.map (zonder hover)

DATA = Path(__file__).parent / "data" / "OPENDATA_MAP_2017-2024_clean.parquet"

# class_accident_code -> groep voor de scores en zwarte punten
LICHT, ZWAAR, DODEN = [4], [3], [1, 2]

# Vaste instelling van een zwart punt (zie aanames.txt onderdeel 5). Wijzig ze nooit stil: leg een wijziging eerst voor.
# Punten per ONGEVAL (niet per slachtoffer: de bron heeft geen slachtoffers), volgens het zwaarste type.
PUNT_LICHT, PUNT_ZWAAR, PUNT_DODEN = 1, 3, 5
AFSTAND = 50  # meter: ongevallen binnen deze afstand worden samengenomen tot één zwart punt
SCORE_DREMPEL = 15  # een zwart punt heeft minstens zoveel punten ...
MIN_TOTAAL = 3  # ... en minstens zoveel ongevallen

st.set_page_config(page_title="Ongevallen", layout="wide", initial_sidebar_state="collapsed")
st.title("Ongevallen")


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

RAW_AANWEZIG = clean_data.RAW.exists()  # op Streamlit Cloud niet: de ruwe data staat niet in git, de parquet wel

# Ruwe data aanwezig, en de opgeschoonde data ontbreekt of is ouder dan de ruwe data of het cleaningscript? Dan opnieuw maken.
# Zonder ruwe data gebruiken we de parquet zoals ze is (bestandstijden na een git clone zeggen niets).
if RAW_AANWEZIG and (
    not DATA.exists()
    or DATA.stat().st_mtime < max(clean_data.RAW.stat().st_mtime, Path(clean_data.__file__).stat().st_mtime)
):
    with st.spinner("Data opschonen uit data/raw (enkele minuten, enkel de eerste keer)..."):
        clean_data.main()
elif not DATA.exists():
    st.error(f"Opgeschoonde data niet gevonden en ook geen ruwe data. Plaats OPENDATA_MAP_2017-2024.txt in `{clean_data.RAW.parent}`.")
    st.stop()

df = laad_data(DATA.stat().st_mtime)

# volgorde van de code: 1 = ernstigst
TYPES = list(
    df[["class_accident_code", "class_accident"]].drop_duplicates().sort_values("class_accident_code")["class_accident"]
)
JAAR_MIN, JAAR_MAX = int(df["jaar"].min()), int(df["jaar"].max())


def punten_per_ongeval(p):
    """Score van elk ongeval: de punten van zijn type (1 / 3 / 5, zoals bij de Vlaamse telling voor gevaarlijke punten)."""
    code = p["class_accident_code"].astype(int)
    return np.select([code.isin(DODEN), code.isin(ZWAAR)], [PUNT_DODEN, PUNT_ZWAAR], default=PUNT_LICHT).astype(float)


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


tab_punten, tab_evolutie = st.tabs(["Kaart", "Evolutie"])


# ---------- Tab Kaart: zwarte punten ----------


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
    spots["score"] = spots["score"].round(1)
    return leden, spots



# Vaste kleurschaal voor de score van een zwart punt, overal dezelfde (elk tabblad, weergave, periode en gebied):
# (ondergrens, kleur, label). De grenzen volgen de verdeling van de scores: de meeste punten liggen tussen 15 en 25.
# Kleuren: de laatste 5 klassen van de Datawrapper-schaal (dof geel naar dof paars), exact gemeten in de afbeelding.
KLEURKLASSEN = [
    (0, "#fbe2af", "minder dan 20"),  # dof geel
    (20, "#fdac7a", "20 tot 24"),  # dof oranje
    (25, "#f6736b", "25 tot 29"),  # zalm
    (30, "#da3c6a", "30 tot 49"),  # karmozijn
    (50, "#a90674", "50 of meer"),  # dof paars
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


def toon_telling(spots):
    """Tabellen met het aantal zwarte punten in de gekozen periode: per gewest en per provincie, elk met een rij
    Totaal (= heel België, dat zo impliciet vermeld wordt).

    Telt altijd heel België, ook als je hieronder op een gebied filtert."""
    if spots.empty:
        spots = pd.DataFrame({"gewest": [], "provincie": []})

    def tel(deel):
        return {"Zwarte punten": len(deel)}

    groepen = [
        ("Gewesten", [(g, spots[spots["gewest"] == g]) for g in sorted(df["gewest"].unique())]),
        ("Provincies", [(p, spots[spots["provincie"] == p]) for p in sorted(df["provincie"].unique())]),
    ]

    # Kolomnamen: vet en wit. Wit is alleen leesbaar in het donkere thema; in het lichte thema blijven ze vet in de
    # gewone tekstkleur. Totaal: doffer grijs (de tekst), niet vet, cursief.
    try:
        donker = st.context.theme.type == "dark"
    except Exception:  # thema onbekend (bv. in een test): veilig kiezen
        donker = False
    KOP = [("font-weight", "bold")] + ([("color", "#ffffff")] if donker else [])
    TOTAAL_CSS = "color: #8c8c8c; font-weight: normal; font-style: italic"

    def totaal_opmaak(rij):
        """De rij 'Totaal' (= heel België): doffer grijs, niet vet, cursief."""
        return [TOTAAL_CSS if rij["Gebied"] == "Totaal" else ""] * len(rij)

    # Twee afgebakende blokken naast elkaar, elk met een eigen kader en titel; onderaan de rij Totaal (= België)
    for kolom, (titel, delen) in zip(st.columns([1, 1.3]), groepen):
        with kolom.container(border=True):
            st.markdown(f"**{titel}**")
            rijen = [{"Gebied": naam, **tel(deel)} for naam, deel in delen] + [{"Gebied": "Totaal", **tel(spots)}]
            stijl = (
                pd.DataFrame(rijen)
                .style.hide(axis="index")
                .set_table_styles([{"selector": "th", "props": KOP}])
                .apply(totaal_opmaak, axis=1)
            )
            st.table(stijl)


PROVINCIE_CODE = {
    "Antwerpen": "ANT", "Limburg": "LIM", "Oost-Vlaanderen": "OVL", "Vlaams-Brabant": "VBR", "West-Vlaanderen": "WVL",
    "Henegouwen": "HEN", "Luik": "LUI", "Luxemburg": "LUX", "Namen": "NAM", "Waals-Brabant": "WBR",
    "Geen provincie (Brussel)": "BRU",
}


def punt_code(provincie, punt_id):
    """Korte code van een zwart punt, bv. ANT-006: de code van de provincie plus het clusternummer.

    Het nummer is de volgorde waarin het punt gevonden werd; het verandert als je periode of instellingen wijzigt."""
    return f"{PROVINCIE_CODE.get(str(provincie), str(provincie)[:3].upper())}-{int(punt_id):03d}"


# Kleur van een individueel ongeval in het ingezoomde zicht van een zwart punt (class_accident_code -> kleur)
ERNST_KLEUREN = [("lichtgewonden", "#f2c200", LICHT), ("zwaargewonden", "#ff7f0e", ZWAAR), ("doden", "#d62728", DODEN)]


def toon_punt_detail(leden, punt_id):
    """Ingezoomde kaart en tabel van de ongevallen in één zwart punt.

    `leden` = de ongevallen in zwarte punten (kolom `cluster`), `punt_id` = het clusternummer."""
    ongevallen = leden[leden["cluster"] == punt_id].sort_values(["jaar", "maand"])
    kleur_van_code = {c: k for _, k, codes in ERNST_KLEUREN for c in codes}
    ongevallen = ongevallen.assign(kleur=ongevallen["class_accident_code"].astype(int).map(kleur_van_code).fillna("#888888"))
    st.markdown(
        " &nbsp;·&nbsp; ".join(
            f"<span style='color:{k}; font-size:1.2em'>●</span> {naam} ({int(ongevallen['class_accident_code'].astype(int).isin(codes).sum())})"
            for naam, k, codes in ERNST_KLEUREN
        ),
        unsafe_allow_html=True,
    )
    st.map(ongevallen, latitude="lat", longitude="lon", color="kleur", size=3)
    st.dataframe(
        ongevallen[["gemeente", "jaar", "maand", "uur", "class_accident", "x_lambert72", "y_lambert72", "lat", "lon"]],
        width="stretch",
        hide_index=True,
        column_config={c: st.column_config.NumberColumn(format="%.1f") for c in ("x_lambert72", "y_lambert72")}
        | {c: st.column_config.NumberColumn(format="%.6f") for c in ("lat", "lon")},
    )
    toon_dekking(len(ongevallen), len(ongevallen), "dit zwarte punt", "Alle ongevallen die bij dit zwarte punt tellen staan op de kaart en in de tabel.")


def toon_kaart_en_tabel(spots, kolommen, dek, leden, straal, onder_kaart=None):
    """`dek` = (weergegeven, totaal, plaats): ongevallen met coördinaten, alle ongevallen, en het gebied.
    `leden` = de individuele ongevallen van deze zwarte punten (kolom `cluster`).
    `onder_kaart` is een functie die direct onder de kaart iets toont (hier: de telling per gebied).

    Elke cirkel heeft als straal de zoekafstand (`straal`, vast: AFSTAND); de kleur toont de score
    volgens vaste klassen. Vlak onder de kaart staat de keuze van één zwart punt (gemeente, sortering, tabel van
    5 rijen); het ingezoomde zicht van het aangeklikte punt staat daaronder."""
    tabel = spots.assign(punt_id=spots.index.to_numpy())  # het clusternummer zit in de index
    tabel["code"] = [punt_code(p, i) for p, i in zip(tabel["provincie"], tabel["punt_id"])]
    kolommen = ["code"] + kolommen

    kaart_klik = None  # id van het zwarte punt dat je op de kaart aanklikt
    if pdk is None:
        st.map(tabel[["lat", "lon"]].assign(kleur=tabel["score"].map(score_kleur)), latitude="lat", longitude="lon", color="kleur", size=straal)
    else:
        bollen = tabel[["lat", "lon", "code", "gemeente", "score", "totaal", "punt_id"]].assign(
            kleur=tabel["score"].map(lambda s: [int(score_kleur(s)[i : i + 2], 16) for i in (1, 3, 5)] + [200]),
            score=tabel["score"].round(0).astype(int),
        )
        laag = pdk.Layer(
            "ScatterplotLayer",
            data=bollen,
            id="punten",
            get_position="[lon, lat]",
            get_radius=straal,
            radius_units="'meters'",  # tekst in pydeck: aanhalingstekens binnen aanhalingstekens, anders een formule
            radius_min_pixels=6,  # uitgezoomd nooit kleiner dan 6 pixels; ingezoomd de echte straal in meters
            get_fill_color="kleur",
            get_line_color=[80, 80, 80],
            stroked=True,
            line_width_min_pixels=1,
            pickable=True,
            auto_highlight=True,
        )
        deck = pdk.Deck(
            layers=[laag],
            initial_view_state=pdk.data_utils.compute_view(bollen[["lon", "lat"]].to_numpy().tolist(), 0.95),
            tooltip={"text": "{code}\n{gemeente}\nScore {score} · {totaal} ongevallen"},
        )
        klik = st.pydeck_chart(deck, on_select="rerun", selection_mode="single-object", key="kaart_punten")
        gekozen_op_kaart = klik.selection.objects.get("punten", []) if klik else []
        if gekozen_op_kaart:
            kaart_klik = gekozen_op_kaart[0]["punt_id"]
    toon_legende()
    toon_dekking(*dek, "De kaart gebruikt enkel ongevallen met bruikbare coördinaten.")
    st.subheader("Selectie Zwart Punt")

    # Het actieve punt: het laatst aangeklikte, op de kaart of in de tabel. Klik op de kaart: de keuzelijst springt
    # naar de gemeente van dat punt en het punt staat bovenaan de tabel (geel). Dit moet vóór de keuzelijst gebeuren.
    sessie = st.session_state
    alle_ids = set(tabel["punt_id"])
    actief = sessie.get("actief_punt") if sessie.get("actief_punt") in alle_ids else None
    bron = sessie.get("actief_bron")
    if actief is None:  # nog niets aangeklikt (of de punten zijn veranderd): het punt met de hoogste score
        actief = tabel.sort_values(["score", "doden", "zwaar"], ascending=False)["punt_id"].iloc[0]
        bron = "kaart"
    if kaart_klik is not None and kaart_klik != sessie.get("vorige_kaart_klik") and kaart_klik in alle_ids:
        actief, bron = kaart_klik, "kaart"
        sessie["punt_gemeente"] = str(tabel.loc[tabel["punt_id"] == kaart_klik, "gemeente"].iloc[0])
    sessie["vorige_kaart_klik"] = kaart_klik

    kies_gemeente, kies_sorteer = st.columns([2, 1])
    gemeente = kies_gemeente.selectbox(
        "Gemeente (typ om te zoeken)",
        ["Alle gemeenten"] + sorted(tabel["gemeente"].astype(str).unique()),
        key="punt_gemeente",
        help="Toont in de tabel hieronder enkel de zwarte punten van deze gemeente (de gemeente waar de meeste ongevallen van het punt liggen).",
    )
    sortering = kies_sorteer.radio(
        "Sorteer op",
        ["Score", "Locatie"],
        horizontal=True,
        help="Score: hoogste score eerst. Locatie: per gemeente, binnen een gemeente van noord naar zuid, zodat "
        "aangrenzende punten meestal onder elkaar staan.",
        key="sorteer_punten",
    )
    if gemeente != "Alle gemeenten":
        tabel = tabel[tabel["gemeente"].astype(str) == gemeente]
    if sortering == "Locatie":
        tabel = tabel.sort_values(["gemeente", "lat"], ascending=[True, False])
    if bron == "kaart" and actief is not None and (tabel["punt_id"] == actief).any():  # aangeklikt punt bovenaan
        tabel = pd.concat([tabel[tabel["punt_id"] == actief], tabel[tabel["punt_id"] != actief]])
    tabel = tabel.reset_index(drop=True)
    is_actief = (tabel["punt_id"] == actief).to_numpy()
    gekozen = st.dataframe(
        tabel[kolommen].style.apply(
            lambda r: ["background-color: rgba(255, 190, 0, 0.35)" if is_actief[r.name] else "" for _ in r], axis=1
        ),
        width="stretch",
        height=5 * 35 + 38,  # kopregel + 5 rijen; scroll voor de rest
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        # De sleutel hangt af van de rijen en hun volgorde: verandert de tabel (andere gemeente, sortering of een punt dat
        # naar boven springt), dan vervalt een oude rij-selectie. Anders wijst die naar een ander punt.
        key=f"tabel_punten_{abs(hash(tuple(tabel['punt_id'])))}",
    )
    st.caption(f"{len(tabel)} zwarte punten. Klik op een rij (of op een bol op de kaart) om dat punt ingezoomd hieronder te zien.")
    rijen = gekozen.selection.rows
    rij_id = tabel.at[rijen[0], "punt_id"] if rijen and rijen[0] < len(tabel) else None
    if rij_id is not None and rij_id != sessie.get("vorige_rij_id"):  # nieuwe klik in de tabel
        actief, bron = rij_id, "rij"
    sessie["vorige_rij_id"] = rij_id
    sessie["actief_punt"], sessie["actief_bron"] = actief, bron
    if actief is not None:
        toon_punt_detail(leden, actief)
    if onder_kaart:
        onder_kaart()
    toon_dekking(*dek, "De zwarte punten zijn gezocht onder ongevallen met bruikbare coördinaten.")


with tab_punten:
    f2 = toon_filters("punten", periode_pijltjes=True, types_filter=False)
    sel, jaren = f2.sel, f2.jaren

    sorteer = ["score", "doden", "zwaar"]  # volgorde van de tabellen

    def voldoet(spots):
        """Een zwart punt heeft naast de score (SCORE_DREMPEL) ook minstens MIN_TOTAAL ongevallen."""
        return spots[spots["totaal"] >= MIN_TOTAAL]

    def spots_voor_periode(p):
        """Zwarte punten (vaste instelling, zie bovenaan) over één periode van ongevallen, gezocht in heel België.

        Wordt ook door het tabblad Evolutie gebruikt, voor elke periode van 3 jaar."""
        cluster = zoek_clusters(
            p["x_lambert72"].to_numpy(), p["y_lambert72"].to_numpy(), punten_per_ongeval(p), AFSTAND, SCORE_DREMPEL
        )
        return voldoet(bouw_spots(p, cluster)[1])

    basis_kolommen = ["gemeente", "score", "totaal", "licht", "zwaar", "doden"]

    plaats2 = plaats_label(f2)
    alle_jaren = list(range(jaren[0], jaren[1] + 1))
    bruikbaar = ["ok", "hersteld"]

    def zonder_standaard(p):
        """Laat de verdachte standaardcoördinaten weg (bv. stadhuizen, zie clean_data.py): ze tellen niet mee voor de
        zwarte punten, wel voor alle andere tellingen."""
        return p[~p["verdacht_coord"]]

    # Zwarte punten zoeken in heel België (alle types ongevallen, via de punten); de filters hierboven bepalen
    # daarna enkel welke punten je op de kaart en in de lijst ziet. Zo klopt de tabel hieronder altijd voor België.
    punten_nl = zonder_standaard(df[df["jaar"].between(*jaren) & df["coord_status"].isin(bruikbaar)]).reset_index(drop=True)
    punten = zonder_standaard(sel[sel["coord_status"].isin(bruikbaar)])  # enkel het gekozen gebied; voor de verhouding weergegeven/totaal
    dek = (len(punten), len(sel), plaats2)

    cluster = zoek_clusters(
        punten_nl["x_lambert72"].to_numpy(), punten_nl["y_lambert72"].to_numpy(), punten_per_ongeval(punten_nl), AFSTAND, SCORE_DREMPEL
    )
    leden, spots_nl = bouw_spots(punten_nl, cluster)
    per_jaar = leden.groupby(["cluster", "jaar"]).size().unstack(fill_value=0).reindex(columns=alle_jaren, fill_value=0)
    per_jaar.columns = [str(j) for j in per_jaar.columns]
    spots_nl = voldoet(spots_nl).join(per_jaar)

    def telling_onder_kaart():
        """De tabel met het aantal zwarte punten (altijd heel België) onder de kaart."""
        st.subheader(f"Aantal zwarte punten, periode {jaren[0]}-{jaren[1]}")
        toon_telling(spots_nl)
        toon_dekking(
            len(punten_nl),
            int(df["jaar"].between(*jaren).sum()),
            "België",
            "Zwarte punten zijn gezocht onder ongevallen met bruikbare coördinaten; deze tabel telt heel België.",
        )

    spots = in_gebied(spots_nl, f2).sort_values(sorteer, ascending=False)

    if spots.empty:
        st.info("Geen zwarte punten in dit gebied en deze periode.")
        telling_onder_kaart()
    else:
        toon_kaart_en_tabel(
            spots, basis_kolommen + list(per_jaar.columns) + ["lat", "lon"], dek, leden, AFSTAND, onder_kaart=telling_onder_kaart
        )

    # Uitleg en vaste instelling staan onder de kaart, zodat boven de kaart enkel de tabel staat.
    with st.expander("ℹ️ Uitleg en vaste instelling"):
        st.markdown(
            "Een **zwart punt** is hier een plek waar binnen een vaste afstand genoeg (ernstige) ongevallen gebeurden. "
            "We lijken op de puntentelling van Vlaanderen voor "
            "[gevaarlijke punten](https://www.vlaanderen.be/mobiliteitsprofessionals/verkeersveiligheid/gevaarlijke-punten) "
            "(3 jaar, score van minstens 15, minstens 3 ongevallen), maar tellen **per ongeval** en niet per slachtoffer: "
            "de bron bevat geen slachtoffers. Alle types ongevallen tellen mee via hun punten. "
            "*De afstand dient enkel om ongevallen tot één punt samen te voegen.*"
        )
        st.caption(
            f"Vaste instelling: afstand {AFSTAND} m · score = {PUNT_LICHT} × licht + {PUNT_ZWAAR} × zwaar + {PUNT_DODEN} × doden, "
            f"per ongeval (niet per slachtoffer). Zwart punt vanaf **{SCORE_DREMPEL} punten** en minstens **{MIN_TOTAAL} ongevallen**. "
            "Ongevallen op verdachte standaardcoördinaten (bv. stadhuizen) tellen niet mee."
        )
        st.caption(
            "De filters gewest, provincie en gemeente bepalen welke zwarte punten op de kaart en in de lijst staan; een "
            "punt hoort bij het gebied waar de meeste van zijn ongevallen liggen. De tabel bovenaan telt altijd heel België."
        )

# ---------- Tab Evolutie ----------
GEWEST_NAAM = {"Vlaams Gewest": "Vlaanderen", "Brussels Hoofdstedelijk Gewest": "Brussels Gewest", "Waals Gewest": "Wallonië"}


STREEP = 6  # lengte van een streepje in pixels (alleen opmaak)


def toon_lijngrafiek(titel, reeksen, periodes, dekking=None):
    """Lijngrafiek met de punten verbonden. `reeksen` = {naam: [aantal per periode]}, `periodes` = [(eindjaar, begin), ...].

    De periode staat op de x-as met het eindjaar (2019 = 2017-2019); de volledige periode staat in de hover.

    Stippellijn-regel (automatisch, per lijnstuk tussen twee opeenvolgende periodes): hebben twee of meer ZICHTBARE reeksen in
    dezelfde grafiek hetzelfde beginpunt EN hetzelfde eindpunt, dan bedekken hun stukken elkaar en worden ze gestippeld, met
    de streepjes om en om zodat je ze allemaal ziet. Kruisen lijnen elkaar enkel, dan blijven ze doorlopend.
    Klik je een reeks in de legende uit, dan wordt de overlap opnieuw bekeken (in de grafiek zelf, niet in Python).

    `dekking` = [% ongevallen met bruikbare coördinaten per periode, heel België]: wordt als dof grijs stippellijntje
    getekend op een eigen as rechts (0-100), los van de reeksen en van de legende."""
    namen = list(reeksen)
    perioden = [str(eind) for eind, _ in periodes]
    volledig = {str(eind): f"{begin}-{eind}" for eind, begin in periodes}

    # Alle punten (voor de bolletjes en de hover)
    punten = pd.DataFrame(
        [
            {"reeks": naam, "periode": perioden[i], "volledig": volledig[perioden[i]], "aantal": reeksen[naam][i]}
            for naam in namen
            for i in range(len(perioden))
        ]
    )

    # Lijnstukken: 2 rijen per reeks en per stuk (periode i -> i+1), met begin- en eindwaarde van het stuk erbij.
    # De overlap zelf berekent de grafiek (zie hieronder), want die moet veranderen als je reeksen in de legende aan/uit zet.
    stukken = pd.DataFrame(
        [
            {
                "reeks": naam,
                "volgorde": volgorde,
                "periode": perioden[j],
                "aantal": reeksen[naam][j],
                "stuk": f"{naam}|{i}",  # elk stuk is een eigen lijntje
                "i": i,
                "begin": reeksen[naam][i],
                "eind": reeksen[naam][i + 1],
            }
            for volgorde, naam in enumerate(namen)
            for i in range(len(perioden) - 1)
            for j in (i, i + 1)
        ]
    )

    # Legende aanklikken: elke klik zet een reeks aan of uit (uitgezette reeksen staan dof in de legende).
    # We starten met alle reeksen aan (value). toggle="true" (een TEKST, geen boolean True!): een klik wisselt enkel die reeks,
    # zonder shift. Vega-Lite gebruikt toggle als uitdrukking; een boolean wordt genegeerd, dan kiest elke klik "enkel deze".
    # empty=False: staan alle reeksen uit, dan toont de grafiek niets (en staan alle namen dof in de legende).
    aan = alt.selection_point(
        fields=["reeks"],
        bind="legend",
        toggle="true",
        empty=False,
        value=[{"reeks": naam} for naam in namen],
    )

    # De y-as ligt vast per grafiek (0 tot de hoogste waarde van ALLE reeksen), ook als je reeksen uitzet. Anders past de as
    # zich aan de zichtbare lijnen aan en lijkt een lage lijn te stijgen als je de hoogste uitzet.
    hoogste = max(1, max(max(aantallen) for aantallen in reeksen.values()))
    x = alt.X("periode:O", title="Periode (eindjaar van 3 jaar)", axis=alt.Axis(labelAngle=0), scale=alt.Scale(domain=perioden))
    y = alt.Y("aantal:Q", title="Aantal zwarte punten", scale=alt.Scale(domain=[0, hoogste], nice=True))
    kleur = alt.Color("reeks:N", title=None, scale=alt.Scale(domain=namen))  # vaste volgorde en kleuren

    # Stippelpatroon per lijnstuk: k = aantal ZICHTBARE reeksen met hetzelfde begin- en eindpunt op dit stuk, r = rang
    # van deze reeks daarbinnen (0 ... k-1). Reeks r tekent zijn streepje op plek r in een periode van k streepjes.
    # k = 1: doorlopend ([1, 0]). Het patroon staat dus voor elke (k, r) vooraf klaar, tot het aantal reeksen.
    patronen = {}
    for k in range(1, len(namen) + 1):
        for r in range(k):
            if k == 1:
                patronen[f"{k}-{r}"] = [1, 0]
            elif r == 0:
                patronen[f"{k}-{r}"] = [STREEP, STREEP * (k - 1)]
            else:
                patronen[f"{k}-{r}"] = [0, STREEP * r, STREEP, STREEP * (k - 1 - r)]

    lijnen = (
        alt.Chart(stukken)
        .transform_filter(aan)  # eerst enkel de aangezette reeksen houden ...
        .transform_joinaggregate(k="distinct(reeks)", groupby=["i", "begin", "eind"])  # ... dan tellen hoeveel er samenvallen
        .transform_window(rang="dense_rank()", sort=[alt.SortField("volgorde")], groupby=["i", "begin", "eind"])
        .transform_calculate(patroon="datum.k + '-' + (datum.rang - 1)")
        .mark_line()
        .encode(
            x=x,
            y=y,
            color=kleur,  # zelfde kleur-encoding als de bolletjes, zodat er één gezamenlijke legende is
            detail="stuk:N",
            strokeDash=alt.StrokeDash(
                "patroon:N", legend=None, scale=alt.Scale(domain=list(patronen), range=list(patronen.values()))
            ),
        )
    )
    bolletjes = (
        alt.Chart(punten)
        .transform_filter(aan)
        .mark_point(filled=True, size=60, opacity=1)
        .encode(
            x=x,
            y=y,
            color=kleur,
            tooltip=[
                alt.Tooltip("reeks:N", title="Reeks"),
                alt.Tooltip("volledig:N", title="Periode"),
                alt.Tooltip("aantal:Q", title="Zwarte punten"),
            ],
        )
        .add_params(aan)
    )
    grafiek = alt.layer(lijnen, bolletjes)
    if dekking is not None:
        # Eigen schaal rechts (0-100 %). Eerst de twee lagen met zwarte punten samen (gedeelde linkeras), dan de dekking
        # ernaast met een onafhankelijke y-schaal, zodat er maar één as links en één rechts staat.
        dek = pd.DataFrame(
            {"periode": perioden, "volledig": [volledig[p] for p in perioden], "procent": dekking}
        )
        y_dek = alt.Y(
            "procent:Q",
            title="Ongevallen met coördinaten (%)",
            scale=alt.Scale(domain=[0, 100]),
            axis=alt.Axis(orient="right", titleColor="gray", labelColor="gray", grid=False),
        )
        dek_lijn = alt.Chart(dek).mark_line(color="gray", opacity=0.45, strokeDash=[3, 3]).encode(x=x, y=y_dek)
        dek_punt = (
            alt.Chart(dek)
            .mark_point(color="gray", opacity=0.45, filled=True, size=30)
            .encode(
                x=x,
                y=y_dek,
                tooltip=[
                    alt.Tooltip("volledig:N", title="Periode"),
                    alt.Tooltip("procent:Q", title="Met coördinaten (%)", format=".1f"),
                ],
            )
        )
        grafiek = alt.layer(grafiek, dek_lijn, dek_punt).resolve_scale(y="independent")
    st.subheader(titel)
    st.altair_chart(grafiek, width="stretch")
    st.caption(
        "Klik op een naam in de legende om een lijn aan of uit te zetten. De schaal blijft dezelfde."
        + (" De dofgrijze stippellijn (as rechts) is het gemiddelde van de 3 jaren van het % ongevallen in heel België met bruikbare coördinaten." if dekking else "")
    )


with tab_evolutie:
    # Voor elke periode van 3 jaar (2017-2019, 2018-2020, ..., 2022-2024) zoeken we de zwarte punten in heel België,
    # met dezelfde vaste instelling als in het tabblad Kaart.
    periodes = [(eind, eind - PERIODE_JAREN + 1) for eind in range(JAAR_MIN + PERIODE_JAREN - 1, JAAR_MAX + 1)]
    per_periode = []
    for eind, begin in periodes:
        in_periode = df["jaar"].between(begin, eind)
        p = zonder_standaard(df[in_periode & df["coord_status"].isin(["ok", "hersteld"])]).reset_index(drop=True)
        per_periode.append({"spots": spots_voor_periode(p), "met_coord": len(p), "alle": int(in_periode.sum())})

    def aantallen(kolom=None, waarde=None):
        """Aantal zwarte punten per periode, voor heel België of voor één gewest of provincie."""
        return [
            len(x["spots"]) if kolom is None else int((x["spots"][kolom] == waarde).sum()) for x in per_periode
        ]

    def provincies_van(gewest):
        return sorted(df.loc[df["gewest"] == gewest, "provincie"].unique())

    # Heel België: eerst het % per jaar, dan per periode het gemiddelde van de 3 jaren (elk jaar telt even zwaar mee)
    procent_per_jaar = df["coord_status"].isin(["ok", "hersteld"]).groupby(df["jaar"]).mean() * 100
    dekking = [procent_per_jaar.loc[begin:eind].mean() for eind, begin in periodes]
    toon_lijngrafiek(
        "België",
        {"Totaal": aantallen()} | {naam: aantallen("gewest", gewest) for gewest, naam in GEWEST_NAAM.items()},
        periodes,
        dekking,
    )
    toon_lijngrafiek(
        "Vlaanderen: provincies",
        {p: aantallen("provincie", p) for p in provincies_van("Vlaams Gewest")},
        periodes,
        dekking,
    )
    toon_lijngrafiek(
        "Wallonië: provincies",
        {p: aantallen("provincie", p) for p in provincies_van("Waals Gewest")},
        periodes,
        dekking,
    )

    toon_dekking(
        int(df["coord_status"].isin(["ok", "hersteld"]).sum()),
        len(df),
        "België",
        "Zwarte punten zijn gezocht onder ongevallen met bruikbare coördinaten (2017-2024).",
    )
    st.caption(
        "Bruikbare coördinaten per periode (gemiddelde van het % van de 3 jaren, zoals de grijze lijn): "
        + " · ".join(f"{eind}: {d:.1f} %".replace(".", ",") for (eind, _), d in zip(periodes, dekking))
        + ". Let op: de dekking stijgt vanaf 2022; een stijging in de grafieken is dus deels een meeteffect (zie aanames.txt 3.4)."
    )
    st.caption(
        f"Instelling zoals in het tabblad Kaart: afstand {AFSTAND} m, zwart punt vanaf {SCORE_DREMPEL} punten en minstens "
        f"{MIN_TOTAAL} ongevallen. Een punt telt mee bij het gewest en de provincie waar de meeste van zijn ongevallen liggen."
    )

# ---------- Bron (onder alle tabbladen) ----------
st.divider()
st.caption(
    "**Bron:** Statbel, verkeersongevallen (open data), 2017-2024: "
    "[OPENDATA_MAP_2017-2024.xlsx](https://statbel.fgov.be/sites/default/files/files/opendata/Accident_XY/OPENDATA_MAP_2017-2024.xlsx). "
    "Opgeschoond en bewerkt in dit project: coördinaten omgerekend naar breedte- en lengtegraad en 2 onmogelijke "
    "waarden hersteld (zie aanames.txt). Ongevallen zonder bruikbare coördinaten staan niet op de kaart."
)
