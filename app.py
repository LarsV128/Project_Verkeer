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

st.set_page_config(page_title="Verkeersongevallen", layout="wide", initial_sidebar_state="collapsed")
st.title("Verkeersongevallen in België, 2017-2024")


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
    """Punten per type, ingesteld via het tandwiel in het tabblad Zwarte punten (standaard 1 / 3 / 5)."""
    s = st.session_state
    return SimpleNamespace(licht=s.get("p_licht", 1), zwaar=s.get("p_zwaar", 3), doden=s.get("p_doden", 5))


def punten_per_ongeval(p):
    """Score van elk ongeval: de punten van zijn type."""
    inst = punten_instellingen()
    code = p["class_accident_code"].astype(int)
    return np.select([code.isin(DODEN), code.isin(ZWAAR)], [inst.doden, inst.zwaar], default=inst.licht).astype(float)


def toon_filters(prefix):
    """Filters bovenaan een tabblad: één gewest, één provincie en/of één gemeente, jaren en types.

    Geeft een object met de gekozen waarden en `sel`, de ongevallen die aan alles voldoen."""
    k1, k2, k3, k4 = st.columns(4)
    gewest = k1.selectbox("Gewest", ["Alle gewesten"] + sorted(df["gewest"].unique()), key=f"{prefix}_gewest")
    sel = df if gewest == "Alle gewesten" else df[df["gewest"] == gewest]

    provincie = k2.selectbox("Provincie", ["Alle provincies"] + sorted(sel["provincie"].unique()), key=f"{prefix}_prov")
    sel = sel if provincie == "Alle provincies" else sel[sel["provincie"] == provincie]

    gemeente = k3.selectbox("Gemeente", ["Alle gemeenten"] + sorted(sel["gemeente"].unique()), key=f"{prefix}_gem")
    sel = sel if gemeente == "Alle gemeenten" else sel[sel["gemeente"] == gemeente]

    jaren = k4.slider("Jaren", JAAR_MIN, JAAR_MAX, (JAAR_MIN, JAAR_MAX), key=f"{prefix}_jaren")
    types = st.multiselect("Type ongeval (class_accident)", TYPES, default=TYPES, key=f"{prefix}_types")

    sel = sel[sel["jaar"].between(*jaren) & sel["class_accident"].isin(types)]
    return SimpleNamespace(sel=sel, gewest=gewest, provincie=provincie, gemeente=gemeente, jaren=jaren, types=types)


def toon_type_overzicht(sel):
    """Aantal ongevallen per type (class_accident) voor de selectie."""
    st.subheader("Overzicht per type ongeval")
    aantal = sel["class_accident"].value_counts().reindex(TYPES, fill_value=0)
    tabel = pd.DataFrame({"type": TYPES, "aantal": aantal.to_numpy()})
    totaal = int(tabel["aantal"].sum())
    tabel["aandeel (%)"] = (tabel["aantal"] / totaal * 100).round(1) if totaal else 0.0
    tabel.loc[len(tabel)] = ["Totaal", totaal, 100.0 if totaal else 0.0]
    st.dataframe(tabel, width="stretch", hide_index=True)


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
        f"Score = {inst.licht} × licht + {inst.zwaar} × zwaar + {inst.doden} × doden (instellen via ⚙️ in het tabblad "
        f"Zwarte punten). Nummering over alle {len(g)} gemeenten in België; positie 1 heeft de hoogste score."
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
    toon(g, titel)


tab_overzicht, tab_punten, tab_fout = st.tabs(["Overzicht", "Zwarte punten", "Foute coördinaten"])

# ---------- Tab 1: overzicht ----------
with tab_overzicht:
    f1 = toon_filters("overzicht")
    sel = f1.sel
    if sel.empty:
        st.warning("Geen ongevallen voor deze selectie. Pas de filters aan.")
    else:
        st.metric("Aantal ongevallen in selectie", f"{len(sel):,}".replace(",", "."))
        toon_type_overzicht(sel)

        st.subheader("Per jaar, per type")
        st.bar_chart(sel.groupby(["jaar", "class_accident"], observed=True).size().unstack(fill_value=0))

    toon_ranglijst(f1)

    if not sel.empty:
        with st.expander("Ruwe rijen van de selectie (eerste 1000)"):
            st.dataframe(sel.head(1000), width="stretch", hide_index=True)

# ---------- Tab 2: zwarte punten ----------
MODUS_VAST = "Vaste locaties, aantallen per jaar"
MODUS_JAAR = "Elk jaar apart"
METHODE_AANTAL = "Aantal ongevallen"
METHODE_SCORE = "Puntenscore"


def bouw_spots(punten, cluster, gewicht):
    """Telt per cluster de ongevallen per type, de score en de gemiddelde plaats.

    Geeft (ongevallen die in een cluster zitten, tabel met één rij per cluster)."""
    leden = punten.assign(cluster=cluster, gewicht=gewicht)
    leden = leden[leden["cluster"] >= 0]
    code = leden["class_accident_code"].astype(int)
    leden = leden.assign(licht=code.isin(LICHT), zwaar=code.isin(ZWAAR), doden=code.isin(DODEN))
    spots = leden.groupby("cluster").agg(
        totaal=("cluster", "size"),
        score=("gewicht", "sum"),
        licht=("licht", "sum"),
        zwaar=("zwaar", "sum"),
        doden=("doden", "sum"),
        lat=("lat", "mean"),
        lon=("lon", "mean"),
        gemeente=("gemeente", lambda s: s.value_counts().index[0]),
    )
    return leden, spots


def toon_kaart_en_tabel(spots, kolommen, maat):
    kaart = spots.assign(grootte=spots[maat] * 15)  # st.map-grootte is in meter
    st.map(kaart, latitude="lat", longitude="lon", size="grootte")
    st.dataframe(spots.reset_index(drop=True)[kolommen], width="stretch", hide_index=True)


with tab_punten:
    f2 = toon_filters("punten")
    sel, jaren = f2.sel, f2.jaren

    st.markdown(
        "Een **zwart punt** is hier een plek waar binnen de gekozen afstand genoeg ongevallen gebeurden. "
        "*Voorlopige definitie; de officiële definitie zoeken we nog op.*"
    )
    r1, r2 = st.columns(2)
    modus = r1.radio("Weergave", [MODUS_VAST, MODUS_JAAR], horizontal=True)
    methode = r2.radio("Wanneer is het een zwart punt?", [METHODE_AANTAL, METHODE_SCORE], horizontal=True)
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

    c1, c2, c3 = st.columns(3)
    straal = c1.select_slider("Afstand (m)", options=[25, 50, 75, 100, 150, 200], value=50)
    min_totaal = c2.number_input(
        f"Min. ongevallen ({eenheid})",
        min_value=2,
        value=3 if per_jaar_modus else 5,
        step=1,
        key="min_jaar" if per_jaar_modus else "min_vast",
        disabled=score_modus,
        help="Wordt gebruikt bij de methode 'Aantal ongevallen'.",
    )

    with c3.popover("⚙️ Instellingen: puntenscore en extra voorwaarden"):
        st.markdown("**Puntenscore**")
        st.caption(
            "Elk ongeval telt mee met de punten van zijn type. Gebruikt bij de methode 'Puntenscore' en voor de "
            "ranglijst van gemeenten in het tabblad Overzicht."
        )
        st.number_input("Punten per ongeval met lichtgewonden", min_value=0, value=1, step=1, key="p_licht")
        st.number_input("Punten per ongeval met zwaargewonden", min_value=0, value=3, step=1, key="p_zwaar")
        st.number_input("Punten per ongeval met doden (incl. dodelijk gewonden)", min_value=0, value=5, step=1, key="p_doden")
        score_drempel = st.number_input(f"Zwart punt vanaf (punten, {eenheid})", min_value=1, value=7, step=1)

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

    inst = punten_instellingen()
    if score_modus:
        st.caption(
            f"Score = {inst.licht} × licht + {inst.zwaar} × zwaar + {inst.doden} × doden. "
            f"Zwart punt vanaf **{score_drempel} punten** ({eenheid})."
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
    drempel = score_drempel if score_modus else min_totaal
    maat = "score" if score_modus else "totaal"
    sorteer = [maat, "doden", "zwaar"]

    def gewichten(p):
        """Gewicht per ongeval: 1 bij de methode 'Aantal ongevallen', anders de punten van het type."""
        return punten_per_ongeval(p) if score_modus else np.ones(len(p))

    def voldoet(spots):
        """Past de extra voorwaarden toe op de gevonden zwarte punten."""
        ok = spots["licht"] >= min_licht
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

    basis_kolommen = ["gemeente"] + (["score"] if score_modus else []) + ["totaal", "licht", "zwaar", "doden"]

    toon_type_overzicht(sel)

    punten = sel[sel["coord_status"].isin(["ok", "hersteld"])].reset_index(drop=True)
    st.caption(
        f"{len(punten):,} van de {len(sel):,} ongevallen in je selectie hebben bruikbare coördinaten "
        f"(de rest ontbreekt of is fout).".replace(",", ".")
    )
    alle_jaren = list(range(jaren[0], jaren[1] + 1))

    if punten.empty:
        st.warning("Geen ongevallen met coördinaten in deze selectie.")

    elif not per_jaar_modus:
        w = gewichten(punten)
        cluster = zoek_clusters(punten["x_lambert72"].to_numpy(), punten["y_lambert72"].to_numpy(), w, straal, drempel)
        leden, spots = bouw_spots(punten, cluster, w)
        per_jaar = (
            leden.groupby(["cluster", "jaar"]).size().unstack(fill_value=0).reindex(columns=alle_jaren, fill_value=0)
        )
        per_jaar.columns = [str(j) for j in per_jaar.columns]
        spots = voldoet(spots).join(per_jaar).sort_values(sorteer, ascending=False)

        m1, m2 = st.columns(2)
        m1.metric("Aantal zwarte punten", f"{len(spots):,}".replace(",", "."))
        m2.metric("Ongevallen in die punten", f"{int(spots['totaal'].sum()):,}".replace(",", "."))

        if spots.empty:
            st.info("Geen zwarte punten met deze drempels. Verlaag een drempel of vergroot de afstand.")
        else:
            toon_kaart_en_tabel(spots, basis_kolommen + list(per_jaar.columns) + ["lat", "lon"], maat)

            st.subheader("Evolutie van één zwart punt")
            top = spots.head(200)
            namen = {}
            for n, (c, r) in enumerate(top.iterrows(), 1):
                extra = f", score {r.score:g}" if score_modus else ""
                namen[c] = f"{n}. {r.gemeente} – {r.totaal} ongevallen{extra}"
            gekozen = st.selectbox(
                f"Kies een zwart punt (top 200 op {maat})", list(namen), format_func=namen.get
            )
            evolutie = (
                leden[leden["cluster"] == gekozen]
                .groupby("jaar")[["licht", "zwaar", "doden"]]
                .sum()
                .reindex(alle_jaren, fill_value=0)
            )
            st.bar_chart(evolutie)

    else:
        delen = []
        for jaar in alle_jaren:
            pj = punten[punten["jaar"] == jaar].reset_index(drop=True)
            if pj.empty:
                continue
            w = gewichten(pj)
            cluster = zoek_clusters(pj["x_lambert72"].to_numpy(), pj["y_lambert72"].to_numpy(), w, straal, drempel)
            _, spots_jaar = bouw_spots(pj, cluster, w)
            delen.append(voldoet(spots_jaar).assign(jaar=jaar))
        spots = pd.concat(delen, ignore_index=True) if delen else pd.DataFrame()

        if spots.empty:
            st.info("Geen zwarte punten met deze drempels. Verlaag een drempel of vergroot de afstand.")
        else:
            samenvatting = (
                spots.groupby("jaar")
                .agg(zwarte_punten=("totaal", "size"), ongevallen_in_punten=("totaal", "sum"), doden=("doden", "sum"))
                .reindex(alle_jaren, fill_value=0)
            )
            st.subheader("Aantal zwarte punten per jaar")
            st.bar_chart(samenvatting["zwarte_punten"])
            st.dataframe(samenvatting, width="stretch")

            jaar_keuze = st.select_slider("Toon de zwarte punten van", options=alle_jaren, value=alle_jaren[-1])
            van_jaar = spots[spots["jaar"] == jaar_keuze].sort_values(sorteer, ascending=False)
            if van_jaar.empty:
                st.info(f"Geen zwarte punten in {jaar_keuze}.")
            else:
                toon_kaart_en_tabel(van_jaar, basis_kolommen + ["lat", "lon"], maat)

# ---------- Tab 3: foute coördinaten ----------
with tab_fout:
    f3 = toon_filters("fout")
    st.markdown(
        "Coördinaten staan in **Lambert72** (meters). Staat een waarde buiten België, dan is ze onmogelijk; "
        "lege waarden ontbreken gewoon. De filters hierboven gelden voor het overzicht per type; de lijst en de "
        "kaarten met herstelde coördinaten daaronder tonen altijd alle ongevallen."
    )

    st.subheader("Overzicht per type ongeval")
    status = pd.crosstab(f3.sel["class_accident"], f3.sel["coord_status"]).reindex(TYPES, fill_value=0)
    status = status.reindex(columns=["ok", "hersteld", "ontbreekt", "onmogelijk"], fill_value=0)
    status.loc["Totaal"] = status.sum()
    status["% bruikbaar"] = ((status["ok"] + status["hersteld"]) / status.sum(axis=1).replace(0, np.nan) * 100).round(1)
    status = status.rename(columns={"ok": "bruikbaar"}).reset_index().rename(columns={"class_accident": "type"})
    st.dataframe(status, width="stretch", hide_index=True)

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
