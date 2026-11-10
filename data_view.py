"""Tabblad Data View van de app Ongevallen.

Los script: wis dit bestand en de app werkt gewoon verder, zonder het tabblad Data View.
app.py geeft alles wat hier nodig is mee via `toon(...)`; die waarden zijn hieronder gewone variabelen
(df, TYPES, straal, ...)."""
import numpy as np
import pandas as pd
import streamlit as st
from pyproj import Transformer


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
        f"instellen via ⚙️ in het tabblad Kaart). Nummering over alle {len(g)} gemeenten in België; "
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


@st.cache_data
def bronrijen(rijen, gewijzigd):
    """Hele rijen (alle kolommen, als tekst) uit de ruwe data, op rijnummer. Het bestand wordt enkel gelezen.

    Rij i van de opgeschoonde data is rij i van de bron (zelfde volgorde); regel 0 van het bestand is de kopregel."""
    gezocht = set(rijen)
    return pd.read_csv(
        clean_data.RAW, sep="|", encoding="utf-8-sig", dtype=str, skiprows=lambda r: r != 0 and (r - 1) not in gezocht
    )


def grensoverlap(leden):
    """Telt per niveau hoeveel zwarte punten ongevallen uit meer dan één gebied bevatten.

    `leden` = de ongevallen in zwarte punten (kolom `cluster`). Geeft een tabel met één rij per niveau."""
    rijen = []
    for niveau in ("gewest", "provincie", "gemeente"):
        per_gebied = leden.groupby(["cluster", niveau], observed=True).size()
        meeste = per_gebied.groupby("cluster").transform("max")
        aantal_gebieden = per_gebied.groupby("cluster").size()
        gelijk = (per_gebied == meeste).groupby("cluster").sum() > 1  # gelijke stand: de keuze van de app is toeval
        rijen.append(
            {
                "niveau": niveau,
                "zwarte punten": len(aantal_gebieden),
                "met ongevallen uit meer dan 1 gebied": int((aantal_gebieden > 1).sum()),
                "waarvan gelijke stand": int(gelijk.sum()),
            }
        )
    return pd.DataFrame(rijen)


def toon(**gedeeld):
    """Bouwt het tabblad Data View. Aanroepen binnen `with tab_data:`, nadat de tabbladen Kaart en Evolutie klaar zijn."""
    globals().update(gedeeld)  # df, TYPES, straal, ... zoals app.py ze meegeeft

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
            # Kolommen blijven allemaal zichtbaar; je filtert op de waarden van een kolom (bv. jaar 2022, gemeente Herentals).
            filterbaar = [c for c in sel.columns if c not in ("x_lambert72", "y_lambert72", "x_bron", "y_bron", "lat", "lon")]
            filter_kolommen = st.multiselect("Filter op kolom", filterbaar, key="viewer_filter_kolommen")
            getoond = sel
            for kolom in filter_kolommen:
                opties = sorted(sel[kolom].dropna().unique().tolist())
                gekozen = st.multiselect(f"{kolom}: waarde(n)", opties, key=f"viewer_waarde_{kolom}")
                if gekozen:
                    getoond = getoond[getoond[kolom].isin(gekozen)]
            st.caption(f"{nl(len(getoond))} van {nl(len(sel))} ongevallen van de selectie.")
            st.dataframe(
                getoond,
                width="stretch",
                height=450,
                hide_index=True,
                column_config={c: st.column_config.NumberColumn(format="%.1f") for c in ("x_lambert72", "y_lambert72")}
                | {c: st.column_config.NumberColumn(format="%.5e") for c in ("x_bron", "y_bron")},
            )
            toon_dekking(len(getoond), len(sel), plaats1, "Zonder filter in deze lijst staan alle rijen van de selectie in de tabel.")

        st.metric("Aantal ongevallen in selectie", nl(len(sel)))
        toon_type_overzicht(sel, plaats1)

    toon_ranglijst(f1)

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

    st.subheader("Hele bronrijen van deze ongevallen")
    st.caption(
        "Alle kolommen van de rijen in `data/raw/OPENDATA_MAP_2017-2024.txt`, zoals ze in de bron staan (onaangeroerd, enkel "
        "gelezen). Zo kan je de coördinaten en de gemeente naast de rest van het ongeval leggen."
    )
    if RAW_AANWEZIG:
        bron = bronrijen(tuple(int(i) for i in fout.index), clean_data.RAW.stat().st_mtime)
        bron.index = [f"{g} (rij {i})" for g, i in zip(fout["gemeente"], fout.index)]
        st.dataframe(bron.T, width="stretch")
        toon_dekking(len(bron), len(fout), "België", "Alle ongevallen uit de tabel hierboven staan ook hier, met alle bronkolommen.")
    else:
        st.info("De ruwe data staat niet online (te groot voor GitHub); deze tabel is enkel zichtbaar in de lokale versie.")

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

    st.divider()
    st.header("Zwarte punten op een grens")
    st.markdown(
        "Een zwart punt voegt ongevallen binnen de gekozen afstand samen, ook over een grens heen. Het punt hoort dan bij "
        "het gebied waar de meeste van zijn ongevallen liggen. Hieronder staat hoeveel punten ongevallen uit meer dan één "
        "gewest, provincie of gemeente bevatten. Bij *gelijke stand* is de keuze van de app toeval."
    )
    cluster_nl = zoek_clusters(
        punten_nl["x_lambert72"].to_numpy(), punten_nl["y_lambert72"].to_numpy(), gewichten(punten_nl), straal, drempel
    )
    leden_nl, spots_alle = bouw_spots(punten_nl, cluster_nl)
    leden_nl = leden_nl[leden_nl["cluster"].isin(voldoet(spots_alle).index)]
    st.dataframe(grensoverlap(leden_nl), width="stretch", hide_index=True)
    st.caption(
        f"Instellingen zoals in het tandwiel van het tabblad Kaart: periode {jaren[0]}-{jaren[1]}, één keer zoeken over "
        f"die jaren (vaste locaties), afstand {straal} m, methode {methode}, zwart punt vanaf {score_drempel} punten en "
        f"minstens {min_totaal} ongevallen."
    )
    toon_dekking(
        len(punten_nl),
        int(df["jaar"].between(*jaren).sum()),
        "België",
        "Zwarte punten zijn gezocht onder ongevallen met bruikbare coördinaten.",
    )

    # Alle zwarte punten met ongevallen uit meer dan één gemeente (die omvatten ook de grenzen van provincie en gewest).
    aantal_gemeenten = leden_nl.groupby("cluster")["gemeente"].nunique()
    grens_ids = aantal_gemeenten[aantal_gemeenten > 1].index
    st.subheader(f"Alle zwarte punten op een gemeentegrens ({len(grens_ids)})")
    if len(grens_ids) == 0:
        st.info("Geen zwarte punten op een grens met deze instellingen.")
    else:
        labels = {}
        for i in grens_ids:
            deel = leden_nl[leden_nl["cluster"] == i]
            verdeling = ", ".join(f"{g} ({n})" for g, n in deel["gemeente"].value_counts().items() if n > 0)
            labels[i] = f"{verdeling} · score {spots_alle.at[i, 'score']:.0f}".replace(".", ",")
        gekozen_punt = st.selectbox(
            "Kies een zwart punt",
            sorted(grens_ids, key=lambda i: -spots_alle.at[i, "score"]),
            format_func=lambda i: labels[i],
            help="Per gemeente staat het aantal ongevallen van dit zwarte punt tussen haakjes.",
        )
        ongevallen = leden_nl[leden_nl["cluster"] == gekozen_punt].sort_values(["gemeente", "jaar", "maand"])
        gemeenten = [g for g in ongevallen["gemeente"].cat.categories if g in set(ongevallen["gemeente"])]
        palet = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
        kleur_van = {g: palet[n % len(palet)] for n, g in enumerate(gemeenten)}
        st.markdown(
            " &nbsp;·&nbsp; ".join(
                f"<span style='color:{kleur_van[g]}; font-size:1.2em'>●</span> {g} ({int((ongevallen['gemeente'] == g).sum())})"
                for g in gemeenten
            ),
            unsafe_allow_html=True,
        )
        st.map(
            ongevallen.assign(kleur=ongevallen["gemeente"].astype(str).map(kleur_van)),
            latitude="lat",
            longitude="lon",
            color="kleur",
            size=3,
        )
        st.dataframe(
            ongevallen[["gemeente", "jaar", "maand", "uur", "class_accident", "x_lambert72", "y_lambert72", "lat", "lon"]],
            width="stretch",
            hide_index=True,
            column_config={c: st.column_config.NumberColumn(format="%.1f") for c in ("x_lambert72", "y_lambert72")}
            | {c: st.column_config.NumberColumn(format="%.6f") for c in ("lat", "lon")},
        )
        toon_dekking(
            len(ongevallen),
            len(ongevallen),
            "dit zwarte punt",
            "Alle ongevallen die bij dit zwarte punt tellen staan op de kaart en in de tabel.",
        )

    st.divider()
    st.header("Veel ongevallen op exact dezelfde coördinaat")
    st.markdown(
        "Plekken waar minstens 10 ongevallen precies dezelfde coördinaat hebben, in de periode van het tabblad Kaart (alle types). "
        "Dat kan een echt gevaarlijke plek zijn, maar ook een standaardcoördinaat (bv. een stadhuis) die een politiezone "
        "invult als de plaats niet bekend is. Klik een rij aan om het hele zwarte punt te zien, met een tabel per coördinaat."
    )
    jaren_c = jaren  # dezelfde periode als in het tabblad Kaart (pijltjes)
    st.caption(f"Periode: {jaren_c[0]}-{jaren_c[1]} (kies ze met de pijltjes in het tabblad Kaart).")
    punten_c = df[df["jaar"].between(*jaren_c) & df["coord_status"].isin(["ok", "hersteld"])].reset_index(drop=True)
    code_c = punten_c["class_accident_code"].astype(int)
    punten_c = punten_c.assign(punt=punten_per_ongeval(punten_c), licht=code_c.isin(LICHT), zwaar=code_c.isin(ZWAAR), doden=code_c.isin(DODEN))
    plekken = (
        punten_c.groupby(["x_lambert72", "y_lambert72"])
        .agg(
            gemeente=("gemeente", lambda s: str(s.value_counts().index[0])),
            ongevallen=("jaar", "size"),
            jaren=("jaar", lambda s: ", ".join(str(j) for j in sorted(s.unique()))),
            licht=("licht", "sum"),
            zwaar=("zwaar", "sum"),
            doden=("doden", "sum"),
            score=("punt", "sum"),
            lat=("lat", "first"),
            lon=("lon", "first"),
        )
        .query("ongevallen >= 10")
        .sort_values("ongevallen", ascending=False)
        .reset_index()
    )
    plekken["score"] = plekken["score"].round(1)
    gekozen_rij = st.dataframe(
        plekken[["gemeente", "ongevallen", "jaren", "licht", "zwaar", "doden", "score", "lat", "lon"]],
        width="stretch",
        height=5 * 35 + 38,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="tabel_samenval",
        column_config={c: st.column_config.NumberColumn(format="%.6f") for c in ("lat", "lon")},
    )
    st.caption(f"{len(plekken)} coördinaten met minstens 10 ongevallen in {jaren_c[0]}-{jaren_c[1]}.")
    rijen_c = gekozen_rij.selection.rows
    if rijen_c and rijen_c[0] < len(plekken):
        gekozen_nr = rijen_c[0]
    else:  # nog niets aangeklikt: standaard de grootste plek in Antwerpen (indien aanwezig), anders de eerste
        antwerpen = np.flatnonzero((plekken["gemeente"] == "Antwerpen").to_numpy())
        gekozen_nr = int(antwerpen[0]) if len(antwerpen) else 0
        st.caption("Standaard getoond: de grootste plek in Antwerpen. Klik een rij aan om een andere plek te zien.")
    if len(plekken):
        rij = plekken.iloc[gekozen_nr]
        # Het hele zwarte punt: zoeken met de instellingen van het tandwiel in Kaart, over de gekozen jaren
        cluster_c = zoek_clusters(
            punten_c["x_lambert72"].to_numpy(), punten_c["y_lambert72"].to_numpy(), gewichten(punten_c), straal, drempel
        )
        hier = (punten_c["x_lambert72"] == rij["x_lambert72"]) & (punten_c["y_lambert72"] == rij["y_lambert72"])
        nummers = set(cluster_c[hier.to_numpy()]) - {-1}
        if not nummers:
            st.info(
                "Deze plek is geen zwart punt met de instellingen van het tandwiel in Kaart (afstand, drempel). "
                "Hieronder staan enkel de ongevallen op deze coördinaat."
            )
            leden_c = punten_c[hier]
        else:
            nummer = max(nummers, key=lambda n: (cluster_c == n).sum())
            leden_c = punten_c[cluster_c == nummer]
            st.markdown(
                f"**Zwart punt met {len(leden_c)} ongevallen, score {leden_c['punt'].sum():.0f}** "
                f"(afstand {straal} m, {jaren_c[0]}-{jaren_c[1]}), op {leden_c.groupby(['x_lambert72', 'y_lambert72']).ngroups} "
                "verschillende coördinaten."
            )
        st.map(leden_c.drop_duplicates(["x_lambert72", "y_lambert72"]), latitude="lat", longitude="lon", size=5, zoom=17)
        per_coord = sorted(leden_c.groupby(["x_lambert72", "y_lambert72"]), key=lambda kv: -len(kv[1]))
        for (x, y), deel in per_coord:
            with st.expander(f"Coördinaat ({deel['lat'].iloc[0]:.6f}, {deel['lon'].iloc[0]:.6f}): {len(deel)} ongevallen", expanded=len(per_coord) <= 3):
                st.dataframe(
                    deel.sort_values(["jaar", "maand", "uur"])[["gemeente", "jaar", "maand", "uur", "class_accident", "x_lambert72", "y_lambert72"]],
                    width="stretch",
                    hide_index=True,
                    height=5 * 35 + 38,
                    column_config={c: st.column_config.NumberColumn(format="%.1f") for c in ("x_lambert72", "y_lambert72")},
                )
        toon_dekking(len(leden_c), len(leden_c), "dit zwarte punt", "Alle ongevallen van dit zwarte punt staan in de tabellen.")
