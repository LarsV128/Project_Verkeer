"""Leest de ruwe ongevallendata uit data/raw/ en schrijft een opgeschoonde versie naar data/.

data/raw/ wordt enkel gelezen, nooit gewijzigd.
Uitvoeren:  python clean_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

# Grenzen van België in Lambert72 (meters), ruim genomen; alles daarbuiten is een invoerfout
X_BEREIK = (14_000, 300_000)
Y_BEREIK = (20_000, 250_000)

MAP = Path(__file__).parent  # paden zijn vast, ongeacht vanuit welke map je het script start
RAW = MAP / "data" / "raw" / "OPENDATA_MAP_2017-2024.txt"
CLEAN = MAP / "data" / "OPENDATA_MAP_2017-2024_clean.parquet"

KOLOMMEN = {
    "DT_YEAR_COLLISION": "jaar",
    "DT_MONTH_COLLISION": "maand",
    "DT_TIME": "uur",
    "CD_NIS": "nis",
    "TX_RGN_COLLISION_NL": "gewest",
    "TX_PROV_COLLISION_NL": "provincie",
    "TX_MUNTY_COLLISION_NL": "gemeente",
    "CD_CLASS_ACCIDENTS": "class_accident_code",
    "TX_CLASS_ACCIDENTS_NL": "class_accident",
    "MS_X_COORD": "x_lambert72",
    "MS_Y_COORD": "y_lambert72",
}


def herstel(waarde, bereik, doel):
    """Zet de komma van een afgeknotte waarde (zoals 7.30077e15) op de plek die in `bereik` valt
    en het dichtst bij `doel` ligt (de mediaan van de andere ongevallen in dezelfde gemeente).
    Geeft None als geen enkele plek past."""
    mantisse = waarde / 10 ** np.floor(np.log10(waarde))  # bv. 7.30077
    kandidaten = [mantisse * 10**k for k in range(3, 7)]
    kandidaten = [k for k in kandidaten if bereik[0] <= k <= bereik[1]]
    return min(kandidaten, key=lambda k: abs(k - doel)) if kandidaten else None


def main():
    # sep='|' en utf-8-sig (het bestand begint met een BOM); alles eerst als tekst inlezen
    df = pd.read_csv(RAW, sep="|", encoding="utf-8-sig", dtype=str, usecols=list(KOLOMMEN))
    df = df.rename(columns=KOLOMMEN)

    df["jaar"] = df["jaar"].astype("int16")
    df["maand"] = df["maand"].astype("int8")
    df["uur"] = pd.to_numeric(df["uur"], errors="coerce").astype("Int8")
    # coördinaten hebben een decimale komma: '149732,99' -> 149732.99
    for kol in ("x_lambert72", "y_lambert72"):
        df[kol] = pd.to_numeric(df[kol].str.replace(",", ".", regex=False), errors="coerce")

    # Coördinaten: 'ontbreekt' (leeg), 'onmogelijk' (buiten België, bv. 1.8e16) of 'ok'.
    ontbreekt = df["x_lambert72"].isna() | df["y_lambert72"].isna()
    in_belgie = df["x_lambert72"].between(*X_BEREIK) & df["y_lambert72"].between(*Y_BEREIK)
    status = pd.Series(np.select([ontbreekt, ~in_belgie], ["ontbreekt", "onmogelijk"], default="ok"), index=df.index)

    # Onmogelijke waarden herstellen (zie herstel()). De originele waarden bewaren we in *_bron.
    df["x_bron"] = np.nan
    df["y_bron"] = np.nan
    for i in df.index[status == "onmogelijk"]:
        referentie = df[(status == "ok") & (df["nis"] == df.at[i, "nis"])]
        x = herstel(df.at[i, "x_lambert72"], X_BEREIK, referentie["x_lambert72"].median())
        y = herstel(df.at[i, "y_lambert72"], Y_BEREIK, referentie["y_lambert72"].median())
        if x is not None and y is not None:
            df.loc[i, ["x_bron", "y_bron"]] = df.loc[i, ["x_lambert72", "y_lambert72"]].to_numpy()
            df.loc[i, ["x_lambert72", "y_lambert72"]] = [x, y]
            status[i] = "hersteld"
            print(f"hersteld: rij {i}, {df.at[i, 'gemeente']}: X={x:.1f} Y={y:.1f}")
    df["coord_status"] = status.astype("category")

    # Lambert72 (EPSG:31370, meters) -> breedtegraad/lengtegraad (EPSG:4326) voor de kaart
    ok = df["coord_status"].isin(["ok", "hersteld"])
    naar_gps = Transformer.from_crs(31370, 4326, always_xy=True)
    lon, lat = naar_gps.transform(df.loc[ok, "x_lambert72"].to_numpy(), df.loc[ok, "y_lambert72"].to_numpy())
    df["lat"] = np.nan
    df["lon"] = np.nan
    df.loc[ok, "lat"] = lat
    df.loc[ok, "lon"] = lon

    # Brussel is een gewest zonder provincie: expliciet label in plaats van een lege waarde
    df["provincie"] = df["provincie"].str.removeprefix("Provincie ")
    df["provincie"] = df["provincie"].fillna("Geen provincie (Brussel)")

    # labels uit de bron beginnen soms met een kleine letter
    df["class_accident"] = df["class_accident"].str.strip().str.capitalize()

    for kol in ("gewest", "provincie", "gemeente", "class_accident"):
        df[kol] = df[kol].astype("category")

    CLEAN.parent.mkdir(exist_ok=True)
    df.to_parquet(CLEAN, index=False)
    print(f"{len(df):,} rijen -> {CLEAN}")


if __name__ == "__main__":
    main()
