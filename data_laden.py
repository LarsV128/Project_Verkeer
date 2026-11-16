"""De opgeschoonde data: waar ze staat, of ze opnieuw gemaakt moet worden, en inlezen. Bevat geen Streamlit.

Het opschonen zelf staat in clean_data.py (data/raw -> data). Dit bestand beslist enkel *wanneer* het nodig is.
"""
from pathlib import Path

import pandas as pd

import clean_data

DATA = Path(__file__).parent / "data" / "OPENDATA_MAP_2017-2024_clean.parquet"

# Op Streamlit Cloud is er geen data/raw: de ruwe data staat niet in git, de parquet wel.
RAW_AANWEZIG = clean_data.RAW.exists()


def opschonen_nodig():
    """Ruwe data aanwezig, en de parquet ontbreekt of is ouder dan de ruwe data of het cleaningscript?

    Zonder ruwe data gebruiken we de parquet zoals ze is (bestandstijden na een git clone zeggen niets)."""
    if not RAW_AANWEZIG:
        return False
    if not DATA.exists():
        return True
    nieuwste_bron = max(clean_data.RAW.stat().st_mtime, Path(clean_data.__file__).stat().st_mtime)
    return DATA.stat().st_mtime < nieuwste_bron


def lees_data():
    """Leest de opgeschoonde parquet in."""
    return pd.read_parquet(DATA)
