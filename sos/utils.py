import os
import base64
from typing import Optional

import pandas as pd

from .config import TEAM_LOGO_DIR


# 1) TEAM_NAME normalization (from your notebook)
TEAM_NAME_MAP = {
    "MACCABI RAPYD TEL AVIV": "MACCABI TEL AVIV",
    "KOSNER BASKONIA VITORIA-GASTEIZ": "BASKONIA VITORIA-GASTEIZ",
    # add more aliases here if you want (e.g. Baskonia sponsor variants)
}


def normalize_team_name(name: str) -> str:
    """Uppercase and normalise known variants."""
    if pd.isna(name):
        return name
    name = str(name).strip().upper()
    return TEAM_NAME_MAP.get(name, name)


# 2) Map TEAM_NAME -> logo filename (your dict)
team_logo_file = {
    "ANADOLU EFES ISTANBUL": "Anadolu Efes_logo.png",
    "KOSNER BASKONIA VITORIA-GASTEIZ": "Baskonia_logo.png",
    "BASKONIA VITORIA-GASTEIZ": "Baskonia_logo.png",
    "CRVENA ZVEZDA MERIDIANBET BELGRADE": "Crvena Zvezda_logo.png",
    "DUBAI BASKETBALL": "Dubai_logo.png",
    "HAPOEL IBI TEL AVIV": "Hapoel Tel Aviv_logo.png",
    "PANATHINAIKOS AKTOR ATHENS": "Panathinaikos_logo.png",
    "VIRTUS BOLOGNA": "Virtus_logo.png",
    "FENERBAHCE BEKO ISTANBUL": "Fenerbahce_logo.png",
    "AS MONACO": "Monaco_logo.png",
    "LDLC ASVEL VILLEURBANNE": "ASVEL_logo.png",
    "MACCABI RAPYD TEL AVIV": "Maccabi_logo.png",
    "MACCABI TEL AVIV": "Maccabi_logo.png",
    "OLYMPIACOS PIRAEUS": "Olympiacos_logo.png",
    "EA7 EMPORIO ARMANI MILAN": "Milan_logo.png",
    "PARTIZAN MOZZART BET BELGRADE": "Partizan_logo.png",
    "FC BARCELONA": "Barcelona_logo.png",
    "FC BAYERN MUNICH": "Bayern_logo.png",
    "REAL MADRID": "Real_logo.png",
    "PARIS BASKETBALL": "Paris_logo.png",
    "ZALGIRIS KAUNAS": "Zalgiris_logo.png",
    "VALENCIA BASKET": "Valencia_logo.png",
}

# base dir (string)
logos_dir = str(TEAM_LOGO_DIR)  # TEAM_LOGO_DIR is "team_logos" in config.py


def team_to_logo_path(team_name: str) -> Optional[str]:
    """
    Map team name -> logo path as STRING (like your notebook).

    We normalise first so "MACCABI RAPYD TEL AVIV" becomes "MACCABI TEL AVIV",
    then look it up in team_logo_file.
    """
    key = normalize_team_name(team_name)
    fname = team_logo_file.get(key)
    if fname is None:
        return None
    path = os.path.join(logos_dir, fname)
    return path if os.path.exists(path) else None


def logo_to_dataurl(path: str | None) -> str | None:
    """Convert a local image file to a data URL for embedding in Altair exports."""
    if not path or not isinstance(path, str):
        return None
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    # Use generic PNG MIME; works fine for most logos
    return f"data:image/png;base64,{b64}"
