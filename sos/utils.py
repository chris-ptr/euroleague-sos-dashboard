import os
import base64
from typing import Optional

import pandas as pd

from .config import TEAM_LOGO_DIR


# Normalized name mapping for consistency across API and schedule sources
TEAM_NAME_MAP = {
    "MACCABI RAPYD TEL AVIV": "MACCABI TEL AVIV",
    "KOSNER BASKONIA VITORIA-GASTEIZ": "BASKONIA VITORIA-GASTEIZ",
}

# 3-letter abbreviations for mobile/compact views
TEAM_ABBREV_MAP = {
    "ANADOLU EFES ISTANBUL": "EFE",
    "FENERBAHCE BEKO ISTANBUL": "FEN",
    "FC BARCELONA": "FCB",
    "REAL MADRID": "RMB",
    "BASKONIA VITORIA-GASTEIZ": "BKN",
    "KOSNER BASKONIA VITORIA-GASTEIZ": "BKN",
    "VALENCIA BASKET": "VAL",
    "OLYMPIACOS PIRAEUS": "OLY",
    "PANATHINAIKOS AKTOR ATHENS": "PAO",
    "MACCABI TEL AVIV": "MTA",
    "MACCABI RAPYD TEL AVIV": "MTA",
    "HAPOEL IBI TEL AVIV": "HTA",
    "PARTIZAN MOZZART BET BELGRADE": "PAR",
    "CRVENA ZVEZDA MERIDIANBET BELGRADE": "CZV",
    "VIRTUS BOLOGNA": "VIR",
    "EA7 EMPORIO ARMANI MILAN": "MIL",
    "AS MONACO": "ASM",
    "LDLC ASVEL VILLEURBANNE": "ASV",
    "PARIS BASKETBALL": "PAR",
    "FC BAYERN MUNICH": "BAY",
    "ZALGIRIS KAUNAS": "ZAL",
    "DUBAI BASKETBALL": "DUB",
}

def team_display_name(team: str, mobile: bool) -> str:
    """Return full name or abbreviation based on UI context."""
    if not isinstance(team, str):
        return team
    if mobile:
        return TEAM_ABBREV_MAP.get(team, team[:3].upper())
    return team


def normalize_team_name(name: str) -> str:
    """Standardize casing and apply alias mapping."""
    if pd.isna(name):
        return name
    name = str(name).strip().upper()
    return TEAM_NAME_MAP.get(name, name)


# Static mapping for local logo files
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

logos_dir = str(TEAM_LOGO_DIR)


def team_to_logo_path(team_name: str) -> Optional[str]:
    """Resolve team name to local logo filesystem path."""
    key = normalize_team_name(team_name)
    fname = team_logo_file.get(key)
    if fname is None:
        return None
    path = os.path.join(logos_dir, fname)
    return path if os.path.exists(path) else None


def logo_to_dataurl(path: str | None) -> str | None:
    """Encode local image to base64 DataURL for Altair embedding."""
    if not path or not isinstance(path, str):
        return None
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/png;base64,{b64}"
