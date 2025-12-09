from pathlib import Path
from typing import Tuple, List

import pandas as pd

from euroleague_api.team_stats import TeamStats
from euroleague_api.game_metadata import GameMetadata

from .utils import normalize_team_name


def load_games_metadata(
    season: int,
    competition_code: str = "E",
    cols_to_keep: List[str] | None = None,
) -> Tuple[pd.DataFrame, TeamStats, GameMetadata]:
    """
    Load EuroLeague season metadata using your API wrappers
    (TeamStats, GameMetadata) and return a cleaned `games_meta` DataFrame.

    Parameters
    ----------
    season : int
        Start year of the season (e.g., 2025 for 2025–26).
    competition_code : str
        "E" for EuroLeague, "U" for EuroCup, etc.

    Returns
    -------
    games_meta : pd.DataFrame
        Columns at least: Round, date, gamecode/gameCode, hometeam, homescore, awayteam, awayscore
    team_stats_api : TeamStats
    metadata_api   : GameMetadata
    """
    if cols_to_keep is None:
        # From your notebook
        cols_to_keep = [
            "Round",
            "date",
            "gamecode",
            "gameCode",
            "hometeam",
            "homecode",
            "homescore",
            "awayteam",
            "awaycode",
            "awayscore",
        ]

    # Initialize API helpers
    team_stats_api = TeamStats(competition_code)
    metadata_api = GameMetadata(competition_code)

    # Load raw metadata from your wrapper
    games_meta = metadata_api.get_gamecodes_season(season)

    # Keep only needed columns
    missing = [c for c in cols_to_keep if c not in games_meta.columns]
    if missing:
        raise ValueError(f"Missing columns in API response: {missing}")

    games_meta = games_meta[cols_to_keep].copy()

    # Normalise team names
    games_meta["hometeam"] = games_meta["hometeam"].apply(normalize_team_name)
    games_meta["awayteam"] = games_meta["awayteam"].apply(normalize_team_name)

    # Some APIs use both gamecode and gameCode; make sure we have a canonical one
    if "gameCode" not in games_meta.columns and "gamecode" in games_meta.columns:
        games_meta["gameCode"] = games_meta["gamecode"]

    return games_meta, team_stats_api, metadata_api


def load_clean_schedule(csv_path: str | Path) -> pd.DataFrame:
    """
    Load the saved 2025–26 schedule and get it ready for next-N-games logic.

    Expected columns in CSV:
        Round, Date, Local_Time, GMT_Time, Home_Team, Away_Team

    Returns
    -------
    DataFrame with a parsed DateTime column and sorted by Round, DateTime.
    """
    df = pd.read_csv(csv_path)

    # Normalise NA-ish values
    df = df.replace({"<NA>": pd.NA, "nan": pd.NA, "None": pd.NA})

    # Parse local tipoff datetime, e.g. "Tuesday, 30 September 2025 20:00"
    df["DateTime"] = pd.to_datetime(
        df["Date"] + " " + df["Local_Time"],
        errors="coerce",
    )

    # Sort for deterministic behaviour
    df = df.sort_values(["Round", "DateTime"]).reset_index(drop=True)

    # Normalise team names
    df["Home_Team"] = df["Home_Team"].apply(normalize_team_name)
    df["Away_Team"] = df["Away_Team"].apply(normalize_team_name)

    return df