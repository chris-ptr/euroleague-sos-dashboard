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
    Fetch and clean EuroLeague season metadata via API wrappers.

    Parameters
    ----------
    season : int
        Start year of the season (e.g., 2025).
    competition_code : str
        Competition identifier ("E" for EuroLeague).

    Returns
    -------
    games_meta : pd.DataFrame
        Cleaned game records (rounds, scores, teams).
    team_stats_api : TeamStats
    metadata_api   : GameMetadata
    """
    if cols_to_keep is None:
        # Default columns required for SOS calculations
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

    # API helper initialization
    team_stats_api = TeamStats(competition_code)
    metadata_api = GameMetadata(competition_code)

    # Fetch raw seasonal data
    games_meta = metadata_api.get_gamecodes_season(season)

    # Validate and filter columns
    missing = [c for c in cols_to_keep if c not in games_meta.columns]
    if missing:
        raise ValueError(f"Missing columns in API response: {missing}")

    games_meta = games_meta[cols_to_keep].copy()

    # Standardize team names for mapping
    games_meta["hometeam"] = games_meta["hometeam"].apply(normalize_team_name)
    games_meta["awayteam"] = games_meta["awayteam"].apply(normalize_team_name)

    # Ensure canonical game identifier exists
    if "gameCode" not in games_meta.columns and "gamecode" in games_meta.columns:
        games_meta["gameCode"] = games_meta["gamecode"]

    return games_meta, team_stats_api, metadata_api


def load_clean_schedule(csv_path: str | Path) -> pd.DataFrame:
    """
    Load and normalize the seasonal schedule for SOS forecasting.

    Expected CSV columns: Round, Date, Local_Time, GMT_Time, Home_Team, Away_Team.
    """
    df = pd.read_csv(csv_path)

    # Clean missing values
    df = df.replace({"<NA>": pd.NA, "nan": pd.NA, "None": pd.NA})

    # Convert to datetime for chronological sorting
    df["DateTime"] = pd.to_datetime(
        df["Date"] + " " + df["Local_Time"],
        errors="coerce",
    )

    # Sort games by round and tip-off time
    df = df.sort_values(["Round", "DateTime"]).reset_index(drop=True)

    # Standardize team names
    df["Home_Team"] = df["Home_Team"].apply(normalize_team_name)
    df["Away_Team"] = df["Away_Team"].apply(normalize_team_name)

    return df