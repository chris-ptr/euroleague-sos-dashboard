from functools import lru_cache
from typing import Tuple, List

import pandas as pd

from euroleague_api.boxscore_data import BoxScoreData
from euroleague_api.game_metadata import GameMetadata
from euroleague_api.schedule import Schedule

from .utils import normalize_team_name


def load_games_metadata(
    season: int,
    competition_code: str = "E",
    cols_to_keep: List[str] | None = None,
) -> Tuple[pd.DataFrame, BoxScoreData, GameMetadata]:
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
    boxscore_api : BoxScoreData
        Used to derive per-game possessions for NetRtg (TeamStats has no
        single-game granularity in euroleague_api >=0.1.0).
    metadata_api : GameMetadata
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
    boxscore_api = BoxScoreData(competition_code)
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

    return games_meta, boxscore_api, metadata_api


def clean_api_schedule(df: pd.DataFrame, regular_season_only: bool = True) -> pd.DataFrame:
    """
    Normalize a raw `Schedule.get_schedule()` frame into the shape
    `build_next_n_games_per_team` expects: Round, DateTime, Home_Team, Away_Team.

    Split out from `load_schedule_from_api` so the reshaping can be tested
    without hitting the network.
    """
    df = df.copy()

    if regular_season_only:
        # Playoffs/Final Four keep numbering past the Regular Season length but
        # have no fixed future fixtures to forecast, so they're dropped here.
        df = df[df["round"] == "RS"].copy()

    df["Round"] = df["gameday"].astype(int)

    # e.g. "Sep 30, 2025" + "20:45" (local tip-off, same basis as the old CSV)
    df["DateTime"] = pd.to_datetime(
        df["date"].str.strip() + " " + df["startime"].str.strip(),
        format="%b %d, %Y %H:%M",
        errors="coerce",
    )

    # Standardize team names
    df["Home_Team"] = df["hometeam"].apply(normalize_team_name)
    df["Away_Team"] = df["awayteam"].apply(normalize_team_name)

    # Sort games by round and tip-off time
    return df.sort_values(["Round", "DateTime"]).reset_index(drop=True)


@lru_cache(maxsize=4)
def _fetch_raw_schedule(season: int, competition_code: str) -> pd.DataFrame:
    """
    Cached raw schedule fetch — one season's schedule is one HTTP call.

    build_round_artifacts builds a Next-N table per N per round, so an
    uncached fetch would re-download the same schedule hundreds of times in a
    single publish run. Callers only ever read the result (clean_api_schedule
    copies before touching it), so sharing the frame is safe. Cached for the
    process lifetime, so a long-lived process won't see mid-run fixture moves.
    """
    return Schedule(competition_code).get_schedule(season)


def load_schedule_from_api(
    season: int,
    competition_code: str = "E",
    regular_season_only: bool = True,
) -> pd.DataFrame:
    """
    Fetch and normalize the seasonal schedule for SOS forecasting.

    Replaces the hand-maintained EL_*_RS_Schedule.csv: the API schedule updates
    itself as fixtures move, so postponed games and home/away swaps stay correct
    without anyone re-exporting a CSV each season.
    """
    raw = _fetch_raw_schedule(season, competition_code)
    return clean_api_schedule(raw, regular_season_only=regular_season_only)
