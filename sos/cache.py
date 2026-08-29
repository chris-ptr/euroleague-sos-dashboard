from pathlib import Path
from typing import Tuple

import pandas as pd
from euroleague_api.boxscore_data import BoxScoreData

from .compute import (
    compute_team_ratings_up_to_round,
    compute_sos_from_netrtg_up_to_round,
    compute_sos_from_winpct_up_to_round,
)
from .fourfactors import compute_four_factors_up_to_round


def _make_parquet_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns that can't be written to Parquet (nested objects like DataFrames),
    and convert Path-like objects to strings if needed.
    """
    df = df.copy()

    bad_cols = []
    for c in df.columns:
        if df[c].dtype == "object":
            if df[c].apply(lambda x: isinstance(x, (pd.DataFrame, dict, list, tuple, set))).any():
                bad_cols.append(c)

    if bad_cols:
        df = df.drop(columns=bad_cols)

    for c in df.columns:
        if df[c].dtype == "object":
            if df[c].apply(lambda x: hasattr(x, "__fspath__")).any():
                df[c] = df[c].astype(str)

    return df


def round_cache_path(cache_dir: Path, round_max: int) -> Path:
    return Path(cache_dir) / f"round_{int(round_max)}.parquet"


# Every table a cache file has to contain to be usable. Adding a table here
# retires every parquet written before it existed: such a file reads back as a
# miss, so compute_for_round recomputes and overwrites it rather than handing
# callers a round that is silently missing a view.
REQUIRED_TABLES = ("team_ratings", "sos_net", "sos_win", "four_factors")


def load_cached_round(
    cache_dir: Path, round_max: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """
    Return (team_ratings, sos_net, sos_win, four_factors) from an existing cache
    file, or None if it is absent or predates one of the tables.
    """
    cache_file = round_cache_path(cache_dir, round_max)
    if not cache_file.exists():
        return None

    df = pd.read_parquet(cache_file)
    tables = {
        name: df[df["TYPE"] == name].drop(columns="TYPE")
        for name in REQUIRED_TABLES
    }
    if any(table.empty for table in tables.values()):
        return None

    # Each table carries only its own columns; the others come back as all-NaN
    # padding from the concat, and would otherwise collide with real column
    # names downstream (every table has a TEAM_NAME, only one has NetRtg).
    return tuple(table.dropna(axis=1, how="all") for table in tables.values())


def compute_for_round(
    cache_dir: Path,
    games_meta: pd.DataFrame,
    season: int,
    boxscore_api: BoxScoreData,
    round_max: int,
    force: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute (or load from cache) team ratings, SOS tables and Four Factors
    through `round_max`.

    Caches the result as a single parquet file per round under `cache_dir`,
    tagged by a TYPE column so all the tables share one file.
    """
    round_max = int(round_max)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not force:
        cached = load_cached_round(cache_dir, round_max)
        if cached is not None:
            return cached

    team_ratings = compute_team_ratings_up_to_round(
        games_meta=games_meta,
        season=season,
        boxscore_api=boxscore_api,
        round_max=round_max,
    )

    sos_net = compute_sos_from_netrtg_up_to_round(
        games_meta=games_meta,
        team_ratings=team_ratings,
        round_max=round_max,
    )

    sos_win = compute_sos_from_winpct_up_to_round(
        games_meta=games_meta,
        round_max=round_max,
    )

    # Re-reads the same box scores compute_team_ratings_up_to_round just read.
    # Free against the on-disk store scripts/recompute_all.py uses; against a
    # bare BoxScoreData it doubles the fetches, which is why that script is the
    # documented way to rebuild the cache.
    four_factors = compute_four_factors_up_to_round(
        games_meta=games_meta,
        season=season,
        boxscore_api=boxscore_api,
        round_max=round_max,
    )

    team_ratings_safe = _make_parquet_safe(team_ratings)
    sos_net_safe = _make_parquet_safe(sos_net)
    sos_win_safe = _make_parquet_safe(sos_win)
    four_factors_safe = _make_parquet_safe(four_factors)

    out = pd.concat(
        [
            team_ratings_safe.assign(TYPE="team_ratings"),
            sos_net_safe.assign(TYPE="sos_net"),
            sos_win_safe.assign(TYPE="sos_win"),
            four_factors_safe.assign(TYPE="four_factors"),
        ],
        ignore_index=True,
    )

    out.to_parquet(round_cache_path(cache_dir, round_max), index=False)

    return team_ratings, sos_net, sos_win, four_factors
