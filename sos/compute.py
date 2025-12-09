from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from euroleague_api.team_stats import TeamStats

from .data import load_clean_schedule
from .utils import normalize_team_name, team_to_logo_path


def compute_team_net_rating(
    team_name: str,
    season: int,
    games_meta: pd.DataFrame,
    team_stats_api: TeamStats,
) -> dict:
    """
    Compute season-to-date OffRtg, DefRtg, NetRtg for a single team,
    using a single possession count per team.

    Uses:
      - games_meta (hometeam/awayteam/homescore/awayscore/gameCode)
      - team_stats_api.get_team_advanced_stats_single_game(season, gamecode)
        where df_adv: row 0 = home, row 1 = away, col 3 = team possessions.
    """
    df = games_meta.copy()
    df = df[
        df["homescore"].notna() &
        df["awayscore"].notna()
    ]

    df = df[
        (df["hometeam"] == team_name) |
        (df["awayteam"] == team_name)
    ].reset_index(drop=True)

    per_game_rows = []
    total_pts_for = 0.0
    total_pts_against = 0.0
    total_poss = 0.0

    for _, row in df.iterrows():
        gamecode = row.get("gameCode") or row.get("gamecode")
        if pd.isna(gamecode):
            continue

        try:
            adv = team_stats_api.get_team_advanced_stats_single_game(season, gamecode)
        except Exception:
            continue
        if adv is None or adv.empty:
            continue

        if row["hometeam"] == team_name:
            idx = 0
            pts_for = float(row["homescore"])
            pts_against = float(row["awayscore"])
        else:
            idx = 1
            pts_for = float(row["awayscore"])
            pts_against = float(row["homescore"])

        poss = float(adv.iloc[idx, 3])
        if poss <= 0 or np.isnan(poss):
            continue

        off = pts_for / poss * 100.0
        deff = pts_against / poss * 100.0
        net = off - deff

        total_pts_for += pts_for
        total_pts_against += pts_against
        total_poss += poss

        per_game_rows.append(
            {
                "gameCode": gamecode,
                "pts_for": pts_for,
                "pts_against": pts_against,
                "poss": poss,
                "OffRtg": off,
                "DefRtg": deff,
                "NetRtg": net,
            }
        )

    if total_poss > 0:
        season_offrtg = total_pts_for / total_poss * 100.0
        season_defrtg = total_pts_against / total_poss * 100.0
        season_netrtg = season_offrtg - season_defrtg
    else:
        season_offrtg = np.nan
        season_defrtg = np.nan
        season_netrtg = np.nan

    per_game_df = pd.DataFrame(per_game_rows)

    return {
        "team": team_name,
        "season": season,
        "OffRtg": season_offrtg,
        "DefRtg": season_defrtg,
        "NetRtg": season_netrtg,
        "games_played": len(per_game_df),
        "per_game": per_game_df,
        "total_pts_for": total_pts_for,
        "total_pts_against": total_pts_against,
        "total_poss": total_poss,
    }


def compute_team_ratings_up_to_round(
    games_meta: pd.DataFrame,
    season: int,
    team_stats_api: TeamStats,
    round_max: int,
    round_col: str = "Round",
) -> pd.DataFrame:
    """
    Compute OffRtg / DefRtg / NetRtg for all teams using games up to and
    including round_max.
    """
    df = games_meta.copy()
    if round_col in df.columns:
        df = df[df[round_col] <= round_max].copy()

    teams = sorted(set(df["hometeam"]).union(df["awayteam"]))
    rows = []
    for team in teams:
        res = compute_team_net_rating(
            team_name=team,
            season=season,
            games_meta=df,
            team_stats_api=team_stats_api,
        )
        rows.append(res)

    team_ratings = pd.DataFrame(rows).rename(
        columns={
            "team": "TEAM_NAME",
            "games_played": "Games",
        }
    )

    return team_ratings


def compute_team_win_pct(games_meta: pd.DataFrame) -> Dict[str, float]:
    """
    Compute basic Win% for each team from games_meta.

    games_meta must contain:
        ['hometeam', 'awayteam', 'homescore', 'awayscore'].
    """
    df = games_meta.copy()
    df = df[
        df["homescore"].notna() &
        df["awayscore"].notna()
    ].reset_index(drop=True)

    teams = sorted(set(df["hometeam"]).union(df["awayteam"]))
    win_pct: Dict[str, float] = {}

    for team in teams:
        g = df[(df["hometeam"] == team) | (df["awayteam"] == team)].copy()
        if g.empty:
            win_pct[team] = np.nan
            continue

        wins = (
            ((g["hometeam"] == team) & (g["homescore"] > g["awayscore"])) |
            ((g["awayteam"] == team) & (g["awayscore"] > g["homescore"]))
        ).sum()

        gp = len(g)
        win_pct[team] = wins / gp if gp > 0 else np.nan

    return win_pct


def compute_sos_from_netrtg_up_to_round(
    games_meta: pd.DataFrame,
    team_ratings: pd.DataFrame,
    round_max: int,
    round_col: str = "Round",
) -> pd.DataFrame:
    """
    Simplified version of your SoS_Net logic:

    SoS_Net(team) = mean(NetRtg of opponents faced up to round_max).
    """
    df = games_meta.copy()
    if round_col in df.columns:
        df = df[df[round_col] <= round_max].copy()
    df = df[
        df["homescore"].notna() &
        df["awayscore"].notna()
    ]

    # Long schedule: one row per (team, opponent, game)
    home_side = df[["gameCode", "hometeam", "awayteam"]].rename(
        columns={"hometeam": "TEAM_NAME", "awayteam": "OPP_NAME"}
    )
    away_side = df[["gameCode", "hometeam", "awayteam"]].rename(
        columns={"awayteam": "TEAM_NAME", "hometeam": "OPP_NAME"}
    )
    schedule_long = pd.concat([home_side, away_side], ignore_index=True)

    net_map = dict(zip(team_ratings["TEAM_NAME"], team_ratings["NetRtg"]))

    schedule_long["OppNetRtg"] = schedule_long["OPP_NAME"].map(net_map)

    sos = (
        schedule_long.groupby("TEAM_NAME", as_index=False)["OppNetRtg"]
        .mean()
        .rename(columns={"OppNetRtg": "SoS_Net"})
    )

    sos = team_ratings[["TEAM_NAME", "Games", "NetRtg"]].merge(
        sos, on="TEAM_NAME", how="left"
    )

    return sos.sort_values("SoS_Net", ascending=False).reset_index(drop=True)


def compute_sos_from_winpct_up_to_round(
    games_meta: pd.DataFrame,
    round_max: int,
    round_col: str = "Round",
) -> pd.DataFrame:
    """
    SoS from Win%:

    SoS(team) = mean(Win% of opponents faced up to round_max).

    Returns:
        DataFrame with columns:
            - TEAM_NAME
            - SoS   (opponents' average win%)
    """
    df = games_meta.copy()
    if round_col in df.columns:
        df = df[df[round_col] <= round_max].copy()

    # keep only games with scores
    df = df[
        df["homescore"].notna() &
        df["awayscore"].notna()
    ].reset_index(drop=True)

    # 1) compute win% per team (using your helper)
    win_pct = compute_team_win_pct(df)   # dict: TEAM_NAME -> win%

    # 2) create long schedule: one row per (team, opponent, game)
    home_side = df[["gameCode", "hometeam", "awayteam"]].rename(
        columns={"hometeam": "TEAM_NAME", "awayteam": "OPP_NAME"}
    )
    away_side = df[["gameCode", "hometeam", "awayteam"]].rename(
        columns={"awayteam": "TEAM_NAME", "hometeam": "OPP_NAME"}
    )
    schedule_long = pd.concat([home_side, away_side], ignore_index=True)

    # 3) map opponent win% and average per TEAM_NAME
    schedule_long["OppWinPct"] = schedule_long["OPP_NAME"].map(win_pct)

    sos = (
        schedule_long.groupby("TEAM_NAME", as_index=False)["OppWinPct"]
        .mean()
        .rename(columns={"OppWinPct": "SoS"})   # <-- name is exactly "SoS"
    )

    # sort (hardest schedule = highest SoS)
    sos = sos.sort_values("SoS", ascending=False).reset_index(drop=True)
    return sos



def build_next_n_games_per_team(
    games: pd.DataFrame,
    current_round: int,
    n_next: int = 5,
) -> Dict[str, pd.DataFrame]:
    """
    For each team, get the next `n_next` games starting from `current_round`
    (inclusive), sorted by Round then DateTime.

    Returns
    -------
    dict(team_name -> DataFrame)
        Each DataFrame has columns: ["Round", "DateTime", "Is_Home", "Opponent"]
    """
    df = games.copy()
    df = df[df["Round"] >= current_round].copy()
    df = df.sort_values(["Round", "DateTime"]).reset_index(drop=True)

    teams = sorted(set(df["Home_Team"]).union(df["Away_Team"]))
    team_to_next: Dict[str, pd.DataFrame] = {}

    for team in teams:
        mask = (df["Home_Team"] == team) | (df["Away_Team"] == team)
        upcoming = df[mask].copy().reset_index(drop=True)

        if upcoming.empty:
            team_to_next[team] = pd.DataFrame(
                columns=["Round", "DateTime", "Is_Home", "Opponent"]
            )
            continue

        upcoming = upcoming.iloc[:n_next].copy()

        def get_opponent(row):
            return row["Away_Team"] if row["Home_Team"] == team else row["Home_Team"]

        def is_home(row):
            return row["Home_Team"] == team

        upcoming["Opponent"] = upcoming.apply(get_opponent, axis=1)
        upcoming["Is_Home"] = upcoming.apply(is_home, axis=1)

        team_to_next[team] = upcoming[
            ["Round", "DateTime", "Is_Home", "Opponent"]
        ].reset_index(drop=True)

    return team_to_next


def compute_sos_net_rating_next5(
    team_to_next_games: Dict[str, pd.DataFrame],
    team_ratings: pd.DataFrame,
) -> Dict[str, float]:
    """
    SoS_Net_nextN(team) = mean(NetRtg of opponents in the next N games).

    `team_ratings` must have columns: ["TEAM_NAME", "NetRtg"].
    """
    net_map = dict(zip(team_ratings["TEAM_NAME"], team_ratings["NetRtg"]))
    sos_net_nextN: Dict[str, float] = {}

    for team, df_next in team_to_next_games.items():
        if df_next.empty:
            sos_net_nextN[team] = float("nan")
            continue

        opp_vals = []
        for _, row in df_next.iterrows():
            opp = row["Opponent"]
            if opp in net_map:
                opp_vals.append(net_map[opp])

        sos_net_nextN[team] = float(sum(opp_vals) / len(opp_vals)) if opp_vals else float("nan")

    return sos_net_nextN


def compute_sos_winpct_next5(
    team_to_next_games: Dict[str, pd.DataFrame],
    team_win_pct: Dict[str, float],
) -> Dict[str, float]:
    """
    SoS_Win_nextN(team) = mean(Win% of opponents in the next N games).

    `team_win_pct` maps "TEAM_NAME" -> Win% in [0, 1].
    """
    sos_win_nextN: Dict[str, float] = {}

    for team, df_next in team_to_next_games.items():
        if df_next.empty:
            sos_win_nextN[team] = float("nan")
            continue

        opp_vals = []
        for _, row in df_next.iterrows():
            opp = row["Opponent"]
            if opp in team_win_pct:
                opp_vals.append(team_win_pct[opp])

        sos_win_nextN[team] = float(sum(opp_vals) / len(opp_vals)) if opp_vals else float("nan")

    return sos_win_nextN


def make_nextN_sos_table(
    current_round: int,
    schedule_path: str | Path,
    games_meta: pd.DataFrame,
    team_ratings: pd.DataFrame,
    n_next: int = 5,
) -> pd.DataFrame:
    """
    Compute Strength of Schedule for the next N games per team (no plotting).

    Uses:
      - normalize_team_name
      - load_clean_schedule
      - build_next_n_games_per_team
      - compute_sos_net_rating_next5
      - compute_team_win_pct
      - compute_sos_winpct_next5

    Returns a DataFrame with columns:
      - Team
      - SoS_Net_nextN
      - SoS_Win_nextN
      - Logo_Path
      - Opp1 ... OppN
      - Opponents   (comma-separated string of up to N opponents)
    """
    # 1) Normalise names in games_meta and ratings
    games_meta = games_meta.copy()
    games_meta["hometeam"] = games_meta["hometeam"].apply(normalize_team_name)
    games_meta["awayteam"] = games_meta["awayteam"].apply(normalize_team_name)

    team_ratings = team_ratings.copy()
    team_ratings["TEAM_NAME"] = team_ratings["TEAM_NAME"].apply(normalize_team_name)

    # 2) Load cleaned schedule
    games_sched = load_clean_schedule(schedule_path)

    # 3) Build next N games per team from current_round
    team_to_next_games = build_next_n_games_per_team(
        games=games_sched,
        current_round=current_round,
        n_next=n_next,
    )

    # 4) SoS based on Net Rating (uses next N games)
    sos_net_nextN = compute_sos_net_rating_next5(
        team_to_next_games=team_to_next_games,
        team_ratings=team_ratings,
    )

    # 5) Win% per team (full season)
    team_win_pct = compute_team_win_pct(games_meta)

    # 6) SoS based on Win% (uses next N games)
    sos_win_nextN = compute_sos_winpct_next5(
        team_to_next_games=team_to_next_games,
        team_win_pct=team_win_pct,
    )

    # 7) Build final table with up to N opponents in wide form
    rows = []
    for team, next_df in team_to_next_games.items():
        if next_df.empty:
            opps = []
        else:
            opps = (
                next_df["Opponent"]
                .astype(str)
                .fillna("")
                .tolist()
            )

        # keep at most N opponents
        opps = [o for o in opps if o][:n_next]

        # pad to length N so we always have Opp1..OppN columns
        while len(opps) < n_next:
            opps.append(None)


        row = {
            "Team": team,
            "SoS_Net_nextN": sos_net_nextN.get(team, float("nan")),
            "SoS_Win_nextN": sos_win_nextN.get(team, float("nan")),
            "Logo_Path": team_to_logo_path(team),  # string or None
            "Opponents": ", ".join([o for o in opps if o]),
        }

        # Opp1..OppN columns
        for idx in range(1, n_next + 1):
            row[f"Opp{idx}"] = opps[idx - 1]

        rows.append(row)

    sos_nextN_df = pd.DataFrame(rows)

    # sort by toughest schedule (highest SoS_Net_nextN) at the top
    sos_nextN_df = sos_nextN_df.sort_values(
        "SoS_Net_nextN", ascending=False
    ).reset_index(drop=True)

    return sos_nextN_df