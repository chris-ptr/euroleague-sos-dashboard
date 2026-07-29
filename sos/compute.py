from typing import Dict, Tuple

import numpy as np
import pandas as pd

from euroleague_api.boxscore_data import BoxScoreData

from .data import load_schedule_from_api
from .utils import normalize_team_name, team_to_logo_path


def compute_team_net_rating(
    team_name: str,
    season: int,
    games_meta: pd.DataFrame,
    boxscore_api: BoxScoreData,
) -> dict:
    """
    Calculate season-to-date efficiency metrics (OffRtg, DefRtg, NetRtg) for a team.

    Aggregates points and possessions across all games played so far.

    Possessions are estimated per game from box-score totals (TeamStats has
    no single-game granularity in euroleague_api >=0.1.0):
        FGA2 + FGA3 + 0.44*FTA - OREB + TOV
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
            box = boxscore_api.get_players_boxscore_stats(season, gamecode)
        except Exception:
            continue
        if box is None or box.empty:
            continue

        is_home = row["hometeam"] == team_name
        if is_home:
            pts_for = float(row["homescore"])
            pts_against = float(row["awayscore"])
        else:
            pts_for = float(row["awayscore"])
            pts_against = float(row["homescore"])

        totals = box[(box["Player"] == "Total") & (box["Home"] == int(is_home))]
        if totals.empty:
            continue
        t = totals.iloc[0]

        # Estimate possessions from box-score totals (see docstring)
        fga = float(t["FieldGoalsAttempted2"]) + float(t["FieldGoalsAttempted3"])
        poss = (
            fga
            + 0.44 * float(t["FreeThrowsAttempted"])
            - float(t["OffensiveRebounds"])
            + float(t["Turnovers"])
        )
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

    # Aggregate season totals
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
    boxscore_api: BoxScoreData,
    round_max: int,
    round_col: str = "Round",
) -> pd.DataFrame:
    """Compute league-wide efficiency metrics through a specific round."""
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
            boxscore_api=boxscore_api,
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
    """Calculate winning percentage for all teams from game metadata."""
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
    Calculate SOS based on opponent Net Rating averages.
    
    Hardest schedule = highest positive SoS_Net (facing strongest opponents).
    """
    df = games_meta.copy()
    if round_col in df.columns:
        df = df[df[round_col] <= round_max].copy()
    df = df[
        df["homescore"].notna() &
        df["awayscore"].notna()
    ]

    # Reshape schedule to long format (one row per team per game)
    home_side = df[["gameCode", "hometeam", "awayteam"]].rename(
        columns={"hometeam": "TEAM_NAME", "awayteam": "OPP_NAME"}
    )
    away_side = df[["gameCode", "hometeam", "awayteam"]].rename(
        columns={"awayteam": "TEAM_NAME", "hometeam": "OPP_NAME"}
    )
    schedule_long = pd.concat([home_side, away_side], ignore_index=True)

    net_map = dict(zip(team_ratings["TEAM_NAME"], team_ratings["NetRtg"]))

    # Map opponent ratings to the schedule
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
    """Calculate SOS based on opponent winning percentage averages."""
    df = games_meta.copy()
    if round_col in df.columns:
        df = df[df[round_col] <= round_max].copy()

    # Filter for completed games
    df = df[
        df["homescore"].notna() &
        df["awayscore"].notna()
    ].reset_index(drop=True)

    # Get current Win% for all teams
    win_pct = compute_team_win_pct(df)

    # Flatten schedule
    home_side = df[["gameCode", "hometeam", "awayteam"]].rename(
        columns={"hometeam": "TEAM_NAME", "awayteam": "OPP_NAME"}
    )
    away_side = df[["gameCode", "hometeam", "awayteam"]].rename(
        columns={"awayteam": "TEAM_NAME", "hometeam": "OPP_NAME"}
    )
    schedule_long = pd.concat([home_side, away_side], ignore_index=True)

    # Aggregate opponent win rates
    schedule_long["OppWinPct"] = schedule_long["OPP_NAME"].map(win_pct)

    sos = (
        schedule_long.groupby("TEAM_NAME", as_index=False)["OppWinPct"]
        .mean()
        .rename(columns={"OppWinPct": "SoS"})
    )

    return sos.sort_values("SoS", ascending=False).reset_index(drop=True)



def build_next_n_games_per_team(
    games: pd.DataFrame,
    current_round: int,
    n_next: int = 5,
) -> Dict[str, pd.DataFrame]:
    """Extract upcoming N games for each team from the schedule."""
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
    """Forecast SOS using Net Ratings of upcoming opponents."""
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
    """Forecast SOS using Winning Percentages of upcoming opponents."""
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
    season: int,
    games_meta: pd.DataFrame,
    team_ratings: pd.DataFrame,
    n_next: int = 5,
    competition_code: str = "E",
) -> pd.DataFrame:
    """Generate a combined SOS forecasting table for the next N games."""
    # Standardize metadata and ratings
    games_meta = games_meta.copy()
    games_meta["hometeam"] = games_meta["hometeam"].apply(normalize_team_name)
    games_meta["awayteam"] = games_meta["awayteam"].apply(normalize_team_name)

    team_ratings = team_ratings.copy()
    team_ratings["TEAM_NAME"] = team_ratings["TEAM_NAME"].apply(normalize_team_name)

    # Load cleaned schedule data
    games_sched = load_schedule_from_api(season, competition_code)

    # Slice upcoming games
    team_to_next_games = build_next_n_games_per_team(
        games=games_sched,
        current_round=current_round,
        n_next=n_next,
    )

    # Forecast efficiency-based SOS
    sos_net_nextN = compute_sos_net_rating_next5(
        team_to_next_games=team_to_next_games,
        team_ratings=team_ratings,
    )

    # Get baseline win percentages
    team_win_pct = compute_team_win_pct(games_meta)

    # Forecast win-rate-based SOS
    sos_win_nextN = compute_sos_winpct_next5(
        team_to_next_games=team_to_next_games,
        team_win_pct=team_win_pct,
    )

    # Build wide-form results table
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

        # Truncate and pad opponent list
        opps = [o for o in opps if o][:n_next]
        while len(opps) < n_next:
            opps.append(None)


        row = {
            "Team": team,
            "SoS_Net_nextN": sos_net_nextN.get(team, float("nan")),
            "SoS_Win_nextN": sos_win_nextN.get(team, float("nan")),
            "Logo_Path": team_to_logo_path(team),
            "Opponents": ", ".join([o for o in opps if o]),
        }

        # Dynamically add Opp1..OppN columns
        for idx in range(1, n_next + 1):
            row[f"Opp{idx}"] = opps[idx - 1]

        rows.append(row)

    sos_nextN_df = pd.DataFrame(rows)

    # Primary sort by efficiency-based difficulty
    sos_nextN_df = sos_nextN_df.sort_values(
        "SoS_Net_nextN", ascending=False
    ).reset_index(drop=True)

    return sos_nextN_df