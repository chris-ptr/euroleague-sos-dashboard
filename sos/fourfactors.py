"""
Dean Oliver's Four Factors, per team, from the box scores the ratings already use.

This is a *team profile*, deliberately kept apart from the SoS pipeline: nothing
here feeds compute_sos_from_netrtg_up_to_round or the Next-N forecast, and the
existing charts are untouched. The Four Factors view sits beside them as its own
tab, answering "what kind of team is this" rather than "how hard is the schedule".

Both sides of a game are in one box score (the Home=1 and Home=0 "Total" rows),
so a team's own factors and the ones it allows come out of the same fetch — and
out of the same on-disk cache scripts/recompute_all.py already fills. No extra
API surface, no second data source.

Factors (offense; the defensive four are the identical formulas applied to the
opponent's totals):

    eFG%  = (FGM + 0.5 * 3PM) / FGA
    TOV%  = TOV / (FGA + 0.44 * FTA + TOV)
    OREB% = OREB / (OREB + Opponent DREB)
    FTR   = FTM / FGA

Ratios are computed from summed season totals, not averaged per game, so a
blowout with 90 shot attempts counts for more than a rock fight with 60 —
the same volume weighting compute_team_net_rating uses for possessions.
"""
from typing import Dict

import numpy as np
import pandas as pd

from euroleague_api.boxscore_data import BoxScoreData


# Oliver's relative importance of the four factors. Used to collapse the eight
# per-factor numbers into one rating per side; see add_four_factor_composites.
FOUR_FACTOR_WEIGHTS = {
    "eFG": 0.40,
    "TOV": 0.25,
    "REB": 0.20,
    "FT": 0.15,
}

# Box-score fields summed per side, per game. Named here rather than inline so
# the team and opponent passes provably read the same columns.
_TOTAL_FIELDS = (
    "FieldGoalsMade2",
    "FieldGoalsAttempted2",
    "FieldGoalsMade3",
    "FieldGoalsAttempted3",
    "FreeThrowsMade",
    "FreeThrowsAttempted",
    "OffensiveRebounds",
    "DefensiveRebounds",
    "Turnovers",
)


def _safe_div(numerator: float, denominator: float) -> float:
    """Ratio, or NaN when the denominator is zero/absent — never a ZeroDivisionError."""
    if denominator is None or denominator <= 0 or np.isnan(denominator):
        return np.nan
    return float(numerator) / float(denominator)


def _side_totals(box: pd.DataFrame, home: bool) -> Dict[str, float] | None:
    """
    The "Total" row for one side of a game, as a plain dict of floats.

    Returns None when the row is missing, which callers treat as "skip this
    game" — the same stance compute_team_net_rating takes for an unusable box.
    """
    rows = box[(box["Player"] == "Total") & (box["Home"] == int(home))]
    if rows.empty:
        return None

    row = rows.iloc[0]
    try:
        return {field: float(row[field]) for field in _TOTAL_FIELDS}
    except (KeyError, TypeError, ValueError):
        return None


def four_factors_from_totals(team: Dict[str, float], opponent: Dict[str, float]) -> Dict[str, float]:
    """
    The eight factors from one team's summed totals and its opponents' summed totals.

    Split out from the aggregation loop so the arithmetic can be tested against
    hand-computed numbers without a box score or an API in sight.

    DREB% is the mirror of the opponent's OREB%: the share of the opponent's
    misses this team cleaned up. It is 1 - OppOREB% by construction, and is
    reported directly because "we grab 72% of available defensive boards" reads
    better on a defensive row than "opponents grab 28% of theirs".
    """
    team_fga = team["FieldGoalsAttempted2"] + team["FieldGoalsAttempted3"]
    team_fgm = team["FieldGoalsMade2"] + team["FieldGoalsMade3"]
    opp_fga = opponent["FieldGoalsAttempted2"] + opponent["FieldGoalsAttempted3"]
    opp_fgm = opponent["FieldGoalsMade2"] + opponent["FieldGoalsMade3"]

    return {
        # Offense: what the team did with the ball.
        "eFG": _safe_div(team_fgm + 0.5 * team["FieldGoalsMade3"], team_fga),
        "TOV_Rate": _safe_div(
            team["Turnovers"],
            team_fga + 0.44 * team["FreeThrowsAttempted"] + team["Turnovers"],
        ),
        "OREB_Pct": _safe_div(
            team["OffensiveRebounds"],
            team["OffensiveRebounds"] + opponent["DefensiveRebounds"],
        ),
        "FT_Rate": _safe_div(team["FreeThrowsMade"], team_fga),

        # Defense: the same four factors, allowed.
        "Opp_eFG": _safe_div(opp_fgm + 0.5 * opponent["FieldGoalsMade3"], opp_fga),
        "Opp_TOV_Rate": _safe_div(
            opponent["Turnovers"],
            opp_fga + 0.44 * opponent["FreeThrowsAttempted"] + opponent["Turnovers"],
        ),
        "DREB_Pct": _safe_div(
            team["DefensiveRebounds"],
            team["DefensiveRebounds"] + opponent["OffensiveRebounds"],
        ),
        "Opp_FT_Rate": _safe_div(opponent["FreeThrowsMade"], opp_fga),
    }


def compute_team_four_factors(
    team_name: str,
    season: int,
    games_meta: pd.DataFrame,
    boxscore_api: BoxScoreData,
) -> dict:
    """
    Season-to-date Four Factors for one team, plus the four it allows.

    `games_meta` is expected to be pre-filtered to the rounds in scope — same
    contract as compute_team_net_rating, which this deliberately mirrors so the
    two can be read side by side.
    """
    df = games_meta.copy()
    df = df[df["homescore"].notna() & df["awayscore"].notna()]
    df = df[
        (df["hometeam"] == team_name) |
        (df["awayteam"] == team_name)
    ].reset_index(drop=True)

    team_totals = {field: 0.0 for field in _TOTAL_FIELDS}
    opp_totals = {field: 0.0 for field in _TOTAL_FIELDS}
    games_counted = 0

    for _, row in df.iterrows():
        gamecode = row.get("gameCode") or row.get("gamecode")
        if pd.isna(gamecode):
            continue

        try:
            box = boxscore_api.get_players_boxscore_stats(season, gamecode)
        except Exception:  # noqa: BLE001 - an unusable game is skipped, as in compute
            continue
        if box is None or box.empty:
            continue

        is_home = row["hometeam"] == team_name
        mine = _side_totals(box, home=is_home)
        theirs = _side_totals(box, home=not is_home)
        # Both sides or neither: a game counted for the team but not the
        # opponent would put OREB% over a denominator missing its other half.
        if mine is None or theirs is None:
            continue

        for field in _TOTAL_FIELDS:
            team_totals[field] += mine[field]
            opp_totals[field] += theirs[field]
        games_counted += 1

    factors = four_factors_from_totals(team_totals, opp_totals)
    return {"TEAM_NAME": team_name, "Games": games_counted, **factors}


def add_four_factor_composites(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the eight factors into one rating per side, using Oliver's weights.

    The four factors are in incompatible units — eFG% spans a few points, FTR is
    a ratio around .25 — so a raw weighted sum would be dominated by whichever
    happens to have the widest numeric spread. Each factor is first standardized
    across the league *at this round* (z-score), then signed so that positive is
    always good for the team, then weighted 40/25/20/15.

    That makes the composite a relative measure by construction: it says how far
    from league-average a team is in units of league spread, so it is comparable
    across teams within a round but not across rounds.

    A factor whose league spread is zero (every team identical, realistically
    only round 1 with a single game each) contributes 0 rather than NaN, which
    would otherwise wipe out the whole composite for every team.
    """
    out = df.copy()

    def z(col: str) -> pd.Series:
        values = pd.to_numeric(out[col], errors="coerce")
        std = values.std(ddof=0)
        if not np.isfinite(std) or std == 0:
            return pd.Series(0.0, index=out.index)
        return (values - values.mean()) / std

    w = FOUR_FACTOR_WEIGHTS

    # Offense: shoot well, don't turn it over, rebound your misses, get to the line.
    out["FF_Off"] = (
        w["eFG"] * z("eFG")
        - w["TOV"] * z("TOV_Rate")
        + w["REB"] * z("OREB_Pct")
        + w["FT"] * z("FT_Rate")
    )

    # Defense: the same four, mirrored — contest shots, force turnovers, close
    # out the defensive glass, and don't send them to the line.
    out["FF_Def"] = (
        -w["eFG"] * z("Opp_eFG")
        + w["TOV"] * z("Opp_TOV_Rate")
        + w["REB"] * z("DREB_Pct")
        - w["FT"] * z("Opp_FT_Rate")
    )

    # Both sides are already signed "higher is better", so the overall rating
    # adds rather than subtracts them — unlike NetRtg, where DefRtg is a raw
    # points-allowed figure that has to be taken away.
    out["FF_Net"] = out["FF_Off"] + out["FF_Def"]

    return out


def compute_four_factors_up_to_round(
    games_meta: pd.DataFrame,
    season: int,
    boxscore_api: BoxScoreData,
    round_max: int,
    round_col: str = "Round",
) -> pd.DataFrame:
    """League-wide Four Factors through a specific round, ranked by overall rating."""
    df = games_meta.copy()
    if round_col in df.columns:
        df = df[df[round_col] <= round_max].copy()

    teams = sorted(set(df["hometeam"]).union(df["awayteam"]))
    rows = [
        compute_team_four_factors(
            team_name=team,
            season=season,
            games_meta=df,
            boxscore_api=boxscore_api,
        )
        for team in teams
    ]

    four_factors = add_four_factor_composites(pd.DataFrame(rows))
    return four_factors.sort_values("FF_Net", ascending=False).reset_index(drop=True)
