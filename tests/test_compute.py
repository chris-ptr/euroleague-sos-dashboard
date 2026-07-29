"""
Tests for sos/compute.py — verify the SOS/rating math against hand-computed
expected values, independent of the live EuroLeague API.
"""

import numpy as np
import pandas as pd
import pytest

from sos.compute import (
    build_next_n_games_per_team,
    compute_sos_from_netrtg_up_to_round,
    compute_sos_from_winpct_up_to_round,
    compute_sos_net_rating_next5,
    compute_sos_winpct_next5,
    compute_team_net_rating,
    compute_team_win_pct,
)


# ---------------------------------------------------------------------------
# compute_team_win_pct
# ---------------------------------------------------------------------------

def test_win_pct_basic():
    games = pd.DataFrame(
        [
            {"hometeam": "A", "awayteam": "B", "homescore": 80, "awayscore": 70},
            {"hometeam": "B", "awayteam": "A", "homescore": 60, "awayscore": 65},
            {"hometeam": "A", "awayteam": "C", "homescore": 50, "awayscore": 55},
        ]
    )

    result = compute_team_win_pct(games)

    # A: won g1 (home), won g2 (away), lost g3 -> 2/3
    assert result["A"] == pytest.approx(2 / 3)
    # B: lost g1, lost g2 -> 0/2
    assert result["B"] == pytest.approx(0.0)
    # C: won g3 -> 1/1
    assert result["C"] == pytest.approx(1.0)


def test_win_pct_ignores_unplayed_games():
    games = pd.DataFrame(
        [
            {"hometeam": "A", "awayteam": "B", "homescore": 80, "awayscore": 70},
            {"hometeam": "A", "awayteam": "B", "homescore": np.nan, "awayscore": np.nan},
        ]
    )

    result = compute_team_win_pct(games)

    assert result["A"] == pytest.approx(1.0)
    assert result["B"] == pytest.approx(0.0)


def test_win_pct_team_with_only_unplayed_games_is_excluded():
    # Teams are derived from the score-filtered frame, so a team whose only
    # game hasn't been played yet doesn't get a (NaN) entry — it's absent
    # entirely.
    games = pd.DataFrame(
        [{"hometeam": "A", "awayteam": "B", "homescore": np.nan, "awayscore": np.nan}]
    )

    result = compute_team_win_pct(games)

    assert result == {}


# ---------------------------------------------------------------------------
# compute_team_net_rating
# ---------------------------------------------------------------------------

def _team_total_row(home, fga2, fga3, fta, oreb, tov):
    """Build a box-score 'Total' row with stats that yield a chosen possession count.

    poss = fga2 + fga3 + 0.44*fta - oreb + tov, matching compute.py's formula.
    Keeping fta/oreb/tov at 0 makes poss = fga2 + fga3 exactly, which is all
    these tests need.
    """
    return {
        "Player": "Total",
        "Home": home,
        "FieldGoalsAttempted2": fga2,
        "FieldGoalsAttempted3": fga3,
        "FreeThrowsAttempted": fta,
        "OffensiveRebounds": oreb,
        "Turnovers": tov,
    }


class _FakeBoxScoreApi:
    """Stand-in for euroleague_api.boxscore_data.BoxScoreData.

    Maps gamecode -> possessions per (home, away) side, mimicking the shape
    of get_players_boxscore_stats's 'Total' rows (Player == "Total", Home in {0, 1}).
    """

    def __init__(self, poss_by_gamecode):
        self._poss_by_gamecode = poss_by_gamecode

    def get_players_boxscore_stats(self, season, gamecode):
        home_poss, away_poss = self._poss_by_gamecode[gamecode]
        return pd.DataFrame(
            [
                _team_total_row(1, home_poss, 0, 0, 0, 0),
                _team_total_row(0, away_poss, 0, 0, 0, 0),
            ]
        )


def test_team_net_rating_aggregates_across_games():
    games_meta = pd.DataFrame(
        [
            {
                "gameCode": 1,
                "hometeam": "A",
                "awayteam": "B",
                "homescore": 80.0,
                "awayscore": 70.0,
            },
            {
                "gameCode": 2,
                "hometeam": "C",
                "awayteam": "A",
                "homescore": 60.0,
                "awayscore": 90.0,
            },
        ]
    )

    # game 1: A home poss=80, B away poss=80
    # game 2: C home poss=75, A away poss=75
    api = _FakeBoxScoreApi({1: (80.0, 80.0), 2: (75.0, 75.0)})

    result = compute_team_net_rating("A", season=2025, games_meta=games_meta, boxscore_api=api)

    # Game 1: A scored 80 on 80 poss (OffRtg 100), allowed 70 (DefRtg 87.5)
    # Game 2: A scored 90 on 75 poss (OffRtg 120), allowed 60 (DefRtg 80)
    total_pts_for = 80.0 + 90.0
    total_pts_against = 70.0 + 60.0
    total_poss = 80.0 + 75.0
    expected_off = total_pts_for / total_poss * 100.0
    expected_def = total_pts_against / total_poss * 100.0

    assert result["games_played"] == 2
    assert result["OffRtg"] == pytest.approx(expected_off)
    assert result["DefRtg"] == pytest.approx(expected_def)
    assert result["NetRtg"] == pytest.approx(expected_off - expected_def)


def test_team_net_rating_skips_zero_possession_games():
    games_meta = pd.DataFrame(
        [
            {
                "gameCode": 1,
                "hometeam": "A",
                "awayteam": "B",
                "homescore": 80.0,
                "awayscore": 70.0,
            }
        ]
    )
    api = _FakeBoxScoreApi({1: (0.0, 0.0)})

    result = compute_team_net_rating("A", season=2025, games_meta=games_meta, boxscore_api=api)

    assert result["games_played"] == 0
    assert np.isnan(result["NetRtg"])


def test_team_net_rating_no_games_played_is_nan():
    games_meta = pd.DataFrame(
        [{"gameCode": 1, "hometeam": "B", "awayteam": "C", "homescore": 80.0, "awayscore": 70.0}]
    )
    api = _FakeBoxScoreApi({1: (80.0, 80.0)})

    result = compute_team_net_rating("A", season=2025, games_meta=games_meta, boxscore_api=api)

    assert result["games_played"] == 0
    assert np.isnan(result["OffRtg"])
    assert np.isnan(result["DefRtg"])
    assert np.isnan(result["NetRtg"])


# ---------------------------------------------------------------------------
# compute_sos_from_netrtg_up_to_round
# ---------------------------------------------------------------------------

def test_sos_from_netrtg_averages_opponent_ratings():
    # A plays B then C; SoS_Net for A should be mean(NetRtg[B], NetRtg[C])
    games_meta = pd.DataFrame(
        [
            {"gameCode": 1, "hometeam": "A", "awayteam": "B", "homescore": 80.0, "awayscore": 70.0, "Round": 1},
            {"gameCode": 2, "hometeam": "C", "awayteam": "A", "homescore": 60.0, "awayscore": 90.0, "Round": 2},
        ]
    )
    team_ratings = pd.DataFrame(
        [
            {"TEAM_NAME": "A", "Games": 2, "NetRtg": 5.0},
            {"TEAM_NAME": "B", "Games": 1, "NetRtg": -10.0},
            {"TEAM_NAME": "C", "Games": 1, "NetRtg": 20.0},
        ]
    )

    result = compute_sos_from_netrtg_up_to_round(games_meta, team_ratings, round_max=2)

    a_row = result[result["TEAM_NAME"] == "A"].iloc[0]
    assert a_row["SoS_Net"] == pytest.approx((-10.0 + 20.0) / 2)

    # sorted descending by SoS_Net
    assert list(result["SoS_Net"]) == sorted(result["SoS_Net"], reverse=True)


def test_sos_from_netrtg_respects_round_cutoff():
    games_meta = pd.DataFrame(
        [
            {"gameCode": 1, "hometeam": "A", "awayteam": "B", "homescore": 80.0, "awayscore": 70.0, "Round": 1},
            {"gameCode": 2, "hometeam": "A", "awayteam": "C", "homescore": 80.0, "awayscore": 70.0, "Round": 2},
        ]
    )
    team_ratings = pd.DataFrame(
        [
            {"TEAM_NAME": "A", "Games": 2, "NetRtg": 5.0},
            {"TEAM_NAME": "B", "Games": 1, "NetRtg": -10.0},
            {"TEAM_NAME": "C", "Games": 1, "NetRtg": 20.0},
        ]
    )

    result = compute_sos_from_netrtg_up_to_round(games_meta, team_ratings, round_max=1)

    a_row = result[result["TEAM_NAME"] == "A"].iloc[0]
    # Only round-1 game (vs B) should count
    assert a_row["SoS_Net"] == pytest.approx(-10.0)


# ---------------------------------------------------------------------------
# compute_sos_from_winpct_up_to_round
# ---------------------------------------------------------------------------

def test_sos_from_winpct_averages_opponent_winpct():
    # A beats B (round 1), loses to C (round 2). B and C's own win% form A's SoS.
    games_meta = pd.DataFrame(
        [
            {"gameCode": 1, "hometeam": "A", "awayteam": "B", "homescore": 80.0, "awayscore": 70.0, "Round": 1},
            {"gameCode": 2, "hometeam": "C", "awayteam": "A", "homescore": 90.0, "awayscore": 60.0, "Round": 2},
            {"gameCode": 3, "hometeam": "B", "awayteam": "C", "homescore": 60.0, "awayscore": 70.0, "Round": 2},
        ]
    )

    result = compute_sos_from_winpct_up_to_round(games_meta, round_max=2)

    # Win%: A = 1/2, B = 0/2, C = 2/2
    a_row = result[result["TEAM_NAME"] == "A"].iloc[0]
    assert a_row["SoS"] == pytest.approx((0.0 + 2 / 2) / 2)


# ---------------------------------------------------------------------------
# build_next_n_games_per_team / next-N forecasts
# ---------------------------------------------------------------------------

def test_build_next_n_games_per_team_orders_and_truncates():
    games = pd.DataFrame(
        [
            {"Round": 5, "DateTime": pd.Timestamp("2026-01-01"), "Home_Team": "A", "Away_Team": "B"},
            {"Round": 6, "DateTime": pd.Timestamp("2026-01-08"), "Home_Team": "C", "Away_Team": "A"},
            {"Round": 7, "DateTime": pd.Timestamp("2026-01-15"), "Home_Team": "A", "Away_Team": "D"},
            {"Round": 4, "DateTime": pd.Timestamp("2025-12-25"), "Home_Team": "A", "Away_Team": "E"},  # past round
        ]
    )

    result = build_next_n_games_per_team(games, current_round=5, n_next=2)

    a_next = result["A"]
    assert len(a_next) == 2
    assert list(a_next["Opponent"]) == ["B", "C"]
    assert list(a_next["Is_Home"]) == [True, False]


def test_build_next_n_games_per_team_excludes_teams_with_no_upcoming_games():
    # Team list is derived after filtering to Round >= current_round, so a
    # team whose only games are in the past doesn't get an empty-frame
    # entry — it's absent entirely.
    games = pd.DataFrame(
        [{"Round": 1, "DateTime": pd.Timestamp("2026-01-01"), "Home_Team": "A", "Away_Team": "B"}]
    )

    result = build_next_n_games_per_team(games, current_round=5, n_next=3)

    assert result == {}


def test_sos_net_rating_next5_averages_upcoming_opponents():
    team_to_next = {
        "A": pd.DataFrame({"Opponent": ["B", "C"]}),
        "B": pd.DataFrame(columns=["Opponent"]),
    }
    team_ratings = pd.DataFrame(
        [
            {"TEAM_NAME": "B", "NetRtg": -5.0},
            {"TEAM_NAME": "C", "NetRtg": 15.0},
        ]
    )

    result = compute_sos_net_rating_next5(team_to_next, team_ratings)

    assert result["A"] == pytest.approx((-5.0 + 15.0) / 2)
    assert np.isnan(result["B"])


def test_sos_winpct_next5_averages_upcoming_opponents():
    team_to_next = {"A": pd.DataFrame({"Opponent": ["B", "C"]})}
    team_win_pct = {"B": 0.25, "C": 0.75}

    result = compute_sos_winpct_next5(team_to_next, team_win_pct)

    assert result["A"] == pytest.approx((0.25 + 0.75) / 2)
