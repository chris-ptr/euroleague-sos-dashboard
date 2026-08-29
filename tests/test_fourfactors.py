"""
Tests for sos/fourfactors.py — the Four Factors math against hand-computed
expected values, independent of the live EuroLeague API.
"""

import numpy as np
import pandas as pd
import pytest

from sos.fourfactors import (
    FOUR_FACTOR_WEIGHTS,
    add_four_factor_composites,
    compute_four_factors_up_to_round,
    compute_team_four_factors,
    four_factors_from_totals,
)


def _totals(
    *,
    fgm2=0, fga2=0, fgm3=0, fga3=0,
    ftm=0, fta=0, oreb=0, dreb=0, tov=0,
):
    """One side's box-score totals, in the shape four_factors_from_totals reads."""
    return {
        "FieldGoalsMade2": float(fgm2),
        "FieldGoalsAttempted2": float(fga2),
        "FieldGoalsMade3": float(fgm3),
        "FieldGoalsAttempted3": float(fga3),
        "FreeThrowsMade": float(ftm),
        "FreeThrowsAttempted": float(fta),
        "OffensiveRebounds": float(oreb),
        "DefensiveRebounds": float(dreb),
        "Turnovers": float(tov),
    }


def _total_row(home, **kwargs):
    t = _totals(**kwargs)
    return {"Player": "Total", "Home": home, **t}


class FakeBoxScoreAPI:
    """
    Stand-in for euroleague_api.boxscore_data.BoxScoreData.

    Maps gamecode -> (home totals kwargs, away totals kwargs), returning the
    two "Total" rows sos.fourfactors reads (Player == "Total", Home in {0, 1}).
    """

    def __init__(self, by_gamecode):
        self._by_gamecode = by_gamecode

    def get_players_boxscore_stats(self, season, gamecode):
        home, away = self._by_gamecode[gamecode]
        return pd.DataFrame([_total_row(1, **home), _total_row(0, **away)])


# ---------------------------------------------------------------------------
# four_factors_from_totals
# ---------------------------------------------------------------------------

def test_four_factors_match_hand_computed_values():
    team = _totals(fgm2=20, fga2=40, fgm3=10, fga3=30, ftm=15, fta=20, oreb=12, dreb=30, tov=14)
    opp = _totals(fgm2=25, fga2=50, fgm3=5, fga3=20, ftm=10, fta=16, oreb=10, dreb=28, tov=18)

    ff = four_factors_from_totals(team, opp)

    # Team: FGA 70, FGM 30, 3PM 10 -> (30 + 5) / 70
    assert ff["eFG"] == pytest.approx(35 / 70)
    # 14 / (70 + 0.44*20 + 14)
    assert ff["TOV_Rate"] == pytest.approx(14 / (70 + 8.8 + 14))
    # 12 / (12 + opponent's 28 defensive boards)
    assert ff["OREB_Pct"] == pytest.approx(12 / 40)
    # 15 made free throws / 70 attempts
    assert ff["FT_Rate"] == pytest.approx(15 / 70)

    # Opponent: FGA 70, FGM 30, 3PM 5 -> (30 + 2.5) / 70
    assert ff["Opp_eFG"] == pytest.approx(32.5 / 70)
    assert ff["Opp_TOV_Rate"] == pytest.approx(18 / (70 + 0.44 * 16 + 18))
    # 30 own defensive boards / (30 + opponent's 10 offensive boards)
    assert ff["DREB_Pct"] == pytest.approx(30 / 40)
    assert ff["Opp_FT_Rate"] == pytest.approx(10 / 70)


def test_dreb_pct_is_the_mirror_of_opponent_oreb_pct():
    team = _totals(fga2=40, fga3=20, oreb=10, dreb=30)
    opp = _totals(fga2=40, fga3=20, oreb=15, dreb=25)

    ours = four_factors_from_totals(team, opp)
    theirs = four_factors_from_totals(opp, team)

    assert ours["DREB_Pct"] == pytest.approx(1.0 - theirs["OREB_Pct"])


def test_zero_attempts_give_nan_not_a_crash():
    empty = _totals()
    ff = four_factors_from_totals(empty, empty)

    assert np.isnan(ff["eFG"])
    assert np.isnan(ff["TOV_Rate"])
    assert np.isnan(ff["OREB_Pct"])
    assert np.isnan(ff["FT_Rate"])


# ---------------------------------------------------------------------------
# compute_team_four_factors
# ---------------------------------------------------------------------------

def test_totals_are_summed_across_games_before_the_ratio_is_taken():
    # The team plays home in game 1 and away in game 2, so this also pins down
    # that the right Home flag is picked for each side.
    games_meta = pd.DataFrame(
        [
            {"gameCode": 1, "hometeam": "A", "awayteam": "B", "homescore": 80.0, "awayscore": 70.0},
            {"gameCode": 2, "hometeam": "C", "awayteam": "A", "homescore": 60.0, "awayscore": 90.0},
        ]
    )
    api = FakeBoxScoreAPI(
        {
            1: (dict(fgm2=20, fga2=40, fgm3=10, fga3=20, dreb=30), dict(fga2=50, dreb=20)),
            2: (dict(fga2=50, dreb=10), dict(fgm2=10, fga2=20, fgm3=10, fga3=20, dreb=25)),
        }
    )

    result = compute_team_four_factors("A", season=2025, games_meta=games_meta, boxscore_api=api)

    # A's totals: FGA (40+20) + (20+20) = 100, FGM 30 + 20 = 50, 3PM 10 + 10 = 20
    assert result["Games"] == 2
    assert result["eFG"] == pytest.approx((50 + 10) / 100)
    # Volume-weighted, not the average of the two per-game eFGs (.5833 and .625).
    assert result["eFG"] != pytest.approx((35 / 60 + 25 / 40) / 2)


def test_game_with_only_one_usable_side_is_skipped_entirely():
    # A box score whose away "Total" row is missing would otherwise contribute
    # the team's shots with no opponent rebounds behind the OREB% denominator.
    class HalfBoxAPI:
        def get_players_boxscore_stats(self, season, gamecode):
            return pd.DataFrame([_total_row(1, fga2=40, oreb=10)])

    games_meta = pd.DataFrame(
        [{"gameCode": 1, "hometeam": "A", "awayteam": "B", "homescore": 80.0, "awayscore": 70.0}]
    )

    result = compute_team_four_factors("A", season=2025, games_meta=games_meta, boxscore_api=HalfBoxAPI())

    assert result["Games"] == 0
    assert np.isnan(result["eFG"])


def test_unplayed_games_are_ignored():
    games_meta = pd.DataFrame(
        [
            {"gameCode": 1, "hometeam": "A", "awayteam": "B", "homescore": 80.0, "awayscore": 70.0},
            {"gameCode": 2, "hometeam": "A", "awayteam": "B", "homescore": np.nan, "awayscore": np.nan},
        ]
    )
    api = FakeBoxScoreAPI({1: (dict(fgm2=20, fga2=40, dreb=30), dict(fga2=40, dreb=20))})

    result = compute_team_four_factors("A", season=2025, games_meta=games_meta, boxscore_api=api)

    assert result["Games"] == 1


# ---------------------------------------------------------------------------
# add_four_factor_composites
# ---------------------------------------------------------------------------

def _league(rows):
    return pd.DataFrame(rows)


def test_composite_signs_each_factor_so_higher_is_always_better():
    # Two teams identical but for one factor at a time; the team on the good
    # side of it must come out ahead every time.
    base = dict(
        eFG=0.52, TOV_Rate=0.14, OREB_Pct=0.30, FT_Rate=0.24,
        Opp_eFG=0.52, Opp_TOV_Rate=0.14, DREB_Pct=0.70, Opp_FT_Rate=0.24,
    )
    better = {
        "eFG": 0.56,            # shoot better
        "TOV_Rate": 0.11,       # cough it up less
        "OREB_Pct": 0.34,       # rebound own misses more
        "FT_Rate": 0.28,        # get to the line more
        "Opp_eFG": 0.48,        # allow worse shooting
        "Opp_TOV_Rate": 0.17,   # force more turnovers
        "DREB_Pct": 0.74,       # close out the defensive glass
        "Opp_FT_Rate": 0.20,    # send them to the line less
    }

    for factor, good_value in better.items():
        good = {**base, "TEAM_NAME": "GOOD", factor: good_value}
        plain = {**base, "TEAM_NAME": "PLAIN"}
        out = add_four_factor_composites(_league([good, plain])).set_index("TEAM_NAME")
        assert out.loc["GOOD", "FF_Net"] > out.loc["PLAIN", "FF_Net"], factor


def test_composite_weights_shooting_above_the_other_factors():
    # Same z-score edge, applied to eFG for one team and to free throws for
    # another: Oliver's weights have to put the shooter ahead.
    base = dict(
        eFG=0.52, TOV_Rate=0.14, OREB_Pct=0.30, FT_Rate=0.24,
        Opp_eFG=0.52, Opp_TOV_Rate=0.14, DREB_Pct=0.70, Opp_FT_Rate=0.24,
    )
    df = _league([
        {**base, "TEAM_NAME": "SHOOTER", "eFG": 0.56},
        {**base, "TEAM_NAME": "FOULED", "FT_Rate": 0.28},
    ])

    out = add_four_factor_composites(df).set_index("TEAM_NAME")

    # With two teams every z-score is exactly +/-1, so each team is a standard
    # deviation up on one factor and a standard deviation down on the other.
    # What is left is the difference between the two weights, and its sign is
    # the whole point: shooting outranks free throws.
    edge = FOUR_FACTOR_WEIGHTS["eFG"] - FOUR_FACTOR_WEIGHTS["FT"]
    assert out.loc["SHOOTER", "FF_Net"] == pytest.approx(edge)
    assert out.loc["FOULED", "FF_Net"] == pytest.approx(-edge)
    assert edge > 0


def test_factor_with_no_league_spread_contributes_zero_not_nan():
    # Every team identical on every factor: a z-score would be 0/0, and one NaN
    # would propagate through the sum and wipe out the whole composite.
    base = dict(
        eFG=0.52, TOV_Rate=0.14, OREB_Pct=0.30, FT_Rate=0.24,
        Opp_eFG=0.52, Opp_TOV_Rate=0.14, DREB_Pct=0.70, Opp_FT_Rate=0.24,
    )
    df = _league([{**base, "TEAM_NAME": "A"}, {**base, "TEAM_NAME": "B"}])

    out = add_four_factor_composites(df)

    assert out["FF_Net"].notna().all()
    assert out["FF_Net"].abs().max() == pytest.approx(0.0)


def test_net_is_the_sum_of_the_two_sides():
    base = dict(
        eFG=0.52, TOV_Rate=0.14, OREB_Pct=0.30, FT_Rate=0.24,
        Opp_eFG=0.52, Opp_TOV_Rate=0.14, DREB_Pct=0.70, Opp_FT_Rate=0.24,
    )
    df = _league([
        {**base, "TEAM_NAME": "A", "eFG": 0.56, "Opp_eFG": 0.48},
        {**base, "TEAM_NAME": "B"},
    ])

    out = add_four_factor_composites(df)

    assert out["FF_Net"].values == pytest.approx((out["FF_Off"] + out["FF_Def"]).values)


# ---------------------------------------------------------------------------
# compute_four_factors_up_to_round
# ---------------------------------------------------------------------------

def test_round_filter_excludes_later_games_and_ranks_by_rating():
    games_meta = pd.DataFrame(
        [
            {"Round": 1, "gameCode": 1, "hometeam": "A", "awayteam": "B",
             "homescore": 80.0, "awayscore": 70.0},
            {"Round": 2, "gameCode": 2, "hometeam": "A", "awayteam": "B",
             "homescore": 90.0, "awayscore": 60.0},
        ]
    )
    api = FakeBoxScoreAPI(
        {
            # A shoots well, B badly — so A must rank first.
            1: (dict(fgm2=30, fga2=50, dreb=30), dict(fgm2=10, fga2=50, dreb=20)),
            2: (dict(fgm2=30, fga2=50, dreb=30), dict(fgm2=10, fga2=50, dreb=20)),
        }
    )

    through_1 = compute_four_factors_up_to_round(games_meta, 2025, api, round_max=1)

    assert list(through_1["TEAM_NAME"]) == ["A", "B"]
    assert (through_1["Games"] == 1).all()
    assert through_1.iloc[0]["TEAM_NAME"] == "A"

    through_2 = compute_four_factors_up_to_round(games_meta, 2025, api, round_max=2)
    assert (through_2["Games"] == 2).all()
