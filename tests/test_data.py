"""
Tests for sos/data.py — verify the raw API schedule is reshaped into the
columns the Next-N forecast expects, without hitting the network.
"""

import pandas as pd
import pytest

from sos.data import clean_api_schedule


def _raw_row(gameday, round_code, date, startime, hometeam, awayteam):
    """One row shaped like euroleague_api Schedule.get_schedule() output."""
    return {
        "gameday": gameday,
        "round": round_code,
        "date": date,
        "startime": startime,
        "hometeam": hometeam,
        "awayteam": awayteam,
    }


def test_clean_api_schedule_maps_columns():
    raw = pd.DataFrame(
        [_raw_row(1, "RS", "Sep 30, 2025", "20:45", "VIRTUS BOLOGNA", "REAL MADRID")]
    )

    result = clean_api_schedule(raw)

    row = result.iloc[0]
    assert row["Round"] == 1
    assert row["Home_Team"] == "VIRTUS BOLOGNA"
    assert row["Away_Team"] == "REAL MADRID"
    assert row["DateTime"] == pd.Timestamp("2025-09-30 20:45")


def test_clean_api_schedule_drops_non_regular_season():
    raw = pd.DataFrame(
        [
            _raw_row(1, "RS", "Sep 30, 2025", "20:45", "A", "B"),
            _raw_row(41, "PO", "Apr 21, 2026", "20:00", "C", "D"),
            _raw_row(47, "FF", "May 24, 2026", "19:00", "E", "F"),
        ]
    )

    result = clean_api_schedule(raw)

    assert list(result["Round"]) == [1]


def test_clean_api_schedule_can_keep_all_phases():
    raw = pd.DataFrame(
        [
            _raw_row(1, "RS", "Sep 30, 2025", "20:45", "A", "B"),
            _raw_row(41, "PO", "Apr 21, 2026", "20:00", "C", "D"),
        ]
    )

    result = clean_api_schedule(raw, regular_season_only=False)

    assert list(result["Round"]) == [1, 41]


def test_clean_api_schedule_normalizes_team_aliases():
    # MACCABI RAPYD TEL AVIV is aliased to MACCABI TEL AVIV in TEAM_NAME_MAP,
    # so schedule names line up with the names used in team_ratings.
    raw = pd.DataFrame(
        [
            _raw_row(
                1, "RS", "Sep 30, 2025", "20:45",
                "MACCABI RAPYD TEL AVIV", "KOSNER BASKONIA VITORIA-GASTEIZ",
            )
        ]
    )

    result = clean_api_schedule(raw)

    assert result.iloc[0]["Home_Team"] == "MACCABI TEL AVIV"
    assert result.iloc[0]["Away_Team"] == "BASKONIA VITORIA-GASTEIZ"


def test_clean_api_schedule_sorts_by_round_then_tipoff():
    raw = pd.DataFrame(
        [
            _raw_row(2, "RS", "Oct 7, 2025", "20:00", "C", "D"),
            _raw_row(1, "RS", "Sep 30, 2025", "21:00", "E", "F"),
            _raw_row(1, "RS", "Sep 30, 2025", "18:00", "A", "B"),
        ]
    )

    result = clean_api_schedule(raw)

    assert list(result["Round"]) == [1, 1, 2]
    # within round 1, the 18:00 tip-off comes before the 21:00 one
    assert list(result["Home_Team"]) == ["A", "E", "C"]


def test_clean_api_schedule_does_not_mutate_input():
    raw = pd.DataFrame(
        [_raw_row(1, "RS", "Sep 30, 2025", "20:45", "A", "B")]
    )
    before = raw.copy()

    clean_api_schedule(raw)

    pd.testing.assert_frame_equal(raw, before)
