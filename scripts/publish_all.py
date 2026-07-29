#!/usr/bin/env python
"""
Regenerate the published chart JSON for *every* cached round.

Reads the already-computed cache/rounds/*.parquet files and rebuilds all the
Vega-Lite artifacts under PUBLISH_DIR, without recomputing anything or calling
the per-game stats API again. Use it when the chart code changes and every
round's spec needs re-rendering; scripts/refresh_and_publish.py handles the
normal incremental case of a single newly-finished round.

Usage:
    python scripts/publish_all.py
"""
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sos.config import (
    DEFAULT_SEASON,
    DEFAULT_COMPETITION,
    CACHE_DIR,
    PUBLISH_DIR,
)
from sos.data import load_games_metadata
from sos.cache import load_cached_round
from sos.presets import NEXT_N_VALUES
from sos.publish import build_round_artifacts, write_artifacts

ROUND_FILE_RE = re.compile(r"round_(\d+)\.parquet$")
NEXT_N_FILE_RE = re.compile(r"^(\d+)\.json$")


def prune_stale_next_n(round_num: int) -> int:
    """
    Delete published next-N specs for values no longer in NEXT_N_VALUES.

    Rebuilding only overwrites the files it generates, so lowering the N cap
    would otherwise leave the old higher-N JSON on disk and reachable by URL.
    """
    next_n_dir = PUBLISH_DIR / f"rounds/{round_num}/next-n"
    removed = 0
    for path in next_n_dir.glob("*.json"):
        m = NEXT_N_FILE_RE.match(path.name)
        if m and int(m.group(1)) not in NEXT_N_VALUES:
            path.unlink()
            removed += 1
    return removed


def main() -> None:
    round_numbers = sorted(
        int(m.group(1))
        for f in CACHE_DIR.glob("round_*.parquet")
        if (m := ROUND_FILE_RE.search(f.name))
    )
    if not round_numbers:
        print(f"No local round cache files found under {CACHE_DIR}. Nothing to publish.")
        return

    print(f"Found {len(round_numbers)} locally-cached round(s): {round_numbers}")

    print("Fetching current EuroLeague game metadata (needed for schedule/next-N tables)...")
    games_meta, _boxscore_api, _ = load_games_metadata(
        season=DEFAULT_SEASON,
        competition_code=DEFAULT_COMPETITION,
    )
    season_label = f"{DEFAULT_SEASON}-{(DEFAULT_SEASON + 1) % 100:02d}"

    for round_num in round_numbers:
        cached = load_cached_round(CACHE_DIR, round_num)
        if cached is None:
            print(f"  round {round_num}: cache file missing/unreadable, skipping")
            continue
        team_ratings, sos_net, sos_win = cached

        print(f"  round {round_num}: building chart JSON...")
        artifacts = build_round_artifacts(
            round_num=round_num,
            season=DEFAULT_SEASON,
            games_meta=games_meta,
            team_ratings=team_ratings,
            sos_net=sos_net,
            sos_win=sos_win,
            season_label=season_label,
        )
        write_artifacts(PUBLISH_DIR, artifacts)
        stale = prune_stale_next_n(round_num)
        if stale:
            print(f"  round {round_num}: pruned {stale} stale next-N spec(s)")

    latest_round = max(round_numbers)
    write_artifacts(
        PUBLISH_DIR,
        {
            "latest.json": {
                "round": latest_round,
                "season": DEFAULT_SEASON,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        },
    )
    print(f"Done. Published rounds 1..{latest_round} to {PUBLISH_DIR}.")


if __name__ == "__main__":
    main()
