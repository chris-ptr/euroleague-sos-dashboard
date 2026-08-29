#!/usr/bin/env python
"""
Pull in any newly-finished EuroLeague rounds and publish their chart JSON.

1. Read the per-round parquet cache in CACHE_DIR (tracked in git) to see which
   rounds are already computed.
2. Fetch current game metadata and detect the latest fully-played round.
3. Compute any rounds that aren't cached yet (sos.cache.compute_for_round writes
   the parquet into CACHE_DIR as a side effect).
4. For newly-computed rounds only: build the precomputed Vega-Lite chart JSON
   (sos.publish) and write it under PUBLISH_DIR.
5. Refresh latest.json so the frontend always knows the current round.

Run locally, then commit and push — Vercel serves PUBLISH_DIR as static files:

    python scripts/refresh_and_publish.py
    git add cache/rounds frontend/data && git commit -m "Publish round N" && git push
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sos.config import (
    DEFAULT_SEASON,
    DEFAULT_COMPETITION,
    TOTAL_SEASON_ROUNDS,
    CACHE_DIR,
    PUBLISH_DIR,
    season_label,
)
from sos.data import load_games_metadata
from sos.rounds import detect_latest_complete_round
from sos.cache import compute_for_round
from sos.publish import build_latest_manifest, build_round_artifacts, write_artifacts


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    existing_rounds = set()
    for path in CACHE_DIR.glob("round_*.parquet"):
        existing_rounds.add(int(path.stem.removeprefix("round_")))
    print(f"Found {len(existing_rounds)} cached round(s): {sorted(existing_rounds)}")

    print("Fetching current EuroLeague game metadata...")
    games_meta, boxscore_api, _ = load_games_metadata(
        season=DEFAULT_SEASON,
        competition_code=DEFAULT_COMPETITION,
    )

    # Cap at the Regular Season length: euroleague_api numbers Playoffs/Final Four
    # rounds continuing past 38, but the Next-N forecast schedule is filtered to the
    # Regular Season, so anything beyond it has no upcoming games to forecast.
    latest_round = min(detect_latest_complete_round(games_meta), TOTAL_SEASON_ROUNDS)
    print(f"Latest complete round (capped at Regular Season length {TOTAL_SEASON_ROUNDS}): {latest_round}")

    label = season_label(DEFAULT_SEASON)
    new_rounds = []

    for round_num in range(1, latest_round + 1):
        is_new = round_num not in existing_rounds
        team_ratings, sos_net, sos_win, four_factors = compute_for_round(
            cache_dir=CACHE_DIR,
            games_meta=games_meta,
            season=DEFAULT_SEASON,
            boxscore_api=boxscore_api,
            round_max=round_num,
        )

        if not is_new:
            continue

        print(f"Round {round_num} is new — publishing...")
        new_rounds.append(round_num)

        artifacts = build_round_artifacts(
            round_num=round_num,
            season=DEFAULT_SEASON,
            games_meta=games_meta,
            team_ratings=team_ratings,
            sos_net=sos_net,
            sos_win=sos_win,
            four_factors=four_factors,
            season_label=label,
        )
        write_artifacts(PUBLISH_DIR, artifacts)

    write_artifacts(
        PUBLISH_DIR,
        {"latest.json": build_latest_manifest(PUBLISH_DIR, DEFAULT_SEASON)},
    )

    if new_rounds:
        print(f"Published {len(new_rounds)} new round(s): {new_rounds}")
    else:
        print("No new rounds — latest.json timestamp refreshed, nothing else to publish.")


if __name__ == "__main__":
    main()
