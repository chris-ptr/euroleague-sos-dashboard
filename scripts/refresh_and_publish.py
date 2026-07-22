#!/usr/bin/env python
"""
GitHub Actions entrypoint: refresh EuroLeague data and publish results to Supabase.

1. Pull the existing per-round parquet cache down from the private `sos-cache`
   Storage bucket (so we never recompute rounds that are already done).
2. Fetch current game metadata and detect the latest fully-played round.
3. Compute any rounds that aren't cached yet (sos.cache.compute_for_round).
4. For newly-computed rounds only: upload the parquet back to `sos-cache`,
   build the precomputed Vega-Lite chart JSON (sos.publish), and upload it to
   the public `sos-public` bucket.
5. Refresh `latest.json` so the frontend always knows the current round.

Requires SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in the environment
(GitHub Actions secrets in CI, or a local .env file for manual runs).
"""
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from sos.config import DEFAULT_SEASON, DEFAULT_COMPETITION, SCHEDULE_FILENAME, TOTAL_SEASON_ROUNDS, SUPABASE_CACHE_BUCKET, SUPABASE_PUBLIC_BUCKET
from sos.data import load_games_metadata
from sos.rounds import detect_latest_complete_round
from sos.cache import compute_for_round, round_cache_path
from sos.publish import build_round_artifacts
from sos.storage import SupabaseStorage


def main() -> None:
    storage = SupabaseStorage()

    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp)

        print("Downloading existing round cache from Supabase...")
        cached_names = storage.list_objects(SUPABASE_CACHE_BUCKET, prefix="rounds")
        existing_rounds = set()
        for name in cached_names:
            if name.startswith("round_") and name.endswith(".parquet"):
                round_num = int(name.removeprefix("round_").removesuffix(".parquet"))
                existing_rounds.add(round_num)
                data = storage.download_bytes(SUPABASE_CACHE_BUCKET, f"rounds/{name}")
                if data is not None:
                    (cache_dir / name).write_bytes(data)
        print(f"Found {len(existing_rounds)} cached round(s): {sorted(existing_rounds)}")

        print("Fetching current EuroLeague game metadata...")
        games_meta, team_stats_api, _ = load_games_metadata(
            season=DEFAULT_SEASON,
            competition_code=DEFAULT_COMPETITION,
        )

        # Cap at the Regular Season length: euroleague_api numbers Playoffs/Final Four
        # rounds continuing past 38, but the Next-N forecast schedule CSV (EL_*_RS_Schedule.csv)
        # only covers the Regular Season, so anything beyond it has no upcoming games to forecast.
        latest_round = min(detect_latest_complete_round(games_meta), TOTAL_SEASON_ROUNDS)
        print(f"Latest complete round (capped at Regular Season length {TOTAL_SEASON_ROUNDS}): {latest_round}")

        season_label = f"{DEFAULT_SEASON}-{(DEFAULT_SEASON + 1) % 100:02d}"
        new_rounds = []

        for round_num in range(1, latest_round + 1):
            is_new = round_num not in existing_rounds
            team_ratings, sos_net, sos_win = compute_for_round(
                cache_dir=cache_dir,
                games_meta=games_meta,
                season=DEFAULT_SEASON,
                team_stats_api=team_stats_api,
                round_max=round_num,
            )

            if not is_new:
                continue

            print(f"Round {round_num} is new — publishing...")
            new_rounds.append(round_num)

            parquet_bytes = round_cache_path(cache_dir, round_num).read_bytes()
            storage.upload_bytes(
                SUPABASE_CACHE_BUCKET,
                f"rounds/round_{round_num}.parquet",
                parquet_bytes,
                content_type="application/octet-stream",
            )

            artifacts = build_round_artifacts(
                round_num=round_num,
                schedule_path=SCHEDULE_FILENAME,
                games_meta=games_meta,
                team_ratings=team_ratings,
                sos_net=sos_net,
                sos_win=sos_win,
                season_label=season_label,
            )
            for path, spec in artifacts.items():
                storage.upload_json(SUPABASE_PUBLIC_BUCKET, path, spec)

        storage.upload_json(
            SUPABASE_PUBLIC_BUCKET,
            "latest.json",
            {
                "round": latest_round,
                "season": DEFAULT_SEASON,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )

        if new_rounds:
            print(f"Published {len(new_rounds)} new round(s): {new_rounds}")
        else:
            print("No new rounds — latest.json timestamp refreshed, nothing else to publish.")


if __name__ == "__main__":
    main()
