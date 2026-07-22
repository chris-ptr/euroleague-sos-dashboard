#!/usr/bin/env python
"""
One-time backfill: run locally (not in CI) to upload the already-computed
local cache/rounds/*.parquet files into Supabase (both the private working
cache and the public precomputed chart JSON), covering all historical rounds
without recomputing anything or calling the per-game stats API again.

After this runs successfully, scripts/refresh_and_publish.py only ever has
to handle newly-finished rounds going forward.

Usage:
    python scripts/seed_supabase.py
"""
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from sos.config import (
    DEFAULT_SEASON,
    DEFAULT_COMPETITION,
    SCHEDULE_FILENAME,
    CACHE_DIR,
    SUPABASE_CACHE_BUCKET,
    SUPABASE_PUBLIC_BUCKET,
)
from sos.data import load_games_metadata
from sos.cache import load_cached_round, round_cache_path
from sos.publish import build_round_artifacts
from sos.storage import SupabaseStorage

ROUND_FILE_RE = re.compile(r"round_(\d+)\.parquet$")


def main() -> None:
    storage = SupabaseStorage()

    round_numbers = sorted(
        int(m.group(1))
        for f in CACHE_DIR.glob("round_*.parquet")
        if (m := ROUND_FILE_RE.search(f.name))
    )
    if not round_numbers:
        print(f"No local round cache files found under {CACHE_DIR}. Nothing to seed.")
        return

    print(f"Found {len(round_numbers)} locally-cached round(s): {round_numbers}")

    print("Fetching current EuroLeague game metadata (needed for schedule/next-N tables)...")
    games_meta, _team_stats_api, _ = load_games_metadata(
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

        print(f"  round {round_num}: uploading parquet + building/uploading chart JSON...")
        parquet_bytes = round_cache_path(CACHE_DIR, round_num).read_bytes()
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

    latest_round = max(round_numbers)
    storage.upload_json(
        SUPABASE_PUBLIC_BUCKET,
        "latest.json",
        {
            "round": latest_round,
            "season": DEFAULT_SEASON,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"Done. Seeded rounds 1..{latest_round}.")


if __name__ == "__main__":
    main()
