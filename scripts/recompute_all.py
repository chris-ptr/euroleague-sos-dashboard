#!/usr/bin/env python
"""
Recompute the per-round parquet cache from scratch, then republish chart JSON.

scripts/refresh_and_publish.py only computes rounds *missing* from CACHE_DIR, and
scripts/publish_all.py only re-renders charts from the parquet already on disk.
This script is the third case: the ratings themselves need recalculating — the
possessions formula changed, the upstream box scores were corrected, or the
cache is suspect — so every round is recomputed with force=True and overwritten.

Box scores are read through a three-layer cache: in-process memo -> on-disk
store -> EuroLeague API. The on-disk store is what makes iterating on the
calculation worthwhile: the ~380 games of a season are fetched once, ever, and
every later run (new possessions formula, new SOS weighting) is pure CPU with
no network at all. It is also crash-safe — each game is written as it arrives,
so a Ctrl+C or a rate-limit wall keeps everything fetched so far.

Without any cache, compute_team_net_rating refetches a game once per team and
every round cutoff replays all earlier games: ~14.8k API calls over ~380
distinct games for a 38-round season.

Usage:
    python scripts/recompute_all.py                     # all rounds, then publish
    python scripts/recompute_all.py --from 20           # rounds 20..latest only
    python scripts/recompute_all.py --from 5 --to 12
    python scripts/recompute_all.py --no-publish        # refresh parquet only
    python scripts/recompute_all.py --offline           # cache only, never fetch
    python scripts/recompute_all.py --refresh-meta      # re-fetch season metadata

The EuroLeague API rate-limits this workload hard, and the 429s outlast a single
run — once throttled, even the metadata endpoint refuses. Both metadata and box
scores are cached to disk as they arrive, so a throttled run loses nothing:
re-run the same command and it picks up only what's still missing.

Testing new calculation methods without touching the live data — CACHE_DIR and
PUBLISH_DIR are read from the environment (see sos/config.py), BOXSCORE_CACHE_DIR
is deliberately *not* redirected so experiments reuse the same downloaded games:

    mkdir -p /tmp/euroleague-test/{cache,publish}
    CACHE_DIR=/tmp/euroleague-test/cache PUBLISH_DIR=/tmp/euroleague-test/publish \
        python scripts/recompute_all.py

Afterwards, commit both trees — Vercel serves PUBLISH_DIR as static files:

    git add cache/rounds frontend/data && git commit -m "Recompute rounds" && git push
"""
import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from requests.exceptions import HTTPError
from tqdm.auto import tqdm
from euroleague_api.boxscore_data import BoxScoreData

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sos.config import (
    DEFAULT_SEASON,
    DEFAULT_COMPETITION,
    TOTAL_SEASON_ROUNDS,
    CACHE_DIR,
    PUBLISH_DIR,
)
from sos.cache import compute_for_round
from sos.data import load_games_metadata
from sos.rounds import detect_latest_complete_round
from sos.publish import build_round_artifacts, write_artifacts

sys.path.insert(0, str(Path(__file__).resolve().parent))
from publish_all import prune_stale_next_n


# Raw per-game box scores, keyed by season+gamecode. Not the round parquet cache
# (CACHE_DIR) and not chart JSON (PUBLISH_DIR) — this is the upstream API's
# output, which never changes once a game is final, so it is worth keeping
# across runs and across experiments with the calculation.
BOXSCORE_CACHE_DIR = Path(os.environ.get("BOXSCORE_CACHE_DIR", ".boxscore_cache"))


class CachingBoxScoreData(BoxScoreData):
    """
    BoxScoreData with the one method sos.compute calls made cached and retrying.

    Overriding get_players_boxscore_stats means every caller inherits the
    behavior, since sos.compute funnels through it for every game of every team
    of every round cutoff.

    Three jobs:

    1. Cache, in two layers. An in-process memo absorbs the ~14.8k calls a
       season's recompute makes over ~380 distinct games. Under it, an on-disk
       store (pickle per game, atomically written) survives the process, so a
       re-run after changing the possessions formula does no network I/O at all
       and finishes in seconds instead of half an hour.

    2. Retry, adaptively. The EuroLeague API rate-limits this workload — a full
       season reliably trips 429 somewhere after ~200 games. A 429 is honored
       via Retry-After when present, and slows the *rest* of the run down
       permanently: once the server has objected, the old pace is known-too-fast.

    3. Account for what still failed. compute_team_net_rating wraps its fetch in
       `except Exception: continue`, so a dropped game silently produces
       understated totals and a run that "succeeds" with wrong ratings. Games
       that exhaust their retries are recorded, swept once more at the end, and
       reported loudly — a parquet cache built from partial data is worse than
       none.
    """

    def __init__(
        self,
        *args,
        cache_dir: Path = BOXSCORE_CACHE_DIR,
        season: int = DEFAULT_SEASON,
        retries: int = 4,
        backoff: float = 2.0,
        pause: float = 0.25,
        offline: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._memo: Dict[Tuple[int, object], pd.DataFrame | None] = {}
        self._cache_dir = Path(cache_dir) / f"{self.competition}{season}"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._retries = retries
        self._backoff = backoff
        self._pause = pause
        self._offline = offline
        self.memo_hits = 0
        self.disk_hits = 0
        self.fetched = 0
        self.retried = 0
        self.throttled = 0
        self.failed: Dict[Tuple[int, object], str] = {}

    def _cache_path(self, gamecode) -> Path:
        return self._cache_dir / f"game_{gamecode}.pkl"

    def _read_disk(self, gamecode) -> pd.DataFrame | None:
        path = self._cache_path(gamecode)
        if not path.exists():
            return None
        try:
            return pd.read_pickle(path)
        except Exception:  # noqa: BLE001 - a corrupt entry is just a cache miss
            path.unlink(missing_ok=True)
            return None

    def _write_disk(self, gamecode, box: pd.DataFrame) -> None:
        # Write-then-rename: a Ctrl+C mid-write leaves the temp file, never a
        # half-written cache entry that would deserialize into wrong stats.
        path = self._cache_path(gamecode)
        tmp = path.with_suffix(".tmp")
        try:
            box.to_pickle(tmp)
            tmp.replace(path)
        except Exception:  # noqa: BLE001 - caching is best-effort, never fatal
            tmp.unlink(missing_ok=True)

    def get_players_boxscore_stats(self, season, gamecode):
        key = (season, gamecode)
        if key in self._memo:
            self.memo_hits += 1
            cached = self._memo[key]
            # Hand back a copy: callers filter and index the frame, and a shared
            # object would let one team's slicing leak into the next.
            return None if cached is None else cached.copy()

        box = self._read_disk(gamecode)
        if box is not None:
            self.disk_hits += 1
            self._memo[key] = box
            return box.copy()

        if self._offline:
            self._memo[key] = None
            self.failed[key] = "not in on-disk cache (--offline)"
            return None

        box = self._fetch(season, gamecode)
        self._memo[key] = box
        if box is not None:
            self.fetched += 1
            self._write_disk(gamecode, box)
            return box.copy()
        return None

    def _fetch(self, season, gamecode) -> pd.DataFrame | None:
        delay = self._backoff
        last_err = None
        for attempt in range(self._retries + 1):
            try:
                box = super().get_players_boxscore_stats(season, gamecode)
                # Be a polite client; unthrottled this loop reliably trips 429.
                time.sleep(self._pause)
                return box
            except Exception as err:  # noqa: BLE001 - reported, not swallowed
                last_err = err
                wait = delay
                if _is_rate_limited(err):
                    self.throttled += 1
                    wait = max(delay, _retry_after(err, default=30.0))
                    # The server has said we are too fast. Slow the whole rest
                    # of the run, not just this retry — otherwise we walk
                    # straight back into the wall on the next game.
                    self._pause = min(self._pause * 1.5, 5.0)
                if attempt < self._retries:
                    self.retried += 1
                    time.sleep(wait)
                    delay *= 2

        # Record the failure for the end-of-run sweep and report. Returning None
        # (rather than raising) matches what sos.compute expects for a game it
        # cannot fetch. Deliberately *not* written to the disk cache, so a later
        # run retries it instead of inheriting the gap.
        self.failed[(season, gamecode)] = f"{type(last_err).__name__}: {last_err}"
        return None

    def prefetch(self, season: int, gamecodes) -> None:
        """
        Fill the cache for a known set of games, one request each, with a bar.

        Replaces BoxScoreData.get_players_boxscore_stats_single_season, which
        fetches every game of the season even when only early rounds are being
        recomputed, concatenates ~380 frames into a result this script throws
        away, and — because its shared collection helper does `df.empty` on
        whatever it gets back — turns a failed fetch into a confusing
        "'NoneType' object has no attribute 'empty'" log line instead of the
        real error.
        """
        todo = [gc for gc in gamecodes if (season, gc) not in self._memo]
        for gamecode in tqdm(todo, desc=f"Season {season} box scores", leave=True):
            self.get_players_boxscore_stats(season, gamecode)

    def sweep_failures(self, season: int, cooldown: float = 60.0) -> None:
        """
        One more pass over games that exhausted their retries, after a cooldown.

        Failures cluster: a rate-limit wall takes out a run of consecutive games
        rather than one here and there. Waiting once and retrying the stragglers
        is far cheaper than re-running the whole script, and it is the difference
        between finishing with a complete cache and finishing with a warning.
        """
        if not self.failed:
            return
        stragglers = sorted(self.failed.keys(), key=str)
        print(
            f"\n{len(stragglers)} game(s) failed; waiting {cooldown:.0f}s "
            f"before a final retry pass..."
        )
        time.sleep(cooldown)
        for season_key, gamecode in stragglers:
            self.failed.pop((season_key, gamecode), None)
            self._memo.pop((season_key, gamecode), None)
            self.get_players_boxscore_stats(season_key, gamecode)


def _is_rate_limited(err: Exception) -> bool:
    return (
        isinstance(err, HTTPError)
        and err.response is not None
        and err.response.status_code == 429
    )


def _retry_after(err: Exception, default: float) -> float:
    """Seconds the server asked us to wait, or `default` if it didn't say."""
    try:
        return float(err.response.headers.get("Retry-After", default))
    except (AttributeError, TypeError, ValueError):
        return default


def load_games_metadata_cached(
    cache_dir: Path,
    season: int,
    competition: str,
    offline: bool = False,
    refresh: bool = False,
    retries: int = 6,
    backoff: float = 15.0,
) -> pd.DataFrame:
    """
    Season metadata, cached to disk beside the box scores.

    This call sits outside CachingBoxScoreData's retry logic and is the first
    thing a run does, which makes it the worst place to be rate-limited: a 429
    here aborts before any work, and on a re-run it burns another request
    against an API that is already refusing them. Caching it means a throttled
    re-run spends its whole budget on box scores, and --offline needs no
    network at all.

    Safe to reuse across runs because a played game's round and final score
    don't change. Pass --refresh-meta after new games are played.
    """
    path = Path(cache_dir) / f"{competition}{season}" / "_games_meta.pkl"

    if not refresh and path.exists():
        print(f"Using cached game metadata ({path}).")
        return pd.read_pickle(path)

    if offline:
        sys.exit(
            f"--offline, but no cached metadata at {path}. Run once without "
            f"--offline to populate it."
        )

    print("Fetching current EuroLeague game metadata...")
    delay = backoff
    for attempt in range(retries + 1):
        try:
            # The BoxScoreData it also returns is discarded — it's stateless
            # (games_meta comes from GameMetadata), so this script builds its
            # own caching/retrying instance instead of wrapping theirs.
            games_meta, _discarded_boxscore_api, _ = load_games_metadata(
                season=season,
                competition_code=competition,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            games_meta.to_pickle(tmp)
            tmp.replace(path)
            return games_meta
        except Exception as err:  # noqa: BLE001 - retried, then re-raised
            if attempt == retries:
                raise
            wait = max(delay, _retry_after(err, default=delay)) if _is_rate_limited(err) else delay
            note = " (rate-limited)" if _is_rate_limited(err) else ""
            print(f"  metadata fetch failed{note} ({err}); retrying in {wait:.0f}s")
            time.sleep(wait)
            delay *= 2


def gamecodes_through_round(games_meta: pd.DataFrame, round_max: int) -> list:
    """
    Played games from round 1 through `round_max`, in round order.

    Round 1 regardless of --from: every round cutoff re-aggregates all earlier
    games, so recomputing round 20 alone still needs rounds 1-20 on hand.
    """
    df = games_meta
    df = df[df["Round"] <= round_max]
    df = df[df["homescore"].notna() & df["awayscore"].notna()]
    codes = df.sort_values("Round")["gameCode"].dropna().unique().tolist()
    return codes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from", dest="from_round", type=int, default=1,
        help="first round to recompute (default: 1)",
    )
    parser.add_argument(
        "--to", dest="to_round", type=int, default=None,
        help="last round to recompute (default: latest complete round)",
    )
    parser.add_argument(
        "--no-publish", action="store_true",
        help="rewrite the parquet cache only, skip chart JSON",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="use only the on-disk box-score cache; never hit the API",
    )
    parser.add_argument(
        "--pause", type=float, default=0.25,
        help="seconds between API requests; raised automatically on 429 (default: 0.25)",
    )
    parser.add_argument(
        "--boxscore-cache", type=Path, default=BOXSCORE_CACHE_DIR,
        help=f"on-disk box-score store (default: {BOXSCORE_CACHE_DIR})",
    )
    parser.add_argument(
        "--refresh-meta", action="store_true",
        help="re-fetch season metadata instead of reusing the cached copy",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    games_meta = load_games_metadata_cached(
        cache_dir=args.boxscore_cache,
        season=DEFAULT_SEASON,
        competition=DEFAULT_COMPETITION,
        offline=args.offline,
        refresh=args.refresh_meta,
    )
    boxscore_api = CachingBoxScoreData(
        DEFAULT_COMPETITION,
        cache_dir=args.boxscore_cache,
        season=DEFAULT_SEASON,
        pause=args.pause,
        offline=args.offline,
    )

    # Same Regular Season cap as refresh_and_publish: euroleague_api numbers
    # Playoffs/Final Four past 38, but the Next-N schedule is RS-only.
    latest_round = min(detect_latest_complete_round(games_meta), TOTAL_SEASON_ROUNDS)
    first = max(1, args.from_round)
    last = min(args.to_round or latest_round, latest_round)
    if first > last:
        print(f"Nothing to do: --from {first} is past the last round {last}.")
        return

    season_label = f"{DEFAULT_SEASON}-{(DEFAULT_SEASON + 1) % 100:02d}"

    gamecodes = gamecodes_through_round(games_meta, last)
    cached_on_disk = sum(
        1 for gc in gamecodes if boxscore_api._cache_path(gc).exists()
    )
    print(
        f"Box scores needed through round {last}: {len(gamecodes)} game(s), "
        f"{cached_on_disk} already cached in {boxscore_api._cache_dir}"
    )
    boxscore_api.prefetch(DEFAULT_SEASON, gamecodes)
    boxscore_api.sweep_failures(DEFAULT_SEASON)

    if boxscore_api.failed:
        print(
            f"\nERROR: {len(boxscore_api.failed)} game(s) could not be fetched. "
            f"Every team that played them would have understated totals, so the "
            f"parquet is not being rewritten. Everything fetched so far is cached "
            f"in {boxscore_api._cache_dir} — re-run to pick up only the missing games."
        )
        for (season, gamecode), err in sorted(boxscore_api.failed.items(), key=str)[:20]:
            print(f"  season {season} gamecode {gamecode}: {err}")
        sys.exit(1)

    print(f"Recomputing rounds {first}..{last} (latest complete: {latest_round})")

    for round_num in range(first, last + 1):
        t0 = time.time()
        team_ratings, sos_net, sos_win = compute_for_round(
            cache_dir=CACHE_DIR,
            games_meta=games_meta,
            season=DEFAULT_SEASON,
            boxscore_api=boxscore_api,
            round_max=round_num,
            force=True,
        )
        elapsed = time.time() - t0

        if args.no_publish:
            print(f"  round {round_num}: parquet rewritten ({elapsed:.1f}s)")
            continue

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
        stale_note = f", pruned {stale} stale next-N spec(s)" if stale else ""
        print(
            f"  round {round_num}: parquet + {len(artifacts)} chart spec(s) "
            f"({elapsed:.1f}s){stale_note}"
        )

    if not args.no_publish:
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

    total = time.time() - started
    print(
        f"\nDone in {total / 60:.1f} min. Box scores: {boxscore_api.fetched} fetched, "
        f"{boxscore_api.disk_hits} from disk, {boxscore_api.memo_hits} from memo "
        f"({boxscore_api.retried} retries, {boxscore_api.throttled} rate-limited)."
    )


if __name__ == "__main__":
    main()
