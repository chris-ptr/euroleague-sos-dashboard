import os
from pathlib import Path

# Competition metadata
DEFAULT_SEASON = 2025
DEFAULT_COMPETITION = "E"


def season_label(season: int) -> str:
    """
    Display name for a season start year: 2025 -> "2025-26".

    One definition, used by the chart titles, the published manifest and the
    Streamlit fallback alike — the label used to be re-derived by hand at each
    call site, which is how a second season would have ended up rendered three
    slightly different ways.
    """
    return f"{int(season)}-{(int(season) + 1) % 100:02d}"

# Input datasets
# Lives inside the published tree so the browser can fetch the same PNGs the
# local Streamlit fallback reads off disk — one copy, no build-time copy step.
TEAM_LOGO_DIR = Path("frontend/data/logos")

# Structural constant: EuroLeague Regular Season length, used only for input
# bounds. The *current* round is never hardcoded — it's always derived live
# via sos.rounds.detect_latest_complete_round().
TOTAL_SEASON_ROUNDS = 38
DEFAULT_N_NEXT = 5

# Per-round parquet cache. Tracked in git and the canonical record of what has
# been computed — scripts/refresh_and_publish.py only computes rounds missing here.
CACHE_DIR = Path(os.environ.get("CACHE_DIR", "cache/rounds"))

# Precomputed chart JSON, written into the frontend tree and served as static
# files by Vercel (whose Root Directory is `frontend`, hence the /data URL prefix).
PUBLISH_DIR = Path(os.environ.get("PUBLISH_DIR", "frontend/data"))
PUBLISH_URL_PREFIX = "/data"


def season_publish_dir(season: int) -> str:
    """
    Where one season's chart JSON lives, relative to PUBLISH_DIR.

    Every artifact is filed under its season so a second season is purely a data
    operation — run the pipeline with a different DEFAULT_SEASON and the site
    picks it up from the manifest with no frontend change. Team logos stay at
    the shared /data/logos, since clubs outlive seasons.
    """
    return f"seasons/{int(season)}"