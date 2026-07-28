import os
from pathlib import Path

# Competition metadata
DEFAULT_SEASON = 2025
DEFAULT_COMPETITION = "E"

# Input datasets
SCHEDULE_FILENAME = "EL_2025_26_EL_RS_Schedule.csv"
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