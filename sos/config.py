import os
from pathlib import Path

# Competition metadata
DEFAULT_SEASON = 2025
DEFAULT_COMPETITION = "E"

# Input datasets
SCHEDULE_FILENAME = "EL_2025_26_EL_RS_Schedule.csv"
TEAM_LOGO_DIR = Path("team_logos")

# Structural constant: EuroLeague Regular Season length, used only for input
# bounds. The *current* round is never hardcoded — it's always derived live
# via sos.rounds.detect_latest_complete_round().
TOTAL_SEASON_ROUNDS = 38
DEFAULT_N_NEXT = 5

# Local working cache (used by scripts/refresh_and_publish.py and
# scripts/seed_supabase.py as scratch space before/after syncing with Supabase)
CACHE_DIR = Path(os.environ.get("CACHE_DIR", "cache/rounds"))

# Supabase Storage config (values only — actual secrets stay in env vars / CI secrets)
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")
SUPABASE_CACHE_BUCKET = os.environ.get("SUPABASE_CACHE_BUCKET", "sos-cache")
SUPABASE_PUBLIC_BUCKET = os.environ.get("SUPABASE_PUBLIC_BUCKET", "sos-public")