import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import time


from sos.config import (
    DEFAULT_SEASON,
    DEFAULT_COMPETITION,
    SCHEDULE_FILENAME,
    TOTAL_SEASON_ROUNDS,
    DEFAULT_N_NEXT,
    CACHE_DIR,
)

from sos.data import load_games_metadata
from sos.rounds import detect_latest_complete_round
from sos.cache import compute_for_round as _compute_for_round_cached
from sos.compute import make_nextN_sos_table

from sos.charts import (
    build_nextN_altair_logos_table,
    make_sos_table_chart,
    make_sos_scatter_and_side_table,
)

from sos.utils import team_to_logo_path, logo_to_dataurl

# NOTE: app.py is a temporary fallback kept only until the static Vercel site
# (see frontend/ and scripts/) is verified end-to-end. It renders the same charts
# locally from cache/rounds/*.parquet, with logos inlined as base64 rather than
# fetched by URL.

# Persistent app state
if "data_ready" not in st.session_state:
    st.session_state.data_ready = False


def _init_state():
    if "base_loaded" not in st.session_state:
        st.session_state.base_loaded = False

    # Cache for per-round SOS metrics: { round: (ratings, sos_net, sos_win) }
    if "round_cache" not in st.session_state:
        st.session_state.round_cache = {}

    if "precomputed_upto" not in st.session_state:
        st.session_state.precomputed_upto = 0

    if "precompute_done" not in st.session_state:
        st.session_state.precompute_done = False

    if "precompute_running" not in st.session_state:
        st.session_state.precompute_running = False


_init_state()


st.warning("This project is migrating to Vercel. You can access the new version here: https://euroleague-sos-dashboard.vercel.app/")

# Page setup
st.set_page_config(page_title="EuroLeague SoS Dashboard", layout="wide")
st.title("EuroLeague Strength of Schedule Dashboard")

# Local schedule file path
schedule_path = SCHEDULE_FILENAME

# Data loading (fetched before the sidebar so the round selector can be
# bounded by the actual latest completed round, not a hardcoded number)
def load_base_data(season: int, competition_code: str):
    if st.session_state.base_loaded:
        return (
            st.session_state.games_meta,
            st.session_state.team_stats_api,
        )

    with st.spinner("Fetching API data..."):
        games_meta, team_stats_api, _metadata_api = load_games_metadata(
            season=season,
            competition_code=competition_code,
        )

    st.session_state.games_meta = games_meta
    st.session_state.team_stats_api = team_stats_api
    st.session_state.base_loaded = True
    return games_meta, team_stats_api


games_meta, team_stats_api = load_base_data(
    season=DEFAULT_SEASON,
    competition_code=DEFAULT_COMPETITION,
)
LATEST_AVAILABLE_ROUND = min(
    TOTAL_SEASON_ROUNDS,
    max(1, detect_latest_complete_round(games_meta)),
)

# Sidebar filters and settings
with st.sidebar:

    st.header("Configuration")

    mobile_mode = st.toggle(
        "Mobile layout",
        key="mobile_mode",
        value=False,
        help="Adjusts chart dimensions and layout for smaller screens.",
    )

    season = st.sidebar.selectbox(
        "Season",
        ["2025"],
        index=0,
        disabled=True,
        help="Only 2025 season is currently supported.",
    )

    competition_code = st.text_input(
        "Competition code",
        value=DEFAULT_COMPETITION,
        disabled=True,
        help='Defaults to EuroLeague (E).',
    )

    current_round = st.number_input(
        "Current round",
        min_value=1,
        max_value=int(LATEST_AVAILABLE_ROUND),
        value=int(LATEST_AVAILABLE_ROUND),
        step=1,
        help="Select the latest round to include in calculations.",
    )

    if current_round == LATEST_AVAILABLE_ROUND:
        st.caption("Showing data up to the latest available round.")

    n_next = st.slider("Number of next games (N)", 1, 10, DEFAULT_N_NEXT, 1)


season_label = f"{int(season)}-{(int(season) + 1) % 100:02d}"
mobile_mode = st.session_state["mobile_mode"]

# Responsive chart presets
if not mobile_mode:
    # Desktop / Tablet
    NEXTN_KWARGS = dict(
        left_col_width=320,
        sos_col_width=110,
        games_col_width=500,
        logo_size_main=24,
        logo_size_opp=24,
        font_size=14,
        title_font_size=19,
    )
    NEXTN_STREAMLIT_WIDTH = "stretch"

    SCATTER_MAIN_W, SCATTER_MAIN_H = 560, 600
    SCATTER_TABLE_W, SCATTER_TABLE_H = 380, 600

    SEASON_KWARGS = dict(
        team_col_width=80,
        net_col_width=190,
        win_col_width=190,
        logo_size=24,
        row_height=26,
        name_font_size=13,
        value_font_size=10,
        font_size=13,
        title_font_size=16,
    )
    SEASON_STREAMLIT_WIDTH = None
else:
    # Mobile
    NEXTN_KWARGS = dict(
        left_col_width=45,
        sos_col_width=65,
        games_col_width=240,
        logo_size_main=20,
        logo_size_opp=24,
        font_size=11,
        title_font_size=13,
    )
    NEXTN_STREAMLIT_WIDTH = "content"

    SCATTER_MAIN_W, SCATTER_MAIN_H = 380, 550
    SCATTER_TABLE_W, SCATTER_TABLE_H = 380, 550

    SEASON_KWARGS = dict(
        team_col_width=52,
        net_col_width=90,
        win_col_width=90,
        logo_size=21,
        row_height=25,
        name_font_size=11,
        value_font_size=9,
        font_size=11,
        title_font_size=13,
    )
    SEASON_STREAMLIT_WIDTH = None


is_small_screen = mobile_mode


def compute_for_round(round_max: int):
    return _compute_for_round_cached(
        cache_dir=CACHE_DIR,
        games_meta=games_meta,
        season=DEFAULT_SEASON,
        team_stats_api=team_stats_api,
        round_max=round_max,
    )


def load_nextN_sos(
    current_round: int,
    schedule_path: str,
    games_meta: pd.DataFrame,
    team_ratings: pd.DataFrame,
    n_next: int,
):
    """Calculate SOS for the upcoming N games."""
    return make_nextN_sos_table(
        current_round=current_round,
        schedule_path=schedule_path,
        games_meta=games_meta,
        team_ratings=team_ratings,
        n_next=n_next,
    )


# Navigation logic
tab_labels = [
    "Info / About Project",
    "Next-N Games SoS",
    "SoS vs Team NetRtg Scatter",
    "NetRtg & Win% Methods Table",
]

selected_tab = st.radio(
    "Navigation",
    tab_labels,
    horizontal=True,
    label_visibility="collapsed",
    key="active_tab",
)

def warm_cache_upto(default_round: int):
    default_round = int(default_round)

    if st.session_state.precompute_done or st.session_state.precompute_running:
        return

    st.session_state.precompute_running = True

    start_r = max(1, st.session_state.precomputed_upto + 1)
    rounds_to_do = list(range(start_r, default_round + 1))

    if rounds_to_do:
        with st.spinner(f"Precomputing rounds 1 → {default_round}..."):
            for r in rounds_to_do:
                compute_for_round(r)

        st.session_state.precomputed_upto = default_round

    st.session_state.precompute_done = True
    st.session_state.precompute_running = False


# Background precomputation
warm_cache_upto(LATEST_AVAILABLE_ROUND)

# --- Tab 0: Project Info ---
if selected_tab == "Info / About Project":
    
    if not is_small_screen:
        st.info(
            "**Viewing on a small screen?**\n\n"
            "This dashboard supports a **Mobile layout** for phones and tablets.\n\n"
            "Toggle it in the **sidebar** to optimize chart sizes for your device.",
            icon="ℹ️",
        )

    st.markdown(
        """
## Project overview
This dashboard analyzes **Strength of Schedule (SoS)** in the EuroLeague by combining:
- Team performance metrics (Net Rating, Win%)
- Official EuroLeague schedule data
- Upcoming opponent difficulty

The objective is to contextualize team performance by answering:
- How difficult has a team’s schedule been?
- How difficult will it be going forward?
"""
    )

    st.markdown("---")

    st.markdown(
        """
## Data sources

### Schedule (Parsed PDF)
The Regular Season schedule was parsed from official EuroLeague documents:
- `EL_2025_26_EL_RS_Schedule.csv`

Used for forecasting future opponent difficulty.

### Team stats (euroleague-api)
Historical performance data fetched via the open-source wrapper:
- **euroleague-api**  
  https://github.com/giasemidis/euroleague_api

This powers our possession-based efficiency metrics and historical SOS calculations.
"""
    )

    st.markdown("---")
    st.markdown("## Core metrics")

    st.markdown("### Net Rating (NetRtg)")
    st.markdown("Measures points differential per 100 possessions:")
    st.latex(r"\text{NetRtg} = \text{OffRtg} - \text{DefRtg}")

    st.markdown(
        """
Where:
- **OffRtg** = points scored per 100 possessions  
- **DefRtg** = points allowed per 100 possessions  

Interpretation:
- Higher NetRtg ⇒ stronger team
- Lower NetRtg ⇒ weaker team
"""
    )

    st.markdown("---")
    st.markdown("## Strength of Schedule (Formulas)")
    st.markdown(
        """
Adapted from **Hack-a-Stat** methodology and applied to EuroLeague datasets.
"""
    )

    st.markdown("### Definitions")
    st.markdown(
        """
- **OppW%**: Opponent Winning Percentage  
- **TmGP**: Team Games Played  
- **OppNetRtg**: Opponent Net Rating  
"""
    )

    st.markdown("### Opponents’ Winning Percentage (OW%)")
    st.latex(r"\text{OW\%} = \frac{\sum_{i=1}^{n} \text{OppW\%}_i}{\text{TmGP}}")

    st.markdown("### Opponents’ Opponents’ Winning Percentage (OOW%)")
    st.latex(r"\text{OOW\%} = \frac{\sum_{i=1}^{n} \text{OW\%}_i}{\text{TmGP}}")

    st.markdown("### Strength of Schedule — Win% based")
    st.latex(r"\text{SoS}_{\text{Win}} = \frac{2 \cdot \text{OW\%} + \text{OOW\%}}{3}")

    st.markdown(
        """
This formula weights direct opponent strength more heavily while reducing noise via secondary opponent metrics.
"""
    )

    st.markdown("---")
    st.markdown("## Net Rating–based Strength of Schedule")
    st.markdown(
        """
Same logic applied to **Net Rating** for an efficiency-based approach.
"""
    )

    st.markdown("### Opponents’ Net Rating (OppNetRtg)")
    st.latex(r"\text{OppNetRtg} = \frac{\sum_{i=1}^{n} \text{NetRtg}_i}{\text{TmGP}}")

    st.markdown("### Opponents’ Opponents’ Net Rating (OONetRtg)")
    st.latex(r"\text{OONetRtg} = \frac{\sum_{i=1}^{n} \text{OppNetRtg}_i}{\text{TmGP}}")

    st.markdown("### Strength of Schedule — Net Rating based")
    st.latex(r"\text{SoS}_{\text{Net}} = \frac{2 \cdot \text{OppNetRtg} + \text{OONetRtg}}{3}")

    st.markdown(
        """
Efficiency-based SOS is less sensitive to close-game variance and remains more stable early in the season.
"""
    )

    st.markdown("---")
    st.markdown("## How to interpret the charts")
    st.markdown(
        """
### Tab 1: Next-N Games (Logo Table)
- Forecasts upcoming SOS difficulty.
- Color-coded cells represent opponent strength (NetRtg).

### Tab 2: SoS(Net) vs NetRtg Scatter
- Contextualizes team performance against schedule difficulty.
- Quadrants highlight overachievers and schedule outliers.

### Tab 3: Season SoS Table
- Comparison between NetRtg-based and Win%-based SOS estimates.
"""
    )

    st.markdown("---")
    st.markdown(
        """
## Implementation notes
- Visualization layer: **Altair**
- Assets: Base64 embedded team logos from `frontend/data/logos/`
- Data persistence: Parquet-based local caching
"""
    )

    st.markdown(
        """
## Roadmap / Limitations
Current scope:
- **Competition:** EuroLeague (E)
- **Season:** 2025

Future updates will include EuroCup support and historical seasonal analysis.
"""
    )

    # Glossary section
    with st.expander("Glossary", expanded=False):
        st.markdown(
            """
### Team efficiency metrics
- **OffRtg**: points scored per 100 possessions  
- **DefRtg**: points allowed per 100 possessions  
"""
        )
        st.latex(r"\text{OffRtg} = \frac{\text{Points Scored}}{\text{Possessions}} \times 100")
        st.latex(r"\text{DefRtg} = \frac{\text{Points Allowed}}{\text{Possessions}} \times 100")
        st.markdown("- **NetRtg**: efficiency differential per 100 possessions")
        st.latex(r"\text{NetRtg} = \text{OffRtg} - \text{DefRtg}")

        st.markdown("---")
        st.markdown(
            """
### SOS Metrics (Win% Based)
- **OppW%**: individual opponent win rate
- **OW%**: average opponent win rate
"""
        )
        st.latex(r"\text{OW\%} = \frac{1}{\text{TmGP}} \sum_{i=1}^{n} \text{OppW\%}_i")
        st.markdown("- **OOW%**: average win rate of opponents' opponents")
        st.latex(r"\text{OOW\%} = \frac{1}{\text{TmGP}} \sum_{i=1}^{n} \text{OW\%}_i")
        st.markdown("- **SoS (Win%)**: composite win-rate based difficulty")
        st.latex(r"\text{SoS}_{\text{Win}} = \frac{2 \cdot \text{OW\%} + \text{OOW\%}}{3}")

        st.markdown("---")
        st.markdown(
            """
### SOS Metrics (NetRtg Based)
- **OppNetRtg**: average opponent efficiency
"""
        )
        st.latex(r"\text{OppNetRtg} = \frac{1}{\text{TmGP}} \sum_{i=1}^{n} \text{NetRtg}_i")
        st.markdown("- **OONetRtg**: average efficiency of opponents' opponents")
        st.latex(r"\text{OONetRtg} = \frac{1}{\text{TmGP}} \sum_{i=1}^{n} \text{OppNetRtg}_i")
        st.markdown("- **SoS (NetRtg)**: composite efficiency-based difficulty")
        st.latex(r"\text{SoS}_{\text{Net}} = \frac{2 \cdot \text{OppNetRtg} + \text{OONetRtg}}{3}")

    # Local development info
    with st.expander("Run locally", expanded=False):
        st.markdown(
            """
- Required files:
  - `EL_2025_26_EL_RS_Schedule.csv`
  - `frontend/data/logos/` directory
- Launch command:
  - `streamlit run app.py`
"""
        )


# --- Tab 1: Next-N SOS Forecast ---
elif selected_tab == "Next-N Games SoS":
    try:
        with st.spinner(f"Forecasting next {int(n_next)} games SOS..."):
            team_ratings, sos_net, sos_win = compute_for_round(current_round)
            sos_nextN_df = load_nextN_sos(
                current_round=int(current_round),
                schedule_path=schedule_path,
                games_meta=games_meta,
                team_ratings=team_ratings,
                n_next=int(n_next),
            )

            nextN_chart = build_nextN_altair_logos_table(
            nextN_df=sos_nextN_df,
            team_ratings=team_ratings,
            team_to_logo_path_fn=team_to_logo_path,
            round_ref=int(current_round),
            n_next=int(n_next),
            **NEXTN_KWARGS,
            mobile_mode=mobile_mode,
        )
        st.altair_chart(nextN_chart, width=NEXTN_STREAMLIT_WIDTH)


    except FileNotFoundError:
        st.error(f"Schedule file missing: {schedule_path}")
        st.stop()
    except Exception as e:
        st.error(f"Computation error: {e}")
        st.stop()



# --- Tab 2: Performance vs Difficulty Scatter ---
elif selected_tab == "SoS vs Team NetRtg Scatter":
    
    with st.spinner("Rendering scatter plot..."):
        team_ratings, sos_net, sos_win = compute_for_round(current_round)
        main_chart, side_table_chart = make_sos_scatter_and_side_table(
        sos_net=sos_net,
        team_ratings=team_ratings,
        team_to_logo_path=team_to_logo_path,
        logo_path_to_url_fn=logo_to_dataurl,
        top_k=5,
        bottom_k=5,
        round_ref=int(current_round),
        season_label=season_label,
        main_w=SCATTER_MAIN_W,
        main_h=SCATTER_MAIN_H,
        table_w=SCATTER_TABLE_W,
        table_h=SCATTER_TABLE_H,
    )

    if mobile_mode:
        st.altair_chart(main_chart, width="stretch")
        st.altair_chart(side_table_chart, width="stretch")
    else:
        col1, col2 = st.columns([3, 2])
        with col1:
            st.altair_chart(main_chart, width="stretch")
        with col2:
            st.altair_chart(side_table_chart, width="content")

# --- Tab 3: Season Metrics Comparison ---
elif selected_tab == "NetRtg & Win% Methods Table":
    with st.spinner("Generating SOS table..."):
        team_ratings, sos_net, sos_win = compute_for_round(current_round)
        sos_table_chart = make_sos_table_chart(
        sos_net=sos_net,
        sos_win=sos_win,
        team_to_logo_path=team_to_logo_path,
        logo_path_to_url_fn=logo_to_dataurl,
        round_ref=int(current_round),
        season_label=season_label,
        **SEASON_KWARGS,
        mobile_mode=mobile_mode,
    )
    st.altair_chart(sos_table_chart, use_container_width=False, theme=None)


