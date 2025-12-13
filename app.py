import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import time
from pathlib import Path


from sos.config import (
    DEFAULT_SEASON,
    DEFAULT_COMPETITION,
    SCHEDULE_FILENAME,
    DEFAULT_CURRENT_ROUND,
    DEFAULT_N_NEXT,
)

from sos.data import load_games_metadata
from sos.compute import (
    compute_team_ratings_up_to_round,
    compute_sos_from_netrtg_up_to_round,
    compute_sos_from_winpct_up_to_round,
    make_nextN_sos_table,
)

from sos.charts import (
    build_nextN_altair_logos_table,
    make_sos_table_chart,
    make_sos_scatter_and_side_table,
)

from sos.utils import team_to_logo_path, logo_to_dataurl

CACHE_DIR = Path("cache/rounds")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _make_parquet_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns that can't be written to Parquet (nested objects like DataFrames),
    and convert Path-like objects to strings if needed.
    """
    df = df.copy()

    # Drop nested / un-serializable columns
    bad_cols = []
    for c in df.columns:
        # anything "object" can hide dicts/dataframes/etc.
        if df[c].dtype == "object":
            # if any cell is a DataFrame/list/dict => drop
            if df[c].apply(lambda x: isinstance(x, (pd.DataFrame, dict, list, tuple, set))).any():
                bad_cols.append(c)

    if bad_cols:
        df = df.drop(columns=bad_cols)

    # Convert Path objects to strings if any slipped in
    for c in df.columns:
        if df[c].dtype == "object":
            if df[c].apply(lambda x: hasattr(x, "__fspath__")).any():
                df[c] = df[c].astype(str)

    return df



if "data_ready" not in st.session_state:
    st.session_state.data_ready = False


def _init_state():
    if "base_loaded" not in st.session_state:
        st.session_state.base_loaded = False

    # Holds per-round computed results:
    # { round_int: (team_ratings_df, sos_net_df, sos_win_df) }
    if "round_cache" not in st.session_state:
        st.session_state.round_cache = {}

    if "precomputed_upto" not in st.session_state:
        st.session_state.precomputed_upto = 0

    if "precompute_done" not in st.session_state:
        st.session_state.precompute_done = False

    if "precompute_running" not in st.session_state:
        st.session_state.precompute_running = False


_init_state()


# --------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------
st.set_page_config(page_title="EuroLeague SoS Dashboard", layout="wide")
st.title("EuroLeague Strength of Schedule Dashboard")



# Schedule path is fixed from config; user cannot change it in the UI
schedule_path = SCHEDULE_FILENAME

# ---------------------------------------------------------
# Sidebar configuration
# ---------------------------------------------------------
with st.sidebar:

    st.header("Configuration")

    mobile_mode = st.toggle(
        "Mobile layout",
        key="mobile_mode",
        value=False,
        help="Uses smaller charts and stack layouts for phones.",
    )

    season = st.sidebar.selectbox(
        "Season",
        ["2025"],
        index=0,
        disabled=True,
        help="Only the 2025 season is available at the moment.",
    )

    competition_code = st.text_input(
        "Competition code",
        value=DEFAULT_COMPETITION,
        disabled=True,
        help='Only "E" (EuroLeague) is available at the moment.',
    )

    LATEST_AVAILABLE_ROUND = DEFAULT_CURRENT_ROUND  # Maybe I should compute it dynamically later

    current_round = st.number_input(
        "Current round",
        min_value=1,
        max_value=int(LATEST_AVAILABLE_ROUND),
        value=min(int(DEFAULT_CURRENT_ROUND), int(LATEST_AVAILABLE_ROUND)),
        step=1,
        help="Enter the latest non completed round number.",
    )

    if current_round == LATEST_AVAILABLE_ROUND:
        st.caption("Showing data up to the latest available round.")

    n_next = st.slider("Number of next games (N)", 1, 10, DEFAULT_N_NEXT, 1)


season_label = f"{int(season)}-{(int(season) + 1) % 100:02d}"
mobile_mode = st.session_state["mobile_mode"]

# ---------------------------------------------------------
# Responsive presets (fast toggle)
# ---------------------------------------------------------
if not mobile_mode:
    # Tablet / Default
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
    SEASON_STREAMLIT_WIDTH = "stretch"

else:
    # Mobile (smaller than tablet)
    NEXTN_KWARGS = dict(
        left_col_width=45,
        sos_col_width=65,
        games_col_width=240,
        logo_size_main=20,
        logo_size_opp=24,
        font_size=11,
        title_font_size=13,
    )
    NEXTN_STREAMLIT_WIDTH = "content"  # allow horizontal scroll

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
    SEASON_STREAMLIT_WIDTH = "content"  # allow scroll


# Treat mobile_mode as phone/tablet layout
is_small_screen = mobile_mode

# ---------------------------------------------------------
# Cached data loading + core computations
# ---------------------------------------------------------
def load_base_data(season: int, competition_code: str):
    if st.session_state.base_loaded:
        return (
            st.session_state.games_meta,
            st.session_state.team_stats_api,
        )

    with st.spinner("Loading base data…"):
        games_meta, team_stats_api, _metadata_api = load_games_metadata(
            season=season,
            competition_code=competition_code,
        )

    st.session_state.games_meta = games_meta
    st.session_state.team_stats_api = team_stats_api
    st.session_state.base_loaded = True
    return games_meta, team_stats_api


games_meta, team_stats_api = load_base_data(
    season=int(season),
    competition_code=competition_code,
)

def compute_for_round(round_max: int):
    round_max = int(round_max)
    cache_file = CACHE_DIR / f"round_{round_max}.parquet"

    # 1) Load from disk if exists
    if cache_file.exists():
        df = pd.read_parquet(cache_file)

        team_ratings = df[df["TYPE"] == "team_ratings"].drop(columns="TYPE")
        sos_net = df[df["TYPE"] == "sos_net"].drop(columns="TYPE")
        sos_win = df[df["TYPE"] == "sos_win"].drop(columns="TYPE")

        return team_ratings, sos_net, sos_win

    # 2) Otherwise compute
    with st.status(f"Computing metrics for Round {round_max}…", expanded=False):
        team_ratings = compute_team_ratings_up_to_round(
            games_meta=games_meta,
            season=int(season),
            team_stats_api=team_stats_api,
            round_max=round_max,
        )

        sos_net = compute_sos_from_netrtg_up_to_round(
            games_meta=games_meta,
            team_ratings=team_ratings,
            round_max=round_max,
        )

        sos_win = compute_sos_from_winpct_up_to_round(
            games_meta=games_meta,
            round_max=round_max,
        )

    team_ratings_safe = _make_parquet_safe(team_ratings)
    sos_net_safe = _make_parquet_safe(sos_net)
    sos_win_safe = _make_parquet_safe(sos_win)

    out = pd.concat(
        [
            team_ratings_safe.assign(TYPE="team_ratings"),
            sos_net_safe.assign(TYPE="sos_net"),
            sos_win_safe.assign(TYPE="sos_win"),
        ],
        ignore_index=True,
    )

    out.to_parquet(cache_file, index=False)

    return team_ratings, sos_net, sos_win


def load_nextN_sos(
    current_round: int,
    schedule_path: str,
    games_meta: pd.DataFrame,
    team_ratings: pd.DataFrame,
    n_next: int,
):
    """
    Wrapper around make_nextN_sos_table.
    Not cached to avoid hashing DataFrames in Streamlit.
    """
    return make_nextN_sos_table(
        current_round=current_round,
        schedule_path=schedule_path,
        games_meta=games_meta,
        team_ratings=team_ratings,
        n_next=n_next,
    )


# ---------------------------------------------------------
# Stateful "tabs" (keeps selection on rerun)
# ---------------------------------------------------------
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

    # Already done for this session
    if st.session_state.precompute_done:
        return

    # Prevent double-running if a rerun happens mid-precompute
    if st.session_state.precompute_running:
        return

    st.session_state.precompute_running = True

    start_r = max(1, st.session_state.precomputed_upto + 1)
    rounds_to_do = list(range(start_r, default_round + 1))

    if rounds_to_do:
        with st.spinner(f"Precomputing rounds 1 → {default_round}…"):
            for r in rounds_to_do:
                compute_for_round(r)

        st.session_state.precomputed_upto = default_round

    st.session_state.precompute_done = True
    st.session_state.precompute_running = False


# Warm cache up to DEFAULT_CURRENT_ROUND (fast slider inside that range)
warm_cache_upto(DEFAULT_CURRENT_ROUND)

# =========================================================
# TAB 0 – Info / About Project
# =========================================================
if selected_tab == "Info / About Project":
    
    if not is_small_screen:
        st.info(
            "**Viewing on a small screen?**\n\n"
            "This dashboard supports a **Mobile layout** for phones and tablets.\n\n"
            "You can enable or disable it at any time from the **sidebar → Mobile layout** toggle "
            "to optimize chart sizes and layout for your device.",
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

### Schedule (official EuroLeague PDF → parsed)
The Regular Season schedule was parsed from the **official EuroLeague schedule PDF** and exported into:
- `EL_2025_26_EL_RS_Schedule.csv`

This file is used to determine future opponents for the **Next-N** analysis.

### Team stats (open-source euroleague-api)
Team ratings and other team-level statistics are calculated with the help of the open-source project:
- **euroleague-api**  
  https://github.com/giasemidis/euroleague_api

This helps me make the dataset for team game data throughout the season and is used to determine opponents played so far, through my helper functions, compute opponent Net Rating, and calculate all components required for the Strength of Schedule metrics.
"""
    )

    st.markdown("---")
    st.markdown("## Core metrics")

    st.markdown("### Net Rating (NetRtg)")
    st.markdown("Net Rating measures how much a team outperforms its opponents per 100 possessions:")
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
The following definitions are adapted from **Hack-a-Stat: Learn a Stat: Strength of Schedule** and applied to EuroLeague data.
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
This formulation:
- Weights direct opponent strength more heavily
- Reduces volatility compared to raw OppW%
"""
    )

    st.markdown("---")
    st.markdown("## Net Rating–based Strength of Schedule")
    st.markdown(
        """
The same can be applied using **Net Rating** instead of Win%.
This is the primary efficiency-based approach used throughout the dashboard.
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
Why NetRtg-based SoS?
- Less sensitive to close-game variance
- More stable early in the season
- Captures how strong opponents actually are
"""
    )

    st.markdown("---")
    st.markdown("## How to interpret the charts")
    st.markdown(
        """
### Tab 1: Next-N Games (Logo Table)
- Left: team logo + team name
- Middle: Next-N SoS (NetRtg)
- Right: next N opponents, colored by opponent NetRtg

### Tab 2: SoS(Net) vs NetRtg Scatter + Side Table
- X-axis: SoS(Net) (reversed: tougher schedules on the left)
- Y-axis: team NetRtg
- Side table: top/bottom NetRtg teams with OffRtg/DefRtg
- Quadrants contextualize performance vs difficulty

### Tab 3: Season SoS Table (NetRtg vs Win%)
Two SoS estimates per team:
- NetRtg-based SoS
- Win%-based SoS
"""
    )

    st.markdown("---")
    st.markdown(
        """
## Implementation notes
- Charts are built using **Altair**
- Team logos are embedded via base64 data URLs
- Local logo files are loaded from: `team_logos/`
"""
    )

    st.markdown(
        """
## Limitations (current version)
Currently locked to:
- **Competition:** EuroLeague (E)
- **Season:** 2025

Support for additional competitions and seasons will be added later.
"""
    )

    # ---- Glossary only on Info tab ----
    with st.expander("Glossary", expanded=False):
        st.markdown(
            """
### Team efficiency metrics
- **OffRtg (Offensive Rating)**: points scored per 100 possessions  
- **DefRtg (Defensive Rating)**: points allowed per 100 possessions  
"""
        )
        st.latex(r"\text{OffRtg} = \frac{\text{Points Scored}}{\text{Possessions}} \times 100")
        st.latex(r"\text{DefRtg} = \frac{\text{Points Allowed}}{\text{Possessions}} \times 100")
        st.markdown("- **NetRtg (Net Rating)**: efficiency differential per 100 possessions")
        st.latex(r"\text{NetRtg} = \text{OffRtg} - \text{DefRtg}")

        st.markdown("---")
        st.markdown(
            """
### Strength of Schedule (SoS) helpers — Win% based
- **OppW%**: opponent winning percentage  
- **OW%**: average opponent winning percentage  
"""
        )
        st.latex(r"\text{OW\%} = \frac{1}{\text{TmGP}} \sum_{i=1}^{n} \text{OppW\%}_i")
        st.markdown("- **OOW%**: opponents’ opponents winning percentage")
        st.latex(r"\text{OOW\%} = \frac{1}{\text{TmGP}} \sum_{i=1}^{n} \text{OW\%}_i")
        st.markdown("- **SoS (Win%)**: weighted opponent difficulty")
        st.latex(r"\text{SoS}_{\text{Win}} = \frac{2 \cdot \text{OW\%} + \text{OOW\%}}{3}")

        st.markdown("---")
        st.markdown(
            """
### Strength of Schedule (SoS) helpers — Net Rating based
- **OppNetRtg**: average Net Rating of opponents  
"""
        )
        st.latex(r"\text{OppNetRtg} = \frac{1}{\text{TmGP}} \sum_{i=1}^{n} \text{NetRtg}_i")
        st.markdown("- **OONetRtg**: opponents’ opponents Net Rating")
        st.latex(r"\text{OONetRtg} = \frac{1}{\text{TmGP}} \sum_{i=1}^{n} \text{OppNetRtg}_i")
        st.markdown("- **SoS (NetRtg)**: efficiency-based schedule difficulty")
        st.latex(r"\text{SoS}_{\text{Net}} = \frac{2 \cdot \text{OppNetRtg} + \text{OONetRtg}}{3}")

    # ---- Run locally only on Info tab ----
    with st.expander("Run locally", expanded=False):
        st.markdown(
            """
- Ensure the schedule CSV exists:
  - `EL_2025_26_EL_RS_Schedule.csv`
- Ensure the logos folder exists:
  - `team_logos/`
- Start the app:
  - `streamlit run app.py`
"""
        )


# =========================================================
# TAB 1 – Next N Games SoS (build_nextN_altair_logos_table)
# =========================================================
elif selected_tab == "Next-N Games SoS":
    try:
        with st.spinner(f"Computing Next-{int(n_next)} games SoS…"):
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
        st.error(f"Schedule file not found: {schedule_path}")
        st.stop()
    except Exception as e:
        st.error(f"Error computing next-N SoS: {e}")
        st.stop()



# =========================================================
# TAB 2 – SoS(NetRtg) vs NetRtg scatter + side table
# =========================================================
elif selected_tab == "SoS vs Team NetRtg Scatter":
    
    with st.spinner("Building scatter + side table…"):
        team_ratings, sos_net, sos_win = compute_for_round(current_round)
        main_chart, side_table_chart = make_sos_scatter_and_side_table(
        sos_net=sos_net,
        team_ratings=team_ratings,
        team_to_logo_path=team_to_logo_path,
        logo_to_dataurl=logo_to_dataurl,
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

# =========================================================
# TAB 3 – Season SoS table (make_sos_table_chart)
# =========================================================
elif selected_tab == "NetRtg & Win% Methods Table":
    with st.spinner("Building SoS table…"):
        team_ratings, sos_net, sos_win = compute_for_round(current_round)
        sos_table_chart = make_sos_table_chart(
        sos_net=sos_net,
        sos_win=sos_win,
        team_to_logo_path=team_to_logo_path,
        logo_to_dataurl=logo_to_dataurl,
        round_ref=int(current_round),
        season_label=season_label,
        **SEASON_KWARGS,
        mobile_mode=mobile_mode,
    )
    st.altair_chart(sos_table_chart, width=SEASON_STREAMLIT_WIDTH)


