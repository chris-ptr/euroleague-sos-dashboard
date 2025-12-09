import streamlit as st
import pandas as pd

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
    make_sos_scatter_with_table,
)

from sos.utils import team_to_logo_path, logo_to_dataurl


# ---------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------
st.set_page_config(page_title="EuroLeague SoS Dashboard", layout="wide")
st.title("EuroLeague Strength of Schedule Dashboard")


# ---------------------------------------------------------
# Sidebar configuration
# ---------------------------------------------------------
with st.sidebar:
    st.header("Configuration")

    season = st.number_input(
        "Season (start year)",
        min_value=2010,
        max_value=2100,
        value=DEFAULT_SEASON,
        step=1,
    )

    competition_code = st.text_input(
        "Competition code",
        value=DEFAULT_COMPETITION,
        help='"E" = EuroLeague, "U" = EuroCup, etc.',
    )

    current_round = st.number_input(
        "Current round (for SoS up to / next-N)",
        min_value=1,
        max_value=40,
        value=DEFAULT_CURRENT_ROUND,
        step=1,
    )

    n_next = st.slider(
        "Number of next games (N)",
        min_value=1,
        max_value=10,
        value=DEFAULT_N_NEXT,
        step=1,
    )

# Schedule path is fixed from config; user cannot change it in the UI
schedule_path = SCHEDULE_FILENAME

season_label = f"{int(season)}-{(int(season) + 1) % 100:02d}"


# ---------------------------------------------------------
# Cached data loading + core computations
# ---------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_meta_and_ratings(
    season: int,
    competition_code: str,
    current_round: int,
):
    """
    Load games metadata, compute team ratings up to current_round,
    and season-to-date SoS (NetRtg + Win%).
    """
    # 1) Games metadata + API object for advanced stats
    games_meta, team_stats_api, _metadata_api = load_games_metadata(
        season=season,
        competition_code=competition_code,
    )

    # 2) Team ratings up to current_round
    team_ratings = compute_team_ratings_up_to_round(
        games_meta=games_meta,
        season=season,
        team_stats_api=team_stats_api,
        round_max=current_round,
    )

    # 3) Season-to-date SoS based on Net Rating
    sos_net = compute_sos_from_netrtg_up_to_round(
        games_meta=games_meta,
        team_ratings=team_ratings,
        round_max=current_round,
    )

    # 4) Season-to-date SoS based on Win% (returns TEAM_NAME + SoS)
    sos_win = compute_sos_from_winpct_up_to_round(
        games_meta=games_meta,
        round_max=current_round,
    )

    return games_meta, team_ratings, sos_net, sos_win


try:
    games_meta, team_ratings, sos_net, sos_win = load_meta_and_ratings(
        season=int(season),
        competition_code=competition_code,
        current_round=int(current_round),
    )
except Exception as e:
    st.error(f"Error loading metadata / computing ratings: {e}")
    st.stop()


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
# Tabs
# ---------------------------------------------------------
tabs = st.tabs(
    [
        f"Next {int(n_next)} Games SoS",
        "Season SoS: Scatter + Table",
        "Season SoS Table Only",
    ]
)

# =========================================================
# TAB 1 – Next N Games SoS (build_nextN_altair_logos_table)
# =========================================================
with tabs[0]:
    st.subheader(
        f"Strength of Schedule – Next {int(n_next)} Games "
        f"(from Round {int(current_round)})"
    )

    try:
        sos_nextN_df = load_nextN_sos(
            current_round=int(current_round),
            schedule_path=schedule_path,
            games_meta=games_meta,
            team_ratings=team_ratings,
            n_next=int(n_next),
        )
    except FileNotFoundError:
        st.error(f"Schedule file not found: {schedule_path}")
        st.stop()
    except Exception as e:
        st.error(f"Error computing next-N SoS: {e}")
        st.stop()

    nextN_chart = build_nextN_altair_logos_table(
        nextN_df=sos_nextN_df,
        team_ratings=team_ratings,
        team_to_logo_path_fn=team_to_logo_path,
        round_ref=int(current_round),
        n_next=int(n_next),
    )
    st.altair_chart(nextN_chart, width="stretch")


# =========================================================
# TAB 2 – SoS(NetRtg) vs NetRtg scatter + side table
#          (make_sos_scatter_with_table)
# =========================================================
with tabs[1]:
    st.subheader(
        f"SoS(NetRtg) vs Team NetRtg – Season-to-date (up to Round {int(current_round)})"
    )

    scatter_chart = make_sos_scatter_with_table(
        sos_net=sos_net,
        team_ratings=team_ratings,
        team_to_logo_path=team_to_logo_path,
        logo_to_dataurl=logo_to_dataurl,
        top_k=5,
        bottom_k=5,
        round_ref=int(current_round),
        season_label=season_label,
        # background handled in the chart (default "#a3a1a1")
    )
    st.altair_chart(scatter_chart)


# =========================================================
# TAB 3 – Season SoS table (make_sos_table_chart)
# =========================================================
with tabs[2]:
    st.subheader(
        f"Season-to-date SoS – NetRtg & Win% (up to Round {int(current_round)})"
    )

    sos_table_chart = make_sos_table_chart(
        sos_net=sos_net,
        sos_win=sos_win,
        team_to_logo_path=team_to_logo_path,
        logo_to_dataurl=logo_to_dataurl,
        round_ref=int(current_round),
        season_label=season_label,
        # background handled inside make_sos_table_chart ("#a3a1a1")
    )
    st.altair_chart(sos_table_chart, width="stretch")
