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
    make_sos_scatter_and_side_table,
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

    
    season = st.sidebar.selectbox(
        "Season",
        ["2025"],
        index=0,
        disabled=True,
        help="Only the 2025 season is available at the moment."
    )

    competition_code = st.text_input(
        "Competition code",
        value=DEFAULT_COMPETITION,
        disabled=True,
        help='Only "E" (EuroLeague) is available at the moment.',
    )

    current_round = st.number_input(
        "Current round",
        min_value=1,
        max_value=40,
        value=DEFAULT_CURRENT_ROUND,
        step=1,
        help="Enter the latest non completed round number.",
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
        "Info / About Project",
        f"Next {int(n_next)} Games SoS",
        "Strength of Schedule vs Team NetRtg Scatter",
        "NetRtg & Win% Methods Table",
    ]
)

# =========================================================
# TAB 0 – Info / About Project
# =========================================================

with tabs[0]:
    st.subheader(
        "A project to calculate schedule difficulty using team strength and upcoming opponents, based on Team Net Rating."
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
    st.markdown(
        "Net Rating measures how much a team outperforms its opponents per 100 possessions:"
    )
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
The following definitions are adapted from **Hack-a-Stat: Learn a Stat: Strength of Schedule ** and applied to EuroLeague data.
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
    st.latex(
        r"""
        \text{OW\%} =
        \frac{\sum_{i=1}^{n} \text{OppW\%}_i}{\text{TmGP}}
        """
    )

    st.markdown("### Opponents’ Opponents’ Winning Percentage (OOW%)")
    st.latex(
        r"""
        \text{OOW\%} =
        \frac{\sum_{i=1}^{n} \text{OW\%}_i}{\text{TmGP}}
        """
    )

    st.markdown("### Strength of Schedule — Win% based")
    st.latex(
        r"""
        \text{SoS}_{\text{Win}} =
        \frac{2 \cdot \text{OW\%} + \text{OOW\%}}{3}
        """
    )

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
    st.latex(
        r"""
        \text{OppNetRtg} =
        \frac{\sum_{i=1}^{n} \text{NetRtg}_i}{\text{TmGP}}
        """
    )

    st.markdown("### Opponents’ Opponents’ Net Rating (OONetRtg)")
    st.latex(
        r"""
        \text{OONetRtg} =
        \frac{\sum_{i=1}^{n} \text{OppNetRtg}_i}{\text{TmGP}}
        """
    )

    st.markdown("### Strength of Schedule — Net Rating based")
    st.latex(
        r"""
        \text{SoS}_{\text{Net}} =
        \frac{2 \cdot \text{OppNetRtg} + \text{OONetRtg}}{3}
        """
    )

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

    st.markdown(
        """
- **NetRtg (Net Rating)**: efficiency differential per 100 possessions  
"""
    )

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

    st.markdown(
        """
- **OOW%**: opponents’ opponents winning percentage  
"""
    )

    st.latex(r"\text{OOW\%} = \frac{1}{\text{TmGP}} \sum_{i=1}^{n} \text{OW\%}_i")

    st.markdown(
        """
- **SoS (Win%)**: weighted opponent difficulty  
"""
    )

    st.latex(r"\text{SoS}_{\text{Win}} = \frac{2 \cdot \text{OW\%} + \text{OOW\%}}{3}")

    st.markdown("---")

    st.markdown(
        """
### Strength of Schedule (SoS) helpers — Net Rating based
- **OppNetRtg**: average Net Rating of opponents  
"""
    )

    st.latex(r"\text{OppNetRtg} = \frac{1}{\text{TmGP}} \sum_{i=1}^{n} \text{NetRtg}_i")

    st.markdown(
        """
- **OONetRtg**: opponents’ opponents Net Rating  
"""
    )

    st.latex(r"\text{OONetRtg} = \frac{1}{\text{TmGP}} \sum_{i=1}^{n} \text{OppNetRtg}_i")

    st.markdown(
        """
- **SoS (NetRtg)**: efficiency-based schedule difficulty  
"""
    )

    st.latex(r"\text{SoS}_{\text{Net}} = \frac{2 \cdot \text{OppNetRtg} + \text{OONetRtg}}{3}")


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
with tabs[1]:
    
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
#          ( make_sos_scatter_and_side_table)
# =========================================================
with tabs[2]:

    main_chart, side_table_chart = make_sos_scatter_and_side_table(
        sos_net=sos_net,
        team_ratings=team_ratings,
        team_to_logo_path=team_to_logo_path,
        logo_to_dataurl=logo_to_dataurl,
        top_k=5,
        bottom_k=5,
        round_ref=int(current_round),
        season_label=season_label,
    )

    col1, col2 = st.columns([3, 2])

    with col1:
        st.altair_chart(main_chart, width="stretch")

    with col2:
        st.altair_chart(side_table_chart, width="content")



# =========================================================
# TAB 3 – Season SoS table (make_sos_table_chart)
# =========================================================
with tabs[3]:

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
