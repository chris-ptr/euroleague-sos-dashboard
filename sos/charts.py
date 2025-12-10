import numpy as np
import pandas as pd
import altair as alt

from .utils import normalize_team_name, logo_to_dataurl
from .utils import team_to_logo_path


def build_nextN_altair_logos_table(
    nextN_df: pd.DataFrame,
    team_ratings: pd.DataFrame,
    team_to_logo_path_fn,
    round_ref: int,
    n_next: int,
):
    """
    Build a table-like Altair chart for the next N games:

    - Left column: team logo + team name
    - Middle column: SoS_Net_nextN (color = green→white→red, text = value)
    - Right block: next N opponents
        * cell color = Opp_NetRtg (continuous green→white→red, 0 neutral)
        * inside each cell: opponent logo only

    Expects nextN_df from make_nextN_sos_table, with columns:
      - Team
      - SoS_Net_nextN
      - Logo_Path   (optional, not strictly required here)
      - Opp1..OppN
    """

    # ---------- STYLE SETTINGS ----------
    ROW_FONT = "Roboto"
    TITLE_FONT = "Arial"
    FONT_SIZE = 15
    TITLE_FONT_SIZE = 18

    LEFT_COL_WIDTH = 300
    SOS_COL_WIDTH = 120
    GAMES_COL_WIDTH = 440

    LOGO_SIZE_MAIN = 24
    LOGO_SIZE_OPP = 26

    # ---------- 1) Map team -> NetRtg (for opponent strength) ----------
    ratings = team_ratings.copy()
    ratings["TEAM_NAME"] = ratings["TEAM_NAME"].apply(normalize_team_name)
    net_map = dict(zip(ratings["TEAM_NAME"], ratings["NetRtg"]))

    # ---------- 2) Wide -> long for the N opponents ----------
    long_rows = []
    for _, row in nextN_df.iterrows():
        team_display = row["Team"]
        team_norm = normalize_team_name(team_display)

        # SoS value: prefer SoS_Net_nextN, fallback to legacy SoS_Net_next5 if present
        sos_net = row.get("SoS_Net_nextN", row.get("SoS_Net_next5", np.nan))

        # Team logo
        team_logo_path = team_to_logo_path_fn(team_norm)
        team_logo_dataurl = logo_to_dataurl(team_logo_path)

        for idx in range(1, n_next + 1):
            col = f"Opp{idx}"
            if col not in row:
                continue
            opp = row[col]
            if pd.isna(opp) or opp is None or opp == "":
                continue

            opp_norm = normalize_team_name(opp)
            opp_netrtg = net_map.get(opp_norm, np.nan)

            opp_logo_path = team_to_logo_path_fn(opp_norm)
            opp_logo_dataurl = logo_to_dataurl(opp_logo_path)

            long_rows.append(
                {
                    "Team": team_norm,
                    "Team_Display": team_display,
                    "Team_Logo_DataURL": team_logo_dataurl,
                    "SoS_Net_nextN": sos_net,
                    "game_idx": idx,
                    "Opponent": opp_norm,
                    "Opp_NetRtg": opp_netrtg,
                    "Opp_Logo_DataURL": opp_logo_dataurl,
                }
            )

    alt_df = pd.DataFrame(long_rows)
    if alt_df.empty:
        raise ValueError("No opponent data found in nextN_df for Altair chart.")

    # ---------- 3) One row per team for left + SoS ----------
    teams_df = (
        alt_df[["Team", "Team_Display", "Team_Logo_DataURL", "SoS_Net_nextN"]]
        .drop_duplicates(subset=["Team"])
        .reset_index(drop=True)
    )

    # Sort by SoS hardest → easiest
    team_order = (
        teams_df.sort_values("SoS_Net_nextN", ascending=False)["Team"].tolist()
    )

    teams_df["SoS_Col"] = ""

    # ---------- 4) Left column: logo + team name ----------
    logo_mark = (
        alt.Chart(teams_df)
        .mark_image(width=LOGO_SIZE_MAIN, height=LOGO_SIZE_MAIN)
        .encode(
            x=alt.value(0),
            y=alt.Y("Team:N", sort=team_order, title=None, axis=None),
            url="Team_Logo_DataURL:N",
            tooltip=[alt.Tooltip("Team_Display:N", title="Team")],
        )
        .properties(width=LEFT_COL_WIDTH)
    )

    team_text = (
        alt.Chart(teams_df)
        .mark_text(
            align="left",
            baseline="middle",
            dx=LOGO_SIZE_MAIN + 10,
            size=FONT_SIZE,
        )
        .encode(
            x=alt.value(0),
            y=alt.Y("Team:N", sort=team_order, title=None, axis=None),
            text="Team_Display:N",
        )
    )

    left_col = logo_mark + team_text

    # ---------- 5) SoS column ----------
    sos_color = alt.Color(
        "SoS_Net_nextN:Q",
        title="SoS (NetRtg)",
        scale=alt.Scale(
            domainMid=0.0,
            range=["green", "white", "red"],  # easy → neutral → hard
        ),
        legend=None,
    )

    sos_rect = (
        alt.Chart(teams_df)
        .mark_rect(stroke="lightgray")
        .encode(
            x=alt.X("SoS_Col:N", title=f"Next {n_next} SoS"),
            y=alt.Y("Team:N", sort=team_order, title=None, axis=None),
            color=sos_color,
            tooltip=[
                alt.Tooltip("Team_Display:N", title="Team"),
                alt.Tooltip("SoS_Net_nextN:Q", title="SoS (NetRtg)", format=".2f"),
            ],
        )
        .properties(width=SOS_COL_WIDTH)
    )

    sos_text = (
        alt.Chart(teams_df)
        .mark_text(size=FONT_SIZE, baseline="middle")
        .encode(
            x=alt.X("SoS_Col:N", title=f"Next {n_next} SoS"),
            y=alt.Y("Team:N", sort=team_order, title=None, axis=None),
            text=alt.Text("SoS_Net_nextN:Q", format=".2f"),
        )
    )

    sos_col = sos_rect + sos_text

    # ---------- 6) Next-N games block ----------
    opp_color = alt.Color(
        "Opp_NetRtg:Q",
        title="Opp NetRtg",
        scale=alt.Scale(domainMid=0.0, range=["green", "white", "red"]),
        legend=None,
    )

    games_rect = (
        alt.Chart(alt_df)
        .mark_rect(stroke="lightgray")
        .encode(
            x=alt.X(
                "game_idx:O",
                title=f"Next {n_next} Games",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y("Team:N", sort=team_order, title=None, axis=None),
            color=opp_color,
            tooltip=[
                alt.Tooltip("Team_Display:N", title="Team"),
                alt.Tooltip("game_idx:O", title="#"),
                alt.Tooltip("Opponent:N", title="Opponent"),
                alt.Tooltip("Opp_NetRtg:Q", title="Opp NetRtg", format=".2f"),
                alt.Tooltip("SoS_Net_nextN:Q", title="Team SoS", format=".2f"),
            ],
        )
        .properties(width=GAMES_COL_WIDTH)
    )

    games_logos = (
        alt.Chart(alt_df)
        .mark_image(width=LOGO_SIZE_OPP, height=LOGO_SIZE_OPP)
        .encode(
            x=alt.X("game_idx:O"),
            y=alt.Y("Team:N", sort=team_order, title=None, axis=None),
            url="Opp_Logo_DataURL:N",
        )
    )

    games_block = games_rect + games_logos

    # ---------- 7) Final layout + title + fonts ----------
    title_txt = (
        f"Strength of Schedule based on NetRtg "
        f"from Round {round_ref - 1} for the next {n_next} games"
    )

    chart = (
        alt.hconcat(
            left_col,
            sos_col,
            games_block,
            spacing=25,
        )
        .resolve_scale(y="shared")
        .properties(
            title=alt.TitleParams(
                title_txt,
                font=TITLE_FONT,
                fontSize=TITLE_FONT_SIZE,
                anchor="start",
            ),
            padding={"left": 20, "right": 20, "top": 20, "bottom": 20},
            background="#a3a1a1",
        )
        .configure_axis(
            labelFont=ROW_FONT,
            titleFont=ROW_FONT,
            labelFontSize=FONT_SIZE,
            titleFontSize=FONT_SIZE,
        )
        .configure_title(
            font=TITLE_FONT,
            fontSize=TITLE_FONT_SIZE,
        )
        .configure_text(
            font=ROW_FONT,
            fontSize=FONT_SIZE,
        )
        .configure_view(
            stroke=None
        )
        .configure_legend(
            labelFont=ROW_FONT,
            titleFont=ROW_FONT,
            labelFontSize=FONT_SIZE,
            titleFontSize=FONT_SIZE,
        )
    )

    return chart

def make_sos_table_chart(
    sos_net: pd.DataFrame,
    sos_win: pd.DataFrame,
    team_to_logo_path,
    logo_to_dataurl,
    *,
    round_ref: int | None = None,
    season_label: str | None = None,
    title: str | None = None,
    logo_size: int = 28,
    padding_inner: float = 0.35,
    padding_outer: float = 0.10,
    row_height: int = 28,
) -> alt.Chart:
    """
    Build the SoS table (logos + SoS_Net bar + SoS_win bar) as an Altair chart.

    sos_net: must have ['TEAM_NAME', 'SoS_Net']
    sos_win: must have ['TEAM_NAME', 'SoS'] (SoS = Win% SoS)
    """

    # ---------- fonts to match Next-N chart ----------
    ROW_FONT = "Roboto"
    TITLE_FONT = "Arial"
    FONT_SIZE = 15
    TITLE_FONT_SIZE = 18

    # ---------- dynamic title ----------
    if title is None:
        if season_label is not None and round_ref is not None:
            title = (
                f"Strength of Schedule {season_label} "
                f"Untill Round {round_ref} \n (Net Rating & Win% Methods Comparison)"
            )
        elif round_ref is not None:
            title = (
                f"EuroLeague Strength of Schedule (Untill Round {round_ref})\n"
                f"(Net Rating vs Win% Methods)"
            )
        else:
            title = "EuroLeague Strength of Schedule\n(Net Rating vs Win% Methods)"

    # --- build combined dataframe (same as before) ---
    df_net = sos_net[["TEAM_NAME", "SoS_Net"]].copy()
    df_win = sos_win[["TEAM_NAME", "SoS"]].copy().rename(columns={"SoS": "SoS_win"})

    combined = df_net.merge(df_win, on="TEAM_NAME", how="inner")
    combined = combined.sort_values("SoS_Net", ascending=False).reset_index(drop=True)

    combined["logo_path"] = combined["TEAM_NAME"].apply(team_to_logo_path)
    combined["logo_url"] = combined["logo_path"].apply(logo_to_dataurl)

    # --- normalization ---
    min_net = float(combined["SoS_Net"].min())
    max_net = float(combined["SoS_Net"].max())
    range_net = max_net - min_net if max_net != min_net else 1.0

    min_win = float(combined["SoS_win"].min())
    max_win = float(combined["SoS_win"].max())
    range_win = max_win - min_win if max_win != min_win else 1.0

    def norm_with_margin(val, vmin, vrange, margin=0.05):
        raw = (val - vmin) / vrange
        raw = max(0.0, min(1.0, raw))
        return margin + (1 - 2 * margin) * raw   # e.g. [0.05, 0.95]

    combined["SoS_Net_norm"] = combined["SoS_Net"].apply(
        lambda v: norm_with_margin(v, min_net, range_net, margin=0.05)
    )
    combined["SoS_win_norm"] = combined["SoS_win"].apply(
        lambda v: norm_with_margin(v, min_win, range_win, margin=0.05)
    )

    team_order = combined["TEAM_NAME"].tolist()

    # common y scale with padding (for spacing between rows)
    y_scale = alt.Scale(paddingInner=padding_inner, paddingOuter=padding_outer)

    # =========================================================
    # LEFT COLUMN: logo + single team name
    # =========================================================

    logo_col = alt.Chart(combined).mark_image(
        width=logo_size,
        height=logo_size,
    ).encode(
        y=alt.Y(
            "TEAM_NAME:N",
            sort=team_order,
            scale=y_scale,
            axis=None,
        ),
        x=alt.value(25),
        url="logo_url:N",
    ).properties(
        width=90,
        height=row_height * len(combined),
        title="TEAM",
    )

    name_col = alt.Chart(combined).mark_text(
        align="left",
        baseline="middle",
        dx=logo_size + 27,  # 55 for size 28; scales if you change logo_size
        fontSize=15,
    ).encode(
        y=alt.Y(
            "TEAM_NAME:N",
            sort=team_order,
            scale=y_scale,
            axis=None,
        ),
        x=alt.value(0),
        text="TEAM_NAME:N",
    )

    left_col = logo_col + name_col

    # =========================================================
    # MIDDLE COLUMN: SoS (NetRtg)
    # =========================================================

    net_bar = alt.Chart(combined).mark_bar(
        stroke="black",      # black outline
        strokeWidth=0.7,
    ).encode(
        y=alt.Y(
            "TEAM_NAME:N",
            sort=team_order,
            scale=y_scale,
            axis=None,
        ),
        x=alt.X(
            "SoS_Net_norm:Q",
            title=None,
            axis=alt.Axis(labels=False, ticks=False),
            scale=alt.Scale(domain=[0, 1]),
        ),
        color=alt.condition(
            "datum.SoS_Net < 0",
            alt.value("green"),
            alt.value("red"),
        ),
    )

    net_text = alt.Chart(combined).mark_text(
        color="black",
        fontSize=11,
        dx=2,
        align="left",
        baseline="middle",
    ).encode(
        y=alt.Y(
            "TEAM_NAME:N",
            sort=team_order,
            scale=y_scale,
            axis=None,
        ),
        x=alt.X("SoS_Net_norm:Q", scale=alt.Scale(domain=[0, 1])),
        text=alt.Text("SoS_Net:Q", format=".2f"),
    )

    net_col = (net_bar + net_text).properties(
        width=220,
        height=row_height * len(combined),
        title="SoS (NetRtg)",
    )

    # =========================================================
    # RIGHT COLUMN: SoS (Win%)
    # =========================================================

    win_bar = alt.Chart(combined).mark_bar(
        stroke="black",      # black outline
        strokeWidth=0.7,
    ).encode(
        y=alt.Y(
            "TEAM_NAME:N",
            sort=team_order,
            scale=y_scale,
            axis=None,
        ),
        x=alt.X(
            "SoS_win_norm:Q",
            title=None,
            axis=alt.Axis(labels=False, ticks=False),
            scale=alt.Scale(domain=[0, 1]),
        ),
        color=alt.condition(
            "datum.SoS_win < 0.50",
            alt.value("green"),
            alt.value("red"),
        ),
    )

    win_text = alt.Chart(combined).mark_text(
        color="black",
        fontSize=11,
        dx=2,
        align="left",
        baseline="middle",
    ).encode(
        y=alt.Y(
            "TEAM_NAME:N",
            sort=team_order,
            scale=y_scale,
            axis=None,
        ),
        x=alt.X("SoS_win_norm:Q", scale=alt.Scale(domain=[0, 1])),
        text=alt.Text("SoS_win:Q", format=".1%"),
    )

    win_col = (win_bar + win_text).properties(
        width=220,
        height=row_height * len(combined),
        title="SoS (Win%)",
    )

    # =========================================================
    # FINAL TABLE
    # =========================================================

    sos_table_chart = alt.hconcat(
        left_col,
        net_col,
        win_col,
    ).resolve_scale(
        y="shared",
    ).properties(
        title=title,
        background="#a3a1a1",
        padding={"left": 20, "right": 20, "top": 20, "bottom": 20},
    ).configure_axis(
        labelFont=ROW_FONT,
        titleFont=ROW_FONT,
        labelFontSize=FONT_SIZE,
        titleFontSize=FONT_SIZE,
    ).configure_title(
        font=TITLE_FONT,
        fontSize=TITLE_FONT_SIZE,
    ).configure_text(
        font=ROW_FONT,
        fontSize=FONT_SIZE,
    ).configure_legend(
        labelFont=ROW_FONT,
        titleFont=ROW_FONT,
        labelFontSize=FONT_SIZE,
        titleFontSize=FONT_SIZE,
    ).configure_view(
        stroke=None,
    )

    return sos_table_chart

# Function to have both tables on the sos scatter
def make_sos_scatter_and_side_table(
    sos_net: pd.DataFrame,
    team_ratings: pd.DataFrame,
    team_to_logo_path,
    logo_to_dataurl,
    *,
    top_k: int = 5,
    bottom_k: int = 5,
    round_ref: int | None = None,
    season_label: str | None = None,
    title: str | None = None,
    background: str = "#a3a1a1",
) -> tuple[alt.Chart, alt.Chart]:
    """
    Build SoS(NetRtg) vs NetRtg scatter + side table,
    but return them as two separate charts (main_chart, table_chart)
    so Streamlit can show them in two columns.
    """

    # ---------- fonts to match Next-N chart ----------
    ROW_FONT = "Roboto"
    TITLE_FONT = "Arial"
    FONT_SIZE = 15
    TITLE_FONT_SIZE = 18

    # ---------- dynamic title ----------
    if title is None:
        if season_label is not None and round_ref is not None:
            title = (
                f"EuroLeague {season_label}: Strength of Schedule (NetRtg) "
                f"vs Team Net Rating (After Round {round_ref - 1})"
            )
        elif round_ref is not None:
            title = (
                f"EuroLeague: Strength of Schedule (NetRtg) "
                f"vs Team Net Rating (After Round {round_ref - 1})"
            )
        else:
            title = "EuroLeague: Strength of Schedule (NetRtg) vs Team Net Rating"

    # =========================================================
    # 0. Combine data + ranking
    # =========================================================
    df = sos_net[["TEAM_NAME", "SoS_Net"]].merge(
        team_ratings[["TEAM_NAME", "NetRtg", "OffRtg", "DefRtg"]],
        on="TEAM_NAME",
        how="inner",
    )

    df["logo_path"] = df["TEAM_NAME"].apply(team_to_logo_path)
    df["logo_url"] = df["logo_path"].apply(logo_to_dataurl)

    # Rank by NetRtg (1 = best)
    df["Rank"] = df["NetRtg"].rank(ascending=False, method="first").astype(int)
    df["Label"] = ""
    df.loc[df["Rank"] <= 4, "Label"] = "#" + df["Rank"].astype(str)
    df.loc[df["Rank"] > len(df) - 4, "Label"] = "#" + df["Rank"].astype(str)

    # for labels on logos (top-4 + bottom-4)
    hi_lo_df = df[(df["Rank"] <= 4) | (df["Rank"] > len(df) - 4)].copy()

    # =========================================================
    # 1. Axis ranges
    # =========================================================
    x_min_raw = float(df["SoS_Net"].min())
    x_max_raw = float(df["SoS_Net"].max())

    abs_max = max(abs(x_min_raw), abs(x_max_raw))
    if abs_max < 0.25:
        abs_max = 0.25

    # reversed axis → hard schedule on the LEFT
    x_domain = [abs_max, -abs_max]

    y_min = float(df["NetRtg"].min())
    y_max = float(df["NetRtg"].max())

    # =========================================================
    # 2. Quadrant labels
    # =========================================================
    x_left = abs_max * 0.65   # tough schedule side
    x_right = -abs_max * 0.65 # easy schedule side
    y_top = y_max * 0.65
    y_bottom = y_min * 0.65

    quad_df = pd.DataFrame([
        {"x": x_left,  "y": y_top,    "label": "Tough Schedule\nOverperforming",  "color": "#E67E22"},
        {"x": x_right, "y": y_top,    "label": "Easy Schedule\nStrong Team",      "color": "#000000"},
        {"x": x_left,  "y": y_bottom, "label": "Tough Schedule\nUnderperforming", "color": "blue"},
        {"x": x_right, "y": y_bottom, "label": "Easy Schedule\nUnderperforming",  "color": "navy"},
    ])

    quad_df["label_lines"] = quad_df["label"].apply(lambda s: s.split("\n"))

    quad_layer = alt.Chart(quad_df).mark_text(
        fontSize=16,
        fontWeight="bold",
        opacity=0.95,
        align="center",
        lineBreak="\n",
        stroke="black",
        strokeWidth=0.25,
    ).encode(
        x="x:Q",
        y="y:Q",
        text="label:N",
        color=alt.Color(
            "color:N",
            legend=None,
        ),
    )

    # =========================================================
    # 3. Scatter + logos + #labels  (MAIN CHART)
    # =========================================================
    base = alt.Chart(df).encode(
        x=alt.X(
            "SoS_Net:Q",
            title="Strength of Schedule (NetRtg)",
            scale=alt.Scale(domain=x_domain, nice=False, zero=False),
        ),
        y=alt.Y(
            "NetRtg:Q",
            title="Team NetRtg",
            scale=alt.Scale(domain=[y_min, y_max], nice=True, zero=False),
        ),
    )

    # thick 0-lines
    rule_y0 = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
        stroke="black", strokeWidth=1.5, opacity=0.9
    ).encode(y="y:Q")

    rule_x0 = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
        stroke="black", strokeWidth=1.5, opacity=0.9
    ).encode(x="x:Q")

    # logos
    logo_layer = base.mark_image(width=40, height=40).encode(
        url="logo_url:N",
        tooltip=[
            alt.Tooltip("TEAM_NAME:N"),
            alt.Tooltip("SoS_Net:Q", format=".3f"),
            alt.Tooltip("NetRtg:Q", format=".2f"),
            alt.Tooltip("Rank:Q"),
        ],
    )

    # Rank labels above logos (only top 4 + bottom 4)
    label_layer = alt.Chart(hi_lo_df).encode(
        x=alt.X("SoS_Net:Q", scale=alt.Scale(domain=x_domain, nice=False, zero=False)),
        y=alt.Y("NetRtg:Q", scale=alt.Scale(domain=[y_min, y_max], nice=True, zero=False)),
    ).mark_text(
        dx=0,
        dy=-32,
        fontSize=12,
        fontWeight="bold",
        color="black",
        stroke="black",
        strokeWidth=0.5,
    ).encode(
        text="Label:N",
    )

    main_chart = (
        rule_x0
        + rule_y0
        + logo_layer
        + label_layer
        + quad_layer
    ).properties(
        width=720,
        height=560,
        title=title,
        background=background,
        padding={"left": 20, "right": 20, "top": 20, "bottom": 20},
    ).configure_axis(
        labelFont=ROW_FONT,
        titleFont=ROW_FONT,
        labelFontSize=FONT_SIZE,
        titleFontSize=FONT_SIZE,
        grid=True,
        gridColor="#d3d3d3",
        gridWidth=0.4,
        domainColor="#555",
        tickColor="#555",
        labelColor="black",
        titleColor="black",
        titlePadding=12,
    ).configure_title(
        font=TITLE_FONT,
        fontSize=TITLE_FONT_SIZE,
    ).configure_text(
        font=ROW_FONT,
        fontSize=FONT_SIZE,
    ).configure_legend(
        labelFont=ROW_FONT,
        titleFont=ROW_FONT,
        labelFontSize=FONT_SIZE,
        titleFontSize=FONT_SIZE,
    )

    # =========================================================
    # 4. SIDE TABLE (Top K & Bottom K) – with logos
    # =========================================================
    top_df = df.nsmallest(top_k, "Rank").copy().sort_values("Rank")
    bottom_df = df.nlargest(bottom_k, "Rank").copy().sort_values("Rank")

    top_df["group"] = "Top NetRtg"
    bottom_df["group"] = "Bottom NetRtg"

    stats_df = pd.concat([top_df, bottom_df], ignore_index=True)

    # order for vertical layout
    stats_df["order"] = range(len(stats_df))

    stats_df["row_text"] = stats_df.apply(
        lambda r: (
            f"#{r['Rank']}  {r['TEAM_NAME']}\n"
            f"Net {r['NetRtg']:.2f}   Off {r['OffRtg']:.1f}   Def {r['DefRtg']:.1f}"
        ),
        axis=1,
    )

    # logo column
    logo_col = alt.Chart(stats_df).mark_image(width=32, height=32).encode(
        y=alt.Y("order:O", axis=None),
        x=alt.value(20),
        url="logo_url:N",
    )

    # text column
    text_col = alt.Chart(stats_df).mark_text(
        align="left",
        fontSize=11,
        fontWeight="bold",
        lineBreak="\n",
    ).encode(
        y=alt.Y("order:O", axis=None),
        x=alt.value(60),
        text="row_text:N",
        color=alt.condition(
            alt.datum.group == "Top NetRtg",
            alt.value("#2c742cb8"),   # color for top rows
            alt.value("#81290e"),     # color for bottom rows
        ),
    )

    table_chart = (logo_col + text_col).properties(
        width=350,
        height=560,
        title=f"Top {top_k} / Bottom {bottom_k} Net Ratings",
        background=background,
        padding={"left": 20, "right": 20, "top": 20, "bottom": 20},
    ).configure_axis(
        labelFont=ROW_FONT,
        titleFont=ROW_FONT,
        labelFontSize=FONT_SIZE,
        titleFontSize=FONT_SIZE,
    ).configure_title(
        font=TITLE_FONT,
        fontSize=TITLE_FONT_SIZE,
    ).configure_text(
        font=ROW_FONT,
        fontSize=FONT_SIZE,
    ).configure_legend(
        labelFont=ROW_FONT,
        titleFont=ROW_FONT,
        labelFontSize=FONT_SIZE,
        titleFontSize=FONT_SIZE,
    ).configure_view(
        stroke="black",
        strokeWidth=1.0,
    )

    return main_chart, table_chart

