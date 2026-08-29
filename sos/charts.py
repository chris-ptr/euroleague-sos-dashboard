import numpy as np
import pandas as pd
import altair as alt

from sos.utils import normalize_team_name, logo_to_dataurl
from sos.utils import team_to_logo_path
from sos.utils import team_display_name
from sos.presets import CHART_FONT


# The site's one diverging scale, written low-to-high for the case where a high
# value is the good one: red at the bottom, green at the top. Named for what the
# color *means* rather than for any one chart, because the underlying quantity
# flips between tables — in the Next-N table green is a weak upcoming opponent,
# in the Four Factors table it is a strong team — while the reading holds either
# way: green is the end a team wants to be at. A chart whose *low* end is the
# good one (schedule difficulty) passes this reversed.
FAVORABILITY_PALETTE = ("red", "white", "green")

# Border around the grouped columns of the Four Factors table. Dark enough to
# hold its own against a saturated red or green cell sitting right under it,
# which a hairline grey does not.
BLOCK_OUTLINE = "#2f2f2f"


def build_nextN_altair_logos_table(
    nextN_df: pd.DataFrame,
    team_ratings: pd.DataFrame,
    team_to_logo_path_fn,
    round_ref: int,
    n_next: int,
    *,
    logo_path_to_url_fn=logo_to_dataurl,
    left_col_width: int = 260,
    sos_col_width: int = 120,
    games_col_width: int = 340,
    logo_size_main: int = 24,
    logo_size_opp: int = 26,
    font_size: int = 15,
    title_font_size: int = 22,
    mobile_mode: bool = False,
) -> alt.Chart:
    """
    Build an Altair heat-map table for upcoming schedule difficulty.
    
    Displays team logos, forecasting SOS, and the next N opponents.
    """

    # Global chart styling
    # Both the site face: these feed .configure_axis/_title/_text/_legend, which
    # are more specific than the spec's top-level config.font and would other-
    # wise keep overriding it back to Roboto/Arial.
    ROW_FONT = CHART_FONT
    TITLE_FONT = CHART_FONT
    FONT_SIZE = font_size
    TITLE_FONT_SIZE = title_font_size

    LEFT_COL_WIDTH = left_col_width
    SOS_COL_WIDTH = sos_col_width
    GAMES_COL_WIDTH = games_col_width

    LOGO_SIZE_MAIN = logo_size_main
    LOGO_SIZE_OPP = logo_size_opp

    # Map team names to NetRtg for opponent color-coding
    ratings = team_ratings.copy()
    ratings["TEAM_NAME"] = ratings["TEAM_NAME"].apply(normalize_team_name)
    net_map = dict(zip(ratings["TEAM_NAME"], ratings["NetRtg"]))

    # Unpivot N opponents into long format for Altair
    long_rows = []
    for _, row in nextN_df.iterrows():
        team_display = row["Team"]
        team_norm = normalize_team_name(team_display)

        sos_net = row.get("SoS_Net_nextN", row.get("SoS_Net_next5", np.nan))

        team_logo_path = team_to_logo_path_fn(team_norm)
        team_logo_dataurl = logo_path_to_url_fn(team_logo_path)

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
            opp_logo_dataurl = logo_path_to_url_fn(opp_logo_path)

            long_rows.append(
                {
                    "TEAM_KEY": team_norm,
                    "TEAM_LABEL": team_display_name(team_norm, mobile_mode),
                    "TEAM_DISPLAY_FULL": team_display,

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

    # Unique team records for the left sidebar layers
    teams_df = (
        alt_df[["TEAM_KEY", "TEAM_LABEL", "TEAM_DISPLAY_FULL", "Team_Logo_DataURL", "SoS_Net_nextN"]]
        .drop_duplicates(subset=["TEAM_KEY"])
        .reset_index(drop=True)
    )

    # Sort teams by schedule difficulty
    team_order = (
        teams_df.sort_values("SoS_Net_nextN", ascending=False)["TEAM_KEY"].tolist()
    )

    teams_df["SoS_Col"] = ""

    # Column 1: Team logos and names
    logo_mark = (
        alt.Chart(teams_df)
        .mark_image(width=LOGO_SIZE_MAIN, height=LOGO_SIZE_MAIN)
        .encode(
            x=alt.value(0),
            y=alt.Y("TEAM_KEY:N", sort=team_order, title=None, axis=None),
            url="Team_Logo_DataURL:N",
            tooltip=[alt.Tooltip("TEAM_DISPLAY_FULL:N", title="Team")],
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
            y=alt.Y("TEAM_KEY:N", sort=team_order, title=None, axis=None),
            text="TEAM_LABEL:N",
        )
    )

    left_col = logo_mark + team_text

    # Column 2: Forecasting SOS (NetRtg)
    sos_color = alt.Color(
        "SoS_Net_nextN:Q",
        title="SoS (NetRtg)",
        scale=alt.Scale(domainMid=0.0, range=list(reversed(FAVORABILITY_PALETTE))),
        legend=None,
    )

    sos_rect = (
        alt.Chart(teams_df)
        .mark_rect(stroke="lightgray")
        .encode(
            x=alt.X("SoS_Col:N", title=f"Next {n_next} SoS"),
            y=alt.Y("TEAM_KEY:N", sort=team_order, title=None, axis=None),
            color=sos_color,
            tooltip=[
                alt.Tooltip("TEAM_DISPLAY_FULL:N", title="Team"),
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
            y=alt.Y("TEAM_KEY:N", sort=team_order, title=None, axis=None),
            text=alt.Text("SoS_Net_nextN:Q", format=".2f"),
        )
    )

    sos_col = sos_rect + sos_text

    # Column 3: Upcoming opponent grid
    opp_color = alt.Color(
        "Opp_NetRtg:Q",
        title="Opp NetRtg",
        scale=alt.Scale(domainMid=0.0, range=list(reversed(FAVORABILITY_PALETTE))),
        legend=None,
    )

    games_rect = (
        alt.Chart(alt_df)
        .mark_rect(stroke="lightgray")
        .encode(
            x=alt.X("game_idx:O", title=f"Next {n_next} Games", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("TEAM_KEY:N", sort=team_order, title=None, axis=None),
            color=opp_color,
            tooltip=[
                alt.Tooltip("TEAM_DISPLAY_FULL:N", title="Team"),
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
            y=alt.Y("TEAM_KEY:N", sort=team_order, title=None, axis=None),
            url="Opp_Logo_DataURL:N",
        )
    )

    games_block = games_rect + games_logos

    # Combine columns into final dashboard layout
    title_txt = (
        f"Strength of Schedule based on NetRtg "
        f"from Round {round_ref - 1} for the next {n_next} games"
    )

    chart = (
        alt.hconcat(left_col, sos_col, games_block, spacing=25)
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
        .configure_title(font=TITLE_FONT, fontSize=TITLE_FONT_SIZE)
        .configure_text(font=ROW_FONT, fontSize=FONT_SIZE)
        .configure_view(stroke=None)
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
    logo_path_to_url_fn,
    *,
    round_ref: int | None = None,
    season_label: str | None = None,
    title: str | None = None,
    logo_size: int = 28,
    padding_inner: float = 0.35,
    padding_outer: float = 0.10,
    row_height: int = 28,
    team_col_width: int = 90,
    net_col_width: int = 220,
    win_col_width: int = 220,
    name_font_size: int = 15,
    value_font_size: int = 11,
    font_size: int = 15,
    title_font_size: int = 18,
    mobile_mode: bool = False,
) -> alt.Chart:
    """Build a comparison table for NetRtg and Win% based SOS metrics."""

    # Typography settings
    # Both the site face: these feed .configure_axis/_title/_text/_legend, which
    # are more specific than the spec's top-level config.font and would other-
    # wise keep overriding it back to Roboto/Arial.
    ROW_FONT = CHART_FONT
    TITLE_FONT = CHART_FONT
    FONT_SIZE = font_size
    TITLE_FONT_SIZE = title_font_size

    # Compose chart title
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

    # Merge SOS datasets
    df_net = sos_net[["TEAM_NAME", "SoS_Net"]].copy()
    df_win = sos_win[["TEAM_NAME", "SoS"]].copy().rename(columns={"SoS": "SoS_win"})

    combined = df_net.merge(df_win, on="TEAM_NAME", how="inner")
    combined = combined.sort_values("SoS_Net", ascending=False).reset_index(drop=True)

    combined["logo_path"] = combined["TEAM_NAME"].apply(team_to_logo_path)
    combined["logo_url"] = combined["logo_path"].apply(logo_path_to_url_fn)

    combined["TEAM_KEY"] = combined["TEAM_NAME"]
    combined["TEAM_LABEL"] = combined["TEAM_NAME"].apply(
        lambda t: team_display_name(t, mobile_mode)
    )

    # Normalize metrics for proportional bar lengths
    min_net = float(combined["SoS_Net"].min())
    max_net = float(combined["SoS_Net"].max())
    range_net = max_net - min_net if max_net != min_net else 1.0

    min_win = float(combined["SoS_win"].min())
    max_win = float(combined["SoS_win"].max())
    range_win = max_win - min_win if max_win != min_win else 1.0

    def norm_with_margin(val, vmin, vrange, margin=0.05):
        raw = (val - vmin) / vrange
        raw = max(0.0, min(1.0, raw))
        return margin + (1 - 2 * margin) * raw

    combined["SoS_Net_norm"] = combined["SoS_Net"].apply(
        lambda v: norm_with_margin(v, min_net, range_net, margin=0.05)
    )
    combined["SoS_win_norm"] = combined["SoS_win"].apply(
        lambda v: norm_with_margin(v, min_win, range_win, margin=0.05)
    )

    team_order = combined["TEAM_KEY"].tolist()
    y_scale = alt.Scale(paddingInner=padding_inner, paddingOuter=padding_outer)

    # Sidebar: Logos and Team labels
    logo_col = (
        alt.Chart(combined)
        .mark_image(width=logo_size, height=logo_size)
        .encode(
            y=alt.Y("TEAM_KEY:N", sort=team_order, scale=y_scale, axis=None),
            x=alt.value(25),
            url="logo_url:N",
        )
        .properties(
            width=team_col_width,
            height=row_height * len(combined),
            title="TEAM",
        )
    )

    name_col = alt.Chart(combined).mark_text(
        align="left",
        baseline="middle",
        dx=logo_size + 27,
        fontSize=name_font_size,
    ).encode(
        y=alt.Y("TEAM_KEY:N", sort=team_order, scale=y_scale, axis=None),
        x=alt.value(0),
        text="TEAM_LABEL:N",
    )

    left_col = logo_col + name_col

    # Bar Chart: SOS (NetRtg)
    net_bar = alt.Chart(combined).mark_bar(
        stroke="black",
        strokeWidth=0.7,
    ).encode(
        y=alt.Y("TEAM_KEY:N", sort=team_order, scale=y_scale, axis=None),
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
        fontSize=value_font_size,
        dx=2,
        align="left",
        baseline="middle",
    ).encode(
        y=alt.Y("TEAM_KEY:N", sort=team_order, scale=y_scale, axis=None),
        x=alt.X("SoS_Net_norm:Q", scale=alt.Scale(domain=[0, 1])),
        text=alt.Text("SoS_Net:Q", format=".2f"),
    )

    net_col = (net_bar + net_text).properties(
        width=net_col_width,
        height=row_height * len(combined),
        title="SoS (NetRtg)",
    )

    # Bar Chart: SOS (Win%)
    win_bar = alt.Chart(combined).mark_bar(
        stroke="black",
        strokeWidth=0.7,
    ).encode(
        y=alt.Y("TEAM_KEY:N", sort=team_order, scale=y_scale, axis=None),
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
        fontSize=value_font_size,
        dx=2,
        align="left",
        baseline="middle",
    ).encode(
        y=alt.Y("TEAM_KEY:N", sort=team_order, scale=y_scale, axis=None),
        x=alt.X("SoS_win_norm:Q", scale=alt.Scale(domain=[0, 1])),
        text=alt.Text("SoS_win:Q", format=".1%"),
    )

    win_col = (win_bar + win_text).properties(
        width=win_col_width,
        height=row_height * len(combined),
        title="SoS (Win%)",
    )

    # Horizontal concatenation of chart columns
    sos_table_chart = (
        alt.hconcat(left_col, net_col, win_col, spacing = 10)
        .resolve_scale(y="shared")
        .properties(
            title=title,
            background="#a3a1a1",
            padding={"left": 20, "right": 20, "top": 20, "bottom": 20},
            autosize=alt.AutoSizeParams(type="pad", resize=True)
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
        .configure_legend(
            labelFont=ROW_FONT,
            titleFont=ROW_FONT,
            labelFontSize=FONT_SIZE,
            titleFontSize=FONT_SIZE,
        )
        .configure_view(stroke=None)
    )

    return sos_table_chart


def make_sos_scatter_and_side_table(
    sos_net: pd.DataFrame,
    team_ratings: pd.DataFrame,
    team_to_logo_path,
    logo_path_to_url_fn,
    *,
    top_k: int = 5,
    bottom_k: int = 5,
    round_ref: int | None = None,
    season_label: str | None = None,
    title: str | None = None,
    background: str = "#a3a1a1",
    main_w: int = 720,
    main_h: int = 560,
    table_w: int = 350,
    table_h: int = 560,
) -> tuple[alt.Chart, alt.Chart]:
    """
    Generate the SOS scatter plot and complementary NetRtg side table.
    """

    # Visual presets
    # Both the site face: these feed .configure_axis/_title/_text/_legend, which
    # are more specific than the spec's top-level config.font and would other-
    # wise keep overriding it back to Roboto/Arial.
    ROW_FONT = CHART_FONT
    TITLE_FONT = CHART_FONT
    FONT_SIZE = 15
    TITLE_FONT_SIZE = 18

    # Dynamic title generation
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

    # Join metrics and rank teams by efficiency
    df = sos_net[["TEAM_NAME", "SoS_Net"]].merge(
        team_ratings[["TEAM_NAME", "NetRtg", "OffRtg", "DefRtg"]],
        on="TEAM_NAME",
        how="inner",
    )

    df["logo_path"] = df["TEAM_NAME"].apply(team_to_logo_path)
    df["logo_url"] = df["logo_path"].apply(logo_path_to_url_fn)

    df["Rank"] = df["NetRtg"].rank(ascending=False, method="first").astype(int)
    df["Label"] = ""
    df.loc[df["Rank"] <= 4, "Label"] = "#" + df["Rank"].astype(str)
    df.loc[df["Rank"] > len(df) - 4, "Label"] = "#" + df["Rank"].astype(str)

    # Top/Bottom teams for overlay labels
    hi_lo_df = df[(df["Rank"] <= 4) | (df["Rank"] > len(df) - 4)].copy()

    # Calculate axis bounds (centered around zero for SOS)
    x_min_raw = float(df["SoS_Net"].min())
    x_max_raw = float(df["SoS_Net"].max())

    abs_max = max(abs(x_min_raw), abs(x_max_raw))
    if abs_max < 0.25:
        abs_max = 0.25

    # Reverse X-axis: harder schedule (higher SOS) on the left
    x_domain = [abs_max, -abs_max]

    y_min = float(df["NetRtg"].min())
    y_max = float(df["NetRtg"].max())

    # Annotate quadrants for performance context
    x_left = abs_max * 0.65   # High SOS zone
    x_right = -abs_max * 0.65 # Low SOS zone
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

    # Base scatter plot layer
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

    # Baseline reference lines
    rule_y0 = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
        stroke="black", strokeWidth=1.5, opacity=0.9
    ).encode(y="y:Q")

    rule_x0 = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
        stroke="black", strokeWidth=1.5, opacity=0.9
    ).encode(x="x:Q")

    # Team logos as markers
    logo_layer = base.mark_image(width=40, height=40).encode(
        url="logo_url:N",
        tooltip=[
            alt.Tooltip("TEAM_NAME:N"),
            alt.Tooltip("SoS_Net:Q", format=".3f"),
            alt.Tooltip("NetRtg:Q", format=".2f"),
            alt.Tooltip("Rank:Q"),
        ],
    )

    # NetRtg rank labels
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
        width=main_w,
        height=main_h,
        title=title,
        background=background,
        padding={"left": 20, "right": 20, "top": 20, "bottom": 20},
    ).configure_axis(
        labelFont=ROW_FONT,
        titleFont=ROW_FONT,
        grid=True,
        gridColor="#d3d3d3",
        gridWidth=0.4,
        labelColor="black",
        titleColor="black",
    )

    # Side Table: Top/Bottom performers
    top_df = df.nsmallest(top_k, "Rank").copy().sort_values("Rank")
    bottom_df = df.nlargest(bottom_k, "Rank").copy().sort_values("Rank")

    top_df["group"] = "Top NetRtg"
    bottom_df["group"] = "Bottom NetRtg"

    stats_df = pd.concat([top_df, bottom_df], ignore_index=True)
    stats_df["order"] = range(len(stats_df))

    # Multi-line summary strings
    stats_df["row_text"] = stats_df.apply(
        lambda r: (
            f"#{r['Rank']}  {r['TEAM_NAME']}\n"
            f"Net {r['NetRtg']:.2f}   Off {r['OffRtg']:.1f}   Def {r['DefRtg']:.1f}"
        ),
        axis=1,
    )

    logo_col = alt.Chart(stats_df).mark_image(width=32, height=32).encode(
        y=alt.Y("order:O", axis=None),
        x=alt.value(20),
        url="logo_url:N",
    )

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
            alt.value("#2c742cb8"),
            alt.value("#81290e"),
        ),
    )

    table_chart = (logo_col + text_col).properties(
        width=table_w,
        height=table_h,
        title=f"Top {top_k} / Bottom {bottom_k} Net Ratings",
        background=background,
        padding={"left": 20, "right": 20, "top": 20, "bottom": 20},
    ).configure_axis(
        labelFont=ROW_FONT,
        titleFont=ROW_FONT,
    ).configure_view(
        stroke="black",
        strokeWidth=1.0,
    )

    return main_chart, table_chart



def make_four_factors_chart(
    four_factors: pd.DataFrame,
    team_to_logo_path,
    logo_path_to_url_fn,
    *,
    round_ref: int | None = None,
    season_label: str | None = None,
    title: str | None = None,
    background: str = "#a3a1a1",
    team_col_width: int = 250,
    factor_col_width: int = 74,
    rating_col_width: int = 110,
    row_height: int = 26,
    logo_size: int = 24,
    name_font_size: int = 13,
    value_font_size: int = 11,
    font_size: int = 13,
    title_font_size: int = 18,
    mobile_mode: bool = False,
) -> alt.Chart:
    """
    Build the Four Factors team-profile table: eight factor columns plus the
    weighted composite, one row per team.

    Every factor gets its own color scale rather than one shared scale, because
    the four live on different numeric ranges — eFG% around .52, FTR around .25
    — and a shared scale would paint the whole FTR column the same shade. Each
    column is instead colored across its own league min/max, so the shading
    answers "good or bad *for this factor*", which is the only comparison that
    means anything across columns.

    Green is always the good end, which for TOV%-on-offense and the three
    allowed factors means the scale runs the other way; `higher_is_better`
    carries that per column so the reversal is data, not four near-identical
    branches.
    """

    # Both the site face: these feed .configure_axis/_title/_text/_legend, which
    # are more specific than the spec's top-level config.font and would other-
    # wise keep overriding it back to Roboto/Arial.
    ROW_FONT = CHART_FONT
    TITLE_FONT = CHART_FONT
    FONT_SIZE = font_size
    TITLE_FONT_SIZE = title_font_size

    if title is None:
        if season_label is not None and round_ref is not None:
            title = (
                f"EuroLeague {season_label}: Four Factors Profile "
                f"(Through Round {round_ref})"
            )
        elif round_ref is not None:
            title = f"EuroLeague Four Factors Profile (Through Round {round_ref})"
        else:
            title = "EuroLeague Four Factors Profile"

    df = four_factors.copy()
    df = df.sort_values("FF_Net", ascending=False).reset_index(drop=True)

    df["logo_url"] = df["TEAM_NAME"].apply(team_to_logo_path).apply(logo_path_to_url_fn)
    df["TEAM_KEY"] = df["TEAM_NAME"]
    df["TEAM_LABEL"] = df["TEAM_NAME"].apply(lambda t: team_display_name(t, mobile_mode))

    team_order = df["TEAM_KEY"].tolist()
    chart_height = row_height * len(df)
    y_enc = alt.Y("TEAM_KEY:N", sort=team_order, title=None, axis=None)

    # (column, header, side, is_percentage, higher_is_better)
    factor_columns = [
        ("eFG", "eFG%", "OFFENSE", True, True),
        ("TOV_Rate", "TOV%", "OFFENSE", True, False),
        ("OREB_Pct", "OR%", "OFFENSE", True, True),
        ("FT_Rate", "FTR", "OFFENSE", False, True),
        ("Opp_eFG", "eFG%", "DEFENSE", True, False),
        ("Opp_TOV_Rate", "TOV%", "DEFENSE", True, True),
        ("DREB_Pct", "DR%", "DEFENSE", True, True),
        ("Opp_FT_Rate", "FTR", "DEFENSE", False, False),
    ]

    # Left column: logo + team name, the shared row key for every column.
    logo_col = (
        alt.Chart(df)
        .mark_image(width=logo_size, height=logo_size)
        .encode(y=y_enc, x=alt.value(14), url="logo_url:N")
        .properties(width=team_col_width, height=chart_height, title="TEAM")
    )

    name_col = (
        alt.Chart(df)
        .mark_text(align="left", baseline="middle", dx=logo_size + 18, fontSize=name_font_size)
        .encode(y=y_enc, x=alt.value(0), text="TEAM_LABEL:N")
    )

    columns = [logo_col + name_col]

    for col, header, side, is_pct, higher_is_better in factor_columns:
        values = pd.to_numeric(df[col], errors="coerce")
        lo, hi = float(values.min()), float(values.max())
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            # One distinct value league-wide (a factor everyone matched, or a
            # column of NaN) leaves no gradient to build; a flat white column
            # is honest, a scale over a zero-width domain is not.
            color = alt.value("#ffffff")
        else:
            # White sits on the league mean, not halfway between the extremes,
            # so a cell's shade answers "better or worse than the field" rather
            # than "near which end of the range". Nudged off either endpoint
            # because a domain has to strictly increase: a lopsided column
            # (nineteen teams tied, one outlier) can put the mean on top of the
            # min or the max.
            mean = float(values.mean())
            span = hi - lo
            mid = min(max(mean, lo + span * 1e-6), hi - span * 1e-6)
            # The Next-N table's own red/white/green, so the two tables read as
            # one site rather than two charts that happen to share a page.
            palette = list(FAVORABILITY_PALETTE)
            color = alt.Color(
                f"{col}:Q",
                scale=alt.Scale(
                    domain=[lo, mid, hi],
                    range=palette if higher_is_better else palette[::-1],
                    # Vega interpolates colors in HCL by default, which does not
                    # pass cleanly through white — the mean cell comes out mint
                    # green instead of neutral. RGB gives the symmetric ramp the
                    # palette is written to describe.
                    interpolate="rgb",
                ),
                legend=None,
            )

        # A constant nominal x gives the column one band the full width of the
        # sub-chart, so the header ends up centered over the cells.
        df[f"{col}_x"] = ""
        fmt = ".1%" if is_pct else ".3f"

        cell = (
            alt.Chart(df)
            .mark_rect(stroke="#8f8f8f", strokeWidth=0.5)
            .encode(
                x=alt.X(f"{col}_x:N", title=None, axis=alt.Axis(labels=False, ticks=False)),
                y=y_enc,
                color=color,
                tooltip=[
                    alt.Tooltip("TEAM_NAME:N", title="Team"),
                    alt.Tooltip(f"{col}:Q", title=f"{side.title()} {header}", format=fmt),
                ],
            )
            .properties(
                width=factor_col_width,
                height=chart_height,
                title=header,
                # Drawn per column because Vega-Lite has no border for a concat
                # group. With spacing=0 inside the block the adjacent borders
                # meet, so four boxes read as one outlined table with rules
                # between the factors — which is the outline we actually want,
                # and it doubles as the column separator.
                view=alt.ViewConfig(stroke=BLOCK_OUTLINE, strokeWidth=1.5),
            )
        )

        label = (
            alt.Chart(df)
            .mark_text(baseline="middle", fontSize=value_font_size, color="black")
            .encode(
                x=alt.X(f"{col}_x:N"),
                y=y_enc,
                text=alt.Text(f"{col}:Q", format=fmt),
            )
        )

        columns.append(cell + label)

    # The composite, shaped exactly like the Next-N table's SoS column: a shaded
    # cell carrying its own number. It sits immediately after the team name so
    # the summary is the first thing on the row and the eight factor columns
    # read as the breakdown behind it.
    #
    # A cell rather than a diverging bar because this column's job is lookup —
    # scanning down it for one team — and a bar puts every near-zero value in
    # the middle of the column where the eye cannot line them up. Nothing is
    # lost: the offense/defense split is in the tooltip.
    df["FF_Net_x"] = ""

    # domainMid rather than an explicit domain: the rating is already centred on
    # zero by construction (league z-scores sum to nothing), so zero is the true
    # neutral point and white lands on it at every round. Reversed against the
    # Next-N scale, where green marks an *easy* schedule — here green marks a
    # good team, so that green keeps meaning "favorable for this team" on both.
    rating_color = alt.Color(
        "FF_Net:Q",
        scale=alt.Scale(domainMid=0.0, range=list(FAVORABILITY_PALETTE)),
        legend=None,
    )

    rating_cell = (
        alt.Chart(df)
        .mark_rect(stroke="#8f8f8f", strokeWidth=0.5)
        .encode(
            x=alt.X("FF_Net_x:N", title=None, axis=alt.Axis(labels=False, ticks=False)),
            y=y_enc,
            color=rating_color,
            tooltip=[
                alt.Tooltip("TEAM_NAME:N", title="Team"),
                alt.Tooltip("FF_Net:Q", title="Four Factor Rating", format="+.2f"),
                alt.Tooltip("FF_Off:Q", title="Offense", format="+.2f"),
                alt.Tooltip("FF_Def:Q", title="Defense", format="+.2f"),
                alt.Tooltip("Games:Q", title="Games"),
            ],
        )
        .properties(
            width=rating_col_width,
            height=chart_height,
            title="4F RATING",
            view=alt.ViewConfig(stroke=BLOCK_OUTLINE, strokeWidth=1.5),
        )
    )

    rating_text = (
        alt.Chart(df)
        .mark_text(baseline="middle", fontSize=value_font_size, fontWeight="bold")
        .encode(
            x=alt.X("FF_Net_x:N"),
            y=y_enc,
            text=alt.Text("FF_Net:Q", format="+.2f"),
        )
    )

    rating_col = rating_cell + rating_text

    # OFFENSE / DEFENSE grouping. The four columns of a side are concatenated
    # into their own block and the banner is that block's title, so it is
    # centered over exactly the columns it describes. A separate banner row of
    # fixed-width text charts cannot manage that: the team column's names
    # overflow their 250px box, Vega grows the group to fit, and every banner
    # after it drifts right of the columns it is supposed to label.
    def side_block(charts: list, label: str) -> alt.HConcatChart:
        return (
            # Zero spacing: the per-column borders have to meet to form one
            # outline. It also buys back 60px of chart width, which is 60px less
            # horizontal swiping on a phone.
            alt.hconcat(*charts, spacing=0)
            .resolve_scale(y="shared", x="independent", color="independent")
            .properties(
                title=alt.TitleParams(
                    label,
                    font=TITLE_FONT,
                    fontSize=FONT_SIZE + 1,
                    fontWeight="bold",
                    anchor="middle",
                )
            )
        )

    chart = (
        alt.hconcat(
            columns[0],
            rating_col,
            side_block(columns[1:5], "OFFENSE"),
            side_block(columns[5:9], "DEFENSE"),
            spacing=18,
        )
        # Independent color scales are what make the eight per-factor palettes
        # work at all: hconcat shares scales by default, so without this every
        # column would be painted from a single merged domain — one column's
        # range applied to another column's numbers.
        .resolve_scale(y="shared", x="independent", color="independent")
        .properties(
            title=alt.TitleParams(
                title, font=TITLE_FONT, fontSize=TITLE_FONT_SIZE, anchor="start"
            ),
            background=background,
            padding={"left": 20, "right": 20, "top": 20, "bottom": 20},
        )
        .configure_axis(
            labelFont=ROW_FONT,
            titleFont=ROW_FONT,
            labelFontSize=FONT_SIZE,
            titleFontSize=FONT_SIZE,
        )
        .configure_title(font=TITLE_FONT, fontSize=TITLE_FONT_SIZE)
        .configure_text(font=ROW_FONT, fontSize=FONT_SIZE)
        .configure_view(stroke=None)
        .configure_legend(
            labelFont=ROW_FONT,
            titleFont=ROW_FONT,
            labelFontSize=FONT_SIZE,
            titleFontSize=FONT_SIZE,
        )
    )

    return chart
