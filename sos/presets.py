"""
Chart sizing presets, ported from the original Streamlit app's desktop layout
(app.py). A single fixed size per chart is published — the frontend scales
each rendered chart to fit its container via CSS/JS, so there's no need to
precompute separate desktop/mobile variants.
"""

# Chart text, matched to the site's own heading/body face so a chart doesn't
# read as a foreign object dropped onto the page. Charts render in the browser
# via vega-embed, so this resolves against the webfont frontend/index.html
# already loads — keep it in step with --font-heading in frontend/styles.css,
# and keep the <link> to Geomini in index.html or this silently falls back.
CHART_FONT = "Geomini, system-ui, sans-serif"

NEXTN_KWARGS = dict(
    left_col_width=320,
    sos_col_width=110,
    games_col_width=500,
    logo_size_main=24,
    logo_size_opp=24,
    font_size=14,
    title_font_size=19,
)

SCATTER_KWARGS = dict(main_w=560, main_h=600, table_w=380, table_h=600)

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

FOURFACTORS_KWARGS = dict(
    team_col_width=250,
    factor_col_width=74,
    rating_col_width=110,
    row_height=26,
    logo_size=24,
    name_font_size=13,
    value_font_size=11,
    font_size=13,
    title_font_size=18,
)

NEXT_N_VALUES = tuple(range(1, 6))
