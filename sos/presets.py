"""
Chart sizing presets, split by desktop/mobile, ported from the original
Streamlit app's sidebar-driven layout logic (app.py).
"""

NEXTN_KWARGS = {
    "desktop": dict(
        left_col_width=320,
        sos_col_width=110,
        games_col_width=500,
        logo_size_main=24,
        logo_size_opp=24,
        font_size=14,
        title_font_size=19,
    ),
    "mobile": dict(
        left_col_width=45,
        sos_col_width=65,
        games_col_width=240,
        logo_size_main=20,
        logo_size_opp=24,
        font_size=11,
        title_font_size=13,
    ),
}

SCATTER_KWARGS = {
    "desktop": dict(main_w=560, main_h=600, table_w=380, table_h=600),
    "mobile": dict(main_w=380, main_h=550, table_w=380, table_h=550),
}

SEASON_KWARGS = {
    "desktop": dict(
        team_col_width=80,
        net_col_width=190,
        win_col_width=190,
        logo_size=24,
        row_height=26,
        name_font_size=13,
        value_font_size=10,
        font_size=13,
        title_font_size=16,
    ),
    "mobile": dict(
        team_col_width=52,
        net_col_width=90,
        win_col_width=90,
        logo_size=21,
        row_height=25,
        name_font_size=11,
        value_font_size=9,
        font_size=11,
        title_font_size=13,
    ),
}

VARIANTS = ("desktop", "mobile")
NEXT_N_VALUES = tuple(range(1, 11))
