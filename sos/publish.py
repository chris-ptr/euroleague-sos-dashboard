"""
Build the precomputed Vega-Lite JSON artifacts for a single round and write them
out as static files under PUBLISH_DIR, where Vercel serves them. Every chart
builder call here is the existing, unmodified sos/charts.py logic — only the
output shape (dict instead of rendered chart) is new. The frontend scales each
rendered chart to fit its container, so only one size per chart is needed.

Published specs reference team logos by site-relative URL, never as inline
base64 — see sos.utils.logo_to_site_url for why that matters.
"""
import json
from pathlib import Path
from typing import Dict

import pandas as pd

from .charts import (
    build_nextN_altair_logos_table,
    make_sos_table_chart,
    make_sos_scatter_and_side_table,
)
from .compute import make_nextN_sos_table
from .utils import team_to_logo_path, logo_to_site_url
from .presets import NEXTN_KWARGS, SCATTER_KWARGS, SEASON_KWARGS, NEXT_N_VALUES


def write_artifacts(out_dir: Path, artifacts: Dict[str, dict]) -> None:
    """Write {relative_path: spec} under out_dir, creating parent dirs."""
    for rel_path, spec in artifacts.items():
        dest = Path(out_dir) / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(spec), encoding="utf-8")


def build_round_artifacts(
    round_num: int,
    schedule_path: str | Path,
    games_meta: pd.DataFrame,
    team_ratings: pd.DataFrame,
    sos_net: pd.DataFrame,
    sos_win: pd.DataFrame,
    season_label: str,
) -> Dict[str, dict]:
    """Return {storage_path: json_serializable_dict} for one round."""
    artifacts: Dict[str, dict] = {}

    for n in NEXT_N_VALUES:
        nextN_df = make_nextN_sos_table(
            current_round=round_num,
            schedule_path=schedule_path,
            games_meta=games_meta,
            team_ratings=team_ratings,
            n_next=n,
        )
        chart = build_nextN_altair_logos_table(
            nextN_df=nextN_df,
            team_ratings=team_ratings,
            team_to_logo_path_fn=team_to_logo_path,
            round_ref=round_num,
            n_next=n,
            logo_path_to_url_fn=logo_to_site_url,
            **NEXTN_KWARGS,
        )
        artifacts[f"rounds/{round_num}/next-n/{n}.json"] = chart.to_dict()

    main_chart, table_chart = make_sos_scatter_and_side_table(
        sos_net=sos_net,
        team_ratings=team_ratings,
        team_to_logo_path=team_to_logo_path,
        logo_path_to_url_fn=logo_to_site_url,
        top_k=5,
        bottom_k=5,
        round_ref=round_num,
        season_label=season_label,
        **SCATTER_KWARGS,
    )
    artifacts[f"rounds/{round_num}/scatter.json"] = {
        "main": main_chart.to_dict(),
        "table": table_chart.to_dict(),
    }

    season_chart = make_sos_table_chart(
        sos_net=sos_net,
        sos_win=sos_win,
        team_to_logo_path=team_to_logo_path,
        logo_path_to_url_fn=logo_to_site_url,
        round_ref=round_num,
        season_label=season_label,
        **SEASON_KWARGS,
    )
    artifacts[f"rounds/{round_num}/season-table.json"] = season_chart.to_dict()

    return artifacts
