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
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import pandas as pd

from .charts import (
    build_nextN_altair_logos_table,
    make_sos_table_chart,
    make_sos_scatter_and_side_table,
    make_four_factors_chart,
)
from .compute import make_nextN_sos_table
from .config import season_label, season_publish_dir
from .utils import team_to_logo_path, logo_to_site_url
from .presets import (
    NEXTN_KWARGS,
    SCATTER_KWARGS,
    SEASON_KWARGS,
    FOURFACTORS_KWARGS,
    NEXT_N_VALUES,
    CHART_FONT,
)


def to_spec(chart) -> dict:
    """
    Serialize a chart, applying the site font to every piece of text in it.

    Set here rather than in sos/charts.py so the chart builders stay about
    chart content: this is the one place every published spec passes through.
    Vega-Lite's top-level `config.font` is the default for all text marks,
    titles and labels, so one key covers axes, legends, headers and marks
    without enumerating them.
    """
    spec = chart.to_dict()
    spec.setdefault("config", {})["font"] = CHART_FONT
    return spec


ROUND_DIR_RE = re.compile(r"^(\d+)$")


def build_latest_manifest(publish_dir: Path, current_season: int) -> dict:
    """
    The manifest the frontend boots from: which seasons exist and how far each got.

    Derived by scanning what has actually been published rather than from a
    hand-kept list, so the season picker can never offer a season whose JSON
    isn't on disk — and so adding one needs no edit here.
    """
    publish_dir = Path(publish_dir)
    seasons = []

    for season_dir in sorted((publish_dir / "seasons").glob("*")):
        if not season_dir.is_dir() or not season_dir.name.isdigit():
            continue
        rounds = [
            int(m.group(1))
            for d in (season_dir / "rounds").glob("*")
            if d.is_dir() and (m := ROUND_DIR_RE.match(d.name))
        ]
        if not rounds:
            continue
        season = int(season_dir.name)
        seasons.append(
            {
                "id": season,
                "label": season_label(season),
                "latest_round": max(rounds),
            }
        )

    seasons.sort(key=lambda entry: entry["id"])

    # Fall back to the season just published if its directory somehow isn't
    # readable, so the site always has at least one selectable season.
    if not any(entry["id"] == int(current_season) for entry in seasons):
        seasons.append(
            {
                "id": int(current_season),
                "label": season_label(current_season),
                "latest_round": 0,
            }
        )

    current = next(entry for entry in seasons if entry["id"] == int(current_season))

    return {
        "season": current["id"],
        "seasons": seasons,
        # Kept at the top level because it is what the round stepper opens on.
        "round": current["latest_round"],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def write_artifacts(out_dir: Path, artifacts: Dict[str, dict]) -> None:
    """Write {relative_path: spec} under out_dir, creating parent dirs."""
    for rel_path, spec in artifacts.items():
        dest = Path(out_dir) / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(spec), encoding="utf-8")


def build_round_artifacts(
    round_num: int,
    season: int,
    games_meta: pd.DataFrame,
    team_ratings: pd.DataFrame,
    sos_net: pd.DataFrame,
    sos_win: pd.DataFrame,
    four_factors: pd.DataFrame,
    season_label: str,
) -> Dict[str, dict]:
    """Return {storage_path: json_serializable_dict} for one round."""
    artifacts: Dict[str, dict] = {}
    base = f"{season_publish_dir(season)}/rounds/{round_num}"

    for n in NEXT_N_VALUES:
        nextN_df = make_nextN_sos_table(
            current_round=round_num,
            season=season,
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
        artifacts[f"{base}/next-n/{n}.json"] = to_spec(chart)

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
    artifacts[f"{base}/scatter.json"] = {
        "main": to_spec(main_chart),
        "table": to_spec(table_chart),
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
    artifacts[f"{base}/season-table.json"] = to_spec(season_chart)

    four_factors_chart = make_four_factors_chart(
        four_factors=four_factors,
        team_to_logo_path=team_to_logo_path,
        logo_path_to_url_fn=logo_to_site_url,
        round_ref=round_num,
        season_label=season_label,
        **FOURFACTORS_KWARGS,
    )
    artifacts[f"{base}/four-factors.json"] = to_spec(four_factors_chart)

    return artifacts
