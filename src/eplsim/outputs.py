"""Writers for everything a run leaves behind.

  seasons parquet    one row per simulated season: every club's place and points
  history parquet    one row per club per run: the headline odds over time
  forecast log       one row per upcoming match: the odds given before kick-off
  feed.json          everything the page draws, plus how the run was made
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .clubs import club, code
from .simulate import SimResult


def seasons_frame(res: SimResult) -> pd.DataFrame:
    cols: dict[str, np.ndarray] = {"season": np.arange(1, res.n_sims + 1, dtype=np.int32)}
    for i, t in enumerate(res.teams):
        cols[f"pos_{code(t)}"] = res.position[:, i].astype(np.int8)
    for i, t in enumerate(res.teams):
        cols[f"pts_{code(t)}"] = res.points[:, i].astype(np.int16)
    return pd.DataFrame(cols)


def write_seasons(path: Path, res: SimResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    seasons_frame(res).to_parquet(path, index=False, compression="zstd")


def upsert(path: Path, rows: list[dict], key: list[str], sort: list[str]) -> pd.DataFrame:
    """Add rows to a parquet table, replacing any existing rows with the same key."""
    path = Path(path)
    new = pd.DataFrame(rows)
    if path.exists() and len(new):
        old = pd.read_parquet(path)
        marker = old[key].merge(new[key].drop_duplicates(), on=key, how="left", indicator=True)["_merge"]
        new = pd.concat([old[(marker == "left_only").to_numpy()], new], ignore_index=True)
    elif path.exists():
        new = pd.read_parquet(path)
    if len(new):
        new = new.sort_values(sort, kind="stable").reset_index(drop=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        new.to_parquet(path, index=False, compression="zstd")
    return new


def history_rows(meta: dict, headline: list[dict]) -> list[dict]:
    keep = ("team", "played", "pts_now", "elo", "exp_pts", "title", "top4", "top5", "relegated")
    return [{"season": meta["season"], "matches_played": meta["matches_played"], "matchweek": meta["matchweek"],
             "data_through": meta["data_through"], "run_at": meta["generated_at"], **{k: r[k] for k in keep}}
            for r in headline]


def forecast_rows(meta: dict, upcoming: dict) -> list[dict]:
    return [{"season": meta["season"], "gameweek": upcoming["gameweek"], "date": f["date"], "home": f["home"],
             "away": f["away"], "p_home": f["home_win"], "p_draw": f["draw"], "p_away": f["away_win"],
             "logged_at": meta["generated_at"], "data_through": meta["data_through"]}
            for f in upcoming["fixtures"]]


def round_floats(obj, digits: int = 4):
    """Round every float in a nested structure and turn numpy scalars into plain Python."""
    if isinstance(obj, dict):
        return {str(k): round_floats(v, digits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_floats(v, digits) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return round(float(obj), digits)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def build_feed(meta: dict, headline: list[dict], matrix: np.ndarray, teams: list[str], sections: dict) -> dict:
    """Assemble the page feed. Clubs are in headline order and the matrix rows follow the same order."""
    idx = {t: i for i, t in enumerate(teams)}
    clubs = [{**r, "code": club(r["team"])["code"], "primary": club(r["team"])["primary"],
              "secondary": club(r["team"])["secondary"]} for r in headline]
    feed = {"meta": meta, "clubs": clubs,
            "matrix": {"teams": [r["team"] for r in headline], "probs": [matrix[idx[r["team"]]].tolist() for r in headline]},
            **sections}
    return round_floats(feed)


def write_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
