"""Writers for everything a run leaves behind.

  seasons parquet    one row per simulated season: every club's place and points
  snapshots parquet  one row per run: when, what data, which settings
  history parquet    one row per club per run: the real table and the model's odds over time
  position parquet   one row per club per final place per run: the whole matrix over time
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


MODEL_VERSION_FIELDS = ("n_sims", "seed", "sigma", "elo", "goals_window_seasons")


def model_version(meta: dict) -> str:
    """A short fingerprint of the frozen settings. If it changes, the model changed."""
    import hashlib
    blob = json.dumps({k: meta[k] for k in MODEL_VERSION_FIELDS}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:10]


def snapshot_row(meta: dict) -> dict:
    """One row per run: when it ran, what data it saw and which settings made it."""
    return {"season": meta["season"], "matches_played": meta["matches_played"], "matchweek": meta["matchweek"],
            "data_through": meta["data_through"], "run_at": meta["generated_at"],
            "reconstructed": bool(meta.get("reconstructed", False)), "n_sims": meta["n_sims"], "seed": meta["seed"],
            "sigma": meta["sigma"], "elo_k": meta["elo"]["k"], "elo_promoted": meta["elo"]["promoted"],
            "goals_window_seasons": meta["goals_window_seasons"], "goal_alpha": meta["goal_params"]["alpha"],
            "goal_beta": meta["goal_params"]["beta"], "goal_gamma": meta["goal_params"]["gamma"],
            "primary_source": meta.get("primary_source"), "run_id": meta.get("run_id"),
            "model_version": model_version(meta),
            **{f"engine_{k}": v for k, v in (meta.get("telemetry") or {}).items()
               if k in ("matches", "goals", "random_draws", "distinct_champions", "distinct_relegated_trios", "seconds",
                        "seasons_incl_match_pricing")}}


def history_rows(meta: dict, headline: list[dict], table: list[dict] | None = None) -> list[dict]:
    """One row per club for this run: where the club really is, and where the model puts it."""
    keep = ("team", "played", "pts_now", "gd_now", "table_pos", "elo", "exp_pts", "most_likely_place",
            "title", "top4", "top5", "relegated")
    real = {r["team"]: {"won": r["w"], "drawn": r["d"], "lost": r["l"], "goals_for": r["gf"], "goals_against": r["ga"]}
            for r in (table or [])}
    return [{"season": meta["season"], "matches_played": meta["matches_played"], "matchweek": meta["matchweek"],
             "data_through": meta["data_through"], "run_at": meta["generated_at"],
             "reconstructed": bool(meta.get("reconstructed", False)), "model_pos": i + 1, **{k: r[k] for k in keep},
             **real.get(r["team"], {})}
            for i, r in enumerate(headline)]


def position_rows(meta: dict, matrix: np.ndarray, teams: list[str]) -> list[dict]:
    """One row per club per final place: the share of seasons that ended there."""
    return [{"season": meta["season"], "matches_played": meta["matches_played"], "matchweek": meta["matchweek"],
             "team": t, "place": j + 1, "share": float(matrix[i, j])}
            for i, t in enumerate(teams) for j in range(matrix.shape[1])]


SERIES = ("table_pos", "pts_now", "model_pos", "exp_pts", "elo", "title", "top4", "top5", "relegated")


def timeline(history: pd.DataFrame, season: str) -> dict:
    """The season so far, one point per matchweek, shaped for the page.

    A matchweek can be run more than once while its matches are being played.
    The run with the most results stands for that matchweek.
    """
    h = history[history["season"] == season]
    last = h.groupby("matchweek")["matches_played"].transform("max")
    h = h[h["matches_played"] == last].sort_values(["matchweek", "team"])
    weeks = h.drop_duplicates("matchweek")
    recon = weeks["reconstructed"].fillna(False).astype(bool) if "reconstructed" in weeks else [False] * len(weeks)
    out = {"season": season,
           "matchweeks": [{"matchweek": int(mw), "matches_played": int(mp), "data_through": str(dt), "reconstructed": bool(rc)}
                          for mw, mp, dt, rc in zip(weeks["matchweek"], weeks["matches_played"], weeks["data_through"], recon)],
           "clubs": {}}
    for team, rows in h.groupby("team"):
        out["clubs"][team] = {k: [None if pd.isna(v) else float(v) for v in rows[k]] for k in SERIES if k in rows}
    return round_floats(out)


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
