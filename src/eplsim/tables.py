"""Tables for the dashboard, derived from what every run already computes.

  matches.parquet        one row per fixture, every season on file
  match_ratings.parquet  one row per played match: both ratings before and after
  points_dist.parquet    one row per club per final points total per run
  clubs.parquet          one row per club ever in the data
  gameweeks.parquet      one row per season per gameweek

Nothing here feeds the model, the page or a published number. Every table can
be rebuilt from the repo at any time, so they are rewritten whole on each live
run, apart from `points_dist`, which is upserted per run like `position_dist`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import clubs, elo
from .data import Match
from .teams import canonical

POINTS_KEY = ["season", "matches_played", "team", "points"]

# Codes for clubs on file that are not in this season, so every club in the data has its own.
# Checked 18 Nov 2026 against the Premier League's data API (footballapi.pulselive.com, the `abbr`
# field); the current twenty in clubs.py match it too. clubs.py keeps them (and colours) for the page.
PAST_CODES = {
    "Birmingham": "BIR", "Blackburn": "BLB", "Blackpool": "BLP", "Bolton": "BOL", "Burnley": "BUR",
    "Cardiff": "CAR", "Huddersfield": "HUD", "Leicester": "LEI", "Luton": "LUT", "Middlesbrough": "MID",
    "Norwich": "NOR", "QPR": "QPR", "Reading": "RDG", "Sheffield United": "SHU", "Southampton": "SOU",
    "Stoke": "STK", "Swansea": "SWA", "Watford": "WAT", "West Brom": "WBA", "West Ham": "WHU",
    "Wigan": "WIG", "Wolves": "WOL",
}


def code(team: str) -> str:
    """The club's code: this season's from clubs.py, past clubs from PAST_CODES."""
    if team in clubs.CLUBS:
        return clubs.CLUBS[team]["code"]
    return PAST_CODES.get(team) or clubs.code(team)


def half_times(paths: list[Path]) -> dict[tuple[str, str, str], tuple[int, int]]:
    """Half-time scores from openfootball files, keyed by (season, home, away). Missing where the file has none."""
    out: dict[tuple[str, str, str], tuple[int, int]] = {}
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        season = re.search(r"(\d{4}-\d{2})", path.name).group(1)
        for m in json.loads(path.read_text(encoding="utf-8"))["matches"]:
            score = m.get("score")
            if isinstance(score, dict) and isinstance(score.get("ht"), list) and len(score["ht"]) == 2:
                out[(season, canonical(m["team1"]), canonical(m["team2"]))] = (int(score["ht"][0]), int(score["ht"][1]))
    return out


def _result(m: Match) -> str | None:
    if not m.played:
        return None
    return "H" if m.home_goals > m.away_goals else "D" if m.home_goals == m.away_goals else "A"


def matches_frame(matches: list[Match], reports: dict[str, dict], ht: dict, postponed: set,
                  current: str) -> pd.DataFrame:
    """Every fixture with its result, half-time score and how many sources backed the score.

    Seasons without a report come from one source, openfootball.
    """
    rows = []
    for m in matches:
        rep = reports.get(m.season)
        key = (m.home, m.away)
        if rep is not None:
            backers = rep.get("backers", {}).get(key, [])
            overruled = any(tuple(o["fixture"]) == key for o in rep.get("overruled", []))
        else:
            backers, overruled = (["openfootball"] if m.played else []), False
        h = ht.get((m.season, m.home, m.away)) if m.played else None
        status = ("played" if m.played else
                  "postponed" if m.season == current and key in postponed else "unplayed")
        rows.append({
            "season": m.season, "gameweek": m.gameweek, "date": m.date, "home": m.home, "away": m.away,
            "home_code": code(m.home), "away_code": code(m.away),
            "home_goals": m.home_goals, "away_goals": m.away_goals, "result": _result(m),
            "ht_home_goals": h[0] if h else None, "ht_away_goals": h[1] if h else None,
            "status": status, "sources_backing": len(backers), "sources": "+".join(backers) or None,
            "overruled": overruled,
        })
    df = pd.DataFrame(rows)
    for c in ("home_goals", "away_goals", "ht_home_goals", "ht_away_goals"):
        df[c] = df[c].astype("Int64")
    return df.sort_values(["season", "date", "home"], kind="stable").reset_index(drop=True)


def match_ratings_frame(matches: list[Match], p: elo.EloParams) -> pd.DataFrame:
    """The rating walk the model runs, one row per played match: before, after, and the expected score."""
    pre, _ = elo.run_elo(matches, p)
    rows = []
    for i, (m, (rh, ra)) in enumerate(zip(elo.played_in_order(matches), pre)):
        nh, na = elo.update(rh, ra, m.home_goals, m.away_goals, p)
        rows.append({
            "season": m.season, "gameweek": m.gameweek, "date": m.date, "home": m.home, "away": m.away,
            "home_goals": m.home_goals, "away_goals": m.away_goals, "order": i + 1,
            "home_elo_pre": rh, "away_elo_pre": ra, "home_elo_post": nh, "away_elo_post": na,
            "home_change": nh - rh, "away_change": na - ra,
            "home_expected": elo.expected_home(rh, ra, p.hfa),
        })
    return pd.DataFrame(rows)


def points_rows(meta: dict, points: np.ndarray, teams: list[str]) -> list[dict]:
    """Share of simulated seasons on each final points total, per club. Only totals that happened."""
    out = []
    n = points.shape[0]
    for i, t in enumerate(teams):
        vals, counts = np.unique(points[:, i], return_counts=True)
        out += [{"season": meta["season"], "matches_played": meta["matches_played"], "matchweek": meta["matchweek"],
                 "team": t, "points": int(v), "share": float(c) / n} for v, c in zip(vals, counts)]
    return out


def clubs_frame(matches: list[Match], seasons: list[str], current: str) -> pd.DataFrame:
    """Every club in the data, with its code, colours and Premier League seasons on file."""
    members: dict[str, set[str]] = {}
    for m in matches:
        for t in (m.home, m.away):
            members.setdefault(t, set()).add(m.season)
    prev = seasons[seasons.index(current) - 1]
    rows = []
    for t, ss in sorted(members.items()):
        c = clubs.club(t)
        rows.append({"team": t, "code": code(t), "primary": c["primary"], "secondary": c["secondary"],
                     "own_colours": t in clubs.CLUBS, "colours_verified": bool(clubs.COLOURS_VERIFIED),
                     "seasons_on_file": len(ss), "first_season": min(ss),
                     "in_current_season": current in ss, "promoted_this_season": current in ss and prev not in ss})
    return pd.DataFrame(rows)


def gameweeks_frame(matches: list[Match], snapshots: pd.DataFrame | None, postponed: set, current: str) -> pd.DataFrame:
    """One row per season per gameweek, and for this season the stored run that stands for it."""
    rows: dict[tuple[str, int], dict] = {}
    for m in matches:
        if not m.gameweek:
            continue
        r = rows.setdefault((m.season, m.gameweek), {"season": m.season, "gameweek": m.gameweek, "first_date": m.date,
                                                     "last_date": m.date, "fixtures": 0, "played": 0, "postponed": 0})
        r["first_date"], r["last_date"] = min(r["first_date"], m.date), max(r["last_date"], m.date)
        r["fixtures"] += 1
        r["played"] += m.played
        r["postponed"] += (not m.played) and m.season == current and (m.home, m.away) in postponed
    stands: dict[int, dict] = {}
    if snapshots is not None and len(snapshots):
        s = snapshots[snapshots["season"] == current]
        for mw, grp in s.groupby("matchweek"):
            top = grp.loc[grp["matches_played"].idxmax()]
            stands[int(mw)] = {"snapshot_matches_played": int(top["matches_played"]),
                               "snapshot_reconstructed": bool(top["reconstructed"])}
    out = []
    for (season, gw), r in sorted(rows.items()):
        done = r["played"] + r["postponed"]
        r["status"] = "complete" if done == r["fixtures"] else "in progress" if r["played"] else "upcoming"
        st = stands.get(gw, {}) if season == current else {}
        r["snapshot_matches_played"] = st.get("snapshot_matches_played")
        r["snapshot_reconstructed"] = st.get("snapshot_reconstructed")
        out.append(r)
    df = pd.DataFrame(out)
    df["snapshot_matches_played"] = df["snapshot_matches_played"].astype("Int64")
    df["snapshot_reconstructed"] = df["snapshot_reconstructed"].astype("boolean")
    return df


def universes_path(out_dir: Path, season: str, matchweek: int) -> Path:
    """One file per gameweek. A later run in the same gameweek overwrites it, so each gameweek keeps one set."""
    return Path(out_dir) / "universes" / season / f"gw{matchweek:02d}.parquet"


def universes_frame(meta: dict, res) -> pd.DataFrame:
    """All simulated seasons of one run, one row per universe per club.

    `universe` is the number the page uses (1 to n_sims). Numbers only mean the
    same season within one gameweek: the next gameweek's run starts from new
    results, so universe 4,817 there is a different season.
    """
    n, k = res.points.shape
    return pd.DataFrame({
        "season": meta["season"], "matchweek": meta["matchweek"], "matches_played": meta["matches_played"],
        "run_at": meta["generated_at"],
        "universe": np.repeat(np.arange(1, n + 1, dtype=np.int32), k),
        "team": np.tile(np.array(res.teams, dtype=object), n),
        "place": res.position.reshape(-1).astype(np.int8),
        "points": res.points.reshape(-1).astype(np.int16),
        "goal_diff": res.goal_diff.reshape(-1).astype(np.int16),
        "goals_for": (res.goals_for.reshape(-1) if res.goals_for is not None
                      else np.full(n * k, -1)).astype(np.int16),
    })


_B36 = "0123456789abcdefghijklmnopqrstuvwxyz"


def site_universes(df: pd.DataFrame, teams: list[str]) -> dict:
    """One gameweek of universes, packed small for the page.

    `place` holds one letter per club per universe (A = 1st ... T = 20th), clubs
    in `teams` order, universes 1 to n in turn. `points` holds two base-36 digits
    per club per universe in the same order. Universe n starts at (n - 1) * len(teams).
    """
    k = len(teams)
    d = df.assign(_i=df["team"].map({t: i for i, t in enumerate(teams)})).sort_values(["universe", "_i"])
    per = d.groupby("universe")["_i"].agg(["nunique", "size"]) if not d["_i"].isna().any() else None
    if per is None or not ((per["nunique"] == k) & (per["size"] == k)).all():
        raise ValueError("universe rows do not cover every club exactly once")
    place = d["place"].to_numpy().astype(int)
    pts = d["points"].to_numpy().astype(int)
    if place.min() < 1 or place.max() > 26 or pts.min() < 0 or pts.max() >= 36 * 36:
        raise ValueError("place or points out of range for packing")
    first = df.iloc[0]
    return {"season": str(first["season"]), "matchweek": int(first["matchweek"]),
            "matches_played": int(first["matches_played"]), "run_at": str(first["run_at"]),
            "n_sims": len(d) // k, "teams": list(teams),
            "place": "".join(chr(64 + p) for p in place),
            "points": "".join(_B36[p // 36] + _B36[p % 36] for p in pts)}


def site_universe_index(site_dir: Path, season: str) -> dict:
    """Which gameweeks have universes on the site, newest last."""
    weeks = []
    for f in sorted((Path(site_dir) / "universes" / season).glob("gw*.json")):
        head = json.loads(f.read_text(encoding="utf-8"))
        weeks.append({"matchweek": head["matchweek"], "matches_played": head["matches_played"],
                      "file": f"universes/{season}/{f.name}"})
    return {"season": season, "weeks": sorted(weeks, key=lambda w: w["matchweek"])}


def write(path: Path, df: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="zstd")
