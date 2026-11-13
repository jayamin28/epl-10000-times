"""The weekly run, start to finish, in one command.

    python -m eplsim.run_week                  fetch, check, simulate, write outputs
    python -m eplsim.run_week --offline        skip the fetch and use the files on disk
    python -m eplsim.run_week --force          run even if no new result has arrived
    python -m eplsim.run_week --archive launch also keep a permanent copy of the 10,000 seasons
    python -m eplsim.run_week --as-of 7        rebuild the picture as it stood after matchweek 7
    python -m eplsim.run_week --backfill       rebuild every earlier matchweek again, then run this one

Every run first fills gaps: a finished matchweek with no snapshot (the first
run of a season, or a week the schedule missed) is rebuilt from the results up
to that week and marked as rebuilt.

Steps: fetch three sources, merge them by majority vote, run the gates, stop
if nothing changed, simulate 10,000 seasons, compute every published number,
write the outputs. Exit code 0 means published or nothing to do. Anything
else means a gate stopped the run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from . import clubs, config, fetch, gates, metrics, outputs, pipeline
from .data import league_table


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _fpl_positions(current_dir: Path, season: str) -> tuple[dict[str, int] | None, int | None]:
    """Positions published by the official feed, and how many results that feed holds."""
    from .data import load_fpl
    from .teams import canonical
    fx, tm = current_dir / "fpl" / f"fixtures_{season}.csv", current_dir / "fpl" / f"teams_{season}.csv"
    if not (fx.exists() and tm.exists()):
        return None, None
    with open(tm, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    positions = {canonical(r["name"]): int(float(r["position"])) for r in rows if (r.get("position") or "0") not in ("", "0")}
    return positions or None, sum(m.played for m in load_fpl(fx, tm, season))


def _results_hash(current) -> str:
    h = hashlib.sha256()
    for m in sorted(current, key=lambda m: (m.home, m.away)):
        h.update(f"{m.gameweek}|{m.date}|{m.home}|{m.away}|{m.home_goals}|{m.away_goals}\n".encode())
    return h.hexdigest()[:16]


def run(data_dir: Path, out_dir: Path, site_dir: Path, current_dir: Path | None = None, do_fetch: bool = True,
        force: bool = False, now: datetime | None = None, n_sims: int = config.N_SIMS, get=fetch.http_get,
        archive: str | None = None, log=print, as_of: int | None = None) -> dict:
    """Run one week. With `as_of`, rebuild an earlier matchweek instead.

    A rebuilt matchweek uses only the results up to that matchweek, with today's
    frozen settings. It adds to the history and writes its own snapshot. It does
    not touch the live feed, the forecast log or the change detector, and it is
    marked as reconstructed: it is what the model would have said, not what it said.
    """
    data_dir, out_dir, site_dir = Path(data_dir), Path(out_dir), Path(site_dir)
    if as_of is not None:
        do_fetch = False
    current_dir = Path(current_dir) if current_dir else data_dir / "current"
    now = now or datetime.now(timezone.utc)
    season = pipeline.CURRENT

    manifest = fetch.fetch_current(current_dir, season, get=get, now=now) if do_fetch else None
    if manifest:
        for name, s in manifest["sources"].items():
            log(f"fetch {name}: {'ok' if s.get('ok') else 'FAILED'} {s.get('origin') or ''} {s.get('error') or ''}".rstrip())

    matches, report = pipeline.load_all(data_dir, current_dir=current_dir, with_report=True)
    rec = report[season]
    if as_of is not None:
        matches = [replace(m, home_goals=None, away_goals=None) if m.season == season and m.gameweek > as_of else m
                   for m in matches]
    current = [m for m in matches if m.season == season]
    played = sum(m.played for m in current)
    if as_of is not None and not played:
        raise ValueError(f"no results up to matchweek {as_of}")
    log(f"reconcile: {played} results. {rec['agreed']} confirmed by 2+ sources, {len(rec['single_source'])} single-source, "
        f"{len(rec['overruled'])} overruled, {len(rec['awaiting_second_source'])} waiting for a second source. "
        f"Primary: {rec['primary']} {rec['sources']}")

    previous = _read_json(out_dir / "run_meta.json")
    postponed = {tuple(p) for p in (_read_json(data_dir / "postponed.json") or {}).get("fixtures", [])}
    positions, fpl_results = _fpl_positions(current_dir, season)
    if as_of is None:
        checks = gates.run_gates(current, league_table(current), now.date(), postponed,
                                 previous["matches_played"] if previous else None, manifest, positions, fpl_results)
    else:
        checks = [gates.fixture_list(current)]
        if checks[0]["status"] == "fail":
            raise gates.GateError(f"fixture_list: {checks[0]['detail']}")
    for c in checks:
        log(f"gate {c['gate']}: {c['status']} ({c['detail']})")

    results_hash = _results_hash(current)
    if as_of is None and previous and previous.get("results_hash") == results_hash and not force:
        log("no new results since the last run, nothing to publish")
        return {"changed": False, "matches_played": played}

    snap = pipeline.snapshot(matches, config.ELO, config.GOALS_WINDOW)

    def sim(forced=None):
        return pipeline.run_sim(snap, n_sims=n_sims, sigma=config.SIGMA, seed=config.SEED, forced=forced)

    res = sim()
    matchweek = max(m.gameweek for m in current if m.played)
    meta = {
        "season": season, "matchweek": matchweek, "matches_played": played, "matches_remaining": len(snap.remaining),
        "data_through": snap.last_result_date, "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_sims": n_sims, "seed": config.SEED, "sigma": config.SIGMA,
        "elo": {"k": config.ELO.k, "home_bonus": config.ELO.hfa, "promoted": config.ELO.promoted},
        "goals_window_seasons": config.GOALS_WINDOW,
        "goal_params": {"alpha": snap.goal_params.alpha, "beta": snap.goal_params.beta, "gamma": snap.goal_params.gamma},
        "sources": rec["sources"], "primary_source": rec["primary"],
        "fpl_origin": (manifest or {}).get("sources", {}).get("fpl", {}).get("origin"),
        "reconciliation": {"confirmed_by_two_or_more": rec["agreed"], "single_source": len(rec["single_source"]),
                           "overruled": len(rec["overruled"]), "waiting_for_second_source": len(rec["awaiting_second_source"])},
        "gates": checks, "tiebreak": res.tiebreak, "level_at_key_places": res.full_ties,
        "noise_note": "Two runs of the same data can differ by about 1.3 points. Changes under 3 points are not news.",
        "badge_colours_verified": clubs.COLOURS_VERIFIED,
        "reconstructed": as_of is not None,
        "run_id": os.environ.get("GITHUB_RUN_ID"), "commit": os.environ.get("GITHUB_SHA"),
    }

    headline = metrics.headline(snap, res)
    nxt = metrics.upcoming(snap)
    log(f"simulated {n_sims} seasons. Working out what next week's {len(nxt['fixtures'])} fixtures are worth")
    swings = metrics.match_swings(snap, sim, nxt["fixtures"])
    leaders = [r["team"] for r in sorted(headline, key=lambda r: (-r["title"], r["team"]))[:2]]

    outputs.upsert(out_dir / "snapshots.parquet", [outputs.snapshot_row(meta)],
                   key=["season", "matches_played"], sort=["season", "matches_played"])
    hist = outputs.upsert(out_dir / "history.parquet", outputs.history_rows(meta, headline, snap.table),
                          key=["season", "matches_played", "team"], sort=["season", "matches_played", "team"])
    earlier = hist[(hist["season"] == season) & (hist["matchweek"] < matchweek)]
    if len(earlier):
        base = earlier[earlier["matches_played"] == earlier["matches_played"].max()]
        before = {r["team"]: r for r in base.to_dict("records")}
        moved = {"since_matchweek": int(base["matchweek"].iloc[0]),
                 "rows": metrics.movers(before, {r["team"]: r for r in headline})}
    else:
        moved = {"since_matchweek": None, "rows": []}

    outputs.upsert(out_dir / "position_dist.parquet", outputs.position_rows(meta, res.position_matrix(), res.teams),
                   key=["season", "matches_played", "team", "place"], sort=["season", "matches_played", "team", "place"])
    log_path = out_dir / "forecast_log.parquet"
    if as_of is None:  # only odds given before kick-off belong in the log
        log_table = outputs.upsert(log_path, outputs.forecast_rows(meta, nxt),
                                   key=["season", "home", "away"], sort=["season", "gameweek", "date", "home"])
    else:
        import pandas as pd
        log_table = pd.read_parquet(log_path) if log_path.exists() else pd.DataFrame()
    known = {(m.home, m.away): (m.home_goals, m.away_goals) for m in current if m.played}
    past = [m for m in matches if m.played and m.season not in pipeline.BURN_IN and m.season != season]
    hg = np.array([m.home_goals for m in past])
    ag = np.array([m.away_goals for m in past])
    base_rates = (float((hg > ag).mean()), float((hg == ag).mean()), float((hg < ag).mean()))
    card = metrics.report_card(log_table[log_table["season"] == season].to_dict("records") if len(log_table) else [],
                               known, base_rates)

    sections = {
        "tiers": metrics.tiers(res), "table_lies": metrics.table_lies(snap, res),
        "magic_numbers": metrics.magic_numbers(res), "points_shape": metrics.points_shape(res),
        "run_in": metrics.run_in(snap), "next": nxt, "swings": swings,
        "wildest": metrics.wildest_season(snap, res), "convergence": metrics.convergence(res),
        "duel": metrics.duel(snap, res, leaders[0], leaders[1]), "movers": moved,
        "universes": metrics.universes(res), "report_card": card,
    }
    feed = outputs.build_feed(meta, headline, res.position_matrix(), res.teams, sections)
    outputs.write_json(site_dir / "snapshots" / f"mw{matchweek:02d}.json", feed)
    outputs.write_json(site_dir / "history.json", outputs.timeline(hist, season))
    if as_of is not None:
        log(f"rebuilt matchweek {matchweek}: {played} played, {len(snap.remaining)} simulated")
        return {"changed": True, "matches_played": played, "matchweek": matchweek, "feed": feed}
    outputs.write_json(site_dir / "feed.json", feed)
    outputs.write_seasons(out_dir / "seasons_latest.parquet", res)
    if archive:
        outputs.write_seasons(out_dir / "archive" / f"seasons_{archive}.parquet", res)
        outputs.write_json(out_dir / "archive" / f"feed_{archive}.json", feed)
    outputs.write_json(out_dir / "run_meta.json", {
        "season": season, "matchweek": matchweek, "matches_played": played, "results_hash": results_hash,
        "data_through": snap.last_result_date, "generated_at": meta["generated_at"]})

    top = headline[0]
    log(f"published matchweek {matchweek}: {played} played, {len(snap.remaining)} simulated. "
        f"Favourite {max(headline, key=lambda r: r['title'])['team']}, top of expected points {top['team']} {top['exp_pts']:.1f}")
    return {"changed": True, "matches_played": played, "matchweek": matchweek, "feed": feed}


def finished_matchweeks(matches, season: str = pipeline.CURRENT, postponed=frozenset()) -> list[int]:
    """Matchweeks whose every fixture has a result, apart from postponed ones."""
    weeks: dict[int, bool] = {}
    for m in matches:
        if m.season == season and m.gameweek and m.gameweek > 0:
            done = m.played or (m.home, m.away) in postponed
            weeks[m.gameweek] = weeks.get(m.gameweek, True) and done
    return sorted(w for w, done in weeks.items() if done)


def backfill(data_dir: Path, out_dir: Path, site_dir: Path, current_dir: Path | None = None, log=print,
             only_missing: bool = False, **kw) -> list[int]:
    """Rebuild finished matchweeks before the latest one, oldest first. Returns the matchweeks rebuilt.

    With `only_missing`, skip any matchweek the history already holds, so a
    snapshot made at the time is never replaced by a rebuilt one.
    """
    cur = Path(current_dir) if current_dir else Path(data_dir) / "current"
    matches = pipeline.load_all(data_dir, current_dir=cur)
    postponed = {tuple(p) for p in (_read_json(Path(data_dir) / "postponed.json") or {}).get("fixtures", [])}
    played = [m.gameweek for m in matches if m.season == pipeline.CURRENT and m.played]
    if not played:
        return []
    latest = max(played)
    todo = [w for w in finished_matchweeks(matches, postponed=postponed) if w < latest]
    hist_path = Path(out_dir) / "history.parquet"
    if only_missing and hist_path.exists():
        import pandas as pd
        h = pd.read_parquet(hist_path, columns=["season", "matchweek"])
        have = set(h.loc[h["season"] == pipeline.CURRENT, "matchweek"].astype(int))
        todo = [w for w in todo if w not in have]
    for mw in todo:
        run(data_dir, out_dir, site_dir, current_dir=cur, as_of=mw, log=log, **kw)
    return todo


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--site", default="site")
    ap.add_argument("--current", default=None, help="folder holding this season's source files (default data/current)")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--archive", default=None)
    ap.add_argument("--sims", type=int, default=config.N_SIMS)
    ap.add_argument("--today", default=None, help="run as if it were this date, YYYY-MM-DD")
    ap.add_argument("--as-of", type=int, default=None, help="rebuild the picture after this matchweek")
    ap.add_argument("--backfill", action="store_true", help="rebuild every earlier matchweek, then run this one")
    a = ap.parse_args(argv)
    now = datetime.combine(date.fromisoformat(a.today), datetime.min.time(), tzinfo=timezone.utc) if a.today else None
    common = dict(current_dir=a.current, now=now, n_sims=a.sims)
    try:
        if a.as_of is not None:
            run(Path(a.data), Path(a.out), Path(a.site), as_of=a.as_of, **common)
            return 0
        out = run(Path(a.data), Path(a.out), Path(a.site), do_fetch=not a.offline, force=a.force,
                  archive=a.archive, **common)
        rebuilt = backfill(Path(a.data), Path(a.out), Path(a.site), only_missing=not a.backfill, **common)
        if rebuilt:
            print(f"rebuilt matchweeks {', '.join(map(str, rebuilt))}")
            out = run(Path(a.data), Path(a.out), Path(a.site), do_fetch=False, force=True, archive=a.archive, **common)
    except gates.GateError as e:
        print(f"STOPPED by a gate, nothing published: {e}", file=sys.stderr)
        return 2
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
            f.write(f"changed={'true' if out['changed'] else 'false'}\n")
            if out["changed"]:
                f.write(f"matchweek={out['matchweek']}\nplayed={out['matches_played']}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
