"""Glue: load every season, tune, fit, backtest, simulate."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from . import goals
from .data import Match, league_table, load_fpl, load_mirror, load_openfootball, reconcile
from .elo import EloParams, played_in_order, run_elo
from .simulate import SimResult, simulate

HISTORY_SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2010, 2025)]  # 2010-11 .. 2024-25
SEASONS = HISTORY_SEASONS + ["2025-26", "2026-27"]
BURN_IN = {"2010-11", "2011-12"}  # ratings still settling, never used for fitting or scoring
TEST_SEASONS = {"2022-23", "2023-24", "2024-25", "2025-26"}  # never used for tuning
CURRENT = "2026-27"


def current_sources(current_dir: Path, season: str = CURRENT) -> dict[str, list[Match]]:
    """Every source found on disk for the live season, freshest first.

    The freshest source sets the fixture list in `reconcile`. Equal counts
    keep the order below, so the official feed leads when it is up to date.
    """
    cur = Path(current_dir)
    found: dict[str, list[Match]] = {}
    fx, tm = cur / "fpl" / f"fixtures_{season}.csv", cur / "fpl" / f"teams_{season}.csv"
    if fx.exists() and tm.exists():
        found["fpl"] = load_fpl(fx, tm, season)
    if (cur / f"fei_{season}" / "teams.csv").exists():
        found["mirror"] = load_mirror(cur / f"fei_{season}", season)
    if (cur / "openfootball" / f"{season}.json").exists():
        found["openfootball"] = load_openfootball(cur / "openfootball" / f"{season}.json", season)
    if not found:
        raise FileNotFoundError(f"No source files for {season} under {cur}")
    return {n: found[n] for n in sorted(found, key=lambda n: -sum(m.played for m in found[n]))}


def load_all(data_dir: Path, current_dir: Path | None = None, with_report: bool = False):
    """All seasons as one list. The two newest seasons are reconciled across sources.

    `data_dir/history` holds finished seasons. The live season is read from
    `current_dir`, by default `data_dir/current`. Tests pass a frozen copy.
    """
    hist = Path(data_dir) / "history"
    cur_dir = Path(current_dir) if current_dir else Path(data_dir) / "current"
    matches: list[Match] = []
    for s in HISTORY_SEASONS:
        matches += load_openfootball(hist / "openfootball" / f"{s}.json", s)

    last, last_report = reconcile({
        "fpl": load_fpl(hist / "fpl" / "fixtures_2025-26.csv", hist / "fpl" / "teams_2025-26.csv", "2025-26"),
        "openfootball": load_openfootball(hist / "openfootball" / "2025-26.json", "2025-26"),
        "mirror": load_mirror(hist / "fei_2025-26", "2025-26"),
    })
    sources = current_sources(cur_dir)
    cur, cur_report = reconcile(sources)
    cur_report["sources"] = {n: sum(m.played for m in ms) for n, ms in sources.items()}
    cur_report["primary"] = next(iter(sources))
    matches += last + cur
    if with_report:
        return matches, {"2025-26": last_report, CURRENT: cur_report}
    return matches


def _design(matches: list[Match], p: EloParams):
    pre, final = run_elo(matches, p)
    played = played_in_order(matches)
    d = np.array([rh - ra for rh, ra in pre])
    hg = np.array([m.home_goals for m in played])
    ag = np.array([m.away_goals for m in played])
    season = np.array([m.season for m in played])
    return d, hg, ag, season, final


def _window(target: str, window: int) -> list[str]:
    """The `window` seasons before `target`, never reaching into the burn-in seasons."""
    i = SEASONS.index(target)
    return [s for s in SEASONS[max(0, i - window):i] if s not in BURN_IN]


@dataclass
class Scored:
    params: EloParams
    tune_loss: float


ELO_GRID = list(itertools.product([4, 6, 8, 10, 12, 14, 16, 20, 24], [0, 30, 60], [1350, 1375, 1400, 1425, 1450]))


def tune_elo(matches: list[Match], grid=None) -> list[Scored]:
    """Score every Elo setting by 1X2 log loss on the tuning seasons only."""
    out: list[Scored] = []
    for k, hfa, prom in grid or ELO_GRID:
        p = EloParams(float(k), float(hfa), float(prom))
        d, hg, ag, season, _ = _design(matches, p)
        tune = ~np.isin(season, list(BURN_IN | TEST_SEASONS | {CURRENT}))
        gp = goals.fit(d[tune], hg[tune], ag[tune])
        ph, pd_, pa = goals.outcome_probs(*goals.rates(d[tune], gp))
        out.append(Scored(p, goals.log_loss_1x2(ph, pd_, pa, hg[tune], ag[tune])))
    out.sort(key=lambda s: s.tune_loss)
    return out


def walk_forward(matches: list[Match], p: EloParams, window: int, targets: list[str]) -> dict:
    """For each target season, fit goals on the `window` seasons before it, then score that season.

    Nothing from a target season is ever used to predict that season.
    """
    d, hg, ag, season, _ = _design(matches, p)
    ph_all, pd_all, pa_all, lam_all, keep = [], [], [], [], []
    for t in targets:
        fit_mask = np.isin(season, _window(t, window))
        tgt = season == t
        gp = goals.fit(d[fit_mask], hg[fit_mask], ag[fit_mask])
        lam_h, lam_a = goals.rates(d[tgt], gp)
        ph, pd_, pa = goals.outcome_probs(lam_h, lam_a)
        ph_all.append(ph)
        pd_all.append(pd_)
        pa_all.append(pa)
        lam_all.append(lam_h + lam_a)
        keep.append(np.flatnonzero(tgt))
    idx = np.concatenate(keep)
    ph, pd_, pa = np.concatenate(ph_all), np.concatenate(pd_all), np.concatenate(pa_all)
    h, a = hg[idx], ag[idx]
    actual = np.where(h > a, 0, np.where(h == a, 1, 2))
    pick = np.argmax(np.vstack([ph, pd_, pa]), axis=0)

    base_mask = np.isin(season, [s for s in SEASONS if s < min(targets) and s not in BURN_IN])
    base = [(hg[base_mask] > ag[base_mask]).mean(), (hg[base_mask] == ag[base_mask]).mean(),
            (hg[base_mask] < ag[base_mask]).mean()]
    n = len(idx)
    return {
        "window": window,
        "matches": n,
        "model_log_loss": goals.log_loss_1x2(ph, pd_, pa, h, a),
        "baseline_log_loss": goals.log_loss_1x2(np.full(n, base[0]), np.full(n, base[1]), np.full(n, base[2]), h, a),
        "accuracy": float((pick == actual).mean()),
        "baseline_accuracy": float((actual == 0).mean()),
        "pred_draw_rate": float(pd_.mean()),
        "actual_draw_rate": float((actual == 1).mean()),
        "pred_goals_per_match": float(np.concatenate(lam_all).mean()),
        "actual_goals_per_match": float((h + a).mean()),
    }


def truncate(matches: list[Match], season: str, cut_matches: int) -> list[Match]:
    """Rewind history to the day the `cut_matches`-th game of `season` was played."""
    played = sorted((m for m in matches if m.season == season and m.played), key=lambda m: m.date)
    cut_date = played[cut_matches - 1].date
    out = [m for m in matches if m.season < season]
    for m in matches:
        if m.season == season:
            known = m.played and m.date <= cut_date
            out.append(m if known else replace(m, home_goals=None, away_goals=None))
    return out


@dataclass
class Snapshot:
    season: str
    teams: list[str]
    table: list[dict]
    remaining: list[tuple[str, str]]
    ratings: dict[str, float]
    goal_params: goals.GoalParams
    played: int
    last_result_date: str
    played_results: list[tuple[str, str, int, int]] = field(default_factory=list)  # home, away, goals
    remaining_matches: list[Match] = field(default_factory=list)  # same order as `remaining`


def snapshot(matches: list[Match], p: EloParams, window: int, season: str = CURRENT) -> Snapshot:
    """Everything the simulator needs for `season`, using only results present in `matches`."""
    d, hg, ag, season_arr, final = _design(matches, p)
    fit_mask = np.isin(season_arr, _window(season, window) + [season])
    gp = goals.fit(d[fit_mask], hg[fit_mask], ag[fit_mask])
    cur = [m for m in matches if m.season == season]
    table = league_table(cur)
    teams = sorted(r["team"] for r in table)
    done = [m for m in cur if m.played]
    # Fixed order, so the same data gives the same seasons whichever source listed the fixtures first.
    todo = sorted((m for m in cur if not m.played), key=lambda m: (m.home, m.away))
    return Snapshot(
        season=season,
        teams=teams,
        table=table,
        remaining=[(m.home, m.away) for m in todo],
        ratings={t: final[t] for t in teams},
        goal_params=gp,
        played=len(done),
        last_result_date=max(m.date for m in done),
        played_results=[(m.home, m.away, m.home_goals, m.away_goals) for m in done],
        remaining_matches=todo,
    )


def run_sim(snap: Snapshot, n_sims: int = 10_000, sigma: float = 35.0, seed: int = 2026,
            forced: dict[tuple[str, str], tuple[int, int]] | None = None) -> SimResult:
    """Simulate the rest of the season. `forced` fixes the score of chosen (home, away) fixtures."""
    by_team = {r["team"]: r for r in snap.table}
    forced_idx = {snap.remaining.index(k): v for k, v in (forced or {}).items()}
    return simulate(
        snap.teams,
        [by_team[t]["pts"] for t in snap.teams],
        [by_team[t]["gd"] for t in snap.teams],
        [by_team[t]["gf"] for t in snap.teams],
        snap.remaining,
        snap.ratings,
        snap.goal_params,
        n_sims=n_sims,
        sigma=sigma,
        seed=seed,
        played=snap.played_results,
        forced=forced_idx,
    )


def season_calibration(matches: list[Match], p: EloParams, window: int, sigmas: list[float],
                       targets: list[str], cut_matches: int = 100, n_sims: int = 2_000) -> list[dict]:
    """Rewind each past season to matchday 10, simulate the rest, compare with what really happened.

    Answers: when the model says 70 percent, does it happen about 70 percent of the time?
    Scored with the Brier score (0 is perfect, lower is better) on three outcomes per club:
    title, top four, relegation.
    """
    cases = []
    for s in targets:
        full = [m for m in matches if m.season == s]
        final_pos = {r["team"]: i + 1 for i, r in enumerate(league_table(full))}
        cases.append((snapshot(truncate(matches, s, cut_matches), p, window, season=s), final_pos))

    out = []
    for sigma in sigmas:
        pred = {"title": [], "top4": [], "relegated": []}
        act = {"title": [], "top4": [], "relegated": []}
        for i, (snap, final_pos) in enumerate(cases):
            res = run_sim(snap, n_sims=n_sims, sigma=sigma, seed=500 + i)
            pos = np.array([final_pos[t] for t in snap.teams])
            n = len(snap.teams)
            pred["title"].append(res.prob_top(1))
            act["title"].append(pos == 1)
            pred["top4"].append(res.prob_top(4))
            act["top4"].append(pos <= 4)
            pred["relegated"].append(res.prob_bottom(3))
            act["relegated"].append(pos > n - 3)
        row = {"sigma": sigma, "club_seasons": sum(len(c[0].teams) for c in cases)}
        allp, alla = [], []
        for k in pred:
            pp, aa = np.concatenate(pred[k]), np.concatenate(act[k]).astype(float)
            row[f"brier_{k}"] = float(np.mean((pp - aa) ** 2))
            allp.append(pp)
            alla.append(aa)
        pp, aa = np.concatenate(allp), np.concatenate(alla)
        row["brier_all"] = float(np.mean((pp - aa) ** 2))
        bins = [(0.0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.0001)]
        row["reliability"] = [
            {"bin": f"{lo:.0%} to {min(hi, 1):.0%}", "n": int(m.sum()),
             "predicted": float(pp[m].mean()) if m.any() else None,
             "happened": float(aa[m].mean()) if m.any() else None}
            for lo, hi in bins
            for m in [(pp >= lo) & (pp < hi)]
        ]
        out.append(row)
    return out
