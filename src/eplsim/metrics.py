"""Every number the page and the posts publish, computed from one set of simulated seasons.

Each function returns plain Python (lists, dicts, floats) so it can go
straight into JSON. Nothing here draws random numbers: same seasons in, same
numbers out.
"""

from __future__ import annotations

import math

import numpy as np

from . import goals
from .pipeline import Snapshot
from .simulate import SimResult

LOCKED_IN, IN_THE_RACE, ALMOST_GONE = 0.85, 0.20, 0.02
MOVER_THRESHOLD = 0.03  # smaller changes are inside the dice noise
RACES = ("title", "top4", "top5", "relegated")
PLACES = {"title": 1, "top4": 4, "top5": 5, "safety": 17}


def race_probs(res: SimResult) -> dict[str, np.ndarray]:
    return {"title": res.prob_top(1), "top4": res.prob_top(4), "top5": res.prob_top(5), "relegated": res.prob_bottom(3)}


def _rank(values, descending: bool = True) -> list[int]:
    """1 = best. Ties share the lower rank number."""
    v = np.asarray(values, dtype=float)
    return [int(((v > x) if descending else (v < x)).sum()) + 1 for x in v]


def headline(snap: Snapshot, res: SimResult) -> list[dict]:
    """One row per club, strongest expected finish first."""
    probs = race_probs(res)
    by_team = {r["team"]: r for r in snap.table}
    table_pos = {r["team"]: i + 1 for i, r in enumerate(snap.table)}
    pm = res.position_matrix()
    rows = []
    for i, t in enumerate(res.teams):
        pts = res.points[:, i]
        rows.append({
            "team": t, "table_pos": table_pos[t], "played": by_team[t]["p"], "pts_now": by_team[t]["pts"],
            "gd_now": by_team[t]["gd"], "elo": float(snap.ratings[t]),
            "exp_pts": float(pts.mean()), "pts_p10": int(np.percentile(pts, 10)), "pts_p90": int(np.percentile(pts, 90)),
            "most_likely_place": int(np.argmax(pm[i]) + 1),
            **{race: float(probs[race][i]) for race in RACES},
        })
    rows.sort(key=lambda r: (-r["exp_pts"], r["team"]))
    return rows


def tiers(res: SimResult) -> dict[str, dict[str, list[str]]]:
    """For each race, sort clubs into four groups by how likely the outcome is."""
    out = {}
    for race, p in race_probs(res).items():
        groups: dict[str, list[tuple[float, str]]] = {"locked_in": [], "in_the_race": [], "on_the_edge": [], "almost_gone": []}
        for prob, team in zip(p, res.teams):
            if prob >= LOCKED_IN:
                groups["locked_in"].append((prob, team))
            elif prob >= IN_THE_RACE:
                groups["in_the_race"].append((prob, team))
            elif prob >= ALMOST_GONE:
                groups["on_the_edge"].append((prob, team))
            else:
                groups["almost_gone"].append((prob, team))
        out[race] = {k: [t for _, t in sorted(v, key=lambda x: (-x[0], x[1]))] for k, v in groups.items()}
    return out


def table_lies(snap: Snapshot, res: SimResult) -> list[dict]:
    """Where the table and the model disagree. Positive gap: the model rates the club higher than the table does."""
    table_pos = {r["team"]: i + 1 for i, r in enumerate(snap.table)}
    exp = res.points.mean(axis=0)
    order = sorted(range(len(res.teams)), key=lambda i: (-exp[i], res.teams[i]))
    model_pos = {res.teams[i]: k + 1 for k, i in enumerate(order)}
    rows = [{"team": t, "table_pos": table_pos[t], "model_pos": model_pos[t], "gap": table_pos[t] - model_pos[t]}
            for t in res.teams]
    rows.sort(key=lambda r: (-abs(r["gap"]), r["team"]))
    return rows


def _dist(arr: np.ndarray) -> dict:
    vals, counts = np.unique(arr, return_counts=True)
    return {"mode": int(vals[np.argmax(counts)]), "median": int(np.median(arr)),
            "p10": int(np.percentile(arr, 10)), "p90": int(np.percentile(arr, 90)),
            "min": int(vals[0]), "share": [float(c) for c in np.bincount(arr - vals[0]) / len(arr)]}


def magic_numbers(res: SimResult) -> dict[str, dict]:
    """Points total of the club finishing 1st, 4th, 5th and 17th. `share[i]` is the share of seasons on `min + i` points."""
    n = len(res.teams)
    return {name: _dist(res.points_of_place(place)) for name, place in PLACES.items() if place <= n}


def points_shape(res: SimResult) -> dict[str, dict]:
    """Each club's spread of final points totals."""
    return {t: _dist(res.points[:, i]) for i, t in enumerate(res.teams)}


def home_edge_in_rating_points(gp: goals.GoalParams) -> float:
    """Home advantage expressed as a rating gap: playing at home is worth this many Elo points."""
    return 200.0 * gp.gamma / gp.beta


def run_in(snap: Snapshot) -> list[dict]:
    """How hard each club's remaining fixtures are. Hardest first.

    Difficulty is the average rating of the opponents still to play, with each
    opponent counted as stronger when the game is away and weaker when it is at home.
    """
    edge = home_edge_in_rating_points(snap.goal_params)
    rows = []
    for t in snap.teams:
        adj, raw, home_games = [], [], 0
        for h, a in snap.remaining:
            if h == t:
                adj.append(snap.ratings[a] - edge)
                raw.append(snap.ratings[a])
                home_games += 1
            elif a == t:
                adj.append(snap.ratings[h] + edge)
                raw.append(snap.ratings[h])
        rows.append({"team": t, "games": len(adj), "home_games": home_games,
                     "avg_opponent_rating": float(np.mean(raw)) if raw else None,
                     "difficulty": float(np.mean(adj)) if adj else None})
    rows.sort(key=lambda r: (-(r["difficulty"] if r["difficulty"] is not None else -math.inf), r["team"]))
    for i, r in enumerate(rows):
        r["rank"] = i + 1
    return rows


def upcoming(snap: Snapshot, gameweek: int | None = None) -> dict:
    """Fixtures of the next gameweek with expected goals and win, draw, loss odds."""
    todo = snap.remaining_matches
    if not todo:
        return {"gameweek": None, "fixtures": []}
    gw = gameweek if gameweek is not None else min(m.gameweek for m in todo if m.gameweek > 0)
    fixtures = sorted((m for m in todo if m.gameweek == gw), key=lambda m: (m.date, m.home))
    d = np.array([snap.ratings[m.home] - snap.ratings[m.away] for m in fixtures])
    lam_h, lam_a = goals.rates(d, snap.goal_params)
    ph, pd_, pa = goals.outcome_probs(lam_h, lam_a)
    return {"gameweek": gw, "fixtures": [
        {"date": m.date, "home": m.home, "away": m.away, "home_xg": float(lam_h[i]), "away_xg": float(lam_a[i]),
         "home_win": float(ph[i]), "draw": float(pd_[i]), "away_win": float(pa[i])}
        for i, m in enumerate(fixtures)]}


def match_swings(snap: Snapshot, sim, fixtures: list[dict], min_swing: float = 0.005) -> dict:
    """What each of the next fixtures is worth.

    `sim(forced)` reruns the season with one result fixed and everything else
    identical, so the difference between two runs is that match and nothing
    else. Fixed scores are one-goal wins (1-0, 0-1) and a 1-1 draw.
    """
    outcomes = {"home_win": (1, 0), "draw": (1, 1), "away_win": (0, 1)}
    per_fixture, best = [], None
    for fx in fixtures:
        key = (fx["home"], fx["away"])
        probs = {name: race_probs(sim({key: score})) for name, score in outcomes.items()}
        entry = {"home": fx["home"], "away": fx["away"], "date": fx["date"], "clubs": {}, "others": []}
        for side, win, lose in (("home", "home_win", "away_win"), ("away", "away_win", "home_win")):
            team = fx[side]
            i = snap.teams.index(team)
            entry["clubs"][team] = {}
            for race in RACES:
                w, d, l = (float(probs[o][race][i]) for o in (win, "draw", lose))
                entry["clubs"][team][race] = {"win": w, "draw": d, "lose": l, "swing": w - l}
                if best is None or abs(w - l) > abs(best["swing"]):
                    best = {"team": team, "opponent": fx["away" if side == "home" else "home"], "venue": side,
                            "date": fx["date"], "race": race, "win": w, "draw": d, "lose": l, "swing": w - l}
        for i, team in enumerate(snap.teams):
            if team in key:
                continue
            for race in RACES:
                change = float(probs["home_win"][race][i] - probs["away_win"][race][i])
                if abs(change) >= min_swing:
                    entry["others"].append({"team": team, "race": race, "if_home_win": float(probs["home_win"][race][i]),
                                            "if_away_win": float(probs["away_win"][race][i]), "change": change})
        entry["others"].sort(key=lambda r: -abs(r["change"]))
        entry["others"] = entry["others"][:5]
        per_fixture.append(entry)
    return {"fixtures": per_fixture, "biggest": best}


def universe(res: SimResult, n: int) -> dict:
    """Season number n, counted from 1: who won it, who finished in the top four, who went down."""
    if not 1 <= n <= res.n_sims:
        raise ValueError(f"Season number must be between 1 and {res.n_sims}")
    order = np.argsort(res.position[n - 1])
    names = [res.teams[i] for i in order]
    return {"season": n, "champion": names[0], "top_four": names[:4], "relegated": names[-3:]}


def universes(res: SimResult) -> dict:
    """Every season's champion and bottom three, packed as letters (A = first club in `teams`)."""
    order = np.argsort(res.position, axis=1)
    letters = np.array([chr(65 + i) for i in range(len(res.teams))])
    return {"teams": list(res.teams), "champion": "".join(letters[order[:, 0]]),
            "relegated": "".join("".join(row) for row in letters[order[:, -3:]])}


def wildest_season(snap: Snapshot, res: SimResult) -> dict:
    """The season won by the least likely champion. Ties go to the lowest season number."""
    title = res.prob_top(1)
    champs = res.champion
    s = int(np.argmin(title[champs]))
    c = int(champs[s])
    team = res.teams[c]
    order = np.argsort(res.position[s])
    table = [{"pos": k + 1, "team": res.teams[i], "pts": int(res.points[s, i]), "gd": int(res.goal_diff[s, i])}
             for k, i in enumerate(order)]
    won = drawn = lost = 0
    best = None
    for f, (h, a) in enumerate(res.fixtures):
        if team not in (h, a):
            continue
        hg, ag = int(res.home_goals[s, f]), int(res.away_goals[s, f])
        mine, theirs, opp = (hg, ag, a) if h == team else (ag, hg, h)
        won += mine > theirs
        drawn += mine == theirs
        lost += mine < theirs
        cand = (mine - theirs, snap.ratings[opp])
        if mine > theirs and (best is None or cand > best[0]):
            best = (cand, {"opponent": opp, "venue": "home" if h == team else "away", "score": f"{mine}-{theirs}"})
    table_pos = {r["team"]: i + 1 for i, r in enumerate(snap.table)}
    return {"season": s + 1, "champion": team, "title_seasons": int((champs == c).sum()), "title_share": float(title[c]),
            "champion_pts": int(res.points[s, c]), "champion_table_pos_now": table_pos[team],
            "champion_pts_now": next(r["pts"] for r in snap.table if r["team"] == team),
            "rest_of_season": {"won": int(won), "drawn": int(drawn), "lost": int(lost)},
            "best_win": best[1] if best else None, "margin": int(res.points[s, c] - res.points[s, order[1]]),
            "table": table}


def convergence(res: SimResult, top: int = 5) -> dict:
    """Title share after the first 10, 100, ... seasons, for the clubs that end up most likely."""
    steps = [n for n in (10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000) if n <= res.n_sims]
    if not steps or steps[-1] != res.n_sims:
        steps.append(res.n_sims)
    champs = res.champion
    final = res.prob_top(1)
    leaders = sorted(range(len(res.teams)), key=lambda i: (-final[i], res.teams[i]))[:top]
    return {"seasons": steps, "share": {res.teams[i]: [float((champs[:n] == i).mean()) for n in steps] for i in leaders}}


def duel(snap: Snapshot, res: SimResult, a: str, b: str) -> dict:
    """Ten model numbers for two clubs, side by side."""
    by_team = {r["team"]: r for r in snap.table}
    probs = race_probs(res)
    difficulty = {r["team"]: r["difficulty"] for r in run_in(snap)}
    ia, ib = res.teams.index(a), res.teams.index(b)
    exp = res.points.mean(axis=0)

    def per_game(key: str) -> dict[str, float]:
        return {t: by_team[t][key] / max(by_team[t]["p"], 1) for t in res.teams}

    series = [
        ("Rating", {t: snap.ratings[t] for t in res.teams}, True, 0),
        ("Points now", {t: by_team[t]["pts"] for t in res.teams}, True, 0),
        ("Goal difference now", {t: by_team[t]["gd"] for t in res.teams}, True, 0),
        ("Goals for per game", per_game("gf"), True, 2),
        ("Goals against per game", per_game("ga"), False, 2),
        ("Expected final points", {t: float(exp[i]) for i, t in enumerate(res.teams)}, True, 1),
        ("Title", {t: float(probs["title"][i]) for i, t in enumerate(res.teams)}, True, 3),
        ("Top four", {t: float(probs["top4"][i]) for i, t in enumerate(res.teams)}, True, 3),
        ("Run-in difficulty", difficulty, False, 0),
    ]
    rows = []
    for label, values, higher_is_better, digits in series:
        ranks = dict(zip(res.teams, _rank([values[t] for t in res.teams], descending=higher_is_better)))
        va, vb = values[a], values[b]
        better = None if va == vb else (a if (va > vb) == higher_is_better else b)
        rows.append({"label": label, "a": round(float(va), digits), "b": round(float(vb), digits),
                     "a_rank": ranks[a], "b_rank": ranks[b], "higher_is_better": higher_is_better, "better": better})
    above = float((res.position[:, ia] < res.position[:, ib]).mean())
    rows.append({"label": "Finishes above the other", "a": round(above, 3), "b": round(1 - above, 3),
                 "a_rank": None, "b_rank": None, "higher_is_better": True,
                 "better": None if above == 0.5 else (a if above > 0.5 else b)})
    return {"a": a, "b": b, "rows": rows}


def movers(before: dict[str, dict], after: dict[str, dict], threshold: float = MOVER_THRESHOLD) -> list[dict]:
    """Clubs whose odds moved by at least `threshold` between two runs. Biggest first."""
    out = []
    for team, now in after.items():
        if team not in before:
            continue
        for race in RACES:
            if race in now and race in before[team]:
                change = now[race] - before[team][race]
                if abs(change) >= threshold - 1e-12:
                    out.append({"team": team, "race": race, "before": float(before[team][race]),
                                "after": float(now[race]), "change": float(change)})
    out.sort(key=lambda r: (-abs(r["change"]), r["team"], r["race"]))
    return out


def report_card(forecasts: list[dict], results: dict[tuple[str, str], tuple[int, int]],
                base_rates: tuple[float, float, float]) -> dict:
    """Score match forecasts that were logged before kick-off against what happened."""
    scored = [(f, results[(f["home"], f["away"])]) for f in forecasts if (f["home"], f["away"]) in results]
    if not scored:
        return {"matches": 0}
    ll = base_ll = brier = right = 0.0
    for f, (hg, ag) in scored:
        p = (f["p_home"], f["p_draw"], f["p_away"])
        k = 0 if hg > ag else 1 if hg == ag else 2
        ll -= math.log(max(p[k], 1e-12))
        base_ll -= math.log(base_rates[k])
        brier += sum((p[j] - (1.0 if j == k else 0.0)) ** 2 for j in range(3))
        right += p[k] == max(p)
    n = len(scored)
    return {"matches": n, "log_loss": ll / n, "baseline_log_loss": base_ll / n, "brier": brier / n, "accuracy": right / n}


def upsets(forecasts: list[dict], results: dict[tuple[str, str], tuple[int, int]], gameweek: int | None = None,
           top: int = 3) -> dict:
    """The played matches whose result the model rated least likely before kick-off.

    Only odds logged before kick-off count, so this is empty until a logged gameweek has been played.
    `gameweek` picks one gameweek; by default the latest logged gameweek with a result.
    """
    played = [f for f in forecasts if (f["home"], f["away"]) in results]
    if gameweek is None and played:
        gameweek = max(int(f["gameweek"]) for f in played)
    rows = []
    for f in played:
        if int(f["gameweek"]) != gameweek:
            continue
        hg, ag = results[(f["home"], f["away"])]
        outcome = "home" if hg > ag else "draw" if hg == ag else "away"
        p = {"home": f["p_home"], "draw": f["p_draw"], "away": f["p_away"]}[outcome]
        rows.append({"home": f["home"], "away": f["away"], "score": f"{hg}-{ag}", "outcome": outcome,
                     "p_before": float(p), "p_home": float(f["p_home"]), "p_draw": float(f["p_draw"]),
                     "p_away": float(f["p_away"])})
    rows.sort(key=lambda r: (r["p_before"], r["home"]))
    return {"gameweek": gameweek, "rows": rows[:top]}


def telemetry(res: SimResult, seconds: float | None = None, reruns: int = 0) -> dict:
    """What the engine did on this run, in countable things.

    `reruns` is the number of extra full runs made to price individual
    matches (three per fixture of the next gameweek).
    """
    hg, ag = res.home_goals.astype(np.int64), res.away_goals.astype(np.int64)
    matches = int(hg.size)
    big_s, big_f = np.unravel_index(np.argmax(np.abs(hg - ag)), hg.shape) if matches else (None, None)
    scores, counts = np.unique(np.minimum(hg, 9) * 10 + np.minimum(ag, 9), return_counts=True) if matches else ([], [])
    common = int(scores[np.argmax(counts)]) if matches else None
    champs = res.champion
    bottom = np.sort(np.argsort(res.position, axis=1)[:, -3:], axis=1)
    return {
        "seasons": res.n_sims,
        "matches": matches,
        "goals": int(hg.sum() + ag.sum()),
        "goals_per_match": float((hg.sum() + ag.sum()) / matches) if matches else None,
        "random_draws": res.random_draws,
        "distinct_champions": int(len(np.unique(champs))),
        "distinct_relegated_trios": int(len({tuple(r) for r in bottom.tolist()})),
        "most_common_score": f"{common // 10}-{common % 10}" if common is not None else None,
        "most_common_score_share": float(counts.max() / matches) if matches else None,
        "biggest_win": None if big_f is None else {
            "home": res.fixtures[big_f][0], "away": res.fixtures[big_f][1],
            "score": f"{int(hg[big_s, big_f])}-{int(ag[big_s, big_f])}", "season": int(big_s) + 1},
        "max_points": {"points": int(res.points.max()), "team": res.teams[int(np.unravel_index(np.argmax(res.points), res.points.shape)[1])]},
        "min_points": {"points": int(res.points.min()), "team": res.teams[int(np.unravel_index(np.argmin(res.points), res.points.shape)[1])]},
        "seconds": None if seconds is None else round(float(seconds), 2),
        "full_runs": 1 + reruns,
        "seasons_incl_match_pricing": res.n_sims * (1 + reruns),
    }
