"""Monte Carlo season simulator.

Real results are locked in. Every remaining fixture is played N times.
Per simulated season:
  1. Each club's rating is nudged by a random amount (sd = sigma Elo points),
     fixed for that season. Ratings are estimates, not facts.
  2. Each remaining match gets expected goals from the goals model.
  3. Goals are drawn from two Poisson distributions.
  4. The final table is ranked the way Premier League Rule C.17 ranks it:
     points, goal difference, goals scored, then points in the matches
     between the clubs concerned, then away goals in those matches.
     Clubs still level after all that would meet in a play-off. The play-off
     is not simulated: they are split at random and counted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .goals import GoalParams, rates


@dataclass
class SimResult:
    teams: list[str]
    position: np.ndarray  # (n_sims, n_teams) finishing place, 1 = champion
    points: np.ndarray  # (n_sims, n_teams) final points
    goal_diff: np.ndarray  # (n_sims, n_teams)
    full_ties: dict[str, int]  # seasons where a key boundary was level on pts, GD and GF
    n_sims: int
    seed: int
    goals_for: np.ndarray | None = None  # (n_sims, n_teams)
    fixtures: list[tuple[str, str]] = field(default_factory=list)  # the simulated matches, in column order
    home_goals: np.ndarray | None = None  # (n_sims, n_fixtures) simulated scores
    away_goals: np.ndarray | None = None
    tiebreak: dict[str, int] = field(default_factory=dict)  # how clubs level on pts, GD and GF were split

    @property
    def champion(self) -> np.ndarray:
        """Index of the title winner in each simulated season."""
        return np.argmin(self.position, axis=1)

    def position_matrix(self) -> np.ndarray:
        """P(team finishes in place k). Rows = teams, cols = places 1..n."""
        n = len(self.teams)
        out = np.zeros((n, n))
        for place in range(1, n + 1):
            out[:, place - 1] = (self.position == place).mean(axis=0)
        return out

    def prob_top(self, k: int) -> np.ndarray:
        return (self.position <= k).mean(axis=0)

    def prob_bottom(self, k: int) -> np.ndarray:
        return (self.position > len(self.teams) - k).mean(axis=0)

    def points_of_place(self, place: int) -> np.ndarray:
        """Points total of whoever finished in `place`, per simulated season."""
        order = np.argsort(self.position, axis=1)
        return np.take_along_axis(self.points, order, axis=1)[:, place - 1]


def head_to_head_order(group: list[int], results: list[tuple[int, int, int, int]]) -> tuple[list[int], str]:
    """Rank clubs that are level on points, goal difference and goals scored.

    `group` is in fallback order (used only if nothing else separates them).
    `results` are (home, away, home_goals, away_goals) for every match of the
    season. Only matches between two clubs of the group count.

    Returns the new order and what decided it: "points", "away_goals" or "level".
    """
    members = set(group)
    pts = dict.fromkeys(group, 0)
    away_goals = dict.fromkeys(group, 0)
    for h, a, hg, ag in results:
        if h in members and a in members:
            if hg > ag:
                pts[h] += 3
            elif ag > hg:
                pts[a] += 3
            else:
                pts[h] += 1
                pts[a] += 1
            away_goals[a] += ag
    ranked = sorted(group, key=lambda t: (-pts[t], -away_goals[t]))  # stable: fallback order survives ties
    if len({pts[t] for t in group}) == len(group):
        how = "points"
    elif len({(pts[t], away_goals[t]) for t in group}) == len(group):
        how = "away_goals"
    else:
        how = "level"
    return ranked, how


def simulate(
    teams: list[str],
    base_points,
    base_gd,
    base_gf,
    fixtures: list[tuple[str, str]],
    ratings: dict[str, float],
    goal_params: GoalParams,
    n_sims: int = 10_000,
    sigma: float = 35.0,
    seed: int = 2026,
    played: list[tuple[str, str, int, int]] | None = None,
    forced: dict[int, tuple[int, int]] | None = None,
) -> SimResult:
    """Simulate the remaining `fixtures` n_sims times.

    `played` are the real results so far (home, away, home_goals, away_goals).
    They are only needed for the head-to-head tie-break.
    `forced` fixes the score of chosen fixtures (by column index) in every
    season, leaving all other randomness untouched. That makes "what if they
    win on Sunday" comparisons clean: the two runs differ in that match only.
    """
    rng = np.random.default_rng(seed)
    idx = {t: i for i, t in enumerate(teams)}
    n_teams, n_fix = len(teams), len(fixtures)
    home = np.array([idx[h] for h, _ in fixtures], dtype=int)
    away = np.array([idx[a] for _, a in fixtures], dtype=int)
    base_r = np.array([ratings[t] for t in teams], dtype=float)

    r = base_r[None, :] + rng.normal(0.0, sigma, size=(n_sims, n_teams))
    lam_h, lam_a = rates(r[:, home] - r[:, away], goal_params)
    hg = rng.poisson(lam_h)
    ag = rng.poisson(lam_a)
    for f, (fh, fa) in (forced or {}).items():
        hg[:, f] = fh
        ag[:, f] = fa

    home_ind = np.zeros((n_fix, n_teams), dtype=np.float64)
    away_ind = np.zeros((n_fix, n_teams), dtype=np.float64)
    home_ind[np.arange(n_fix), home] = 1.0
    away_ind[np.arange(n_fix), away] = 1.0

    draw = (hg == ag).astype(np.float64)
    home_pts = 3.0 * (hg > ag) + draw
    away_pts = 3.0 * (ag > hg) + draw
    hgf, agf = hg.astype(np.float64), ag.astype(np.float64)

    points = np.asarray(base_points, dtype=np.float64)[None, :] + home_pts @ home_ind + away_pts @ away_ind
    gf = np.asarray(base_gf, dtype=np.float64)[None, :] + hgf @ home_ind + agf @ away_ind
    gd = np.asarray(base_gd, dtype=np.float64)[None, :] + (hgf - agf) @ home_ind + (agf - hgf) @ away_ind

    points = np.rint(points).astype(np.int64)
    gf = np.rint(gf).astype(np.int64)
    gd = np.rint(gd).astype(np.int64)

    # One sortable integer per club: points, then goal difference, then goals for.
    key = points * 10_000_000 + (gd + 1_000) * 1_000 + gf
    jitter = rng.random(size=key.shape)  # fallback order for clubs nothing else separates
    order = np.argsort(-(key + jitter), axis=1)

    sorted_key = np.take_along_axis(key, order, axis=1)
    boundaries = {"title (1st v 2nd)": 1, "top 4 (4th v 5th)": 4, "top 5 (5th v 6th)": 5}
    boundaries["relegation (17th v 18th)"] = n_teams - 3
    full_ties = {
        name: int((sorted_key[:, b - 1] == sorted_key[:, b]).sum())
        for name, b in boundaries.items()
        if 0 < b < n_teams
    }

    # Rule C.17: clubs level on points, goal difference and goals scored go to head to head.
    level = sorted_key[:, :-1] == sorted_key[:, 1:]
    tied_sims = np.flatnonzero(level.any(axis=1))
    tiebreak = {"seasons_with_clubs_fully_level": int(len(tied_sims)), "groups": 0,
                "split_by_head_to_head_points": 0, "split_by_head_to_head_away_goals": 0, "left_to_a_play_off": 0}
    real = [(idx[h], idx[a], int(x), int(y)) for h, a, x, y in (played or [])]
    for s in tied_sims:
        season_results = None
        j = 0
        while j < n_teams - 1:
            if not level[s, j]:
                j += 1
                continue
            k = j
            while k < n_teams - 1 and level[s, k]:
                k += 1
            if season_results is None:
                season_results = real + list(zip(home.tolist(), away.tolist(), hg[s].tolist(), ag[s].tolist()))
            ranked, how = head_to_head_order(order[s, j:k + 1].tolist(), season_results)
            order[s, j:k + 1] = ranked
            tiebreak["groups"] += 1
            tiebreak[{"points": "split_by_head_to_head_points", "away_goals": "split_by_head_to_head_away_goals",
                      "level": "left_to_a_play_off"}[how]] += 1
            j = k + 1

    position = np.empty_like(order)
    rows = np.arange(n_sims)[:, None]
    position[rows, order] = np.arange(1, n_teams + 1)[None, :]

    return SimResult(teams, position, points, gd, full_ties, n_sims, seed, goals_for=gf, fixtures=list(fixtures),
                     home_goals=hg.astype(np.int16), away_goals=ag.astype(np.int16), tiebreak=tiebreak)
