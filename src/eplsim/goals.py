"""Goals model: Elo gap in, expected goals and scoreline probabilities out.

    log(home goals rate) = alpha + gamma + beta * d / 400
    log(away goals rate) = alpha         - beta * d / 400

d is home Elo minus away Elo (no home bonus added, gamma carries home
advantage in goals). Three numbers, fitted by maximum likelihood on real
matches. Scorelines come from two independent Poisson draws.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson


@dataclass(frozen=True)
class GoalParams:
    alpha: float
    beta: float
    gamma: float


def rates(d, p: GoalParams):
    d = np.asarray(d, dtype=float)
    return np.exp(p.alpha + p.gamma + p.beta * d / 400.0), np.exp(p.alpha - p.beta * d / 400.0)


def fit(d, home_goals, away_goals) -> GoalParams:
    d = np.asarray(d, dtype=float) / 400.0
    hg = np.asarray(home_goals, dtype=float)
    ag = np.asarray(away_goals, dtype=float)

    def nll(x):
        a, b, g = x
        lh = a + g + b * d
        la = a - b * d
        return float(np.sum(np.exp(lh) - hg * lh + np.exp(la) - ag * la))

    res = minimize(nll, x0=np.array([0.2, 0.5, 0.2]), method="BFGS")
    if not res.success and res.status != 2:  # status 2: precision loss at an already converged point
        raise RuntimeError(f"Poisson fit did not converge: {res.message}")
    return GoalParams(*map(float, res.x))


def score_matrix(lam_home: float, lam_away: float, max_goals: int = 10) -> np.ndarray:
    """P(home scores i, away scores j) for i, j in 0..max_goals."""
    g = np.arange(max_goals + 1)
    return np.outer(poisson.pmf(g, lam_home), poisson.pmf(g, lam_away))


def outcome_probs(lam_home, lam_away, max_goals: int = 10):
    """Vectorised home win, draw, away win probabilities, renormalised after truncation."""
    lam_home = np.atleast_1d(np.asarray(lam_home, dtype=float))
    lam_away = np.atleast_1d(np.asarray(lam_away, dtype=float))
    g = np.arange(max_goals + 1)
    ph = poisson.pmf(g[None, :], lam_home[:, None])
    pa = poisson.pmf(g[None, :], lam_away[:, None])
    joint = ph[:, :, None] * pa[:, None, :]
    home = np.tril(np.ones((max_goals + 1, max_goals + 1)), -1)
    draw = np.eye(max_goals + 1)
    p_home = (joint * home).sum(axis=(1, 2))
    p_draw = (joint * draw).sum(axis=(1, 2))
    p_away = (joint * home.T).sum(axis=(1, 2))
    total = p_home + p_draw + p_away
    return p_home / total, p_draw / total, p_away / total


def log_loss_1x2(p_home, p_draw, p_away, home_goals, away_goals) -> float:
    hg = np.asarray(home_goals)
    ag = np.asarray(away_goals)
    p = np.where(hg > ag, p_home, np.where(hg == ag, p_draw, p_away))
    return float(-np.mean(np.log(np.clip(p, 1e-12, 1.0))))
