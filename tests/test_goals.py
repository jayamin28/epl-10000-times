import math

import numpy as np
import pytest

from eplsim.goals import GoalParams, fit, log_loss_1x2, outcome_probs, rates, score_matrix

GP = GoalParams(alpha=0.24, beta=0.68, gamma=0.16)


def test_score_matrix_covers_nearly_all_probability():
    m = score_matrix(1.6, 1.2)
    assert m.shape == (11, 11) and 0.9999 < m.sum() <= 1.0


def test_outcome_probabilities_sum_to_one():
    ph, pd_, pa = outcome_probs([0.3, 1.5, 4.0], [3.5, 1.5, 0.2])
    assert ph + pd_ + pa == pytest.approx(np.ones(3))


def test_outcome_probs_agree_with_score_matrix():
    m = score_matrix(1.7, 1.1)
    ph, pd_, pa = outcome_probs(1.7, 1.1)
    total = m.sum()
    assert ph[0] == pytest.approx(np.tril(m, -1).sum() / total)
    assert pd_[0] == pytest.approx(np.trace(m) / total)
    assert pa[0] == pytest.approx(np.triu(m, 1).sum() / total)


def test_bigger_rating_gap_means_better_home_chances():
    d = np.array([-300, -100, 0, 100, 300])
    ph, _, pa = outcome_probs(*rates(d, GP))
    assert np.all(np.diff(ph) > 0) and np.all(np.diff(pa) < 0)


def test_home_advantage_exists_between_equal_teams():
    lam_h, lam_a = rates(0, GP)
    assert lam_h > lam_a
    ph, _, pa = outcome_probs(lam_h, lam_a)
    assert ph[0] > pa[0]


def test_fit_recovers_known_parameters():
    """Generate fake matches from known settings, then check the fit finds those settings again."""
    rng = np.random.default_rng(7)
    true = GoalParams(alpha=0.20, beta=0.70, gamma=0.25)
    d = rng.normal(0, 120, size=40_000)
    lam_h, lam_a = rates(d, true)
    got = fit(d, rng.poisson(lam_h), rng.poisson(lam_a))
    assert got.alpha == pytest.approx(true.alpha, abs=0.02)
    assert got.beta == pytest.approx(true.beta, abs=0.03)
    assert got.gamma == pytest.approx(true.gamma, abs=0.02)


def test_log_loss_of_a_coin_flip_three_ways():
    third = np.full(4, 1 / 3)
    assert log_loss_1x2(third, third, third, [1, 0, 2, 2], [0, 0, 2, 3]) == pytest.approx(math.log(3))


def test_log_loss_rewards_being_right():
    hg, ag = np.array([2, 0]), np.array([0, 1])
    good = log_loss_1x2(np.array([0.8, 0.1]), np.array([0.1, 0.1]), np.array([0.1, 0.8]), hg, ag)
    bad = log_loss_1x2(np.array([0.1, 0.8]), np.array([0.1, 0.1]), np.array([0.8, 0.1]), hg, ag)
    assert good < bad
