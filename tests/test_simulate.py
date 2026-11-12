import time

import numpy as np
import pytest

from eplsim import config, pipeline
from eplsim.goals import GoalParams
from eplsim.simulate import simulate

GP = GoalParams(alpha=0.24, beta=0.68, gamma=0.16)
FOUR = ["A", "B", "C", "D"]
DOUBLE_ROUND_ROBIN = [(h, a) for h in FOUR for a in FOUR if h != a]


def _fresh(ratings, n_sims=4_000, sigma=0.0, seed=1):
    return simulate(FOUR, [0] * 4, [0] * 4, [0] * 4, DOUBLE_ROUND_ROBIN, ratings, GP, n_sims=n_sims, sigma=sigma, seed=seed)


def test_same_seed_gives_identical_seasons(snap):
    a = pipeline.run_sim(snap, n_sims=2_000, seed=42)
    b = pipeline.run_sim(snap, n_sims=2_000, seed=42)
    assert np.array_equal(a.position, b.position) and np.array_equal(a.points, b.points)


def test_different_seed_gives_different_seasons(snap):
    a = pipeline.run_sim(snap, n_sims=2_000, seed=42)
    b = pipeline.run_sim(snap, n_sims=2_000, seed=43)
    assert not np.array_equal(a.position, b.position)


def test_every_season_is_a_full_ranking(result):
    n = len(result.teams)
    assert np.array_equal(np.sort(result.position, axis=1), np.tile(np.arange(1, n + 1), (result.n_sims, 1)))


def test_position_matrix_rows_and_columns_sum_to_one(result):
    m = result.position_matrix()
    assert m.sum(axis=0) == pytest.approx(np.ones(20)) and m.sum(axis=1) == pytest.approx(np.ones(20))


def test_headline_probabilities_add_up(result):
    assert result.prob_top(1).sum() == pytest.approx(1.0)
    assert result.prob_top(4).sum() == pytest.approx(4.0)
    assert result.prob_bottom(3).sum() == pytest.approx(3.0)


def test_points_are_conserved(snap, result):
    """Each simulated game hands out 3 points (a win) or 2 (a draw). No points appear or vanish."""
    base = sum(r["pts"] for r in snap.table)
    total = result.points.sum(axis=1)
    m = len(snap.remaining)
    assert total.min() >= base + 2 * m and total.max() <= base + 3 * m
    by_team = {r["team"]: r["pts"] for r in snap.table}
    assert np.all(result.points >= np.array([by_team[t] for t in snap.teams])[None, :])
    left = np.array([sum(t in f for f in snap.remaining) for t in snap.teams])
    assert np.all(result.points <= np.array([by_team[t] for t in snap.teams])[None, :] + 3 * left[None, :])


def test_one_match_awards_exactly_3_0_or_1_1_or_0_3():
    """The points rule itself, on the smallest possible league: two clubs, one game."""
    res = simulate(["A", "B"], [0, 0], [0, 0], [0, 0], [("A", "B")], {"A": 1500.0, "B": 1500.0}, GP,
                   n_sims=5_000, sigma=0.0, seed=5)
    outcomes = {tuple(int(x) for x in row) for row in np.unique(res.points, axis=0)}
    assert outcomes == {(3, 0), (1, 1), (0, 3)}
    won, drew = res.points[:, 0] == 3, res.points[:, 0] == 1
    assert np.all(res.goal_diff[won, 0] > 0) and np.all(res.goal_diff[drew, 0] == 0)
    assert np.all(res.goal_diff[~won & ~drew, 0] < 0)


def test_away_win_is_worth_three_points():
    res = simulate(["A", "B"], [0, 0], [0, 0], [0, 0], [("A", "B")], {"A": 900.0, "B": 2400.0}, GP,
                   n_sims=2_000, sigma=0.0, seed=5)
    assert (res.points[:, 1] == 3).mean() > 0.99 and (res.points[:, 0] == 0).mean() > 0.99


def test_goal_difference_sums_to_zero_every_season(result):
    assert np.all(result.goal_diff.sum(axis=1) == 0)


def test_more_points_always_ranks_higher(result):
    order = np.argsort(result.position, axis=1)
    sorted_pts = np.take_along_axis(result.points, order, axis=1)
    assert np.all(np.diff(sorted_pts, axis=1) <= 0)


def test_goal_difference_breaks_ties_on_points(result):
    order = np.argsort(result.position, axis=1)
    pts = np.take_along_axis(result.points, order, axis=1)
    gd = np.take_along_axis(result.goal_diff, order, axis=1)
    level = pts[:, :-1] == pts[:, 1:]
    assert level.any()
    assert np.all(gd[:, :-1][level] >= gd[:, 1:][level])


def test_finished_season_has_one_outcome():
    """No games left: the table is the table. Order is points, then goal difference, then goals scored."""
    res = simulate(FOUR, base_points=[10, 10, 10, 4], base_gd=[5, 5, 2, -12], base_gf=[9, 12, 20, 3],
                   fixtures=[], ratings=dict.fromkeys(FOUR, 1500.0), goal_params=GP, n_sims=200)
    #  B beats A on goals scored, both beat C on goal difference, D is last on points.
    assert np.all(res.position == np.array([2, 1, 3, 4])[None, :])


def test_remaining_fixtures_each_played_once(snap):
    res = pipeline.run_sim(snap, n_sims=500, seed=3)
    by_team = {r["team"]: r for r in snap.table}
    games_left = np.array([38 - by_team[t]["p"] for t in snap.teams])
    assert games_left.sum() == 2 * len(snap.remaining)
    max_gain = res.points - np.array([by_team[t]["pts"] for t in snap.teams])[None, :]
    assert np.all(max_gain <= 3 * games_left[None, :])


def test_much_stronger_team_usually_wins_the_league():
    res = _fresh({"A": 1900.0, "B": 1500.0, "C": 1500.0, "D": 1500.0})
    title = res.prob_top(1)
    assert title[0] > 0.85 and np.all(title[1:] < 0.1)


def test_equal_teams_get_equal_chances():
    res = _fresh(dict.fromkeys(FOUR, 1500.0), n_sims=40_000)
    assert res.prob_top(1) == pytest.approx(np.full(4, 0.25), abs=0.015)


def test_rating_uncertainty_makes_the_favourite_less_certain():
    ratings = {"A": 1600.0, "B": 1500.0, "C": 1500.0, "D": 1500.0}
    sure = _fresh(ratings, n_sims=20_000, sigma=0.0).prob_top(1)[0]
    unsure = _fresh(ratings, n_sims=20_000, sigma=150.0).prob_top(1)[0]
    assert unsure < sure - 0.03


def test_ten_thousand_seasons_run_in_seconds(snap):
    t0 = time.perf_counter()
    pipeline.run_sim(snap, n_sims=10_000)
    assert time.perf_counter() - t0 < 5.0


def test_dice_noise_is_small_enough_to_publish(snap):
    """Run the same 10,000 seasons 12 times with different dice. Odds should move about a point, no more."""
    runs = [pipeline.run_sim(snap, n_sims=10_000, sigma=config.SIGMA, seed=100 + i) for i in range(12)]
    for name, probs in (("title", [r.prob_top(1) for r in runs]), ("top4", [r.prob_top(4) for r in runs]),
                        ("relegated", [r.prob_bottom(3) for r in runs])):
        sd = np.array(probs).std(axis=0, ddof=1).max()
        assert sd < 0.01, f"{name}: worst sd {sd:.4f}"
