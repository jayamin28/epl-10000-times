"""Clubs level on points, goal difference and goals scored: Premier League Rule C.17.

Order of the rule: points, goal difference, goals scored, then points in the
matches between the level clubs, then away goals in those matches, then a
play-off. No real season since 2010 ended with two clubs fully level, so
the head-to-head steps are tested on hand-built cases and the goals-scored
step on the three real seasons where it decided a place.
"""

import numpy as np
import pytest

from eplsim.data import league_table
from eplsim.goals import GoalParams
from eplsim.pipeline import SEASONS
from eplsim.simulate import head_to_head_order, simulate

GP = GoalParams(alpha=0.24, beta=0.68, gamma=0.16)
A, B, C, D = 0, 1, 2, 3


def test_head_to_head_points_decide():
    # A beat B home and away
    order, how = head_to_head_order([B, A], [(A, B, 1, 0), (B, A, 0, 2)])
    assert (order, how) == ([A, B], "points")


def test_head_to_head_away_goals_decide_when_points_are_level():
    # A won 1-0 at home, B won 2-1 at home: three points each. A scored once away, B never.
    order, how = head_to_head_order([B, A], [(A, B, 1, 0), (B, A, 2, 1)])
    assert (order, how) == ([A, B], "away_goals")


def test_two_identical_draws_leave_the_clubs_level_and_keep_the_fallback_order():
    results = [(A, B, 1, 1), (B, A, 1, 1)]
    assert head_to_head_order([B, A], results) == ([B, A], "level")
    assert head_to_head_order([A, B], results) == ([A, B], "level")


def test_matches_against_other_clubs_do_not_count():
    # B thrashed outsiders, but A won the only matches that count
    results = [(A, B, 1, 0), (B, A, 0, 1), (B, C, 9, 0), (D, B, 0, 9), (A, C, 0, 5)]
    assert head_to_head_order([B, A], results) == ([A, B], "points")


def test_three_clubs_use_a_mini_league_of_their_matches():
    results = [
        (A, B, 2, 0), (B, A, 0, 1),  # A takes 6 from B
        (A, C, 1, 1), (C, A, 0, 0),  # A and C share 2 each
        (B, C, 3, 0), (C, B, 0, 1),  # B takes 6 from C
        (A, D, 0, 4), (D, C, 0, 6),  # outsiders, ignored
    ]
    order, how = head_to_head_order([C, B, A], results)
    assert order == [A, B, C]  # mini-league points: A 8, B 6, C 2
    assert how == "points"


def test_three_clubs_partly_split_is_reported_as_level():
    # A beats both. B and C draw twice with the same score: nothing splits those two.
    results = [(A, B, 1, 0), (B, A, 0, 1), (A, C, 1, 0), (C, A, 0, 1), (B, C, 1, 1), (C, B, 1, 1)]
    order, how = head_to_head_order([C, B, A], results)
    assert order == [A, C, B] and how == "level"


def _forced_league(played, fixtures, forced, seed):
    """Two clubs, one real match already in the table, one match still to play."""
    (h, a, hg, ag), = played
    pts = {h: 3 if hg > ag else 1 if hg == ag else 0, a: 3 if ag > hg else 1 if hg == ag else 0}
    gd, gf = {h: hg - ag, a: ag - hg}, {h: hg, a: ag}
    return simulate(["A", "B"], [pts["A"], pts["B"]], [gd["A"], gd["B"]], [gf["A"], gf["B"]], fixtures,
                    {"A": 1500.0, "B": 1500.0}, GP,
                    n_sims=200, sigma=65.0, seed=seed, played=played, forced=forced)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_simulation_uses_head_to_head_from_real_and_simulated_matches(seed):
    """First leg is a real result (A won 1-0). Second leg is simulated and fixed at 2-1 to B.

    Both clubs end on 3 points, goal difference 0, 2 goals. A scored an away goal, B did not.
    A must finish first in every season, whatever the random fallback order says.
    """
    res = _forced_league(played=[("A", "B", 1, 0)], fixtures=[("B", "A")], forced={0: (2, 1)}, seed=seed)
    assert (res.points == 3).all() and (res.goal_diff == 0).all()
    assert (res.position[:, 0] == 1).all()
    assert res.tiebreak["split_by_head_to_head_away_goals"] == 200 and res.tiebreak["left_to_a_play_off"] == 0


def test_without_a_tiebreak_the_same_league_would_be_a_coin_flip():
    """The guard for the test above: two mirror-image draws really are left to chance."""
    res = _forced_league(played=[("A", "B", 1, 1)], fixtures=[("B", "A")], forced={0: (1, 1)}, seed=5)
    share = (res.position[:, 0] == 1).mean()
    assert 0.35 < share < 0.65
    assert res.tiebreak["left_to_a_play_off"] == 200


def test_tiebreak_counts_add_up(result):
    t = result.tiebreak
    assert t["groups"] == t["split_by_head_to_head_points"] + t["split_by_head_to_head_away_goals"] + t["left_to_a_play_off"]
    assert 0 < t["seasons_with_clubs_fully_level"] <= t["groups"]
    assert t["seasons_with_clubs_fully_level"] < 0.05 * result.n_sims  # rare, as in real life


def test_tiebreak_leaves_every_season_a_full_ranking(result):
    n = len(result.teams)
    assert np.array_equal(np.sort(result.position, axis=1), np.tile(np.arange(1, n + 1), (result.n_sims, 1)))


def test_a_club_never_finishes_below_one_with_fewer_points(result):
    order = np.argsort(result.position, axis=1)
    pts = np.take_along_axis(result.points, order, axis=1)
    assert (np.diff(pts, axis=1) <= 0).all()


@pytest.mark.parametrize("season, above, below, goals", [
    ("2011-12", "West Brom", "Swansea", (45, 44)),
    ("2016-17", "Leicester", "Stoke", (48, 41)),
    ("2020-21", "Newcastle", "Wolves", (46, 36)),
])
def test_real_ties_on_points_and_goal_difference_go_to_goals_scored(matches, season, above, below, goals):
    table = league_table([m for m in matches if m.season == season])
    row = {r["team"]: r for r in table}
    pos = {r["team"]: i for i, r in enumerate(table)}
    assert (row[above]["pts"], row[above]["gd"]) == (row[below]["pts"], row[below]["gd"])
    assert (row[above]["gf"], row[below]["gf"]) == goals
    assert pos[above] + 1 == pos[below]


def test_no_real_season_since_2010_ended_with_clubs_fully_level(matches):
    """Why the head-to-head steps cannot be checked against history."""
    for season in SEASONS[:-1]:
        keys = [(r["pts"], r["gd"], r["gf"]) for r in league_table([m for m in matches if m.season == season])]
        assert len(set(keys)) == len(keys), season
