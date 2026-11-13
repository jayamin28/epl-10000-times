"""Every published number, checked on a four-club league small enough to work out by hand."""

import math

import numpy as np
import pytest

from eplsim import metrics, pipeline
from eplsim.data import Match
from eplsim.goals import GoalParams
from eplsim.pipeline import Snapshot
from eplsim.simulate import SimResult

TEAMS = ["Ajax", "Bern", "Cork", "Dax"]
GP = GoalParams(alpha=0.2, beta=0.8, gamma=0.2)

# Ten hand-written seasons. Row = season, column = club in TEAMS order.
POSITION = np.array([
    [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4],  # Ajax win six
    [2, 1, 3, 4], [2, 1, 3, 4], [2, 1, 4, 3],                                              # Bern win three
    [3, 2, 1, 4],                                                                          # Cork win one (season 10)
])
POINTS = np.array([
    [12, 9, 6, 3], [12, 9, 6, 3], [12, 9, 6, 3], [12, 9, 6, 3], [13, 9, 6, 3], [13, 9, 6, 3],
    [9, 12, 6, 3], [9, 12, 6, 3], [9, 12, 3, 6],
    [6, 9, 10, 3],
])
FIXTURES = [("Cork", "Ajax"), ("Cork", "Dax"), ("Bern", "Ajax")]
HOME_GOALS = np.zeros((10, 3), dtype=np.int16)
AWAY_GOALS = np.zeros((10, 3), dtype=np.int16)
HOME_GOALS[9] = [3, 1, 0]  # season 10: Cork beat Ajax 3-0 and Dax 1-0
AWAY_GOALS[9] = [0, 0, 0]


@pytest.fixture(scope="module")
def res():
    return SimResult(TEAMS, POSITION, POINTS, np.zeros_like(POINTS), {}, 10, 1, goals_for=POINTS.copy(),
                     fixtures=FIXTURES, home_goals=HOME_GOALS, away_goals=AWAY_GOALS, tiebreak={})


@pytest.fixture(scope="module")
def mini():
    """Table now: Bern 6, Ajax 4, Cork 3, Dax 1. Ajax are rated highest."""
    table = [
        {"team": "Bern", "p": 2, "w": 2, "d": 0, "l": 0, "gf": 4, "ga": 1, "pts": 6, "gd": 3},
        {"team": "Ajax", "p": 2, "w": 1, "d": 1, "l": 0, "gf": 3, "ga": 1, "pts": 4, "gd": 2},
        {"team": "Cork", "p": 2, "w": 1, "d": 0, "l": 1, "gf": 2, "ga": 3, "pts": 3, "gd": -1},
        {"team": "Dax", "p": 2, "w": 0, "d": 1, "l": 1, "gf": 1, "ga": 5, "pts": 1, "gd": -4},
    ]
    remaining = [Match("x", 3, "2030-01-10", "Cork", "Ajax", None, None), Match("x", 3, "2030-01-09", "Cork", "Dax", None, None),
                 Match("x", 4, "2030-01-17", "Bern", "Ajax", None, None)]
    return Snapshot(season="x", teams=TEAMS, table=table, remaining=[(m.home, m.away) for m in remaining],
                    ratings={"Ajax": 1700.0, "Bern": 1600.0, "Cork": 1500.0, "Dax": 1400.0}, goal_params=GP, played=4,
                    last_result_date="2030-01-02", played_results=[], remaining_matches=remaining)


def test_race_probabilities(res):
    p = metrics.race_probs(res)
    assert p["title"].tolist() == [0.6, 0.3, 0.1, 0.0]
    assert p["top4"].tolist() == [1.0, 1.0, 1.0, 1.0]
    assert p["relegated"].tolist() == [0.4, 0.7, 0.9, 1.0]  # bottom three of four clubs: everyone but the champion
    assert res.prob_bottom(1).tolist() == [0.0, 0.0, 0.1, 0.9]


def test_headline_rows(mini, res):
    rows = metrics.headline(mini, res)
    assert [r["team"] for r in rows] == ["Ajax", "Bern", "Cork", "Dax"]  # by expected points
    ajax = rows[0]
    assert ajax["exp_pts"] == pytest.approx((12 * 4 + 13 * 2 + 9 * 3 + 6) / 10)  # 10.7
    assert (ajax["table_pos"], ajax["pts_now"], ajax["played"], ajax["gd_now"]) == (2, 4, 2, 2)
    assert ajax["title"] == 0.6 and ajax["most_likely_place"] == 1
    assert rows[3]["most_likely_place"] == 4 and rows[3]["exp_pts"] == pytest.approx(3.3)


def test_tiers_use_the_published_cut_offs(res):
    t = metrics.tiers(res)["title"]
    assert t == {"locked_in": [], "in_the_race": ["Ajax", "Bern"], "on_the_edge": ["Cork"], "almost_gone": ["Dax"]}
    assert (metrics.LOCKED_IN, metrics.IN_THE_RACE, metrics.ALMOST_GONE) == (0.85, 0.20, 0.02)


def test_tier_boundaries_are_inclusive():
    pos = np.array([[1, 2]] * 17 + [[2, 1]] * 3)  # 85% and 15%
    r = SimResult(["A", "B"], pos, np.zeros_like(pos), np.zeros_like(pos), {}, 20, 1)
    t = metrics.tiers(r)["title"]
    assert t["locked_in"] == ["A"] and t["on_the_edge"] == ["B"]


def test_table_lies_ranks_the_biggest_disagreement_first(mini, res):
    rows = metrics.table_lies(mini, res)
    by = {r["team"]: r for r in rows}
    assert by["Ajax"] == {"team": "Ajax", "table_pos": 2, "model_pos": 1, "gap": 1}
    assert by["Bern"]["gap"] == -1 and by["Cork"]["gap"] == 0 and by["Dax"]["gap"] == 0
    assert [r["team"] for r in rows[:2]] == ["Ajax", "Bern"]


def test_magic_numbers(res):
    m = metrics.magic_numbers(res)
    title = m["title"]  # champion's points: 12 x6, 13 x2, 12 x... -> 12,12,12,12,13,13,12,12,12,10
    assert (title["mode"], title["min"], title["median"]) == (12, 10, 12)
    assert title["share"] == pytest.approx([0.1, 0.0, 0.7, 0.2])
    assert sum(title["share"]) == pytest.approx(1.0)
    assert m["top4"]["mode"] == 3  # the club finishing fourth has 3 points in every season
    assert "safety" not in m  # no 17th place in a four-club league


def test_points_shape(res):
    s = metrics.points_shape(res)["Ajax"]
    assert (s["min"], s["mode"]) == (6, 12)
    assert s["share"][0] == pytest.approx(0.1) and s["share"][12 - 6] == pytest.approx(0.4) and s["share"][13 - 6] == pytest.approx(0.2)


def test_home_edge_in_rating_points():
    # home and away rates differ by gamma on top of beta*d/400 each way: gamma = 2*beta*edge/400
    assert metrics.home_edge_in_rating_points(GP) == pytest.approx(50.0)
    lam_h, lam_a = metrics.goals.rates(np.array([-50.0]), GP)
    assert lam_h[0] == pytest.approx(lam_a[0])  # a home side 50 points weaker is an even match


def test_run_in(mini):
    rows = {r["team"]: r for r in metrics.run_in(mini)}
    # Ajax: away to Cork (1500 + 50) and away to Bern (1600 + 50)
    assert rows["Ajax"]["difficulty"] == pytest.approx(1600.0) and rows["Ajax"]["avg_opponent_rating"] == pytest.approx(1550.0)
    assert (rows["Ajax"]["games"], rows["Ajax"]["home_games"]) == (2, 0)
    # Cork: home to Ajax (1700 - 50) and home to Dax (1400 - 50)
    assert rows["Cork"]["difficulty"] == pytest.approx(1500.0) and rows["Cork"]["home_games"] == 2
    assert rows["Bern"]["difficulty"] == pytest.approx(1650.0)
    assert rows["Dax"]["difficulty"] == pytest.approx(1550.0)
    assert [rows[t]["rank"] for t in ("Bern", "Ajax", "Dax", "Cork")] == [1, 2, 3, 4]  # hardest first


def test_upcoming_is_the_next_gameweek_in_date_order(mini):
    nxt = metrics.upcoming(mini)
    assert nxt["gameweek"] == 3
    assert [(f["home"], f["away"]) for f in nxt["fixtures"]] == [("Cork", "Dax"), ("Cork", "Ajax")]
    for f in nxt["fixtures"]:
        assert f["home_win"] + f["draw"] + f["away_win"] == pytest.approx(1.0, abs=1e-6)
    cork_dax, cork_ajax = nxt["fixtures"]
    assert cork_dax["home_win"] > 0.5 > cork_ajax["home_win"]
    assert cork_dax["home_xg"] == pytest.approx(math.exp(0.2 + 0.2 + 0.8 * 100 / 400))


def test_universe_lookup(res):
    assert metrics.universe(res, 10) == {"season": 10, "champion": "Cork", "top_four": ["Cork", "Bern", "Ajax", "Dax"],
                                         "relegated": ["Bern", "Ajax", "Dax"]}
    assert metrics.universe(res, 1)["champion"] == "Ajax"
    with pytest.raises(ValueError):
        metrics.universe(res, 0)
    with pytest.raises(ValueError):
        metrics.universe(res, 11)


def test_packed_universes_agree_with_the_lookup(res):
    u = metrics.universes(res)
    assert u["champion"] == "AAAAAABBBC" and len(u["relegated"]) == 30
    for n in range(1, 11):
        one = metrics.universe(res, n)
        assert u["teams"][ord(u["champion"][n - 1]) - 65] == one["champion"]
        assert [u["teams"][ord(c) - 65] for c in u["relegated"][3 * (n - 1):3 * n]] == one["relegated"]


def test_wildest_season_is_the_least_likely_champion(mini, res):
    w = metrics.wildest_season(mini, res)
    assert (w["season"], w["champion"], w["title_seasons"], w["title_share"]) == (10, "Cork", 1, 0.1)
    assert (w["champion_pts"], w["margin"], w["champion_table_pos_now"], w["champion_pts_now"]) == (10, 1, 3, 3)
    assert w["rest_of_season"] == {"won": 2, "drawn": 0, "lost": 0}
    assert w["best_win"] == {"opponent": "Ajax", "venue": "home", "score": "3-0"}
    assert [r["team"] for r in w["table"]] == ["Cork", "Bern", "Ajax", "Dax"]


def test_convergence_ends_on_the_headline_number(res):
    c = metrics.convergence(res, top=2)
    assert c["seasons"] == [10] and c["share"] == {"Ajax": [0.6], "Bern": [0.3]}


def test_convergence_on_the_real_run(result):
    c = metrics.convergence(result)
    assert c["seasons"][-1] == result.n_sims and len(c["share"]) == 5
    title = dict(zip(result.teams, result.prob_top(1)))
    for team, path in c["share"].items():
        assert path[-1] == pytest.approx(title[team])


def test_duel(mini, res):
    d = metrics.duel(mini, res, "Ajax", "Bern")
    rows = {r["label"]: r for r in d["rows"]}
    assert len(d["rows"]) == 10
    assert (rows["Rating"]["a"], rows["Rating"]["b"], rows["Rating"]["better"]) == (1700.0, 1600.0, "Ajax")
    assert rows["Points now"]["better"] == "Bern" and rows["Points now"]["b_rank"] == 1
    assert rows["Goals against per game"]["a"] == 0.5 and rows["Goals against per game"]["better"] is None  # both 0.5
    assert rows["Title"]["a"] == 0.6 and rows["Title"]["b"] == 0.3
    assert rows["Run-in difficulty"]["better"] == "Ajax"  # 1600 against 1650: lower is easier
    assert rows["Finishes above the other"]["a"] == 0.6 and rows["Finishes above the other"]["b"] == 0.4


def test_movers_ignore_changes_inside_the_noise():
    before = {"A": {"title": 0.30, "top4": 0.80, "top5": 0.9, "relegated": 0.0}, "B": {"title": 0.10, "top4": 0.5, "top5": 0.6, "relegated": 0.01}}
    after = {"A": {"title": 0.36, "top4": 0.82, "top5": 0.9, "relegated": 0.0}, "B": {"title": 0.07, "top4": 0.40, "top5": 0.6, "relegated": 0.01},
             "C": {"title": 0.5, "top4": 0.5, "top5": 0.5, "relegated": 0.5}}  # C has no earlier row
    rows = metrics.movers(before, after)
    assert [(r["team"], r["race"], round(r["change"], 2)) for r in rows] == [("B", "top4", -0.10), ("A", "title", 0.06), ("B", "title", -0.03)]
    assert metrics.MOVER_THRESHOLD == 0.03


def test_report_card_by_hand():
    forecasts = [{"home": "A", "away": "B", "p_home": 0.5, "p_draw": 0.3, "p_away": 0.2},
                 {"home": "C", "away": "D", "p_home": 0.2, "p_draw": 0.2, "p_away": 0.6},
                 {"home": "E", "away": "F", "p_home": 0.4, "p_draw": 0.3, "p_away": 0.3}]  # not played yet
    results = {("A", "B"): (2, 0), ("C", "D"): (1, 1)}
    card = metrics.report_card(forecasts, results, (0.45, 0.25, 0.30))
    assert card["matches"] == 2
    assert card["log_loss"] == pytest.approx(-(math.log(0.5) + math.log(0.2)) / 2)
    assert card["baseline_log_loss"] == pytest.approx(-(math.log(0.45) + math.log(0.25)) / 2)
    assert card["brier"] == pytest.approx(((0.25 + 0.09 + 0.04) + (0.04 + 0.64 + 0.36)) / 2)
    assert card["accuracy"] == 0.5
    assert metrics.report_card(forecasts[2:], results, (0.45, 0.25, 0.30)) == {"matches": 0}


# Match swings need real simulated seasons, so they run on the live snapshot with fewer seasons.

@pytest.fixture(scope="module")
def swing_setup(snap):
    def sim(forced=None):
        return pipeline.run_sim(snap, n_sims=1_500, sigma=65.0, seed=7, forced=forced)
    nxt = metrics.upcoming(snap)
    return snap, sim, nxt, metrics.match_swings(snap, sim, nxt["fixtures"][:3])


def test_forcing_a_result_changes_that_match_and_nothing_else(swing_setup):
    snap, sim, nxt, _ = swing_setup
    fx = nxt["fixtures"][0]
    key = (fx["home"], fx["away"])
    col = snap.remaining.index(key)
    free, fixed = sim(), sim({key: (1, 0)})
    assert (fixed.home_goals[:, col] == 1).all() and (fixed.away_goals[:, col] == 0).all()
    others = [c for c in range(len(snap.remaining)) if c != col]
    assert np.array_equal(free.home_goals[:, others], fixed.home_goals[:, others])
    assert np.array_equal(free.away_goals[:, others], fixed.away_goals[:, others])


def test_winning_never_hurts_and_losing_never_helps(swing_setup):
    _, _, _, swings = swing_setup
    for entry in swings["fixtures"]:
        for team, races in entry["clubs"].items():
            for race in ("title", "top4", "top5"):
                assert races[race]["win"] >= races[race]["draw"] >= races[race]["lose"], (team, race)
            r = races["relegated"]
            assert r["win"] <= r["draw"] <= r["lose"], team


def test_biggest_swing_is_the_largest_of_all(swing_setup):
    _, _, _, swings = swing_setup
    every = [abs(r["swing"]) for e in swings["fixtures"] for races in e["clubs"].values() for r in races.values()]
    b = swings["biggest"]
    assert abs(b["swing"]) == pytest.approx(max(every)) and abs(b["swing"]) > 0.02
    assert b["swing"] == pytest.approx(b["win"] - b["lose"])


def test_next_fixtures_on_the_live_snapshot(snap):
    nxt = metrics.upcoming(snap)
    assert nxt["gameweek"] == 11 and len(nxt["fixtures"]) == 10
    clubs = {f["home"] for f in nxt["fixtures"]} | {f["away"] for f in nxt["fixtures"]}
    assert len(clubs) == 20


def test_telemetry_counts_what_the_engine_did(snap, result):
    t = metrics.telemetry(result, seconds=0.5, reruns=30)
    n_fix = len(snap.remaining)
    assert t["matches"] == result.n_sims * n_fix
    assert t["random_draws"] == result.n_sims * (2 * len(result.teams) + 2 * n_fix)  # noise, two scores, tie order
    assert t["goals"] == int(result.home_goals.sum()) + int(result.away_goals.sum())
    assert t["distinct_champions"] == int((result.prob_top(1) > 0).sum())
    assert 1 <= t["distinct_relegated_trios"] <= result.n_sims
    assert (t["full_runs"], t["seasons_incl_match_pricing"]) == (31, 31 * result.n_sims)
    hg, ag = (int(v) for v in t["biggest_win"]["score"].split("-"))
    s, col = t["biggest_win"]["season"] - 1, result.fixtures.index((t["biggest_win"]["home"], t["biggest_win"]["away"]))
    assert (result.home_goals[s, col], result.away_goals[s, col]) == (hg, ag)
    assert abs(hg - ag) == int(abs(result.home_goals.astype(int) - result.away_goals.astype(int)).max())
    assert t["max_points"]["points"] == result.points.max() and t["min_points"]["points"] == result.points.min()


def test_telemetry_on_the_hand_built_league(res):
    t = metrics.telemetry(res)
    assert (t["matches"], t["goals"], t["distinct_champions"]) == (30, 4, 3)
    assert t["biggest_win"] == {"home": "Cork", "away": "Ajax", "score": "3-0", "season": 10}
    assert t["most_common_score"] == "0-0" and t["seconds"] is None
