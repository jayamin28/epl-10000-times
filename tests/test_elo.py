import pytest

from eplsim.data import Match
from eplsim.elo import START, EloParams, expected_home, margin_multiplier, played_in_order, run_elo, update

P = EloParams(k=20.0, hfa=0.0, promoted=1400.0)


def test_expected_scores_of_both_sides_sum_to_one():
    assert expected_home(1600, 1500, 0) + expected_home(1500, 1600, 0) == pytest.approx(1.0)


def test_equal_teams_are_even_and_home_bonus_helps():
    assert expected_home(1500, 1500, 0) == pytest.approx(0.5)
    assert expected_home(1500, 1500, 60) > 0.5


def test_hand_calculated_updates():
    assert update(1500, 1500, 1, 0, P) == pytest.approx((1510.0, 1490.0))  # 20 * 1.0 * (1 - 0.5)
    assert update(1500, 1500, 3, 0, P) == pytest.approx((1517.5, 1482.5))  # margin multiplier 1.75
    assert update(1500, 1500, 1, 1, P) == pytest.approx((1500.0, 1500.0))  # draw between equals


@pytest.mark.parametrize("gd, mult", [(0, 1.0), (1, 1.0), (-1, 1.0), (2, 1.5), (3, 1.75), (4, 1.875), (-5, 2.0)])
def test_margin_multiplier(gd, mult):
    assert margin_multiplier(gd) == pytest.approx(mult)


def test_updates_are_zero_sum():
    h, a = update(1712.3, 1455.9, 0, 4, EloParams(k=12, hfa=60, promoted=1400))
    assert h + a == pytest.approx(1712.3 + 1455.9)


def test_upset_moves_ratings_more_than_expected_result():
    fav_wins = update(1700, 1400, 1, 0, P)[0] - 1700
    fav_loses = 1700 - update(1700, 1400, 0, 1, P)[0]
    assert fav_loses > fav_wins > 0


def _season(season, teams, score=(1, 0)):
    out, day = [], 1
    for h in teams:
        for a in teams:
            if h != a:
                out.append(Match(season, 1, f"{season[:4]}-09-{day:02d}", h, a, *score))
                day = day % 28 + 1
    return out


def test_first_match_uses_starting_ratings_not_its_own_result():
    ms = _season("2010-11", ["A", "B", "C"])
    pre, _ = run_elo(ms, P)
    assert pre[0] == (START, START)


def test_promoted_club_enters_at_promoted_rating_and_others_carry_over():
    ms = _season("2010-11", ["A", "B", "C"]) + _season("2011-12", ["A", "B", "D"])
    pre, final = run_elo(ms, P)
    ordered = played_in_order(ms)
    _, end_first = run_elo(_season("2010-11", ["A", "B", "C"]), P)

    def rating_before_first_game(club):
        i = next(i for i, m in enumerate(ordered) if m.season == "2011-12" and club in (m.home, m.away))
        return pre[i][0] if ordered[i].home == club else pre[i][1]

    assert rating_before_first_game("D") == 1400.0
    assert rating_before_first_game("A") == pytest.approx(end_first["A"])
    assert rating_before_first_game("B") == pytest.approx(end_first["B"])
    assert final["C"] == pytest.approx(end_first["C"])  # relegated clubs keep their last rating on file


def test_ratings_within_a_season_are_conserved(matches):
    """Zero-sum updates mean the 20 clubs' total rating is the same before and after a season's games."""
    from eplsim import config
    season = "2018-19"
    upto_prev = [m for m in matches if m.season < season]
    _, before = run_elo(upto_prev, config.ELO)
    _, after = run_elo([m for m in matches if m.season <= season], config.ELO)
    clubs = {t for m in matches if m.season == season for t in (m.home, m.away)}
    prev_clubs = {t for m in matches if m.season == "2017-18" for t in (m.home, m.away)}
    start_total = sum(before[t] if t in prev_clubs else config.ELO.promoted for t in clubs)
    assert sum(after[t] for t in clubs) == pytest.approx(start_total, abs=1e-6)
