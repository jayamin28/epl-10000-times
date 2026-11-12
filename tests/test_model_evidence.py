"""Evidence that the frozen settings in config.py are still supported by the data.

If one of these fails after a data update, the setting needs a second look.
It does not get loosened to make the test pass.
"""

import pytest

from eplsim import config, pipeline

SELECT = ["2015-16", "2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22"]
CALIB = [s for s in pipeline.SEASONS if "2014-15" <= s <= "2025-26"]
TOL = 0.001


@pytest.fixture(scope="module")
def tuning(matches):
    return pipeline.tune_elo(matches)


@pytest.fixture(scope="module")
def held_out(matches):
    return pipeline.walk_forward(matches, config.ELO, config.GOALS_WINDOW, sorted(pipeline.TEST_SEASONS))


@pytest.fixture(scope="module")
def calibration(matches):
    rows = pipeline.season_calibration(matches, config.ELO, config.GOALS_WINDOW,
                                       [0.0, 20.0, 35.0, 50.0, 65.0, 80.0, 100.0], CALIB)
    return {r["sigma"]: r for r in rows}


def test_frozen_elo_is_as_good_as_the_best_setting_tried(tuning):
    best = tuning[0].tune_loss
    frozen = next(s for s in tuning if s.params == config.ELO)
    assert frozen.tune_loss - best < TOL


def test_best_elo_setting_is_not_on_the_edge_of_the_search(tuning):
    """An optimum on the edge means the search did not look far enough."""
    ks = sorted({k for k, _, _ in pipeline.ELO_GRID})
    proms = sorted({p for _, _, p in pipeline.ELO_GRID})
    best = tuning[0].params
    assert ks[0] < best.k < ks[-1] and proms[0] < best.promoted < proms[-1]


def test_home_bonus_setting_barely_matters(tuning):
    """PLAN D3 claims the Elo home bonus changes log loss by under 0.0003 at the frozen K and promoted rating."""
    same = [s.tune_loss for s in tuning if (s.params.k, s.params.promoted) == (config.ELO.k, config.ELO.promoted)]
    assert len(same) == 3 and max(same) - min(same) < 0.0003


def test_frozen_window_is_as_good_as_the_best_window(matches):
    losses = {w: pipeline.walk_forward(matches, config.ELO, w, SELECT)["model_log_loss"] for w in (2, 3, 4, 6, 8, 20)}
    assert losses[config.GOALS_WINDOW] - min(losses.values()) < TOL


def test_model_beats_base_rates_on_seasons_it_never_saw(held_out):
    assert held_out["matches"] == 1520
    assert held_out["baseline_log_loss"] - held_out["model_log_loss"] > 0.05
    assert held_out["accuracy"] > held_out["baseline_accuracy"] + 0.05


def test_draw_rate_is_close_on_held_out_seasons(held_out):
    assert abs(held_out["pred_draw_rate"] - held_out["actual_draw_rate"]) < 0.02


def test_goals_level_gap_stays_inside_the_documented_limit(held_out):
    """Known limitation: scoring rose in 2022 to 2026 and a trailing fit lags it. Bound: 0.25 goals a match."""
    assert abs(held_out["pred_goals_per_match"] - held_out["actual_goals_per_match"]) < 0.25


def test_frozen_sigma_is_as_good_as_the_best_sigma(calibration):
    best = min(r["brier_all"] for r in calibration.values())
    assert calibration[config.SIGMA]["brier_all"] - best < TOL


def test_season_forecasts_beat_knowing_nothing(calibration):
    """Brier score of always saying the base rate: title 1/20, top four 4/20, relegation 3/20."""
    r = calibration[config.SIGMA]
    assert r["brier_title"] < 0.05 * 0.95 * 0.75
    assert r["brier_top4"] < 0.20 * 0.80 * 0.60
    assert r["brier_relegated"] < 0.15 * 0.85 * 0.60


def test_stated_odds_match_what_happened(calibration):
    """In every well-filled bin, what the model said and what happened are within 8 points."""
    for b in calibration[config.SIGMA]["reliability"]:
        if b["n"] >= 40:
            assert abs(b["predicted"] - b["happened"]) < 0.08, b
