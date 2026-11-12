"""The checks that decide whether a run may publish."""

from dataclasses import replace
from datetime import date

import pytest

from eplsim import gates
from eplsim.data import league_table

TODAY = date(2026, 11, 11)


@pytest.fixture(scope="module")
def current(matches):
    return [m for m in matches if m.season == "2026-27"]


def _status(r):
    return r["status"]


def test_live_fixture_list_is_complete(current):
    assert _status(gates.fixture_list(current)) == "pass"


def test_fixture_list_catches_a_missing_a_repeated_and_a_reversed_fixture(current):
    assert _status(gates.fixture_list(current[:-1])) == "fail"
    assert _status(gates.fixture_list(current[:-1] + [current[0]])) == "fail"
    flipped = replace(current[-1], home=current[-1].away, away=current[-1].home)
    assert _status(gates.fixture_list(current[:-1] + [flipped])) == "fail"  # a club with 20 home games


def test_no_result_is_overdue_on_the_frozen_day(current):
    assert _status(gates.no_stale_results(current, TODAY, set())) == "pass"


def test_a_three_day_old_match_without_a_result_stops_the_run(current):
    i = next(k for k, m in enumerate(current) if m.played)
    old = current[i]
    lost = replace(old, date="2026-11-08", home_goals=None, away_goals=None)
    r = gates.no_stale_results(current[:i] + [lost] + current[i + 1:], TODAY, set())
    assert _status(r) == "fail" and f"{old.home} v {old.away}" in r["detail"] and "postponed.json" in r["detail"]


def test_a_match_from_the_last_two_days_is_given_time(current):
    i = next(k for k, m in enumerate(current) if m.played)
    recent = replace(current[i], date="2026-11-09", home_goals=None, away_goals=None)
    assert _status(gates.no_stale_results(current[:i] + [recent] + current[i + 1:], TODAY, set())) == "pass"


def test_a_postponed_match_on_the_allowlist_is_exempt(current):
    i = next(k for k, m in enumerate(current) if m.played)
    old = current[i]
    lost = replace(old, date="2026-10-01", home_goals=None, away_goals=None)
    season = current[:i] + [lost] + current[i + 1:]
    assert _status(gates.no_stale_results(season, TODAY, {(old.home, old.away)})) == "pass"
    assert _status(gates.no_stale_results(season, TODAY, {(old.away, old.home)})) == "fail"  # the reverse fixture is a different match


def test_results_may_grow_but_never_shrink():
    assert _status(gates.results_never_vanish(100, None)) == "skipped"
    assert _status(gates.results_never_vanish(100, 100)) == "pass"
    assert _status(gates.results_never_vanish(110, 100)) == "pass"
    assert _status(gates.results_never_vanish(99, 100)) == "fail"


def test_two_of_three_sources_must_answer():
    def manifest(*ok):
        return {"sources": {n: {"ok": v} for n, v in zip(("fpl", "mirror", "openfootball"), ok)}}
    assert _status(gates.sources_reachable(manifest(True, True, True))) == "pass"
    assert _status(gates.sources_reachable(manifest(True, False, True))) == "pass"
    assert _status(gates.sources_reachable(manifest(False, False, True))) == "fail"
    assert _status(gates.sources_reachable(None)) == "skipped"


def test_official_positions_agree_warn_or_skip(current):
    table = league_table(current)
    played = sum(m.played for m in current)
    official = {r["team"]: i + 1 for i, r in enumerate(table)}
    assert _status(gates.table_matches_official_positions(table, official, played, played)) == "pass"

    swapped = dict(official)
    a, b = table[0]["team"], table[5]["team"]
    swapped[a], swapped[b] = swapped[b], swapped[a]
    r = gates.table_matches_official_positions(table, swapped, played, played)
    assert _status(r) == "warn" and not r["blocking"] and a in r["detail"]

    assert _status(gates.table_matches_official_positions(table, official, played - 10, played)) == "skipped"
    assert _status(gates.table_matches_official_positions(table, None, played, played)) == "skipped"
    assert _status(gates.table_matches_official_positions(table, {t: 0 for t in official}, played, played)) == "skipped"


def test_clubs_level_on_everything_may_swap_places():
    table = [{"team": "A", "pts": 10, "gd": 2, "gf": 5}, {"team": "B", "pts": 7, "gd": 0, "gf": 4},
             {"team": "C", "pts": 7, "gd": 0, "gf": 4}] + [{"team": f"T{i}", "pts": 0, "gd": -i, "gf": 0} for i in range(17)]
    official = {r["team"]: i + 1 for i, r in enumerate(table)}
    official["B"], official["C"] = 3, 2
    assert _status(gates.table_matches_official_positions(table, official, 30, 30)) == "pass"


def test_run_gates_passes_on_the_frozen_season(current):
    out = gates.run_gates(current, league_table(current), TODAY, set(), None, None, None, None)
    assert [r["gate"] for r in out] == ["fixture_list", "sources_reachable", "no_stale_results", "results_never_vanish",
                                        "table_matches_official_positions"]
    assert not any(r["status"] == "fail" for r in out)


def test_run_gates_names_every_blocking_failure(current):
    bad_manifest = {"sources": {"fpl": {"ok": False}, "mirror": {"ok": True}, "openfootball": {"ok": False}}}
    with pytest.raises(gates.GateError) as e:
        gates.run_gates(current[:-1], league_table(current), TODAY, set(), 250, bad_manifest, None, None)
    msg = str(e.value)
    assert "fixture_list" in msg and "results_never_vanish" in msg and "sources_reachable" in msg


def test_a_warning_does_not_stop_the_run(current):
    table = league_table(current)
    played = sum(m.played for m in current)
    official = {r["team"]: i + 1 for i, r in enumerate(table)}
    a, b = table[0]["team"], table[5]["team"]
    official[a], official[b] = official[b], official[a]
    out = gates.run_gates(current, table, TODAY, set(), None, None, official, played)
    assert out[-1]["status"] == "warn"
