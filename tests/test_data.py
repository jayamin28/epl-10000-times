"""Data layer: every trap found in the sources has a test here."""

import csv
import json
from collections import Counter
from datetime import date, timedelta

import pytest

from eplsim import pipeline
from eplsim.data import Match, league_table, load_fpl, load_mirror, load_openfootball, reconcile
from eplsim.teams import canonical


# ---- parsing traps ---------------------------------------------------------

def test_openfootball_goalless_draw_in_bare_list_form_is_a_result(tmp_path):
    """openfootball writes a 0-0 draw as a bare list [0, 0], and gives an unplayed game no score at all.

    The first version of the loader read the bare list as "not played yet" and dropped real results.
    """
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"matches": [
        {"round": "Matchday 1", "date": "2026-08-21", "team1": "Arsenal FC", "team2": "Chelsea FC",
         "score": {"ft": [3, 0]}},
        {"round": "Matchday 2", "date": "2026-08-28", "team1": "Chelsea FC", "team2": "Arsenal FC",
         "score": [0, 0]},
        {"round": "Matchday 3", "date": "2026-09-05", "team1": "Arsenal FC", "team2": "Liverpool FC"},
    ]}))
    ms = load_openfootball(p, "2026-27")
    assert [m.played for m in ms] == [True, True, False]
    assert (ms[0].home_goals, ms[0].away_goals) == (3, 0)
    assert (ms[1].home_goals, ms[1].away_goals) == (0, 0)


@pytest.mark.parametrize("season, results, goalless", [("2025-26", 380, 27), ("2026-27", 50, 5)])
def test_openfootball_goalless_draws_are_the_official_feeds_goalless_draws(data_dir, frozen_dir, season, results, goalless):
    """Proof on real data that the bare lists are results: they are exactly the 0-0 draws in the FPL feed."""
    raw = frozen_dir if season == pipeline.CURRENT else data_dir / "history"
    of = load_openfootball(raw / "openfootball" / f"{season}.json", season)
    fpl = load_fpl(raw / "fpl" / f"fixtures_{season}.csv", raw / "fpl" / f"teams_{season}.csv", season)
    of_played = {(m.home, m.away) for m in of if m.played}
    assert len(of_played) == results
    of_00 = {(m.home, m.away) for m in of if (m.home_goals, m.away_goals) == (0, 0)}
    fpl_00 = {(m.home, m.away) for m in fpl if (m.home_goals, m.away_goals) == (0, 0)}
    assert len(of_00) == goalless
    assert of_00 == fpl_00 & of_played


def test_unknown_team_name_raises():
    with pytest.raises(KeyError):
        canonical("Real Madrid")


@pytest.mark.parametrize("raw, want", [
    ("Manchester United FC", "Man United"), ("Man Utd", "Man United"), ("manchester-united", "Man United"),
    ("AFC Bournemouth", "Bournemouth"), ("afc-bournemouth", "Bournemouth"), ("Sunderland AFC", "Sunderland"),
    ("Nott'm Forest", "Nottingham Forest"), ("Spurs", "Tottenham"), ("brighton-hove-albion", "Brighton"),
    ("Brighton & Hove Albion FC", "Brighton"), ("Hull City AFC", "Hull"), ("Wolverhampton Wanderers", "Wolves"),
])
def test_team_spellings_map_to_one_name(raw, want):
    assert canonical(raw) == want


def _write_mirror(folder, home_code, away_code, match_id, finished="True", hs="2.0", as_="1.0"):
    folder.mkdir(exist_ok=True)
    with open(folder / "teams.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["code", "id", "name"])
        w.writerows([[3, 1, "Arsenal"], [8, 6, "Chelsea"], [14, 14, "Liverpool"]])
    with open(folder / "pl_matches_gw1.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gameweek", "kickoff_time", "home_team", "home_score", "away_score", "away_team", "finished", "match_id"])
        w.writerow([1, "2026-08-21T19:00:00", home_code, hs, as_, away_code, finished, match_id])


def test_mirror_resolves_club_codes(tmp_path):
    _write_mirror(tmp_path / "m", "3.0", "8.0", "26-27-prem-arsenal-vs-chelsea")
    (m,) = load_mirror(tmp_path / "m", "2026-27")
    assert (m.home, m.away, m.home_goals, m.away_goals) == ("Arsenal", "Chelsea", 2, 1)


def test_mirror_rejects_code_that_disagrees_with_match_id(tmp_path):
    """Codes are the only team field. If one is wrong the match_id slug catches it."""
    _write_mirror(tmp_path / "m", "14.0", "8.0", "26-27-prem-arsenal-vs-chelsea")
    with pytest.raises(ValueError, match="disagree"):
        load_mirror(tmp_path / "m", "2026-27")


def test_mirror_rejects_finished_without_score(tmp_path):
    _write_mirror(tmp_path / "m", "3.0", "8.0", "26-27-prem-arsenal-vs-chelsea", finished="True", hs="", as_="")
    with pytest.raises(ValueError, match="finished flag"):
        load_mirror(tmp_path / "m", "2026-27")


# ---- reconciliation --------------------------------------------------------

def _m(hg, ag):
    return [Match("2025-26", 8, "2025-10-19", "Liverpool", "Man United", hg, ag)]


def test_reconcile_majority_overrules_the_odd_one_out():
    merged, rep = reconcile({"a": _m(1, 2), "b": _m(1, 2), "c": _m(0, 1)})
    assert (merged[0].home_goals, merged[0].away_goals) == (1, 2)
    assert len(rep["overruled"]) == 1 and rep["overruled"][0]["kept_from"] == ["a", "b"]


def test_reconcile_refuses_to_guess_without_a_majority():
    with pytest.raises(ValueError, match="no majority"):
        reconcile({"a": _m(1, 2), "b": _m(0, 1)})


def test_reconcile_flags_single_source_results():
    merged, rep = reconcile({"a": _m(1, 2), "b": _m(None, None)})
    assert merged[0].played and rep["single_source"] == [("Liverpool", "Man United")]


def test_reconcile_secondary_source_alone_cannot_invent_a_result():
    """If the primary says not played, one other source is not enough. It waits for a second."""
    merged, rep = reconcile({"a": _m(None, None), "b": _m(0, 0)})
    assert not merged[0].played and rep["awaiting_second_source"] == [("Liverpool", "Man United")]


def test_reconcile_two_agreeing_sources_fill_a_gap_in_the_primary():
    merged, rep = reconcile({"a": _m(None, None), "b": _m(1, 2), "c": _m(1, 2)})
    assert (merged[0].home_goals, merged[0].away_goals) == (1, 2) and rep["awaiting_second_source"] == []


def test_real_bad_score_in_mirror_is_caught(matches, report):
    """The mirror has Liverpool 0-1 Man United (19 Oct 2025). Two other sources say 1-2."""
    rep = report["2025-26"]
    assert [o["fixture"] for o in rep["overruled"]] == [("Liverpool", "Man United")]
    m = next(m for m in matches if m.season == "2025-26" and (m.home, m.away) == ("Liverpool", "Man United"))
    assert (m.home_goals, m.away_goals) == (1, 2)


def test_no_disagreement_between_sources_this_season(report):
    assert report[pipeline.CURRENT]["overruled"] == []
    assert report[pipeline.CURRENT]["awaiting_second_source"] == []


# ---- shape of every season -------------------------------------------------

def test_every_past_season_is_complete(matches):
    for s in pipeline.SEASONS[:-1]:
        ms = [m for m in matches if m.season == s]
        assert len(ms) == 380 and all(m.played for m in ms), s
        home, away = Counter(m.home for m in ms), Counter(m.away for m in ms)
        assert len(home) == 20 and set(home.values()) == {19} and set(away.values()) == {19}, s


def test_current_season_fixture_list(matches):
    ms = [m for m in matches if m.season == pipeline.CURRENT]
    assert len(ms) == 380 and len({(m.home, m.away) for m in ms}) == 380
    home, away = Counter(m.home for m in ms), Counter(m.away for m in ms)
    assert len(home) == 20 and set(home.values()) == {19} and set(away.values()) == {19}
    assert sum(m.played for m in ms) == 100


# ---- truth checks against facts from outside the data ----------------------

@pytest.mark.parametrize("season, champion, points", [
    ("2011-12", "Man City", 89),  # won on goal difference, level on points with Man United
    ("2015-16", "Leicester", 81),
    ("2017-18", "Man City", 100),
    ("2019-20", "Liverpool", 99),
])
def test_known_champions(matches, season, champion, points):
    table = league_table([m for m in matches if m.season == season])
    assert (table[0]["team"], table[0]["pts"]) == (champion, points)


def test_goal_difference_decides_2011_12(matches):
    table = league_table([m for m in matches if m.season == "2011-12"])
    assert table[1]["team"] == "Man United" and table[1]["pts"] == table[0]["pts"]
    assert table[0]["gd"] > table[1]["gd"]


def test_current_table_matches_independent_standings(matches, frozen_dir):
    """All 100 results this season, checked in aggregate against a standings page fetched separately."""
    ours = {r["team"]: r for r in league_table([m for m in matches if m.season == pipeline.CURRENT])}
    with open(frozen_dir / "reference_table.csv", newline="") as f:
        ref = list(csv.DictReader(f))
    assert len(ref) == 20
    for r in ref:
        o = ours[canonical(r["team"])]
        assert {k: o[k] for k in ("p", "w", "d", "l", "gf", "ga", "gd", "pts")} == \
               {k: int(r[k]) for k in ("p", "w", "d", "l", "gf", "ga", "gd", "pts")}, r["team"]


def test_snapshot_is_not_stale(matches, frozen_dir):
    """Every game dated at least two days before the fetch must have a result."""
    fetched = date.fromisoformat(json.loads((frozen_dir / "manifest.json").read_text())["fetched_on"])
    cutoff = (fetched - timedelta(days=2)).isoformat()
    missing = [m for m in matches if m.season == pipeline.CURRENT and m.date <= cutoff and not m.played]
    assert missing == []
