"""Fetching three sources, with the network replaced by a stand-in that serves the frozen files."""

import csv
import json

import pytest

from eplsim import fetch, pipeline
from eplsim.data import load_fpl, load_mirror, load_openfootball

SEASON = "2026-27"


def _header(path):
    with open(path, newline="", encoding="utf-8") as f:
        return next(csv.reader(f))


def test_live_api_is_used_first_and_reads_back_identically(tmp_path, fake_get, frozen_dir):
    m = fetch.fetch_current(tmp_path, SEASON, get=fake_get())
    assert m["sources"]["fpl"]["ok"] and m["sources"]["fpl"]["origin"] == "api"
    got = load_fpl(tmp_path / "fpl" / f"fixtures_{SEASON}.csv", tmp_path / "fpl" / f"teams_{SEASON}.csv", SEASON)
    want = load_fpl(frozen_dir / "fpl" / f"fixtures_{SEASON}.csv", frozen_dir / "fpl" / f"teams_{SEASON}.csv", SEASON)
    assert got == want and len(got) == 380 and sum(x.played for x in got) == 60


def test_api_down_falls_back_to_the_github_mirror_of_the_same_feed(tmp_path, fake_get, frozen_dir):
    m = fetch.fetch_current(tmp_path, SEASON, get=fake_get(fail=("api",)))
    fpl = m["sources"]["fpl"]
    assert fpl["ok"] and fpl["origin"] == "mirror"
    assert [t["origin"] for t in fpl["tried"]] == ["api", "mirror"] and not fpl["tried"][0]["ok"]
    got = load_fpl(tmp_path / "fpl" / f"fixtures_{SEASON}.csv", tmp_path / "fpl" / f"teams_{SEASON}.csv", SEASON)
    assert got == load_fpl(frozen_dir / "fpl" / f"fixtures_{SEASON}.csv", frozen_dir / "fpl" / f"teams_{SEASON}.csv", SEASON)


def test_all_three_sources_round_trip(tmp_path, fake_get, frozen_dir, data_dir):
    m = fetch.fetch_current(tmp_path, SEASON, get=fake_get())
    assert all(s["ok"] for s in m["sources"].values())
    assert load_mirror(tmp_path / f"fei_{SEASON}", SEASON) == load_mirror(frozen_dir / f"fei_{SEASON}", SEASON)
    assert load_openfootball(tmp_path / "openfootball" / f"{SEASON}.json", SEASON) == \
        load_openfootball(frozen_dir / "openfootball" / f"{SEASON}.json", SEASON)
    fresh = [x for x in pipeline.load_all(data_dir, current_dir=tmp_path) if x.season == SEASON]
    frozen = [x for x in pipeline.load_all(data_dir, current_dir=frozen_dir) if x.season == SEASON]
    assert fresh == frozen


def test_only_facts_are_saved(tmp_path, fake_get):
    """The stand-in serves extra columns (player prices, ratings, difficulty). None of them reach disk."""
    fetch.fetch_current(tmp_path, SEASON, get=fake_get(fail=("api",)))
    assert _header(tmp_path / "fpl" / f"fixtures_{SEASON}.csv") == fetch.FPL_FIXTURE_FIELDS
    assert _header(tmp_path / "fpl" / f"teams_{SEASON}.csv") == fetch.FPL_TEAM_FIELDS
    assert _header(tmp_path / f"fei_{SEASON}" / "pl_matches_gw1.csv") == fetch.MIRROR_MATCH_FIELDS
    assert _header(tmp_path / f"fei_{SEASON}" / "teams.csv") == fetch.MIRROR_TEAM_FIELDS
    on_disk = " ".join(p.read_text(encoding="utf-8") for p in tmp_path.rglob("*.csv"))
    assert "home_team_elo" not in on_disk and "pulse_id" not in on_disk and "strength" not in on_disk


def test_api_rows_drop_everything_but_the_facts():
    fixtures = [{"code": 1, "event": 1, "finished": True, "finished_provisional": True, "id": 9, "kickoff_time": "2026-08-21T19:00:00Z",
                 "team_a": 2, "team_a_score": 0, "team_h": 1, "team_h_score": 3, "stats": [{"x": 1}], "minutes": 90},
                {"code": 2, "event": None, "finished": False, "finished_provisional": False, "id": 10, "kickoff_time": None,
                 "team_a": 1, "team_a_score": None, "team_h": 2, "team_h_score": None, "stats": [], "minutes": 0}]
    teams = {"teams": [{"code": 3, "id": 1, "name": "Arsenal", "short_name": "ARS", "position": 1, "strength": 5}], "elements": [1, 2]}
    fx, tm = fetch.fpl_api_rows(fixtures, teams)
    assert list(fx[0]) == fetch.FPL_FIXTURE_FIELDS and list(tm[0]) == fetch.FPL_TEAM_FIELDS
    assert (fx[0]["finished"], fx[1]["finished"], fx[1]["team_h_score"]) == ("True", "False", None)


@pytest.mark.parametrize("bad", [{"fixtures": []}, "not json at all"])
def test_api_rows_reject_an_unexpected_shape(bad):
    with pytest.raises((ValueError, TypeError)):
        fetch.fpl_api_rows(bad, {"teams": []})


def test_a_source_that_drops_a_field_we_need_is_marked_failed(tmp_path, fake_get):
    def edit(url, rows):
        if "FPL-Elo-Insights" in url and url.endswith("matches.csv"):
            return [{k: v for k, v in r.items() if k != "home_score"} for r in rows]
        return rows
    m = fetch.fetch_current(tmp_path, SEASON, get=fake_get(edit=edit))
    assert not m["sources"]["mirror"]["ok"] and "home_score" in m["sources"]["mirror"]["error"]
    assert m["sources"]["fpl"]["ok"] and m["sources"]["openfootball"]["ok"]
    assert not (tmp_path / f"fei_{SEASON}").exists()


def test_a_short_fixture_list_is_refused(tmp_path, fake_get):
    def edit(url, data):
        if url.endswith("/fixtures/") or url.endswith("fixtures.csv"):
            return data[:-1]
        if "openfootball" in url:
            return {**data, "matches": data["matches"][:-1]}
        return data
    m = fetch.fetch_current(tmp_path, SEASON, get=fake_get(edit=edit))
    assert not m["sources"]["fpl"]["ok"] and "379" in m["sources"]["fpl"]["tried"][0]["error"]
    assert not m["sources"]["openfootball"]["ok"]
    assert not (tmp_path / "fpl").exists() and not (tmp_path / "openfootball").exists()


def test_an_unreachable_source_keeps_its_previous_files(tmp_path, fake_get):
    fetch.fetch_current(tmp_path, SEASON, get=fake_get())
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file() and p.name != "manifest.json"}
    m = fetch.fetch_current(tmp_path, SEASON, get=fake_get(fail=("api", "vaastav", "mirror", "openfootball")))
    assert [s["ok"] for s in m["sources"].values()] == [False, False, False]
    assert {p: p.read_bytes() for p in before} == before
    assert json.loads((tmp_path / "manifest.json").read_text())["sources"]["mirror"]["ok"] is False


def test_manifest_records_when_and_what(tmp_path, fake_get):
    from datetime import datetime, timezone
    m = fetch.fetch_current(tmp_path, SEASON, get=fake_get(), now=datetime(2026, 11, 22, 7, 0, tzinfo=timezone.utc))
    assert m["fetched_at"] == "2026-11-22T07:00:00Z" and m["fetched_on"] == "2026-11-22" and m["season"] == SEASON
    assert all(len(s["sha"]) == 16 for s in m["sources"].values())
    again = fetch.fetch_current(tmp_path, SEASON, get=fake_get())
    assert {k: v["sha"] for k, v in again["sources"].items()} == {k: v["sha"] for k, v in m["sources"].items()}
    assert m["sources"]["mirror"]["gameweek_files"] == 38


def test_whistle_results_count_before_the_feed_confirms_them(tmp_path, fake_get, frozen_dir):
    """`finished` turns true hours after full time. `finished_provisional` turns true at the whistle."""
    def edit(url, data):
        if url.endswith("/fixtures/"):
            first_unplayed = next(fx for fx in data if not fx["finished"])
            first_unplayed.update(finished_provisional=True, team_h_score=2, team_a_score=2)
        return data
    fetch.fetch_current(tmp_path, SEASON, get=fake_get(edit=edit))
    got = load_fpl(tmp_path / "fpl" / f"fixtures_{SEASON}.csv", tmp_path / "fpl" / f"teams_{SEASON}.csv", SEASON)
    assert sum(x.played for x in got) == 61
