"""The check that decides whether the full run starts."""

import ast
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from eplsim import trigger

ROOT = Path(__file__).resolve().parents[1]
UTC = timezone.utc


def fx(gw, home, away, kickoff):
    return {"gameweek": gw, "home": home, "away": away,
            "kickoff": datetime.fromisoformat(kickoff).replace(tzinfo=UTC) if kickoff else None}


FIXTURES = [
    fx(10, "1", "2", "2026-11-08T16:30:00"),
    fx(11, "3", "4", "2026-11-21T12:30:00"),
    fx(11, "5", "6", "2026-11-23T20:00:00"),
    fx(12, "1", "3", "2026-11-29T16:30:00"),
]


def feed(finished_by_event):
    """A fake live feed: {gameweek: [finished flags]}. Counts how often it is asked."""
    calls = []

    def get(url):
        calls.append(url)
        rows = [{"event": g, "finished": False, "finished_provisional": f,
                 "team_h_score": 1 if f else None, "team_a_score": 0 if f else None}
                for g, flags in finished_by_event.items() for f in flags]
        return json.dumps(rows).encode()
    get.calls = calls
    return get


def at(s):
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


def test_international_break_stops_without_asking_the_feed():
    get = feed({11: [False, False]})
    d = trigger.decide(FIXTURES, 10, at("2026-11-17T07:00:00"), get=get)
    assert not d["due"] and d["gameweek"] == 11 and "not over yet" in d["reason"]
    assert get.calls == []


def test_mid_gameweek_stops_without_asking_the_feed():
    get = feed({11: [True, False]})
    d = trigger.decide(FIXTURES, 10, at("2026-11-22T23:00:00"), get=get)
    assert not d["due"] and get.calls == []


def test_last_whistle_needs_the_full_margin():
    # last kick-off 20:00, so the gameweek counts as over from 22:30
    assert trigger.decide(FIXTURES, 10, at("2026-11-23T22:29:00"), get=feed({11: [True, True]}))["due"] is False
    assert trigger.decide(FIXTURES, 10, at("2026-11-23T22:30:00"), get=feed({11: [True, True]}))["due"] is True


def test_finished_gameweek_starts_the_run_once_the_feed_agrees():
    get = feed({11: [True, True]})
    d = trigger.decide(FIXTURES, 10, at("2026-11-24T07:00:00"), get=get)
    assert d == {"due": True, "gameweek": 11, "reason": "gameweek 11 finished, starting the full run"}
    assert len(get.calls) == 1


def test_feed_not_caught_up_waits_for_the_next_check():
    d = trigger.decide(FIXTURES, 10, at("2026-11-24T07:00:00"), get=feed({11: [True, False]}))
    assert not d["due"] and "not closed it yet" in d["reason"]


def test_unreadable_feed_runs_anyway_so_the_gates_decide():
    def broken(url):
        raise RuntimeError("GET failed: timed out")
    d = trigger.decide(FIXTURES, 10, at("2026-11-24T07:00:00"), get=broken)
    assert d["due"] and "gates decide" in d["reason"]


def test_published_gameweek_is_not_run_again():
    get = feed({11: [True, True]})
    d = trigger.decide(FIXTURES, 11, at("2026-11-25T07:00:00"), get=get)
    assert not d["due"] and d["gameweek"] == 12 and get.calls == []


def test_a_missed_gameweek_catches_up_on_the_latest():
    get = feed({11: [True, True], 12: [True]})
    d = trigger.decide(FIXTURES, 10, at("2026-12-01T07:00:00"), get=get)
    assert d["due"] and d["gameweek"] == 12


def test_postponed_and_undated_fixtures_do_not_hold_a_gameweek_open():
    late = FIXTURES + [fx(11, "7", "8", "2027-02-10T20:00:00"), fx(11, "9", "10", None)]
    assert trigger.gameweek_ends(late, postponed={("7", "8")})[11] == at("2026-11-23T22:30:00")
    assert trigger.gameweek_ends(late)[11] == at("2027-02-10T22:30:00")


def test_runs_started_by_hand_or_push_always_go_ahead():
    for event in ("push", "workflow_dispatch"):
        assert trigger.decide(FIXTURES, 10, at("2026-11-17T07:00:00"), event=event)["due"]


def test_whistle_without_a_score_is_not_finished():
    def get(url):
        return json.dumps([{"event": 11, "finished": True, "team_h_score": None, "team_a_score": None}]).encode()
    assert trigger.feed_says_finished(11, get) is False


def test_season_over():
    d = trigger.decide(FIXTURES, 12, at("2027-06-01T07:00:00"), get=feed({}))
    assert not d["due"] and "season over" in d["reason"]


def test_real_fixture_list_during_the_international_break():
    fixtures = trigger.load_fixtures(ROOT / "data" / "current" / "fpl" / "fixtures_2026-27.csv")
    assert len(fixtures) == 380
    ends = trigger.gameweek_ends(fixtures)
    assert len(ends) == 38 and ends[11] == at("2026-11-23T22:30:00")
    d = trigger.decide(fixtures, 10, at("2026-11-17T07:00:00"), get=feed({}))
    assert not d["due"] and d["gameweek"] == 11


def test_main_writes_the_decision_for_the_workflow(tmp_path, monkeypatch, capsys):
    out = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    assert trigger.main(["--offline", "--event", "schedule", "--now", "2026-11-17T07:00:00Z"]) == 0
    assert out.read_text() == "due=false\ngameweek=11\n"
    assert "not over yet" in capsys.readouterr().out


def test_check_uses_only_the_standard_library():
    """The check runs before anything is installed, so it must not import a package."""
    allowed = {"__future__", "argparse", "csv", "json", "os", "sys", "collections", "datetime", "pathlib",
               "hashlib", "io", "time", "urllib", "fetch"}
    for name in ("trigger.py", "fetch.py"):
        tree = ast.parse((ROOT / "src" / "eplsim" / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [(node.module or "").split(".")[0] or "fetch"]
            else:
                continue
            assert set(mods) <= allowed, f"{name} imports {mods}"


def test_workflow_runs_the_check_before_the_full_run():
    wf = (ROOT / ".github" / "workflows" / "weekly.yml").read_text()
    assert "python3 -m eplsim.trigger" in wf
    assert "needs: check" in wf and "needs.check.outputs.due == 'true'" in wf
    assert '"0 7 * * *"' not in wf
