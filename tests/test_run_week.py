"""The weekly command end to end, and the files it writes.

Runs on the frozen season with 2,000 simulated seasons to stay quick. The
checks are about wiring and bookkeeping, which do not depend on the count.
"""

import json
import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from eplsim import clubs, config, gates, metrics, outputs, pipeline, run_week

NOW = datetime(2026, 11, 11, 7, 0, tzinfo=timezone.utc)
SIMS = 2_000
QUIET = {"log": lambda *_: None}


def _strict(text: str):
    def bad(x):
        raise ValueError(f"not valid JSON: {x}")
    return json.loads(text, parse_constant=bad)


@pytest.fixture(scope="module")
def week(tmp_path_factory, data_dir, frozen_dir):
    base = tmp_path_factory.mktemp("week")
    out = run_week.run(data_dir, base / "outputs", base / "site", current_dir=frozen_dir, do_fetch=False, now=NOW,
                       n_sims=SIMS, **QUIET)
    return {"base": base, "out": base / "outputs", "site": base / "site", "result": out,
            "feed_text": (base / "site" / "feed.json").read_text(encoding="utf-8")}


@pytest.fixture(scope="module")
def feed(week):
    return _strict(week["feed_text"])


def _rerun(week, data_dir, frozen_dir, **kw):
    """Run again in a copy of the first run's folders, so tests cannot disturb each other."""
    base = week["base"].parent / f"copy_{len(list(week['base'].parent.iterdir()))}"
    shutil.copytree(week["base"], base)
    args = {"current_dir": frozen_dir, "do_fetch": False, "now": NOW, "n_sims": SIMS, **QUIET, **kw}
    return base, lambda: run_week.run(data_dir, base / "outputs", base / "site", **args)


def test_first_run_publishes(week):
    assert week["result"]["changed"] and week["result"]["matchweek"] == 10 and week["result"]["matches_played"] == 100
    for f in ("history.parquet", "forecast_log.parquet", "seasons_latest.parquet", "run_meta.json"):
        assert (week["out"] / f).exists(), f


def test_feed_has_every_section_the_page_draws(feed):
    assert set(feed) == {"meta", "clubs", "matrix", "tiers", "table_lies", "magic_numbers", "points_shape", "run_in",
                         "next", "swings", "wildest", "convergence", "duel", "movers", "universes", "report_card"}
    meta = feed["meta"]
    assert (meta["season"], meta["matchweek"], meta["matches_played"], meta["matches_remaining"]) == ("2026-27", 10, 100, 280)
    assert (meta["n_sims"], meta["seed"], meta["sigma"]) == (SIMS, config.SEED, config.SIGMA)
    assert meta["data_through"] == "2026-11-08" and meta["generated_at"] == "2026-11-11T07:00:00Z"
    assert [g["gate"] for g in meta["gates"]][:1] == ["fixture_list"] and not any(g["status"] == "fail" for g in meta["gates"])
    assert meta["badge_colours_verified"] is clubs.COLOURS_VERIFIED
    assert set(meta["tiebreak"]) >= {"groups", "left_to_a_play_off"}


def test_feed_clubs_carry_odds_and_badge_colours(feed):
    assert len(feed["clubs"]) == 20
    exp = [c["exp_pts"] for c in feed["clubs"]]
    assert exp == sorted(exp, reverse=True)
    for c in feed["clubs"]:
        assert set(c) >= {"team", "code", "primary", "secondary", "table_pos", "pts_now", "exp_pts", "title", "top4", "top5", "relegated"}
        assert len(c["code"]) == 3 and c["primary"].startswith("#") and len(c["primary"]) == 7
        assert 0 <= c["title"] <= c["top4"] <= c["top5"] <= 1 and 0 <= c["relegated"] <= 1
    assert sum(c["title"] for c in feed["clubs"]) == pytest.approx(1.0, abs=2e-3)
    assert sum(c["relegated"] for c in feed["clubs"]) == pytest.approx(3.0, abs=2e-3)
    assert len({c["code"] for c in feed["clubs"]}) == 20


def test_feed_matrix_is_square_and_sums_to_one(feed):
    m = np.array(feed["matrix"]["probs"])
    assert m.shape == (20, 20) and feed["matrix"]["teams"] == [c["team"] for c in feed["clubs"]]
    assert m.sum(axis=1) == pytest.approx(np.ones(20), abs=2e-3) and m.sum(axis=0) == pytest.approx(np.ones(20), abs=2e-3)
    for row, c in zip(m, feed["clubs"]):
        assert row[0] == pytest.approx(c["title"], abs=1e-4) and row[:4].sum() == pytest.approx(c["top4"], abs=5e-4)


def test_feed_numbers_equal_the_parquet_store(week, feed):
    seasons = pd.read_parquet(week["out"] / "seasons_latest.parquet")
    assert len(seasons) == SIMS and seasons["season"].tolist() == list(range(1, SIMS + 1))
    for c in feed["clubs"]:
        pos, pts = seasons[f"pos_{c['code']}"], seasons[f"pts_{c['code']}"]
        assert (pos == 1).mean() == pytest.approx(c["title"], abs=1e-4)
        assert (pos <= 4).mean() == pytest.approx(c["top4"], abs=1e-4)
        assert (pos <= 5).mean() == pytest.approx(c["top5"], abs=1e-4)
        assert (pos >= 18).mean() == pytest.approx(c["relegated"], abs=1e-4)
        assert pts.mean() == pytest.approx(c["exp_pts"], abs=1e-4)
    pos_cols = [c for c in seasons if c.startswith("pos_")]
    assert (np.sort(seasons[pos_cols].to_numpy(), axis=1) == np.arange(1, 21)).all()


def test_universe_lookup_on_the_page_equals_the_parquet_store(week, feed):
    seasons = pd.read_parquet(week["out"] / "seasons_latest.parquet")
    u = feed["universes"]
    assert len(u["champion"]) == SIMS and len(u["relegated"]) == 3 * SIMS and len(u["teams"]) == 20
    for n in (1, 2, 777, SIMS):
        row = seasons.iloc[n - 1]
        champ = u["teams"][ord(u["champion"][n - 1]) - 65]
        assert row[f"pos_{clubs.code(champ)}"] == 1
        down = [u["teams"][ord(ch) - 65] for ch in u["relegated"][3 * (n - 1):3 * n]]
        assert [int(row[f"pos_{clubs.code(t)}"]) for t in down] == [18, 19, 20]


def test_history_holds_one_row_per_club(week, feed):
    h = pd.read_parquet(week["out"] / "history.parquet")
    assert len(h) == 20 and set(h["matchweek"]) == {10} and set(h["matches_played"]) == {100}
    by = h.set_index("team")
    for c in feed["clubs"]:
        assert by.loc[c["team"], "title"] == pytest.approx(c["title"], abs=1e-4)


def test_forecast_log_holds_next_weeks_odds(week, feed):
    log = pd.read_parquet(week["out"] / "forecast_log.parquet")
    assert len(log) == 10 and set(log["gameweek"]) == {11}
    assert (log[["p_home", "p_draw", "p_away"]].sum(axis=1) - 1).abs().max() < 1e-6
    assert set(zip(log["home"], log["away"])) == {(f["home"], f["away"]) for f in feed["next"]["fixtures"]}
    assert feed["report_card"] == {"matches": 0}  # nothing forecast before kick-off has been played yet


def test_swings_cover_every_fixture_of_the_next_gameweek(feed):
    s = feed["swings"]
    assert len(s["fixtures"]) == 10 and s["biggest"]["team"] in {c["team"] for c in feed["clubs"]}
    assert abs(s["biggest"]["swing"]) >= max(abs(r["swing"]) for e in s["fixtures"] for c in e["clubs"].values() for r in c.values()) - 1e-9


def test_nothing_new_means_nothing_published(week, data_dir, frozen_dir):
    base, run = _rerun(week, data_dir, frozen_dir, now=datetime(2026, 11, 12, 7, 0, tzinfo=timezone.utc))
    assert run() == {"changed": False, "matches_played": 100}
    assert (base / "site" / "feed.json").read_text(encoding="utf-8") == week["feed_text"]


def test_a_forced_rerun_gives_the_same_numbers(week, data_dir, frozen_dir, feed):
    base, run = _rerun(week, data_dir, frozen_dir, force=True)
    assert run()["changed"]
    again = _strict((base / "site" / "feed.json").read_text(encoding="utf-8"))
    gates_now = {g["gate"]: g["status"] for g in again["meta"].pop("gates")}
    gates_then = {g["gate"]: g["status"] for g in feed["meta"]["gates"]}
    assert (gates_then["results_never_vanish"], gates_now["results_never_vanish"]) == ("skipped", "pass")
    assert again == {**feed, "meta": {k: v for k, v in feed["meta"].items() if k != "gates"}}
    assert len(pd.read_parquet(base / "outputs" / "history.parquet")) == 20  # replaced, not appended


def test_results_that_vanish_stop_the_run_and_leave_the_page_alone(week, data_dir, frozen_dir):
    base, run = _rerun(week, data_dir, frozen_dir, force=True)
    meta = json.loads((base / "outputs" / "run_meta.json").read_text())
    (base / "outputs" / "run_meta.json").write_text(json.dumps({**meta, "matches_played": 110}))
    with pytest.raises(gates.GateError, match="results_never_vanish"):
        run()
    assert (base / "site" / "feed.json").read_text(encoding="utf-8") == week["feed_text"]


def test_an_overdue_result_stops_the_run(week, data_dir, frozen_dir):
    _, run = _rerun(week, data_dir, frozen_dir, force=True, now=datetime(2026, 11, 25, 7, 0, tzinfo=timezone.utc))
    with pytest.raises(gates.GateError, match="no_stale_results"):
        run()


def test_unreachable_sources_stop_the_run(week, data_dir, frozen_dir, fake_get):
    base, _ = _rerun(week, data_dir, frozen_dir)
    current = base / "current"
    shutil.copytree(frozen_dir, current)
    dead = fake_get(fail=("api", "vaastav", "mirror", "openfootball"))
    with pytest.raises(gates.GateError, match="sources_reachable"):
        run_week.run(data_dir, base / "outputs", base / "site", current_dir=current, do_fetch=True, now=NOW,
                     n_sims=SIMS, get=dead, **QUIET)


def test_a_run_with_a_live_fetch_matches_the_offline_run(week, data_dir, fake_get, feed):
    base = week["base"].parent / "online"
    out = run_week.run(data_dir, base / "outputs", base / "site", current_dir=base / "current", do_fetch=True, now=NOW,
                       n_sims=SIMS, get=fake_get(), **QUIET)
    assert out["changed"]
    online = out["feed"]
    assert online["meta"]["fpl_origin"] == "api"
    assert [g["status"] for g in online["meta"]["gates"] if g["gate"] == "sources_reachable"] == ["pass"]
    assert online["clubs"] == feed["clubs"] and online["matrix"] == feed["matrix"]


def test_movers_compare_with_the_previous_matchweek(week, data_dir, frozen_dir, feed):
    base, run = _rerun(week, data_dir, frozen_dir, force=True)
    h = pd.read_parquet(base / "outputs" / "history.parquet")
    earlier = h.copy()
    earlier["matchweek"], earlier["matches_played"] = 9, 90
    leader = feed["clubs"][0]["team"]
    earlier.loc[earlier["team"] == leader, "title"] -= 0.10
    earlier.loc[earlier["team"] == feed["clubs"][1]["team"], "title"] += 0.02  # inside the noise, must not show
    pd.concat([earlier, h]).to_parquet(base / "outputs" / "history.parquet", index=False)
    moved = run()["feed"]["movers"]
    assert moved["since_matchweek"] == 9
    assert [(r["team"], r["race"]) for r in moved["rows"]] == [(leader, "title")]
    assert moved["rows"][0]["change"] == pytest.approx(0.10, abs=1e-3)
    assert len(pd.read_parquet(base / "outputs" / "history.parquet")) == 40


def test_report_card_scores_forecasts_logged_before_kick_off(week, data_dir, frozen_dir, matches):
    base, run = _rerun(week, data_dir, frozen_dir, force=True)
    log = pd.read_parquet(base / "outputs" / "forecast_log.parquet")
    done = next(m for m in matches if m.season == "2026-27" and m.played and m.home_goals > m.away_goals)
    old = log.iloc[[0]].copy()
    old["home"], old["away"], old["gameweek"] = done.home, done.away, done.gameweek
    old["p_home"], old["p_draw"], old["p_away"] = 0.5, 0.3, 0.2
    pd.concat([old, log]).to_parquet(base / "outputs" / "forecast_log.parquet", index=False)
    card = run()["feed"]["report_card"]
    assert card["matches"] == 1 and card["accuracy"] == 1.0
    assert card["log_loss"] == pytest.approx(-np.log(0.5), abs=1e-4)
    assert len(pd.read_parquet(base / "outputs" / "forecast_log.parquet")) == 11


def test_archive_keeps_a_permanent_copy(week, data_dir, frozen_dir):
    base, run = _rerun(week, data_dir, frozen_dir, force=True, archive="launch")
    run()
    assert (base / "outputs" / "archive" / "seasons_launch.parquet").exists()
    assert (base / "outputs" / "archive" / "feed_launch.json").exists()


def test_command_line_reports_to_the_workflow(week, data_dir, frozen_dir, tmp_path, monkeypatch, capsys):
    gh = tmp_path / "gh_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(gh))
    args = ["--data", str(data_dir), "--out", str(tmp_path / "o"), "--site", str(tmp_path / "s"), "--current", str(frozen_dir),
            "--offline", "--sims", "300", "--today", "2026-11-11"]
    assert run_week.main(args) == 0
    assert gh.read_text() == "changed=true\nmatchweek=10\nplayed=100\n"
    # the first run of a season fills in every earlier matchweek on its own
    assert sorted(p.name for p in (tmp_path / "s" / "snapshots").iterdir()) == [f"mw{n:02d}.json" for n in range(1, 11)]
    t = _strict((tmp_path / "s" / "history.json").read_text())
    assert [w["reconstructed"] for w in t["matchweeks"]] == [True] * 9 + [False]
    assert _strict((tmp_path / "s" / "feed.json").read_text())["movers"]["since_matchweek"] == 9
    assert run_week.main(args) == 0
    assert gh.read_text().endswith("changed=false\n")
    assert run_week.main(args + ["--today", "2026-12-25", "--force"]) == 2
    assert "STOPPED by a gate" in capsys.readouterr().err


def test_default_run_size_is_the_frozen_setting():
    import inspect
    assert inspect.signature(run_week.run).parameters["n_sims"].default == config.N_SIMS == 10_000


def test_upsert_replaces_rows_with_the_same_key(tmp_path):
    p = tmp_path / "t.parquet"
    outputs.upsert(p, [{"k": 1, "v": 1.0}, {"k": 2, "v": 2.0}], key=["k"], sort=["k"])
    out = outputs.upsert(p, [{"k": 2, "v": 9.0}, {"k": 3, "v": 3.0}], key=["k"], sort=["k"])
    assert out.to_dict("records") == [{"k": 1, "v": 1.0}, {"k": 2, "v": 9.0}, {"k": 3, "v": 3.0}]
    assert outputs.upsert(p, [], key=["k"], sort=["k"]).to_dict("records") == out.to_dict("records")


def test_feed_rounding_turns_numpy_into_plain_json():
    raw = {"a": np.float64(0.123456789), "b": [np.int64(3), np.bool_(True)], "c": (1.0, "x"), 5: None}
    assert outputs.round_floats(raw) == {"a": 0.1235, "b": [3, True], "c": [1.0, "x"], "5": None}
    json.dumps(outputs.round_floats(raw))


def test_full_size_feed_matches_the_headline_numbers(snap, result):
    """The same numbers the page shows, straight from 10,000 seasons, for the launch card."""
    rows = {r["team"]: r for r in metrics.headline(snap, result)}
    assert rows["Liverpool"]["title"] == pytest.approx(0.0539, abs=1e-4)
    assert rows["Arsenal"]["title"] == pytest.approx(0.6631, abs=1e-4)


def test_fixture_order_in_the_source_does_not_change_the_seasons(matches, snap):
    """Found on 12 Nov: the live feed listed fixtures in a different order from the frozen copy,
    and the same results gave Liverpool 540 titles instead of 575. Same data must give the same answer."""
    import random
    shuffled = list(matches)
    random.Random(3).shuffle(shuffled)
    cur = [m for m in shuffled if m.season == "2026-27"]
    rest = [m for m in matches if m.season != "2026-27"]
    other = pipeline.snapshot(rest + cur, config.ELO, config.GOALS_WINDOW)
    assert other.remaining == snap.remaining == sorted(snap.remaining)
    a = pipeline.run_sim(snap, n_sims=500, sigma=config.SIGMA, seed=config.SEED)
    b = pipeline.run_sim(other, n_sims=500, sigma=config.SIGMA, seed=config.SEED)
    assert np.array_equal(a.position, b.position)


# ---- the season as a data model: one snapshot per matchweek, kept forever ----

def test_every_run_leaves_a_snapshot_and_a_timeline(week, feed):
    snap_file = week["site"] / "snapshots" / "mw10.json"
    assert _strict(snap_file.read_text(encoding="utf-8")) == feed
    t = _strict((week["site"] / "history.json").read_text(encoding="utf-8"))
    assert t["matchweeks"] == [{"matchweek": 10, "matches_played": 100, "data_through": "2026-11-08", "reconstructed": False}]
    assert len(t["clubs"]) == 20
    for c in feed["clubs"]:
        s = t["clubs"][c["team"]]
        assert (s["table_pos"], s["pts_now"]) == ([c["table_pos"]], [c["pts_now"]])
        assert s["title"][0] == pytest.approx(c["title"], abs=1e-4) and s["top4"][0] == pytest.approx(c["top4"], abs=1e-4)
    assert sorted(t["clubs"][c["team"]]["model_pos"][0] for c in feed["clubs"]) == list(range(1, 21))


def test_position_history_holds_every_club_and_place(week, feed):
    d = pd.read_parquet(week["out"] / "position_dist.parquet")
    assert len(d) == 400 and set(d["place"]) == set(range(1, 21))
    assert d.groupby("team")["share"].sum().sub(1).abs().max() < 1e-9
    title = d[d["place"] == 1].set_index("team")["share"]
    for c in feed["clubs"]:
        assert title[c["team"]] == pytest.approx(c["title"], abs=1e-4)


@pytest.fixture(scope="module")
def rebuilt(week, data_dir, frozen_dir):
    base, _ = _rerun(week, data_dir, frozen_dir)
    before = {f: (base / f).read_bytes() for f in ("site/feed.json", "outputs/run_meta.json", "outputs/forecast_log.parquet",
                                                    "outputs/seasons_latest.parquet")}
    out = run_week.run(data_dir, base / "outputs", base / "site", current_dir=frozen_dir, now=NOW, n_sims=SIMS, as_of=7, **QUIET)
    return {"base": base, "before": before, "out": out}


def test_rebuilding_a_matchweek_uses_only_results_up_to_it(rebuilt, matches):
    meta = rebuilt["out"]["feed"]["meta"]
    assert (meta["matchweek"], meta["matches_played"], meta["matches_remaining"]) == (7, 70, 310)
    assert meta["reconstructed"] is True
    last = max(m.date for m in matches if m.season == "2026-27" and m.played and m.gameweek <= 7)
    assert meta["data_through"] == last
    pts = {c["team"]: c["pts_now"] for c in rebuilt["out"]["feed"]["clubs"]}
    from eplsim.data import league_table
    table = league_table([m for m in matches if m.season == "2026-27" and m.gameweek <= 7])
    assert pts == {r["team"]: r["pts"] for r in table}
    assert rebuilt["out"]["feed"]["next"]["gameweek"] == 8


def test_rebuilding_never_touches_the_live_page_or_the_forecast_log(rebuilt):
    base = rebuilt["base"]
    for f, content in rebuilt["before"].items():
        assert (base / f).read_bytes() == content, f
    assert (base / "site" / "snapshots" / "mw07.json").exists()
    assert _strict((base / "site" / "snapshots" / "mw10.json").read_text())["meta"]["reconstructed"] is False


def test_timeline_grows_by_one_matchweek_in_order(rebuilt, feed):
    base = rebuilt["base"]
    t = _strict((base / "site" / "history.json").read_text(encoding="utf-8"))
    assert [(w["matchweek"], w["matches_played"], w["reconstructed"]) for w in t["matchweeks"]] == [(7, 70, True), (10, 100, False)]
    for series in t["clubs"].values():
        assert all(len(v) == 2 for v in series.values())
    liv = t["clubs"]["Liverpool"]
    assert liv["title"][1] == pytest.approx(next(c for c in feed["clubs"] if c["team"] == "Liverpool")["title"], abs=1e-4)
    assert liv["pts_now"][0] < liv["pts_now"][1]
    assert len(pd.read_parquet(base / "outputs" / "history.parquet")) == 40
    assert len(pd.read_parquet(base / "outputs" / "position_dist.parquet")) == 800


def test_timeline_keeps_the_fullest_run_of_a_matchweek():
    def rows(mw, played, title):
        return [{"season": "s", "matchweek": mw, "matches_played": played, "data_through": "d", "reconstructed": False,
                 "team": t, "table_pos": i + 1, "pts_now": played, "model_pos": i + 1, "exp_pts": 50.0, "elo": 1500.0,
                 "title": title, "top4": 0.5, "top5": 0.6, "relegated": 0.1} for i, t in enumerate(("A", "B"))]
    h = pd.DataFrame(rows(11, 105, 0.2) + rows(11, 110, 0.3) + rows(10, 100, 0.1) + [{**rows(11, 120, 0.9)[0], "season": "other"}])
    t = outputs.timeline(h, "s")
    assert [(w["matchweek"], w["matches_played"]) for w in t["matchweeks"]] == [(10, 100), (11, 110)]
    assert t["clubs"]["A"]["title"] == [0.1, 0.3] and t["clubs"]["B"]["table_pos"] == [2.0, 2.0]


def test_backfill_rebuilds_every_earlier_matchweek(data_dir, frozen_dir, tmp_path):
    rebuilt = run_week.backfill(data_dir, tmp_path / "o", tmp_path / "s", current_dir=frozen_dir, now=NOW, n_sims=200, **QUIET)
    assert rebuilt == list(range(1, 10))
    assert sorted(p.name for p in (tmp_path / "s" / "snapshots").iterdir()) == [f"mw{n:02d}.json" for n in range(1, 10)]
    t = _strict((tmp_path / "s" / "history.json").read_text())
    assert [w["matches_played"] for w in t["matchweeks"]] == [10 * n for n in range(1, 10)]
    assert not (tmp_path / "s" / "feed.json").exists()


def test_gap_filling_never_replaces_a_snapshot_made_at_the_time(week, data_dir, frozen_dir):
    base, _ = _rerun(week, data_dir, frozen_dir)
    kw = dict(current_dir=frozen_dir, now=NOW, n_sims=SIMS, **QUIET)
    assert run_week.backfill(data_dir, base / "outputs", base / "site", only_missing=True, **kw) == list(range(1, 10))
    assert run_week.backfill(data_dir, base / "outputs", base / "site", only_missing=True, **kw) == []
    snaps = pd.read_parquet(base / "outputs" / "snapshots.parquet")
    assert snaps["matchweek"].tolist() == list(range(1, 11))
    assert snaps["reconstructed"].tolist() == [True] * 9 + [False]


def test_finished_matchweeks_respect_postponed_matches(matches):
    from dataclasses import replace
    cur = [m for m in matches if m.season == "2026-27"]
    assert run_week.finished_matchweeks(cur) == list(range(1, 11))
    i = next(k for k, m in enumerate(cur) if m.gameweek == 4)
    gone = replace(cur[i], home_goals=None, away_goals=None)
    hole = cur[:i] + [gone] + cur[i + 1:]
    assert 4 not in run_week.finished_matchweeks(hole)
    assert 4 in run_week.finished_matchweeks(hole, postponed={(gone.home, gone.away)})


def test_snapshot_table_records_how_each_run_was_made(week, feed):
    snaps = pd.read_parquet(week["out"] / "snapshots.parquet")
    assert len(snaps) == 1
    row = snaps.iloc[0]
    assert (row["season"], row["matches_played"], row["matchweek"], bool(row["reconstructed"])) == ("2026-27", 100, 10, False)
    assert (row["n_sims"], row["seed"], row["sigma"], row["elo_k"]) == (SIMS, config.SEED, config.SIGMA, config.ELO.k)
    assert row["model_version"] == outputs.model_version(feed["meta"]) and len(row["model_version"]) == 10
    assert outputs.model_version({**feed["meta"], "sigma": 35.0}) != row["model_version"]


def test_history_holds_the_real_table_beside_the_model(week, matches):
    from eplsim.data import league_table
    h = pd.read_parquet(week["out"] / "history.parquet").set_index("team")
    for r in league_table([m for m in matches if m.season == "2026-27"]):
        got = h.loc[r["team"]]
        assert (got["won"], got["drawn"], got["lost"], got["goals_for"], got["goals_against"], got["pts_now"]) == \
            (r["w"], r["d"], r["l"], r["gf"], r["ga"], r["pts"])


def test_the_workflow_commits_every_file_the_run_writes(week, root):
    """Found on 12 Nov: snapshots were written but never committed, so past matchweeks
    would have disappeared from the live page after the next run."""
    import re
    yml = (root / ".github" / "workflows" / "weekly.yml").read_text(encoding="utf-8")
    added = re.search(r"git add ([^\n]+)", yml).group(1).split()
    written = [p.relative_to(week["base"]) for p in week["base"].rglob("*") if p.is_file()]
    for rel in written:
        path = ("outputs" if rel.parts[0] == "outputs" else "site") + "/" + "/".join(rel.parts[1:])
        assert any(path == a or path.startswith(a.rstrip("/") + "/") for a in added), path
