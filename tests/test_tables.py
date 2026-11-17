"""The dashboard tables: keys, sums, and agreement with the tables the model already writes.

Runs the frozen season once live and rebuilds two earlier matchweeks, with
2,000 simulated seasons to stay quick.
"""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from eplsim import clubs, config, data, metrics, outputs, pipeline, run_week, tables

NOW = datetime(2026, 11, 11, 7, 0, tzinfo=timezone.utc)
SIMS = 2_000
QUIET = {"log": lambda *_: None}
SEASON = pipeline.CURRENT


@pytest.fixture(scope="module")
def built(tmp_path_factory, data_dir, frozen_dir):
    base = tmp_path_factory.mktemp("tables")
    out, site = base / "outputs", base / "site"
    for mw in (4, 9):
        run_week.run(data_dir, out, site, current_dir=frozen_dir, as_of=mw, now=NOW, n_sims=SIMS, **QUIET)
    run_week.run(data_dir, out, site, current_dir=frozen_dir, do_fetch=False, now=NOW, n_sims=SIMS, **QUIET)
    read = lambda name: pd.read_parquet(out / f"{name}.parquet")  # noqa: E731
    return {"out": out, "base": base, "read": read}


def test_every_table_is_written(built):
    for name in ("matches", "match_ratings", "points_dist", "clubs", "gameweeks"):
        assert (built["out"] / f"{name}.parquet").exists(), name


def test_matches_one_row_per_fixture_with_consistent_results(built):
    m = built["read"]("matches")
    assert not m.duplicated(["season", "home", "away"]).any()
    cur = m[m.season == SEASON]
    assert len(cur) == 380 and (cur.status == "played").sum() == 100
    assert m.season.min() == "2010-11"
    played = m[m.status == "played"]
    hg, ag = played.home_goals.astype(int), played.away_goals.astype(int)
    assert (played.result == np.where(hg > ag, "H", np.where(hg == ag, "D", "A"))).all()
    assert m.loc[m.status != "played", ["home_goals", "result"]].isna().all().all()
    assert (played.sources_backing >= 1).all() and (cur.loc[cur.status == "played", "sources_backing"] >= 1).all()
    ht = played.dropna(subset=["ht_home_goals"])
    assert len(ht) > 4000  # openfootball half-time scores
    assert ((ht.ht_home_goals <= ht.home_goals) & (ht.ht_away_goals <= ht.away_goals)).all()
    assert set(m.home_code) <= set(built["read"]("clubs").code)


def test_backers_do_not_change_what_reconcile_keeps(data_dir, frozen_dir):
    matches, report = pipeline.load_all(data_dir, current_dir=frozen_dir, with_report=True)
    rep = report[SEASON]
    played = [x for x in matches if x.season == SEASON and x.played]
    assert set(rep["backers"]) == {(x.home, x.away) for x in played}
    assert sum(len(b) >= 2 for b in rep["backers"].values()) == rep["agreed"] + len(rep["overruled"])


def test_ratings_after_each_week_match_the_stored_history(built):
    r, h = built["read"]("match_ratings"), built["read"]("history")
    assert not r.duplicated(["season", "home", "away"]).any() and r.order.is_unique
    assert np.allclose(r.home_change, -r.away_change)
    cur = r[r.season == SEASON]
    for mp, snap in h[h.season == SEASON].groupby("matches_played"):
        mw = int(snap.matchweek.iloc[0])
        upto = cur[cur.gameweek <= mw]
        assert len(upto) == mp
        last = {}
        for row in upto.sort_values("order").itertuples():
            last[row.home], last[row.away] = row.home_elo_post, row.away_elo_post
        for row in snap.itertuples():
            assert last[row.team] == pytest.approx(row.elo, abs=1e-9), (mp, row.team)


def test_a_table_rebuilt_from_matches_equals_the_stored_real_table(built):
    m, h = built["read"]("matches"), built["read"]("history")
    cur = m[m.season == SEASON]
    for mp, snap in h[h.season == SEASON].groupby("matches_played"):
        mw = int(snap.matchweek.iloc[0])
        rows = [data.Match(SEASON, int(x.gameweek), x.date, x.home, x.away,
                           None if pd.isna(x.home_goals) or x.gameweek > mw else int(x.home_goals),
                           None if pd.isna(x.away_goals) or x.gameweek > mw else int(x.away_goals))
                for x in cur.itertuples()]
        table = {t["team"]: t for t in data.league_table(rows)}
        for row in snap.itertuples():
            t = table[row.team]
            assert (t["pts"], t["gd"], t["w"], t["d"], t["l"], t["gf"], t["ga"]) == (
                row.pts_now, row.gd_now, row.won, row.drawn, row.lost, row.goals_for, row.goals_against)


def test_points_shares_sum_to_one_and_average_to_expected_points(built):
    p, h = built["read"]("points_dist"), built["read"]("history")
    assert not p.duplicated(tables.POINTS_KEY).any()
    assert set(p.matches_played) == set(h[h.season == SEASON].matches_played) == {40, 90, 100}
    sums = p.groupby(["matches_played", "team"]).share.sum()
    assert np.allclose(sums, 1.0)
    mean = (p.points * p.share).groupby([p.matches_played, p.team]).sum()
    exp = h.set_index(["matches_played", "team"]).exp_pts
    assert np.allclose(mean.sort_index(), exp.loc[mean.sort_index().index])


def test_points_of_the_live_run_match_the_saved_seasons(built):
    p, s = built["read"]("points_dist"), pd.read_parquet(built["out"] / "seasons_latest.parquet")
    live = p[p.matches_played == 100]
    for team in ("Arsenal", "Liverpool", "Coventry"):
        col = s[f"pts_{clubs.code(team)}"].to_numpy()
        vals, counts = np.unique(col, return_counts=True)
        got = live[live.team == team].sort_values("points")
        assert list(got.points) == list(vals) and np.allclose(got.share, counts / len(col))


def test_clubs_codes_unique_and_current_season_complete(built):
    c = built["read"]("clubs")
    assert c.team.is_unique and c.code.is_unique
    cur = c[c.in_current_season]
    assert len(cur) == 20 and cur.own_colours.all() and cur.promoted_this_season.sum() == 3


def test_gameweeks_add_up_and_point_at_the_run_that_stands_for_them(built):
    g = built["read"]("gameweeks")
    assert not g.duplicated(["season", "gameweek"]).any()
    cur = g[g.season == SEASON].set_index("gameweek")
    assert cur.fixtures.sum() == 380 and len(cur) == 38
    assert (cur.loc[1:10, "status"] == "complete").all() and (cur.loc[11:, "status"] == "upcoming").all()
    assert cur.loc[10, "snapshot_matches_played"] == 100 and not cur.loc[10, "snapshot_reconstructed"]
    assert cur.loc[9, "snapshot_matches_played"] == 90 and cur.loc[9, "snapshot_reconstructed"]
    assert pd.isna(cur.loc[5, "snapshot_matches_played"])


def test_forecast_log_carries_expected_goals_and_ratings(built, data_dir, frozen_dir):
    f, h = built["read"]("forecast_log"), built["read"]("history")
    assert {"home_xg", "away_xg", "home_elo", "away_elo"} <= set(f.columns) and f[["home_xg", "home_elo"]].notna().all().all()
    elo = h[h.matches_played == 100].set_index("team").elo
    assert np.allclose(f.home_elo, elo.loc[f.home].to_numpy()) and np.allclose(f.away_elo, elo.loc[f.away].to_numpy())
    assert ((f.p_home + f.p_draw + f.p_away) > 0.99).all()


def test_fill_points_restores_a_missing_run_exactly(built, data_dir, frozen_dir):
    path = built["out"] / "points_dist.parquet"
    before = pd.read_parquet(path)
    before[before.matches_played != 90].to_parquet(path, index=False)
    assert run_week.fill_points(data_dir, built["out"], current_dir=frozen_dir, **QUIET) == [90]
    after = pd.read_parquet(path)
    pd.testing.assert_frame_equal(after.reset_index(drop=True), before.reset_index(drop=True))


def test_fill_points_skips_a_run_it_cannot_reproduce(tmp_path, built, data_dir, frozen_dir):
    import shutil
    out = tmp_path / "outputs"
    shutil.copytree(built["out"], out)
    pts = pd.read_parquet(out / "points_dist.parquet")
    pts[pts.matches_played != 40].to_parquet(out / "points_dist.parquet", index=False)
    pos = pd.read_parquet(out / "position_dist.parquet")
    pos.loc[(pos.matches_played == 40) & (pos.place == 1), "share"] += 1e-6  # stored places no longer reproducible
    pos.to_parquet(out / "position_dist.parquet", index=False)
    logs = []
    assert run_week.fill_points(data_dir, out, current_dir=frozen_dir, log=logs.append) == []
    assert any("did not reproduce" in line for line in logs)
    assert 40 not in set(pd.read_parquet(out / "points_dist.parquet").matches_played)
