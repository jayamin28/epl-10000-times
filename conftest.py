"""Shared fixtures. Heavy objects are built once per test session."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from eplsim import config, pipeline  # noqa: E402


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


FROZEN = ROOT / "tests" / "fixtures" / "gw10"  # the live season as fetched on 11 Nov 2026


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return ROOT / "data"


@pytest.fixture(scope="session")
def frozen_dir() -> Path:
    """A frozen copy of the live season, so tests give the same answer every week."""
    return FROZEN


@pytest.fixture(scope="session")
def loaded(data_dir, frozen_dir):
    return pipeline.load_all(data_dir, current_dir=frozen_dir, with_report=True)


@pytest.fixture(scope="session")
def matches(loaded):
    return loaded[0]


@pytest.fixture(scope="session")
def report(loaded):
    return loaded[1]


@pytest.fixture(scope="session")
def snap(matches):
    return pipeline.snapshot(matches, config.ELO, config.GOALS_WINDOW)


@pytest.fixture(scope="session")
def result(snap):
    return pipeline.run_sim(snap, n_sims=config.N_SIMS, sigma=config.SIGMA, seed=config.SEED)


def make_get(frozen: Path = FROZEN, fail: tuple[str, ...] = (), edit=None):
    """A stand-in for the network: serves the frozen files in the shape each source really uses.

    `fail` lists sources that cannot be reached: "api", "vaastav", "mirror", "openfootball".
    `edit(url, rows_or_json)` may change what a source serves, to imitate a broken feed.
    """
    import csv
    import io
    import json
    import re

    def rows(path):
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def as_csv(rs, extra):
        out = io.StringIO()
        fields = list(rs[0]) + list(extra)
        w = csv.DictWriter(out, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rs:
            w.writerow({**r, **extra})
        return out.getvalue().encode()

    def num(v):
        return None if v == "" else int(float(v))

    def get(url: str) -> bytes:
        def down(name):
            if name in fail:
                raise RuntimeError(f"GET {url} failed: unreachable")

        if "fantasy.premierleague.com" in url:
            down("api")
            if url.endswith("/fixtures/"):
                data = [{"code": int(r["code"]), "event": num(r["event"]), "finished": r["finished"] == "True",
                         "finished_provisional": r["finished_provisional"] == "True", "id": int(r["id"]),
                         "kickoff_time": r["kickoff_time"] or None, "team_a": int(r["team_a"]),
                         "team_a_score": num(r["team_a_score"]), "team_h": int(r["team_h"]),
                         "team_h_score": num(r["team_h_score"]), "minutes": 90, "stats": [{"identifier": "goals_scored"}],
                         "team_h_difficulty": 3} for r in rows(frozen / "fpl" / "fixtures_2026-27.csv")]
            else:
                data = {"events": [], "elements": [{"web_name": "Someone", "now_cost": 55}],
                        "teams": [{**r, "code": int(r["code"]), "id": int(r["id"]), "position": int(r["position"]),
                                   "strength": 3, "points": 0} for r in rows(frozen / "fpl" / "teams_2026-27.csv")]}
            return json.dumps(edit(url, data) if edit else data).encode()
        if "vaastav" in url:
            down("vaastav")
            name = "fixtures" if url.endswith("fixtures.csv") else "teams"
            rs = rows(frozen / "fpl" / f"{name}_2026-27.csv")
            return as_csv(edit(url, rs) if edit else rs, {"pulse_id": "1", "strength": "3"})
        if "FPL-Elo-Insights" in url:
            down("mirror")
            if url.endswith("teams.csv"):
                rs = rows(frozen / "fei_2026-27" / "teams.csv")
            else:
                gw = re.search(r"/GW(\d+)/matches\.csv$", url).group(1)
                rs = rows(frozen / "fei_2026-27" / f"pl_matches_gw{gw}.csv")
            return as_csv(edit(url, rs) if edit else rs, {"home_team_elo": "1500.0", "tournament": "prem"})
        if "openfootball" in url:
            down("openfootball")
            data = json.loads((frozen / "openfootball" / "2026-27.json").read_text(encoding="utf-8"))
            return json.dumps(edit(url, data) if edit else data).encode()
        raise AssertionError(f"unexpected URL {url}")

    return get


@pytest.fixture(scope="session")
def fake_get():
    return make_get
