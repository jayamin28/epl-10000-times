"""Fetch this season's fixtures and results from three sources.

  fpl           the official Fantasy Premier League feed. Live API first, a
                public GitHub mirror of the same feed if the API cannot be reached.
  mirror        olbauday/FPL-Elo-Insights, a community dataset on GitHub.
  openfootball  openfootball/football.json, public domain.

What is saved is a facts-only extract: fixture, date, score and the ids
needed to read them. The hash of each original download is kept in the
manifest so a run can be traced back to exactly what the source served.

A source that cannot be reached keeps its previous files and is marked failed
in the manifest. The run goes on with whatever is on disk, and the gates
decide whether that is good enough to publish.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

FPL_FIXTURE_FIELDS = ["code", "event", "finished", "finished_provisional", "id", "kickoff_time",
                      "team_a", "team_a_score", "team_h", "team_h_score"]
FPL_TEAM_FIELDS = ["code", "id", "name", "short_name", "position"]
MIRROR_MATCH_FIELDS = ["gameweek", "kickoff_time", "home_team", "home_score", "away_score", "away_team",
                       "finished", "match_id"]
MIRROR_TEAM_FIELDS = ["code", "id", "name"]

FPL_API = "https://fantasy.premierleague.com/api"
VAASTAV = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
MIRROR = "https://raw.githubusercontent.com/olbauday/FPL-Elo-Insights/main/data"
OPENFOOTBALL = "https://raw.githubusercontent.com/openfootball/football.json/master"


def http_get(url: str, timeout: int = 30, retries: int = 2) -> bytes:
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "epl-10000-times (github.com/jayamin28/epl-10000-times)"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError, OSError) as e:  # HTTPError is a URLError
            last = e
            if isinstance(e, urllib.error.HTTPError) and e.code in (403, 404):
                break
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed: {last}")


def _sha(*blobs: bytes) -> str:
    h = hashlib.sha256()
    for b in blobs:
        h.update(b)
    return h.hexdigest()[:16]


def _csv_rows(blob: bytes) -> list[dict]:
    return list(csv.DictReader(io.StringIO(blob.decode("utf-8-sig"))))


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})


def trim(rows: list[dict], fields: list[str]) -> list[dict]:
    """Keep only the fields this project reads. Raises if a source drops one we need."""
    if rows:
        missing = [f for f in fields if f not in rows[0]]
        if missing:
            raise ValueError(f"Source no longer has the fields {missing}")
    return [{k: r.get(k) for k in fields} for r in rows]


def fpl_api_rows(fixtures_json: list[dict], bootstrap_json: dict) -> tuple[list[dict], list[dict]]:
    """Turn the two live FPL API responses into the same rows the CSV mirror holds."""
    if not isinstance(fixtures_json, list) or "teams" not in bootstrap_json:
        raise ValueError("FPL API response is not in the expected shape")
    fixtures = trim([{k: v for k, v in fx.items() if k != "stats"} for fx in fixtures_json], FPL_FIXTURE_FIELDS)
    for fx in fixtures:
        for flag in ("finished", "finished_provisional"):
            fx[flag] = "True" if fx[flag] in (True, "True", "true") else "False"
    return fixtures, trim(bootstrap_json["teams"], FPL_TEAM_FIELDS)


def _fetch_fpl(dest: Path, season: str, get) -> dict:
    info: dict = {"ok": False, "origin": None, "tried": []}
    for origin in ("api", "mirror"):
        try:
            if origin == "api":
                raw_fx, raw_tm = get(f"{FPL_API}/fixtures/"), get(f"{FPL_API}/bootstrap-static/")
                fixtures, teams = fpl_api_rows(json.loads(raw_fx), json.loads(raw_tm))
            else:
                raw_fx, raw_tm = get(f"{VAASTAV}/{season}/fixtures.csv"), get(f"{VAASTAV}/{season}/teams.csv")
                fixtures = trim(_csv_rows(raw_fx), FPL_FIXTURE_FIELDS)
                teams = trim(_csv_rows(raw_tm), FPL_TEAM_FIELDS)
            if len(fixtures) != 380 or len(teams) != 20:
                raise ValueError(f"expected 380 fixtures and 20 clubs, got {len(fixtures)} and {len(teams)}")
            _write_csv(dest / "fpl" / f"fixtures_{season}.csv", FPL_FIXTURE_FIELDS, fixtures)
            _write_csv(dest / "fpl" / f"teams_{season}.csv", FPL_TEAM_FIELDS, teams)
            info.update(ok=True, origin=origin, sha=_sha(raw_fx, raw_tm))
            info["tried"].append({"origin": origin, "ok": True})
            return info
        except Exception as e:  # noqa: BLE001 - any failure means: try the next origin
            info["tried"].append({"origin": origin, "ok": False, "error": str(e)[:200]})
    return info


def _fetch_mirror(dest: Path, season: str, get) -> dict:
    long_season = f"20{season[2:4]}-20{season[5:7]}"  # 2026-27 -> 2026-2027
    base = f"{MIRROR}/{long_season}"
    try:
        raw_teams = get(f"{base}/teams.csv")
        teams = trim(_csv_rows(raw_teams), MIRROR_TEAM_FIELDS)
        weeks: dict[int, list[dict]] = {}
        blobs = [raw_teams]
        for gw in range(1, 39):
            url = f"{base}/{urllib.parse.quote('By Tournament/Premier League')}/GW{gw}/matches.csv"
            try:
                blob = get(url)
            except RuntimeError:
                continue  # a gameweek folder that does not exist yet
            weeks[gw] = trim(_csv_rows(blob), MIRROR_MATCH_FIELDS)
            blobs.append(blob)
        n = sum(len(v) for v in weeks.values())
        if n != 380:
            raise ValueError(f"expected 380 fixtures across gameweeks, got {n} in {len(weeks)} files")
        folder = dest / f"fei_{season}"
        _write_csv(folder / "teams.csv", MIRROR_TEAM_FIELDS, teams)
        for gw, rows in weeks.items():
            _write_csv(folder / f"pl_matches_gw{gw}.csv", MIRROR_MATCH_FIELDS, rows)
        return {"ok": True, "sha": _sha(*blobs), "gameweek_files": len(weeks)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)[:200]}


def _fetch_openfootball(dest: Path, season: str, get) -> dict:
    try:
        blob = get(f"{OPENFOOTBALL}/{season}/en.1.json")
        if len(json.loads(blob)["matches"]) != 380:
            raise ValueError("expected 380 fixtures")
        path = dest / "openfootball" / f"{season}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
        return {"ok": True, "sha": _sha(blob)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)[:200]}


def fetch_current(dest: Path, season: str, get=http_get, now: datetime | None = None) -> dict:
    """Refresh every source under `dest` and write `dest/manifest.json`. Returns the manifest."""
    dest = Path(dest)
    now = now or datetime.now(timezone.utc)
    manifest = {
        "season": season,
        "fetched_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fetched_on": now.strftime("%Y-%m-%d"),
        "sources": {
            "fpl": _fetch_fpl(dest, season, get),
            "mirror": _fetch_mirror(dest, season, get),
            "openfootball": _fetch_openfootball(dest, season, get),
        },
    }
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
