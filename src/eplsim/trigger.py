"""Decide whether the full weekly run is worth starting.

    python -m eplsim.trigger                 print the decision, and write it to GITHUB_OUTPUT
    python -m eplsim.trigger --calendar      when each gameweek still to come should be over

The full run installs packages, runs every test, downloads three sources and
simulates 10,000 seasons. That is only worth doing when a gameweek has
finished. This check uses nothing but the standard library and the files
already in the repository, so it takes seconds:

1. Read the fixture list and the last published gameweek.
2. If no gameweek after it has had its last kick-off at least OVER_AFTER ago,
   stop. That covers mid-gameweek and international breaks, with no download.
3. Otherwise ask the official feed one question: has every fixture in that
   gameweek finished? Yes: start the full run. No: try again at the next check.
   If the feed cannot be reached, start the full run anyway and let its gates
   decide, because a missed gameweek is worse than a wasted run.

Postponed fixtures (data/postponed.json) and fixtures with no date do not hold
a gameweek open. Runs started by hand or by a push always go ahead.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

SEASON = "2026-27"
OVER_AFTER = timedelta(hours=2, minutes=30)  # kick-off to final whistle, with stoppage time to spare
FEED = "https://fantasy.premierleague.com/api/fixtures/"


def _time(s: str) -> datetime | None:
    s = (s or "").strip()
    if not s:
        return None
    t = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def load_fixtures(path: Path) -> list[dict]:
    """The committed copy of the official fixture list: gameweek, teams, kick-off."""
    with open(path, newline="", encoding="utf-8") as f:
        return [{"gameweek": int(r["event"]), "home": r["team_h"], "away": r["team_a"],
                 "kickoff": _time(r["kickoff_time"])}
                for r in csv.DictReader(f) if (r.get("event") or "").strip()]


def gameweek_ends(fixtures: list[dict], postponed: set = frozenset()) -> dict[int, datetime]:
    """When each gameweek should be over: its last dated kick-off plus OVER_AFTER."""
    last: dict[int, datetime] = {}
    for fx in fixtures:
        if fx["kickoff"] is None or (fx["home"], fx["away"]) in postponed:
            continue
        g = fx["gameweek"]
        last[g] = max(last.get(g, fx["kickoff"]), fx["kickoff"])
    return {g: t + OVER_AFTER for g, t in sorted(last.items())}


def feed_says_finished(gameweek: int, get) -> bool:
    """Ask the live feed whether every fixture of one gameweek has finished."""
    def done(r):  # same rule as the pipeline: a final or provisional whistle, with a score
        flagged = any(r.get(k) in (True, "True", "true") for k in ("finished", "finished_provisional"))
        return flagged and r.get("team_h_score") is not None and r.get("team_a_score") is not None
    rows = [r for r in json.loads(get(FEED)) if r.get("event") == gameweek]
    return bool(rows) and all(done(r) for r in rows)


def decide(fixtures: list[dict], published: int, now: datetime, event: str = "schedule",
           postponed: set = frozenset(), get=None) -> dict:
    """Return {'due': bool, 'gameweek': int | None, 'reason': str}. Never raises for a network failure."""
    if event != "schedule":
        return {"due": True, "gameweek": None, "reason": f"started by {event}, always runs"}
    ends = gameweek_ends(fixtures, postponed)
    over = [g for g, t in ends.items() if g > published and t <= now]
    if not over:
        upcoming = [(g, t) for g, t in ends.items() if g > published]
        if not upcoming:
            return {"due": False, "gameweek": None, "reason": "season over, nothing left to publish"}
        g, t = upcoming[0]
        return {"due": False, "gameweek": g,
                "reason": f"gameweek {g} is not over yet (expected after {t:%a %d %b %H:%M} UTC)"}
    g = max(over)
    if get is None:
        return {"due": True, "gameweek": g, "reason": f"gameweek {g} should be over; no feed to ask"}
    try:
        if feed_says_finished(g, get):
            return {"due": True, "gameweek": g, "reason": f"gameweek {g} finished, starting the full run"}
        return {"due": False, "gameweek": g, "reason": f"gameweek {g} should be over but the feed has not closed it yet"}
    except Exception as e:  # noqa: BLE001 - a feed we cannot read must not hide a finished gameweek
        return {"due": True, "gameweek": g, "reason": f"gameweek {g} should be over; feed unreadable ({str(e)[:80]}), "
                                                       "starting the full run so its gates decide"}


def published_gameweek(path: Path) -> int:
    if not path.exists():
        return 0
    return int(json.loads(path.read_text(encoding="utf-8")).get("matchweek") or 0)


def _postponed(path: Path) -> set:
    if not path.exists():
        return set()
    return {tuple(p) for p in json.loads(path.read_text(encoding="utf-8")).get("fixtures", [])}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--event", default=os.environ.get("GITHUB_EVENT_NAME", "schedule"))
    ap.add_argument("--now", default=None, help="decide as if it were this moment, ISO 8601 UTC")
    ap.add_argument("--offline", action="store_true", help="do not ask the feed")
    ap.add_argument("--calendar", action="store_true", help="list when each gameweek still to come should be over")
    a = ap.parse_args(argv)
    data = Path(a.data)
    fixtures = load_fixtures(data / "current" / "fpl" / f"fixtures_{SEASON}.csv")
    published = published_gameweek(Path(a.out) / "run_meta.json")
    postponed = _postponed(data / "postponed.json")
    now = _time(a.now) if a.now else datetime.now(timezone.utc)
    if a.calendar:
        for g, t in gameweek_ends(fixtures, postponed).items():
            if g > published:
                print(f"gameweek {g:>2}: over after {t:%a %d %b %Y %H:%M} UTC")
        return 0
    get = None
    if not a.offline:
        from .fetch import http_get
        get = http_get
    d = decide(fixtures, published, now, a.event, postponed, get)
    print(f"published gameweek {published}. {d['reason']}")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
            f.write(f"due={'true' if d['due'] else 'false'}\ngameweek={d['gameweek'] or ''}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
