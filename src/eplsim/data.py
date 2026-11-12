"""Loaders for the three result sources, normalised to one Match record.

History (2010/11 to 2024/25): openfootball/football.json, one JSON per season.
Last season and this one: three sources merged by `reconcile`: the official
FPL feed, the FPL-Elo-Insights mirror and openfootball.

Traps handled here, each covered by a test:
- openfootball stores a result as {"ft": [h, a]}, except goalless draws in its
  2025/26 and 2026/27 files, which are a bare list [0, 0]. Unplayed games have
  no score key at all. Both shapes are results. (An earlier version of this
  loader took the bare list for a placeholder and dropped 32 real 0-0 draws.)
- The mirror stores teams as FPL club *codes* in float form ("43.0"), not ids.
  They are resolved through that season's teams.csv and cross-checked
  against the club slugs in match_id.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .teams import canonical


@dataclass(frozen=True)
class Match:
    season: str  # "2026-27"
    gameweek: int
    date: str  # ISO date, YYYY-MM-DD
    home: str
    away: str
    home_goals: int | None  # None when not played yet
    away_goals: int | None

    @property
    def played(self) -> bool:
        return self.home_goals is not None


def load_openfootball(path: Path, season: str) -> list[Match]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out: list[Match] = []
    for m in raw["matches"]:
        score = m.get("score")
        hg = ag = None
        if isinstance(score, dict) and isinstance(score.get("ft"), list) and len(score["ft"]) == 2:
            hg, ag = int(score["ft"][0]), int(score["ft"][1])
        elif isinstance(score, list) and len(score) == 2 and all(isinstance(x, int) for x in score):
            hg, ag = score[0], score[1]
        gw = int(re.search(r"(\d+)", m["round"]).group(1))
        out.append(Match(season, gw, m["date"], canonical(m["team1"]), canonical(m["team2"]), hg, ag))
    return out


def _gw_number(p: Path) -> int:
    return int(re.search(r"gw(\d+)", p.name).group(1))


def load_mirror(folder: Path, season: str) -> list[Match]:
    """Load a season from the FPL-Elo-Insights layout (pl_matches_gwN.csv + teams.csv)."""
    folder = Path(folder)
    with open(folder / "teams.csv", newline="", encoding="utf-8") as f:
        code_to_name = {int(float(r["code"])): canonical(r["name"]) for r in csv.DictReader(f)}

    out: list[Match] = []
    seen: set[str] = set()
    for p in sorted(folder.glob("pl_matches_gw*.csv"), key=_gw_number):
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["match_id"] in seen:
                    raise ValueError(f"Duplicate match_id {r['match_id']}")
                seen.add(r["match_id"])
                home = code_to_name[int(float(r["home_team"]))]
                away = code_to_name[int(float(r["away_team"]))]

                slug = re.match(r"\d\d-\d\d-prem-(.+)-vs-(.+)$", r["match_id"])
                if not slug or (canonical(slug.group(1)), canonical(slug.group(2))) != (home, away):
                    raise ValueError(f"Team codes disagree with match_id: {r['match_id']} -> {home} v {away}")

                finished = r["finished"].strip().lower() == "true"
                has_score = r["home_score"] != "" and r["away_score"] != ""
                if finished != has_score:
                    raise ValueError(f"finished flag and score disagree: {r['match_id']}")
                hg = int(float(r["home_score"])) if finished else None
                ag = int(float(r["away_score"])) if finished else None
                out.append(Match(season, _gw_number(p), r["kickoff_time"][:10], home, away, hg, ag))
    return out


def load_fpl(fixtures_csv: Path, teams_csv: Path, season: str) -> list[Match]:
    """Load the official FPL fixtures feed (vaastav mirror layout: team ids, not codes).

    The live FPL API returns the same fields, so this is also the shape the
    production fetch will produce.
    """
    with open(teams_csv, newline="", encoding="utf-8") as f:
        id_to_name = {int(float(r["id"])): canonical(r["name"]) for r in csv.DictReader(f)}
    out: list[Match] = []
    with open(fixtures_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            # `finished` turns true hours after full time, `finished_provisional` at the whistle.
            flags = (r["finished"], r.get("finished_provisional") or "")
            finished = any(f.strip().lower() == "true" for f in flags) and r["team_h_score"] != "" and r["team_a_score"] != ""
            hg = int(float(r["team_h_score"])) if finished else None
            ag = int(float(r["team_a_score"])) if finished else None
            gw = int(float(r["event"])) if r["event"] else 0
            out.append(Match(season, gw, (r["kickoff_time"] or "")[:10], id_to_name[int(float(r["team_h"]))],
                             id_to_name[int(float(r["team_a"]))], hg, ag))
    return out


def reconcile(sources: dict[str, list[Match]]) -> tuple[list[Match], dict]:
    """Merge several sources for one season into one trusted list.

    The first source in `sources` sets the fixture list and wins when it is the
    only one with a result. Where sources disagree on a score, the majority
    wins. A disagreement with no majority raises, because guessing a score is
    worse than stopping. A secondary source on its own can never create a
    result the primary does not have: it waits for a second source.
    """
    names = list(sources)
    idx = {n: season_keys(ms) for n, ms in sources.items()}
    primary = sources[names[0]]
    out: list[Match] = []
    report = {"agreed": 0, "single_source": [], "overruled": [], "unplayed": 0, "awaiting_second_source": []}
    for m in primary:
        key = (m.home, m.away)
        votes: dict[tuple[int, int], list[str]] = {}
        for n in names:
            other = idx[n].get(key)
            if other is not None and other.played:
                votes.setdefault((other.home_goals, other.away_goals), []).append(n)
        if votes and not m.played and sum(len(b) for b in votes.values()) < 2:
            report["awaiting_second_source"].append(key)
            votes = {}
        if not votes:
            report["unplayed"] += 1
            out.append(Match(m.season, m.gameweek, m.date, m.home, m.away, None, None))
            continue
        ranked = sorted(votes.items(), key=lambda kv: -len(kv[1]))
        if len(ranked) > 1 and len(ranked[0][1]) == len(ranked[1][1]):
            raise ValueError(f"Sources disagree with no majority on {key}: {votes}")
        score, backers = ranked[0]
        if len(ranked) > 1:
            report["overruled"].append({"fixture": key, "kept": score, "kept_from": backers,
                                        "rejected": {str(s): b for s, b in ranked[1:]}})
        elif len(backers) == 1:
            report["single_source"].append(key)
        else:
            report["agreed"] += 1
        out.append(Match(m.season, m.gameweek, m.date, m.home, m.away, score[0], score[1]))
    return out, report


def season_keys(matches: list[Match]) -> dict[tuple[str, str], Match]:
    """Index a single season by (home, away). Each pairing happens once per season."""
    idx: dict[tuple[str, str], Match] = {}
    for m in matches:
        key = (m.home, m.away)
        if key in idx:
            raise ValueError(f"Fixture appears twice in {m.season}: {key}")
        idx[key] = m
    return idx


def compare_sources(a: list[Match], b: list[Match]) -> dict:
    """Compare played results between two sources for the same season."""
    ia, ib = season_keys(a), season_keys(b)
    both = [k for k in ia if k in ib and ia[k].played and ib[k].played]
    mismatches = [
        (k, (ia[k].home_goals, ia[k].away_goals), (ib[k].home_goals, ib[k].away_goals))
        for k in both
        if (ia[k].home_goals, ia[k].away_goals) != (ib[k].home_goals, ib[k].away_goals)
    ]
    return {"compared": len(both), "mismatches": mismatches, "fixtures_only_in_a": sorted(set(ia) - set(ib))}


def league_table(matches: list[Match]) -> list[dict]:
    """Current table from played matches: points, then goal difference, then goals for."""
    rows: dict[str, dict] = {}
    for m in matches:
        for t in (m.home, m.away):
            rows.setdefault(t, {"team": t, "p": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0, "pts": 0})
        if not m.played:
            continue
        h, a = rows[m.home], rows[m.away]
        h["p"] += 1
        a["p"] += 1
        h["gf"] += m.home_goals
        h["ga"] += m.away_goals
        a["gf"] += m.away_goals
        a["ga"] += m.home_goals
        if m.home_goals > m.away_goals:
            h["w"] += 1
            a["l"] += 1
            h["pts"] += 3
        elif m.home_goals < m.away_goals:
            a["w"] += 1
            h["l"] += 1
            a["pts"] += 3
        else:
            h["d"] += 1
            a["d"] += 1
            h["pts"] += 1
            a["pts"] += 1
    table = list(rows.values())
    for r in table:
        r["gd"] = r["gf"] - r["ga"]
    table.sort(key=lambda r: (-r["pts"], -r["gd"], -r["gf"], r["team"]))
    return table
