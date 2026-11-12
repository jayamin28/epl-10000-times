"""Checks that must hold before a run is allowed to publish.

A failed blocking gate stops the run. Last week's page stays up and the
workflow goes red, which sends an email. Publishing late beats publishing wrong.
"""

from __future__ import annotations

from collections import Counter
from datetime import date, timedelta

from .data import Match


class GateError(RuntimeError):
    """One or more blocking gates failed."""


def _result(gate: str, ok: bool, detail: str, blocking: bool = True) -> dict:
    return {"gate": gate, "status": "pass" if ok else ("fail" if blocking else "warn"), "detail": detail,
            "blocking": blocking}


def fixture_list(current: list[Match]) -> dict:
    """380 fixtures, 20 clubs, every club 19 at home and 19 away, no pairing twice."""
    home, away = Counter(m.home for m in current), Counter(m.away for m in current)
    pairs = {(m.home, m.away) for m in current}
    ok = (len(current) == 380 and len(pairs) == 380 and len(home) == 20 and len(away) == 20
          and set(home.values()) == {19} and set(away.values()) == {19} and not any(h == a for h, a in pairs))
    return _result("fixture_list", ok, f"{len(current)} fixtures, {len(pairs)} distinct, {len(home)} clubs")


def no_stale_results(current: list[Match], today: date, postponed: set[tuple[str, str]], max_age_days: int = 2) -> dict:
    """Every match dated more than `max_age_days` ago has a result, unless it is on the postponed list."""
    cutoff = (today - timedelta(days=max_age_days)).isoformat()
    missing = [(m.date, m.home, m.away) for m in current
               if not m.played and m.date and m.date < cutoff and (m.home, m.away) not in postponed]
    detail = "no result is overdue" if not missing else (
        f"{len(missing)} overdue: " + "; ".join(f"{d} {h} v {a}" for d, h, a in sorted(missing)[:5])
        + ". If a match was postponed, add it to data/postponed.json")
    return _result("no_stale_results", not missing, detail)


def results_never_vanish(played_now: int, played_before: int | None) -> dict:
    """A source that loses results it had last time is broken."""
    if played_before is None:
        return {"gate": "results_never_vanish", "status": "skipped", "detail": "first run", "blocking": True}
    return _result("results_never_vanish", played_now >= played_before,
                   f"{played_now} results now, {played_before} at the last run")


def sources_reachable(manifest: dict | None) -> dict:
    """At least two of the three sources answered on this fetch."""
    if not manifest or "sources" not in manifest:
        return {"gate": "sources_reachable", "status": "skipped", "detail": "offline run, nothing was fetched",
                "blocking": True}
    ok = [n for n, s in manifest["sources"].items() if s.get("ok")]
    return _result("sources_reachable", len(ok) >= 2, f"reached: {', '.join(ok) or 'none'}")


def table_matches_official_positions(table: list[dict], fpl_positions: dict[str, int] | None,
                                     fpl_results: int | None, results: int) -> dict:
    """Our table order against the positions the official feed publishes.

    Only meaningful when the official feed holds as many results as we do.
    Clubs level on points, goal difference and goals scored may swap places.
    Not blocking yet: how quickly the feed updates this field after a match is
    not known, so a mismatch is recorded as a warning.
    """
    name = "table_matches_official_positions"
    if not fpl_positions or len(set(fpl_positions.values())) != 20 or fpl_results != results:
        why = "official feed has no usable positions" if fpl_results == results else \
            f"official feed holds {fpl_results} results, we hold {results}"
        return {"gate": name, "status": "skipped", "detail": why, "blocking": False}
    ours = {r["team"]: i + 1 for i, r in enumerate(table)}
    key = {r["team"]: (r["pts"], r["gd"], r["gf"]) for r in table}
    wrong = []
    for team, pos in ours.items():
        theirs = fpl_positions.get(team)
        if theirs != pos:
            holder = next((t for t, p in ours.items() if p == theirs), None)
            if holder is None or key[holder] != key[team]:
                wrong.append(f"{team}: ours {pos}, feed {theirs}")
    return _result(name, not wrong, "all 20 positions agree" if not wrong else "; ".join(wrong[:5]), blocking=False)


def run_gates(current: list[Match], table: list[dict], today: date, postponed: set[tuple[str, str]],
              played_before: int | None, manifest: dict | None, fpl_positions: dict[str, int] | None,
              fpl_results: int | None) -> list[dict]:
    """Run every gate. Raises GateError naming the blocking gates that failed."""
    played = sum(m.played for m in current)
    results = [
        fixture_list(current),
        sources_reachable(manifest),
        no_stale_results(current, today, postponed),
        results_never_vanish(played, played_before),
        table_matches_official_positions(table, fpl_positions, fpl_results, played),
    ]
    failed = [r for r in results if r["status"] == "fail"]
    if failed:
        raise GateError("; ".join(f"{r['gate']}: {r['detail']}" for r in failed))
    return results
