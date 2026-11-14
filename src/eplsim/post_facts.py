"""Every number a weekly post is allowed to use, written out once from the run's feed.

    python -m eplsim.post_facts                         fact sheet for the latest run, to stdout
    python -m eplsim.post_facts --out posts/facts.md    write it to a file
    python -m eplsim.post_facts --check draft.md        list any number in a draft that the fact sheet does not hold

The writer (a person or an AI agent) picks the angles and the words. The numbers come only from here,
already rounded and phrased the way posts use them ("66%", "1 in 19", "2 of 10,000"). The check is
deliberately blunt: a number in the draft that appears nowhere on the fact sheet fails, so an invented or
miscopied figure cannot slip through. It cannot tell a right number used in the wrong place; a human
review still reads every post.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

RACE_NAMES = {"title": "win the league", "top4": "finish top four", "top5": "finish top five", "relegated": "go down"}
ALWAYS_ALLOWED = {"10000", "1", "2010"}  # "1 to 10,000", "since 2010"


def pct(p: float) -> str:
    """Odds as posts print them: whole percent from 10%, one decimal below that, 'under 0.1%' at the bottom."""
    if p >= 0.995:
        return "100%" if p >= 0.99995 else "over 99%"
    if p >= 0.10:
        return f"{round(p * 100)}%"
    if p >= 0.001:
        return f"{p * 100:.1f}%"
    return "under 0.1%" if p > 0 else "0%"


def one_in(p: float) -> str:
    """'1 in N' for odds under one half, the way rule C4.7 asks. Above one half, use the miss instead."""
    if p <= 0:
        return "never in 10,000 seasons"
    return f"1 in {round(1 / p):,}"


def seasons(p: float, n: int = 10_000) -> str:
    return f"{round(p * n):,} of {n:,}"


def odds(p: float, n: int = 10_000) -> str:
    """Every way a post may phrase one probability."""
    parts = [pct(p), seasons(p, n)]
    if 0 < p < 0.5:
        parts.append(one_in(p))
    elif 0.5 <= p < 1:
        parts.append(f"misses {one_in(1 - p)}")
    return ", ".join(parts)


def ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def fact_sheet(feed: dict) -> str:
    m = feed["meta"]
    n = int(m["n_sims"])
    gw = int(m["matchweek"])
    L = [f"# Fact sheet: after Gameweek {gw}", "",
         f"Data through {m['data_through']}. {m['matches_played']} results played, {m['matches_remaining']} matches "
         f"simulated, {n:,} seasons, seed {m['seed']}. Run {m['generated_at']}.", "",
         "Use only the numbers on this sheet. Say \"after Gameweek " + str(gw) + "\" on every card and in the text.", "",
         "## Every club", "",
         "| Club | Table now | Points | Model rank | Win the league | Top four | Go down | Expected points |",
         "|---|---|---|---|---|---|---|---|"]
    for i, c in enumerate(feed["clubs"]):
        L.append(f"| {c['team']} | {ordinal(int(c['table_pos']))} | {c['pts_now']} | {ordinal(i + 1)} | "
                 f"{odds(c['title'], n)} | {odds(c['top4'], n)} | {odds(c['relegated'], n)} | {round(c['exp_pts'])} |")

    up = feed.get("upsets") or {}
    L += ["", f"## Upset of the week (odds logged before kick-off)", ""]
    if up.get("rows"):
        L.append(f"Gameweek {up['gameweek']}:")
        for r in up["rows"]:
            who = {"home": f"{r['home']} won", "away": f"{r['away']} won", "draw": "a draw"}[r["outcome"]]
            L.append(f"- {r['home']} {r['score']} {r['away']}: {who}. Before kick-off the model gave that "
                     f"{pct(r['p_before'])} ({one_in(r['p_before'])}).")
    else:
        L.append("None yet. Odds are logged before kick-off from Gameweek 11 on, so the first upset runs after Gameweek 11.")

    mv = feed.get("movers") or {}
    L += ["", f"## Biggest moves since Gameweek {mv.get('since_matchweek')} (3 points or more)", ""]
    for r in (mv.get("rows") or [])[:8]:
        L.append(f"- {r['team']}, {RACE_NAMES.get(r['race'], r['race'])}: {pct(r['before'])} to {pct(r['after'])} "
                 f"({'up' if r['change'] > 0 else 'down'} {round(abs(r['change']) * 100)} points)")
    if not mv.get("rows"):
        L.append("No move of 3 points or more. Rule C4.1: don't headline a smaller one.")

    L += ["", "## The table lies (table place against model rank)", ""]
    for r in (feed.get("table_lies") or [])[:3]:
        L.append(f"- {r['team']}: {ordinal(int(r['table_pos']))} in the table, {ordinal(int(r['model_pos']))} on the model")

    w = feed.get("wildest") or {}
    if w:
        rs = w["rest_of_season"]
        L += ["", "## Wildest universe", "",
              f"Universe #{w['season']:,}: {w['champion']} win the league with {w['champion_pts']} points, "
              f"{w['margin']} clear. They are {ordinal(int(w['champion_table_pos_now']))} now on {w['champion_pts_now']} points. "
              f"Rest of season: won {rs['won']}, drew {rs['drawn']}, lost {rs['lost']}. Best win {w['best_win']['score']} "
              f"{'at home to' if w['best_win']['venue'] == 'home' else 'away at'} {w['best_win']['opponent']}. "
              f"{w['champion']} won the league in {w['title_seasons']} of {n:,} seasons."]
        for i, row in enumerate((w.get("table") or [])[:5]):
            L.append(f"- {ordinal(i + 1)} {row.get('team')} {row.get('pts')} pts")

    mn = feed.get("magic_numbers") or {}
    L += ["", "## Magic numbers (points it takes)", ""]
    for key, label in (("title", "Win the league"), ("top4", "Finish top four"), ("top5", "Finish top five"),
                       ("safety", "Stay up (17th)")):
        if key in mn:
            d = mn[key]
            L.append(f"- {label}: {d['mode']} most often, usually {d['p10']} to {d['p90']} (8 seasons in 10)")

    big = (feed.get("swings") or {}).get("biggest")
    if big:
        L += ["", "## The match that matters most next", "",
              f"{big['team']} v {big['opponent']} ({big['venue']}, {big['date']}). {big['team']}'s chance to "
              f"{RACE_NAMES.get(big['race'], big['race'])}: {pct(big['win'])} if they win, {pct(big['draw'])} if they draw, "
              f"{pct(big['lose'])} if they lose."]

    t = m.get("telemetry") or {}
    if t:
        L += ["", "## Engine", "",
              f"{t.get('matches', 0):,} simulated matches, {t.get('goals', 0):,} goals, {t.get('random_draws', 0):,} random "
              f"draws, {t.get('distinct_champions')} different champions, {t.get('distinct_relegated_trios')} different "
              f"relegated trios."]
    L += ["", "Every post ends: \"Analysis for fun. Not betting advice.\"", ""]
    return "\n".join(L)


_NUM = re.compile(r"1 in \d[\d,]*|\d[\d,]*(?:\.\d+)?%?")


def numbers(text: str) -> set[str]:
    """Every number in a text, commas dropped, so '10,000' and '10000' match. '1 in 19' stays one token."""
    return {tok.replace(",", "") for tok in _NUM.findall(text)}


def check(draft: str, sheet: str) -> list[str]:
    """Numbers in the draft that the fact sheet does not hold. An empty list means the draft passes."""
    allowed = numbers(sheet) | ALWAYS_ALLOWED
    allowed |= {tok.split(" in ")[1] for tok in allowed if tok.startswith("1 in ")}
    return sorted(numbers(draft) - allowed, key=lambda s: (len(s), s))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--feed", default="site/feed.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--check", default=None, help="a draft post to check against the fact sheet")
    a = ap.parse_args(argv)
    sheet = fact_sheet(json.loads(Path(a.feed).read_text(encoding="utf-8")))
    if a.check:
        bad = check(Path(a.check).read_text(encoding="utf-8"), sheet)
        if bad:
            print("Not on the fact sheet: " + ", ".join(bad))
            return 1
        print("Every number in the draft is on the fact sheet.")
        return 0
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(sheet, encoding="utf-8")
    else:
        sys.stdout.write(sheet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
