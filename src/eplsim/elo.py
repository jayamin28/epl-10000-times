"""Club Elo ratings computed from league results only.

Why our own instead of ClubElo: the ClubElo API is not reachable from the build
environment and the public mirror stopped updating in January 2026. Computing
it from results we already hold removes the dependency and makes every rating
reproducible from the repo.

Rules
- Everyone starts at START in the first season on file.
- A club that was not in the previous season (promoted) enters at `promoted`.
- Expected home score uses a home-field bonus `hfa` in Elo points.
- Updates are scaled by a goal-margin multiplier (World Football Elo style).
- Updates are zero sum: what the home side gains, the away side loses.
"""

from __future__ import annotations

from dataclasses import dataclass

from .data import Match

START = 1500.0


@dataclass(frozen=True)
class EloParams:
    k: float = 20.0
    hfa: float = 60.0
    promoted: float = 1400.0


def expected_home(r_home: float, r_away: float, hfa: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-(r_home + hfa - r_away) / 400.0))


def margin_multiplier(goal_diff: int) -> float:
    n = abs(goal_diff)
    if n <= 1:
        return 1.0
    if n == 2:
        return 1.5
    return 1.75 + (n - 3) / 8.0


def update(r_home: float, r_away: float, home_goals: int, away_goals: int, p: EloParams) -> tuple[float, float]:
    exp = expected_home(r_home, r_away, p.hfa)
    actual = 1.0 if home_goals > away_goals else 0.5 if home_goals == away_goals else 0.0
    delta = p.k * margin_multiplier(home_goals - away_goals) * (actual - exp)
    return r_home + delta, r_away - delta


def run_elo(matches: list[Match], p: EloParams) -> tuple[list[tuple[float, float]], dict[str, float]]:
    """Walk through played matches in date order.

    Returns the pre-match (home, away) ratings for every played match, in the
    order processed, and the final rating of every club ever seen. Pre-match
    ratings never include the match itself, so they are safe to use as model
    inputs for that match.
    """
    played = sorted((m for m in matches if m.played), key=lambda m: (m.season, m.date, m.home))
    seasons = sorted({m.season for m in matches})
    members = {s: {t for m in matches if m.season == s for t in (m.home, m.away)} for s in seasons}

    ratings: dict[str, float] = {}
    pre: list[tuple[float, float]] = []
    current = None
    for m in played:
        if m.season != current:
            first = current is None
            current = m.season
            for t in sorted(members[current]):
                prev = seasons[seasons.index(current) - 1] if not first else None
                if first:
                    ratings[t] = START
                elif t not in members[prev]:
                    ratings[t] = p.promoted
        rh, ra = ratings[m.home], ratings[m.away]
        pre.append((rh, ra))
        ratings[m.home], ratings[m.away] = update(rh, ra, m.home_goals, m.away_goals, p)
    return pre, ratings


def played_in_order(matches: list[Match]) -> list[Match]:
    """Same ordering run_elo uses, so pre-match ratings line up with matches."""
    return sorted((m for m in matches if m.played), key=lambda m: (m.season, m.date, m.home))
