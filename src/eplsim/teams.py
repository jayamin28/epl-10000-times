"""Canonical team names.

Every source spells clubs differently (openfootball: "Manchester United FC",
FPL: "Man Utd", mirror slug: "manchester-united"). Everything is mapped to one
canonical short name here. Unknown names raise, so a new spelling can never
slip through as a 21st club.
"""

from __future__ import annotations

import re

_ALIASES: dict[str, str] = {
    "arsenal": "Arsenal",
    "aston villa": "Aston Villa",
    "birmingham city": "Birmingham",
    "blackburn rovers": "Blackburn",
    "blackpool": "Blackpool",
    "bolton wanderers": "Bolton",
    "bournemouth": "Bournemouth",
    "brentford": "Brentford",
    "brighton": "Brighton",
    "brighton & hove albion": "Brighton",
    "brighton hove albion": "Brighton",
    "burnley": "Burnley",
    "cardiff city": "Cardiff",
    "chelsea": "Chelsea",
    "coventry city": "Coventry",
    "crystal palace": "Crystal Palace",
    "everton": "Everton",
    "fulham": "Fulham",
    "huddersfield town": "Huddersfield",
    "hull city": "Hull",
    "ipswich town": "Ipswich",
    "leeds": "Leeds",
    "leeds united": "Leeds",
    "leicester city": "Leicester",
    "liverpool": "Liverpool",
    "luton town": "Luton",
    "man city": "Man City",
    "manchester city": "Man City",
    "man utd": "Man United",
    "manchester united": "Man United",
    "middlesbrough": "Middlesbrough",
    "newcastle": "Newcastle",
    "newcastle united": "Newcastle",
    "norwich city": "Norwich",
    "nott'm forest": "Nottingham Forest",
    "nottingham forest": "Nottingham Forest",
    "queens park rangers": "QPR",
    "reading": "Reading",
    "sheffield united": "Sheffield United",
    "southampton": "Southampton",
    "spurs": "Tottenham",
    "tottenham hotspur": "Tottenham",
    "stoke city": "Stoke",
    "sunderland": "Sunderland",
    "swansea city": "Swansea",
    "watford": "Watford",
    "west bromwich albion": "West Brom",
    "west ham": "West Ham",
    "west ham united": "West Ham",
    "wigan athletic": "Wigan",
    "wolves": "Wolves",
    "wolverhampton wanderers": "Wolves",
}


def canonical(name: str) -> str:
    """Map any source spelling to the canonical club name. Raises on unknown."""
    key = name.strip().lower().replace("-", " ")
    key = re.sub(r"^afc\s+", "", key)
    key = re.sub(r"\s+a?fc$", "", key)
    key = re.sub(r"\s+", " ", key)
    try:
        return _ALIASES[key]
    except KeyError:
        raise KeyError(f"Unknown team name: {name!r} (normalised: {key!r})") from None
