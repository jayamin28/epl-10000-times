"""Three-letter codes and badge colours for each club.

Badges are plain discs: primary colour fill, secondary colour ring. No crest
artwork is used anywhere in this project. Colour never identifies a club on
its own (seven of the twenty are red), so the code always sits beside it.

Colours are close matches to each club's own palette. `verified` is False
until a value has been checked against the club's published colours.
"""

from __future__ import annotations

CLUBS: dict[str, dict] = {
    "Arsenal": {"code": "ARS", "primary": "#EF0107", "secondary": "#9C824A"},
    "Aston Villa": {"code": "AVL", "primary": "#670E36", "secondary": "#95BFE5"},
    "Bournemouth": {"code": "BOU", "primary": "#DA291C", "secondary": "#000000"},
    "Brentford": {"code": "BRE", "primary": "#E30613", "secondary": "#FFFFFF"},
    "Brighton": {"code": "BHA", "primary": "#0057B8", "secondary": "#FFCD00"},
    "Chelsea": {"code": "CHE", "primary": "#034694", "secondary": "#DBA111"},
    "Coventry": {"code": "COV", "primary": "#78BFEA", "secondary": "#FFFFFF"},
    "Crystal Palace": {"code": "CRY", "primary": "#1B458F", "secondary": "#C4122E"},
    "Everton": {"code": "EVE", "primary": "#003399", "secondary": "#FFFFFF"},
    "Fulham": {"code": "FUL", "primary": "#FFFFFF", "secondary": "#000000"},
    "Hull": {"code": "HUL", "primary": "#F18A01", "secondary": "#000000"},
    "Ipswich": {"code": "IPS", "primary": "#0044A9", "secondary": "#DE2C37"},
    "Leeds": {"code": "LEE", "primary": "#FFFFFF", "secondary": "#1D428A"},
    "Liverpool": {"code": "LIV", "primary": "#C8102E", "secondary": "#00B2A9"},
    "Man City": {"code": "MCI", "primary": "#6CABDD", "secondary": "#1C2C5B"},
    "Man United": {"code": "MUN", "primary": "#DA291C", "secondary": "#FBE122"},
    "Newcastle": {"code": "NEW", "primary": "#241F20", "secondary": "#FFFFFF"},
    "Nottingham Forest": {"code": "NFO", "primary": "#DD0000", "secondary": "#FFFFFF"},
    "Sunderland": {"code": "SUN", "primary": "#EB172B", "secondary": "#211E1F"},
    "Tottenham": {"code": "TOT", "primary": "#132257", "secondary": "#FFFFFF"},
}

COLOURS_VERIFIED = False  # set True once every value is checked against the clubs' published colours


def club(team: str) -> dict:
    """Code and colours for a club. Unknown clubs get a neutral badge and a code from their name."""
    if team in CLUBS:
        return CLUBS[team]
    letters = "".join(ch for ch in team.upper() if ch.isalpha())
    return {"code": letters[:3], "primary": "#8A8A8A", "secondary": "#FFFFFF"}


def code(team: str) -> str:
    return club(team)["code"]
