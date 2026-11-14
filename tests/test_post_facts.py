import json
from pathlib import Path

import pytest

from eplsim import post_facts as pf

FEED = Path(__file__).resolve().parents[1] / "site" / "feed.json"


def test_odds_are_phrased_the_way_posts_print_them():
    assert pf.pct(0.6631) == "66%"
    assert pf.pct(0.0785) == "7.8%"
    assert pf.pct(0.0002) == "under 0.1%"
    assert pf.pct(0.0) == "0%"
    assert pf.one_in(0.0539) == "1 in 19"
    assert pf.one_in(0.0002) == "1 in 5,000"
    assert pf.seasons(0.6631) == "6,631 of 10,000"
    assert pf.odds(0.6631) == "66%, 6,631 of 10,000, misses 1 in 3"
    assert pf.odds(0.0539) == "5.4%, 539 of 10,000, 1 in 19"
    assert [pf.ordinal(n) for n in (1, 2, 3, 4, 11, 12, 13, 21, 22)] == ["1st", "2nd", "3rd", "4th", "11th", "12th",
                                                                         "13th", "21st", "22nd"]


def test_check_passes_numbers_on_the_sheet_and_flags_the_rest():
    sheet = "Arsenal 66%, misses 1 in 3. Liverpool 1 in 19. Man United 18th, 7.8%. Hull 40%. 11 different champions."
    assert pf.check("After Gameweek 10", sheet) == ["10"]  # 10 is not on this sheet
    assert pf.check("Arsenal win it in 66% of seasons, Liverpool 1 in 19, from 1 to 10,000.", sheet) == []
    assert pf.check("Arsenal win it in 71% and Liverpool 1 in 22.", sheet) == ["71%", "1 in 22"]
    # a "1 in N" must be on the sheet as a phrase; a bare 13 elsewhere does not make "1 in 13" true
    assert pf.check("United are 18th but go down 1 in 13.", sheet + " about 13 seasons") == ["1 in 13"]
    assert pf.check("United go down 1 in 13.", sheet + " misses 1 in 13") == []


@pytest.mark.skipif(not FEED.exists(), reason="needs a committed feed")
def test_fact_sheet_from_the_committed_feed_holds_every_headline_number():
    feed = json.loads(FEED.read_text(encoding="utf-8"))
    sheet = pf.fact_sheet(feed)
    gw = feed["meta"]["matchweek"]
    assert f"# Fact sheet: after Gameweek {gw}" in sheet
    top = feed["clubs"][0]
    assert f"| {top['team']} |" in sheet and pf.pct(top["title"]) in sheet
    assert "Analysis for fun. Not betting advice." in sheet
    assert "—" not in sheet
    # a draft built only from the sheet passes its own check
    assert pf.check(sheet, sheet) == []
