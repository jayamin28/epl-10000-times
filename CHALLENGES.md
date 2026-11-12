# What went wrong, and what I did about it

A build log for [The Premier League, 10,000 times](README.md). Every entry is something that broke, surprised me or could not be fixed, with the fix and where to check it.

## Data

**The rating I wanted was not there.** The plan was to use ClubElo ratings. The API was unreachable from my build environment and the public mirror stopped updating in January 2026. I computed my own Elo from every league result since 2010 and tuned it on seasons the model had never seen. Check: `src/eplsim/elo.py`, `tests/test_model_evidence.py`.

**One source had the wrong score.** The only up-to-date source had Liverpool 0-1 Man United for 19 Oct 2025. The real score was 1-2. Fix: three sources, majority vote, and a hard stop when there is no majority. Check: `reconcile` in `src/eplsim/data.py`.

**I dropped 32 real goalless draws, and wrote a test that protected the mistake.** One source writes results as an object with a full-time score, except 0-0 draws, which are a bare `[0, 0]`. I read that as "not played yet", skipped it, and wrote a test enforcing it. I caught it hours later while fact-checking a number for this log. No published number moved, because the other two sources carried those games. That was luck in the design, not skill in the parsing. Check: `tests/test_data.py`.

**Clubs stored as decimal codes.** One source stores each club as a number like `43.0`. It is a club code, not the club id used elsewhere in the same repository. I resolved codes through that season's club list, then checked all 760 fixtures against the club names inside each match id.

**56 spellings for 42 clubs.** One source alone spells its clubs 56 ways across seasons. Others add "Man Utd", "Spurs" and "Nott'm Forest". There is now one canonical name per club and a loader that refuses any spelling it has not seen, so a typo can never become a 21st club. Check: `src/eplsim/teams.py`.

**The freshest source was the least official.** The two better-known sources were weeks behind. The current one is a hobby repository, and it is the one with the wrong score above. I use it, but never alone.

**A source with no licence.** Before publishing I checked what each source allows. One has no licence file at all, which means no permission to copy. I store only match facts from it (fixture, date, score) and none of its ratings or statistics. A test fails if anything else reaches disk. Check: `data/NOTICE.md`, `tests/test_fetch.py`.

## Model

**The best setting sat on the edge.** The first tuning run picked the lowest K it was allowed to try. An optimum on the boundary means the search stopped too early. I widened the grid from 60 to 135 settings until the best sat inside it. A test fails if it ever lands on the edge again.

**A borrowed number did not transfer.** The video that inspired this uses 35 Elo points of rating uncertainty. That is on a different rating scale. I rewound 12 past seasons to matchday 10, simulated each forward and scored the odds against what happened. The score was nearly flat across settings, so I took the more humble value, 65.

**The bias I could not fix.** On unseen seasons the model predicts fewer goals than were scored: about 2.81 a match against 2.95. The league got higher scoring from 2022. I tried six training windows and none removed it. It is measured, monitored and written on the page as a limit.

**A rule with nothing to test it against.** Clubs level on points, goal difference and goals scored are split by head to head. In 16 seasons of data that has never happened once. I tested it with hand-built cases, plus the three real seasons where the step before it decided a place. Check: `tests/test_tiebreak.py`.

## Trusting the numbers

**Same data, different answer.** Two files holding the same 100 results gave Liverpool 575 title wins in one run and 540 in the other. The cause: the sources list fixtures in a different order, and the random numbers were dealt out in fixture order. Fixtures are now sorted before any dice are rolled, and a test shuffles the input and demands identical seasons. Check: `tests/test_run_week.py`.

**I was wrong about the noise.** I assumed two runs on the same data would differ by about 1 percentage point. Measured over 30 runs it is closer to 1.3, with gaps over 2 between the luckiest and unluckiest run. The rule on the page: a change under 3 points is not news.

**A test suite that had never failed.** The first full run passed. A suite that has never failed has proved nothing. I injected deliberate bugs one at a time: an away win worth 2 points, a minority source winning a vote, relegation counting the bottom two. Planning them exposed one test that would have let a real bug through. It was fixed first. All 15 are now caught.

**Seven of twenty clubs are red.** The page uses club-colour discs instead of crests. Seven clubs are red, so colour alone cannot tell them apart. The three-letter code carries identity and colour supports it.
