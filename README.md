# The Premier League, 10,000 times

After every matchweek, this project plays out the rest of the Premier League season 10,000 times and counts what happens: who wins the title, who makes the top four, who goes down.

**Live page:** https://jayamin28.github.io/epl-10000-times/

Inspired by [Bink Sport's video](https://www.youtube.com/watch?v=JSAjJsbysVk), which did this for the Champions League.

## How it works

1. **Rate every club.** An Elo rating, updated after every league match since 2010.
2. **Turn the rating gap into goals.** A Poisson model gives each side an expected number of goals. Home advantage is included.
3. **Roll the dice.** Every remaining match is played with random goals around those expectations. That is one possible season.
4. **Repeat 10,000 times** and count. Ties are split the way the league's own rules say: points, goal difference, goals scored, then head to head.

Results already played are fixed. Only the remaining matches are simulated.

## Run it

```
pip install -r requirements.txt
python -m pytest                              # the test suite
PYTHONPATH=src python -m eplsim.run_week      # fetch, check, simulate, write outputs
```

| Output | What it holds |
|---|---|
| `site/feed.json` | Everything the page draws |
| `outputs/seasons_latest.parquet` | All 10,000 seasons: every club's place and points |
| `outputs/history.parquet` | Headline odds per club, per matchweek |
| `outputs/forecast_log.parquet` | Match odds logged before kick-off, scored later |

A scheduled GitHub Action runs this every morning. It publishes only when a new result has arrived and every check passes.

## Checks before anything is published

- The fixture list is complete: 380 matches, 20 clubs, 19 home and 19 away each.
- At least two of three data sources answered.
- No finished match is missing its result (postponed matches go in `data/postponed.json`).
- No result has disappeared since the last run.
- The test suite passes.

If a check fails, the last good page stays up.

## What it cannot see

- Injuries, transfers, managers, fixture congestion. It knows results only.
- It predicts slightly fewer goals than are really scored.
- Two runs on the same data differ by about 1 percentage point. A change under 3 points is noise.

## What went wrong

The bugs, the wrong assumptions and the one bias I could not fix are in [`CHALLENGES.md`](CHALLENGES.md).

## Data and credits

See [`data/NOTICE.md`](data/NOTICE.md). Sources: openfootball (public domain), the official Fantasy Premier League fixtures feed, and FPL-Elo-Insights.

Club discs on the page use approximate club colours and three-letter codes. They are not official crests. This project is not affiliated with the Premier League or any club.

Analysis for fun. Not betting advice.
