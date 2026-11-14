# Working on this repo with an AI agent

This file is for any AI coding agent (Claude Code, Codex, Cursor and others) opened in this repo, and for people who want to know where the AI fits in.

## The one rule

**The numbers are made by code, never by the agent.** The daily pipeline is plain, deterministic Python on a schedule: same data in, same answer out, every time. An agent may change that code through a normal commit with passing tests. It never writes, edits or estimates a published number by hand.

Where an agent does help:

| Job | Who does it |
|---|---|
| Fetch, check, rate, simulate, publish (every morning) | The pipeline in `src/eplsim`, run by `.github/workflows/weekly.yml`. No AI involved. |
| Building and changing that pipeline | A person with an AI coding agent, through commits, tests and review |
| Turning each week's run into a post: angles, words, a number check | The `weekly-post` agent (`.claude/agents/weekly-post.md`), with a person approving before anything is posted |

## How the pipeline runs

1. Wake at 07:00 UTC, run the tests. Any failure stops the run.
2. Fetch this season's results from three sources: the official FPL feed (with a public mirror as fallback), FPL-Elo-Insights and openfootball (`src/eplsim/fetch.py`).
3. Keep a result only when at least two sources agree (`data.reconcile`).
4. Gates (`src/eplsim/gates.py`): complete fixture list, two of three sources reachable, no overdue results, no results vanished. A failure stops the run and the last good page stays up.
5. No new results: stop. Otherwise rebuild any finished gameweek that has no snapshot, then rate every club (Elo since 2010), fit the goals model, play the rest of the season 10,000 times with seed 2026, and count.
6. Save a snapshot, commit, deploy the page.

The data model is in `docs/DATA_MODEL.md`. The challenges met on the way are in `CHALLENGES.md`.

## Commands

```bash
pip install -r requirements.txt
python -m pytest                                   # must pass before any commit
PYTHONPATH=src python -m eplsim.run_week --offline --force   # rerun from the files on disk
PYTHONPATH=src python -m eplsim.run_week --as-of 7            # the picture as it stood after Gameweek 7
PYTHONPATH=src python -m eplsim.post_facts                    # this week's fact sheet for posts
PYTHONPATH=src python -m eplsim.post_facts --check draft.md   # fail if a draft uses a number not on the sheet
```

## Rules for changing code

- Run the full test suite and show its result before calling a change done.
- Don't change the frozen settings in `src/eplsim/config.py` (seed, sigma, Elo, goals window, number of seasons) to make a result look better. Changing them changes `model_version`, and every earlier snapshot was made with the old one.
- Never hand-edit anything under `outputs/`, `site/feed.json`, `site/history.json` or `site/snapshots/`. The scheduled run owns them.
- Odds that go in the forecast log are logged before kick-off only. Don't backfill them.
- The launch record in `outputs/archive/` is the baseline the season is scored against. Never change it.
- If a fix breaks something, say so and fix it. Don't work around a failing test.
- No club crests and no real match photos anywhere. Clubs are shown by name, three-letter code and colours.

## Writing posts

Use the `weekly-post` agent. In short: build the fact sheet, pick the angles, draft, check every number against the sheet, hand the draft to a person. Nothing is posted without them.
