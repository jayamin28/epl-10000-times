# Data model

Every run adds a snapshot. Nothing is overwritten except a rerun of the same data. So any question about the season can be asked "as of" any matchweek: where a club stood, what the model gave them, and how both changed.

## Grain and keys

One snapshot = one run of the model on one state of the results. Its key is `(season, matches_played)`. Matchweek is an attribute, not the key: while a matchweek is being played, the daily run can take several snapshots of it (Saturday's results in, Sunday's not yet). For a week-by-week view, take the snapshot with the most results in each matchweek, which is what `site/history.json` does.

```
snapshots ─┬─< history         (one row per club)
           ├─< position_dist   (one row per club per final place)
           ├─< points_dist     (one row per club per final points total)
           └─< forecast_log    (one row per fixture, live runs only)

clubs ──< matches ──< match_ratings    (dimensions and match facts, every season on file)
gameweeks ── points to the snapshot that stands for each gameweek
```

Join on `team` (or `home` / `away`) for clubs, `(season, gameweek)` for gameweeks, `(season, matches_played)` for snapshots.

## Tables

All in `outputs/`, as Parquet.

### `snapshots.parquet`: one row per run

| Column | Meaning |
|---|---|
| `season`, `matches_played` | Key |
| `matchweek` | Latest matchweek with a result |
| `data_through` | Date of the last result the run saw |
| `run_at` | When the run happened (UTC) |
| `reconstructed` | `true` if rebuilt later from the results up to that week. `false` if made at the time |
| `n_sims`, `seed`, `sigma`, `elo_k`, `elo_promoted`, `goals_window_seasons` | The frozen settings |
| `goal_alpha`, `goal_beta`, `goal_gamma` | The goals model as fitted for that run |
| `model_version` | Fingerprint of the settings. If it changes between two snapshots, compare them with care |
| `primary_source`, `run_id` | Which source set the fixture list, and the GitHub run that made it |

### `history.parquet`: one row per club per run

The real table and the model, side by side.

| Real, as it stood | Model, at that moment |
|---|---|
| `table_pos`, `played`, `won`, `drawn`, `lost` | `elo` (strength rating) |
| `goals_for`, `goals_against`, `gd_now`, `pts_now` | `exp_pts` (average final points), `model_pos` (rank by it) |
| | `most_likely_place` |
| | `title`, `top4`, `top5`, `relegated` (share of 10,000 seasons) |

Key: `(season, matches_played, team)`.

### `position_dist.parquet`: one row per club per final place per run

`share` is the fraction of the 10,000 seasons in which `team` finished in `place`. Twenty rows per club, and each club's rows sum to 1. This is the full matrix behind the heatmap, kept for every snapshot.

Key: `(season, matches_played, team, place)`.

### `forecast_log.parquet`: match odds given before kick-off

Live runs only. A rebuilt snapshot never writes here, because odds made after the match are not a forecast. Used to score the model in public.

Key: `(season, home, away)`. The latest odds before kick-off win.

### `points_dist.parquet`: one row per club per final points total per run

`share` is the fraction of the 10,000 seasons in which `team` finished on `points`. Only totals that happened are stored, so each club's rows sum to 1 and `sum(points * share)` equals `history.exp_pts`. Runs made before this table existed were filled by simulating them again; a run is filled only when the new simulation reproduces its stored `position_dist` exactly, so the rows come from the same 10,000 seasons.

Key: `(season, matches_played, team, points)`.

### `matches.parquet`: one row per fixture, every season on file

`gameweek`, `date`, `home`, `away`, `home_code`, `away_code`, `home_goals`, `away_goals`, `result` (H, D or A), `ht_home_goals`, `ht_away_goals` (half-time, where openfootball has it), `status` (played, unplayed, postponed), `sources_backing` (how many sources gave the kept score), `sources`, `overruled` (a source was outvoted). Seasons before 2025-26 come from one source, openfootball. Rewritten whole on each live run.

Key: `(season, home, away)`.

### `match_ratings.parquet`: one row per played match

Both clubs' rating before and after the match, the change, the expected home score from the ratings, and `order`, the order the ratings walk processed the match. The same walk the model runs, so a club's rating after its last match of a gameweek equals `history.elo` for that gameweek.

Key: `(season, home, away)`.

### `clubs.parquet`: one row per club in the data

`code`, colours (`own_colours` is false for past clubs, which get a neutral badge), `colours_verified`, `seasons_on_file`, `first_season`, `in_current_season`, `promoted_this_season`.

Key: `team`.

### `gameweeks.parquet`: one row per season per gameweek

`first_date`, `last_date`, `fixtures`, `played`, `postponed`, `status` (complete, in progress, upcoming), and for this season `snapshot_matches_played` and `snapshot_reconstructed`: the stored run that stands for the gameweek.

Key: `(season, gameweek)`.

### `universes/<season>/gwNN.parquet`: every simulated season, one file per gameweek

One row per universe per club: `universe` (1 to 10,000, the same number the page uses), `team`, `place`, `points`, `goal_diff`, `goals_for`, plus the run's `season`, `matchweek`, `matches_played` and `run_at`. Written by live runs from Gameweek 10 on (no earlier weeks). A later run in the same gameweek overwrites that gameweek's file, so each gameweek keeps exactly one set: the one after its last result. About 0.7 MB a gameweek.

Universe numbers only mean the same season within one gameweek. The next gameweek's run starts from new results, so universe 4,817 there is a different season. Always show a universe with its gameweek.

Key: `(season, matchweek, universe, team)`.

For the page, each live run also writes `site/universes/<season>/gwNN.json` (the same gameweek, packed: `place` is one letter per club per universe, A = 1st to T = 20th; `points` is two base-36 digits per club per universe; clubs in `teams` order; universe n starts at (n - 1) x 20) and `site/universes/index.json`, the list of gameweeks the page can offer. The page's universe picker reads them.

The forecast log also carries `home_xg`, `away_xg`, `home_elo` and `away_elo` from 17 Nov 2026: the expected goals and the ratings behind each set of odds.

### Also written

| File | What |
|---|---|
| `seasons_latest.parquet` | All 10,000 simulated seasons of the latest run: every club's place and points |
| `site/feed.json` | Everything the page shows for the latest run |
| `site/snapshots/mwNN.json` | The same, frozen for each matchweek. The page's matchweek picker reads these |
| `site/history.json` | Every club's series over the season, one point per matchweek, for the timeline charts |

## Rebuilt matchweeks

The model went live after matchweek 10. Matchweeks 1 to 9 were rebuilt afterwards: for each one the run sees only the results up to that week and uses today's settings. They are what the model would have said, not what it said, and they are marked `reconstructed = true` everywhere. Any finished matchweek that has no snapshot (for example if a scheduled run fails) is filled the same way on the next run. A snapshot made at the time is never replaced by a rebuilt one.

## Example: a club's season, as of every matchweek

```python
import pandas as pd

h = pd.read_parquet("outputs/history.parquet")
h = h[(h.season == "2026-27") & (h.team == "Liverpool")]
weekly = h.loc[h.groupby("matchweek").matches_played.idxmax()]
print(weekly[["matchweek", "table_pos", "pts_now", "model_pos", "exp_pts", "title", "top4", "reconstructed"]])
```

The same in SQL (DuckDB reads Parquet directly):

```sql
select matchweek, table_pos, pts_now, model_pos, round(title * 100, 1) as title_pct, reconstructed
from 'outputs/history.parquet'
where season = '2026-27' and team = 'Liverpool'
qualify matches_played = max(matches_played) over (partition by matchweek)
order by matchweek;
```
