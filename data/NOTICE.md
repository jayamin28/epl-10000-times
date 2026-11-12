# Data in this folder

Everything here is match facts: who played whom, when, and the score.

| Folder | Source | Terms |
|---|---|---|
| `history/openfootball`, `current/openfootball` | [openfootball/football.json](https://github.com/openfootball/football.json) | Public domain (CC0). Stored as fetched. |
| `history/fpl`, `current/fpl` | The official Fantasy Premier League fixtures feed. The live API when reachable, otherwise the [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) mirror (MIT licence, copyright Vaastav Anand). | Trimmed to fixture, kick-off time, score and ids. No player data. |
| `history/fei_*`, `current/fei_*` | [olbauday/FPL-Elo-Insights](https://github.com/olbauday/FPL-Elo-Insights) | That repository publishes no licence file. Only fixture, date and score are kept here. Its ratings and statistics are not copied or used. |

`current/manifest.json` records when each source was fetched and a hash of what it served.

A result is used only when the sources agree. Where they disagree, the majority wins and the disagreement is recorded.

If you maintain one of these sources and want something changed or removed, open an issue.
