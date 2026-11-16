# SupeRR Selector Season 2 — Pressure Performance Analysis

This repository script analyses the supplied IPL and T20I ball-by-ball data, calculates batting and bowling pressure ratings, validates player samples, tests alternative metric weights, and exports ranked Top 10 lists with supporting evidence.

## Source data

Place these six CSV files in a folder (for example, `data/`):

- `ball_by_ball.csv`
- `matches.csv`
- `competitions.csv`
- `teams.csv`
- `playing_eleven.csv`
- `competition_team_player_mapping.csv`

Do not commit the competition-provided raw data to a public repository unless its sharing terms permit it.

## Install dependencies

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python supeer_selector_analysis.py --input-dir ./data --output-dir ./outputs
```

Optional pressure cutoff (default is the 80th percentile within each format and innings):

```bash
python supeer_selector_analysis.py --input-dir ./data --output-dir ./outputs --pressure-quantile 0.75
```

## Main outputs

- `SupeRR_Top10_Batters.csv`
- `SupeRR_Top10_Bowlers.csv`
- `SupeRR_Top10_Rankings.csv`
- `SupeRR_Batter_Top10_Evidence.csv`
- `SupeRR_Bowler_Top10_Evidence.csv`
- `batter_sensitivity_analysis.csv`
- `bowler_sensitivity_analysis.csv`
- `batter_match_consistency.csv`
- `bowler_match_consistency.csv`
- `SupeRR_Batter_Rankings.png`
- `SupeRR_Bowler_Rankings.png`
- `Methodology_Summary.txt`

The analysis also writes `all_batter_format_rankings.csv` and `all_bowler_format_rankings.csv`. These contain the eligible format-level ratings used by the dashboard's IPL and T20I leaderboard filters.

The Pressure Performance Rating (PPR) is a custom, explainable index—not an official cricket statistic or a probability. Review the methodology notes, assumptions, sample sizes and sensitivity results before using the rankings in a competition submission.

## Dashboard

The dashboard reads files from `outputs/`. Generate the outputs first, then run:

```bash
streamlit run app.py
```

The format selector uses format-specific ratings for IPL and T20I. “All formats” uses the combined player ranking. The data audit distinguishes unmatched played players from unplayed squad entries and reports duplicate player-match keys without silently collapsing source records.
