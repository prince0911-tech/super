# SupeRR Selector

Season 2 pressure performance analysis for IPL and T20I cricket. The project includes a Python analysis pipeline and a Streamlit dashboard for exploring batter and bowler rankings.

**Live dashboard:** [superr.streamlit.app](https://superr.streamlit.app)

## What it does

- Scores pressure at delivery level from match context.
- Compares batting and bowling performance in pressure and normal situations.
- Builds format-specific and combined Pressure Performance Ratings (PPR).
- Checks sample eligibility, rank sensitivity, and match-level consistency.
- Exports rankings, evidence tables, methodology notes, audit summaries, and charts.

PPR is a custom index, not an official cricket statistic or a probability. Interpret it alongside the sample sizes, assumptions, and sensitivity results.

## Project files

```text
app.py                         Streamlit dashboard
supeer_selector_analysis.py   Analysis pipeline
requirements.txt               Python dependencies
data/                          Supplied source CSV files
outputs/                       Generated rankings and dashboard data
```

The analysis expects these files in `data/`: `ball_by_ball.csv`, `matches.csv`, `competitions.csv`, `teams.csv`, `playing_eleven.csv`, and `competition_team_player_mapping.csv`.

## Install

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the analysis

```bash
python supeer_selector_analysis.py --input-dir ./data --output-dir ./outputs
```

The pressure cutoff defaults to the 80th percentile within each format and innings. To use a different cutoff, for example 75%:

```bash
python supeer_selector_analysis.py --input-dir ./data --output-dir ./outputs --pressure-quantile 0.75
```

The pipeline writes combined and format-specific rankings, pressure comparisons, sensitivity and consistency tables, audit files, evidence tables, plots, and methodology notes into `outputs/`.

## Run the dashboard

Generate the analysis outputs first, then start Streamlit:

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`. The dashboard supports player search, batter and bowler views, combined or format-specific rankings, pressure comparisons, consistency, sensitivity, methodology, and data audit views.

## Deploy with Streamlit Community Cloud

1. Select this GitHub repository in Streamlit Community Cloud.
2. Set the app file path to `app.py`.
3. Deploy. Keep `requirements.txt` and the `outputs/` folder in the repository.

## Large generated file

`outputs/pressure_scored_deliveries.csv` is larger than GitHub's regular per-file limit, so this repository stores a compressed snapshot at `outputs/pressure_scored_deliveries.csv.gz`. Running the analysis regenerates the uncompressed CSV. The dashboard does not require this delivery-level file.

## Data and interpretation

The included source data is supplied with this project. Confirm that its terms allow your intended redistribution before using it in another public repository. Historical rankings can include players who are no longer active; no active-player filter is applied.
