# SupeRR Selector Season 2 — Interactive Dashboard

An interactive Streamlit dashboard for exploring the supplied IPL and T20I pressure-performance analysis.

## Features

- Top 10 batter and bowler PPR leaderboards
- Search and explore individual players
- Pressure versus normal performance comparisons
- IPL and T20I format-specific rankings and comparisons
- Match-level consistency analysis
- Ranking sensitivity analysis
- Methodology and pressure-distribution notes

## Project structure

```text
SupeRR_Selector_Dashboard/
├── app.py
├── requirements.txt
├── README_SupeRR.md
├── .gitignore
└── outputs/
    ├── SupeRR_Top10_Batters.csv
    ├── SupeRR_Top10_Bowlers.csv
    ├── all_batter_rankings.csv
    ├── all_bowler_rankings.csv
    ├── all_batter_format_rankings.csv
    ├── all_bowler_format_rankings.csv
    ├── batter_pressure_comparison.csv
    ├── bowler_pressure_comparison.csv
    ├── batter_sensitivity_analysis.csv
    ├── bowler_sensitivity_analysis.csv
    ├── batter_match_consistency.csv
    ├── bowler_match_consistency.csv
    ├── analysis_summary.csv
    ├── pressure_distribution_summary.csv
    └── Methodology_Summary.txt
```

## Run locally on macOS

Open Terminal and navigate into the extracted project folder.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will display a local address, usually:

```text
http://localhost:8501
```

To stop the dashboard, press `Control + C` in Terminal. To leave the virtual environment, run:

```bash
deactivate
```

## Publish the dashboard and get a public URL

### 1. Create a GitHub repository

Create a repository named `supeer-selector-dashboard` on GitHub.

### 2. Upload the project

Upload `app.py`, `requirements.txt`, this README, and the complete `outputs/` folder. Keep the folder structure shown above. Generate current outputs with `python supeer_selector_analysis.py --input-dir ./data --output-dir ./outputs` before deployment so the format-specific rankings and audit summary are present.

### 3. Deploy with Streamlit Community Cloud

1. Visit https://share.streamlit.io/
2. Sign in with GitHub.
3. Choose **Create app**.
4. Select your repository and branch.
5. Set the main file path to `app.py`.
6. Select **Deploy**.

Streamlit Community Cloud will build the app and provide a shareable `*.streamlit.app` URL.

## Requirements

The required Python packages are listed in `requirements.txt`. They include Streamlit, Pandas, NumPy, and Plotly.

## Data and privacy

This dashboard package is intended to contain selected analysis outputs, not the original raw ball-by-ball source data or a local virtual environment. Before making the GitHub repository public, confirm that the competition's data-sharing terms permit publishing the included outputs.

Do not upload API keys, passwords, access tokens, or other secrets.

## Interpretation note

PPR (Pressure Performance Rating) is a custom analytical index, not an official cricket statistic or a probability. Interpret rankings alongside match counts, eligibility rules, methodology assumptions, and sensitivity results. The historical rankings may include players who are no longer active; do not describe them as current-player rankings unless an active-player eligibility filter has been applied.

The generated `outputs/pressure_scored_deliveries.csv` exceeds GitHub's regular per-file size limit. A repository snapshot may store it as `outputs/pressure_scored_deliveries.csv.gz`; the analysis script recreates the uncompressed CSV when run.
