"""Streamlit dashboard for the SupeRR Selector pressure analysis."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="SupeRR Selector",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

OUTPUTS = Path(__file__).parent / "outputs"


@st.cache_data
def read_csv(filename: str) -> pd.DataFrame:
    path = OUTPUTS / filename
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error(f"Could not read `{filename}`: {exc}")
        return pd.DataFrame()


@st.cache_data
def read_text(filename: str) -> str:
    path = OUTPUTS / filename
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def summary_value(summary: pd.DataFrame, metric: str, default=0):
    if summary.empty or not {"metric", "value"}.issubset(summary.columns):
        return default
    values = summary.loc[summary["metric"].astype(str).eq(metric), "value"]
    return values.iloc[0] if not values.empty else default


def number(value) -> str:
    try:
        return f"{int(float(value)):,}"
    except (TypeError, ValueError):
        return str(value)


def filtered_player_name(df: pd.DataFrame, query: str) -> pd.DataFrame:
    if query and "player_name" in df:
        return df[df["player_name"].astype(str).str.contains(query, case=False, na=False)]
    return df


# Inputs
summary = read_csv("analysis_summary.csv")
matches = read_csv("eligible_matches.csv")
pressure_distribution = read_csv("pressure_distribution_summary.csv")
methodology = read_text("Methodology_Summary.txt")
data_audit = read_csv("data_quality_audit.csv")

# Navigation and filters
with st.sidebar:
    st.title("🏏 SupeRR Selector")
    st.caption("Season 2 · Pressure performance analytics")
    st.divider()
    category = st.radio("Player category", ["Batters", "Bowlers"])
    selected_format = st.selectbox("Competition format", ["All formats", "IPL", "T20I"])
    rows_to_show = st.slider("Leaderboard rows", min_value=5, max_value=10, value=10)
    player_search = st.text_input("Search players", placeholder="Player name")

is_batter = category == "Batters"
role_key = "batter" if is_batter else "bowler"
data_format = {"IPL": "T20", "T20I": "T20I"}.get(selected_format)
if is_batter:
    top_file = "SupeRR_Top10_Batters.csv"
    ranking_file = "all_batter_rankings.csv"
    format_ranking_file = "all_batter_format_rankings.csv"
else:
    top_file = "SupeRR_Top10_Bowlers.csv"
    ranking_file = "all_bowler_rankings.csv"
    format_ranking_file = "all_bowler_format_rankings.csv"

comparison = read_csv(f"{role_key}_pressure_comparison.csv")
consistency = read_csv(f"{role_key}_match_consistency.csv")
sensitivity = read_csv(f"{role_key}_sensitivity_analysis.csv")
rankings = read_csv(ranking_file)
top_ten = read_csv(top_file)

if selected_format != "All formats":
    format_rankings = read_csv(format_ranking_file)
    if format_rankings.empty:
        rankings, top_ten = pd.DataFrame(), pd.DataFrame()
    else:
        format_rankings = format_rankings[
            format_rankings["match_format"].astype(str).str.upper().eq(data_format)
        ].copy()
        rankings = format_rankings.rename(columns={"format_PPR": "PPR"})
        top_ten = rankings.head(10).copy()

    format_consistency = read_csv(f"{role_key}_format_match_consistency.csv")
    format_sensitivity = read_csv(f"{role_key}_format_sensitivity_analysis.csv")
    if not format_consistency.empty:
        consistency = format_consistency[
            format_consistency["match_format"].astype(str).str.upper().eq(data_format)
        ]
    if not format_sensitivity.empty:
        sensitivity = format_sensitivity[
            format_sensitivity["match_format"].astype(str).str.upper().eq(data_format)
        ]
    if not comparison.empty:
        comparison = comparison[
            comparison["match_format"].astype(str).str.upper().eq(data_format)
        ]

rankings = filtered_player_name(rankings, player_search)
top_ten = filtered_player_name(top_ten, player_search)

# Page heading
st.title("Pressure performance")
st.caption(
    "Compare player performance in high-pressure deliveries with performance in normal situations. "
    "PPR is a custom index, not an official cricket statistic."
)
st.divider()

# Summary metrics reflect the selected format and role.
match_count = int(matches["match_id"].nunique()) if not matches.empty and "match_id" in matches else int(summary_value(summary, "eligible_matches"))
ipl_count = int(summary_value(summary, "eligible_ipl_matches"))
t20i_count = int(summary_value(summary, "eligible_t20i_matches"))
if selected_format != "All formats" and not matches.empty and "competition_type" in matches:
    match_count = int(
        matches.loc[
            matches["competition_type"].astype(str).str.upper().eq(selected_format),
            "match_id",
        ].nunique()
    )
    ipl_count = match_count if selected_format == "IPL" else 0
    t20i_count = match_count if selected_format == "T20I" else 0
record_count = int(rankings["player_id"].nunique()) if not rankings.empty and "player_id" in rankings else 0

metric_columns = st.columns(4)
metric_columns[0].metric("Eligible player-format records", number(record_count))
metric_columns[1].metric("Matches analysed", number(match_count))
metric_columns[2].metric("IPL matches", number(ipl_count))
metric_columns[3].metric("T20I matches", number(t20i_count))

st.subheader(f"{selected_format} · {category}")
leaderboard, chart_column = st.columns([1, 1], gap="large")
with leaderboard:
    st.markdown(f"#### Top {rows_to_show} by PPR")
    if top_ten.empty:
        st.info("No ranking rows match these filters. Confirm the analysis outputs are available.")
    else:
        display = top_ten.head(rows_to_show).copy()
        for col in ["PPR", "mean_reliability"]:
            if col in display:
                display[col] = pd.to_numeric(display[col], errors="coerce").round(2)
        display_columns = [
            col for col in ["Rank", "player_name", "PPR", "pressure_matches", "normal_matches", "mean_reliability"]
            if col in display.columns
        ]
        display = display[display_columns].rename(
            columns={
                "player_name": "Player",
                "pressure_matches": "Pressure matches",
                "normal_matches": "Normal matches",
                "mean_reliability": "Reliability",
            }
        )
        st.dataframe(display, width="stretch", hide_index=True)

with chart_column:
    st.markdown("#### PPR leaderboard")
    if not top_ten.empty:
        chart_data = top_ten.head(rows_to_show).sort_values("PPR", ascending=True)
        chart = px.bar(
            chart_data,
            x="PPR",
            y="player_name",
            orientation="h",
            text="PPR",
            labels={"PPR": "Pressure Performance Rating", "player_name": "Player"},
            color_discrete_sequence=["#E87524"],
        )
        chart.update_traces(texttemplate="%{text:.1f}", textposition="outside", cliponaxis=False)
        chart.update_layout(height=420, margin=dict(l=5, r=35, t=10, b=10), yaxis_title="")
        st.plotly_chart(chart, width="stretch")

# Analysis detail views
player_tab, comparison_tab, reliability_tab, methodology_tab = st.tabs(
    ["Player explorer", "Pressure comparison", "Consistency & sensitivity", "Methodology & audit"]
)

with player_tab:
    st.subheader("Player explorer")
    if rankings.empty or "player_name" not in rankings:
        st.info("No eligible players match the current filters.")
    else:
        players = sorted(rankings["player_name"].dropna().astype(str).unique())
        if players:
            player = st.selectbox("Choose a player", players)
            record = rankings[rankings["player_name"].astype(str).eq(player)].iloc[0]
            stats = st.columns(4)
            stats[0].metric("Rank", number(record.get("Rank", "—")))
            stats[1].metric("PPR", f"{float(record['PPR']):.2f}")
            stats[2].metric("Pressure matches", number(record.get("pressure_matches", "—")))
            stats[3].metric("Normal matches", number(record.get("normal_matches", "—")))
            st.markdown("#### Ranking record")
            st.dataframe(pd.DataFrame([record]), width="stretch", hide_index=True)

            player_comparison = comparison[
                comparison.get("player_name", pd.Series(dtype=str)).astype(str).eq(player)
            ] if not comparison.empty else pd.DataFrame()
            if not player_comparison.empty:
                st.markdown("#### Pressure and normal results")
                columns = (
                    ["match_format", "pressure_strike_rate", "normal_strike_rate", "sr_change", "pressure_matches", "normal_matches"]
                    if is_batter
                    else ["match_format", "pressure_economy", "normal_economy", "economy_improvement", "pressure_matches", "normal_matches"]
                )
                st.dataframe(
                    player_comparison[[col for col in columns if col in player_comparison]],
                    width="stretch",
                    hide_index=True,
                )

with comparison_tab:
    st.subheader("Pressure versus normal performance")
    if comparison.empty:
        st.info("No comparison data is available for the current filters.")
    else:
        pressure_col = "pressure_strike_rate" if is_batter else "pressure_economy"
        normal_col = "normal_strike_rate" if is_batter else "normal_economy"
        metric_label = "Strike rate" if is_batter else "Economy rate"
        if {pressure_col, normal_col, "player_name"}.issubset(comparison.columns):
            plot_data = comparison[["player_name", pressure_col, normal_col]].melt(
                id_vars="player_name", var_name="Situation", value_name=metric_label
            )
            plot_data["Situation"] = plot_data["Situation"].map(
                {pressure_col: "Under pressure", normal_col: "Normal"}
            )
            chart = px.scatter(
                plot_data,
                x="Situation",
                y=metric_label,
                color="Situation",
                hover_name="player_name",
                color_discrete_map={"Under pressure": "#E87524", "Normal": "#16877A"},
            )
            st.plotly_chart(chart, width="stretch")
        st.dataframe(comparison, width="stretch", hide_index=True)

with reliability_tab:
    st.subheader("Match-level consistency")
    if consistency.empty:
        st.info("No consistency data is available for this selection.")
    else:
        st.dataframe(
            consistency.sort_values("consistency_pct", ascending=False),
            width="stretch",
            hide_index=True,
        )

    st.subheader("Ranking sensitivity")
    st.caption("Top-10 appearances across the tested alternative weighting schemes.")
    if sensitivity.empty:
        st.info("No sensitivity data is available for this selection.")
    else:
        st.dataframe(
            sensitivity.sort_values(["top10_count", "average_rank"], ascending=[False, True]),
            width="stretch",
            hide_index=True,
        )

with methodology_tab:
    st.subheader("Methodology")
    if methodology:
        st.text(methodology)
    else:
        st.warning("Methodology notes are missing from the outputs folder.")
    if not pressure_distribution.empty:
        st.subheader("Pressure delivery distribution")
        st.dataframe(pressure_distribution, width="stretch", hide_index=True)
    with st.expander("Data quality audit"):
        if data_audit.empty:
            st.info("Data audit file is missing.")
        else:
            st.dataframe(data_audit, width="stretch", hide_index=True)
        audit_metrics = {
            "Unmatched played XI players": "unmatched_playing_xi_players",
            "Unmatched unplayed XI entries": "unmatched_unplayed_xi_players",
            "Duplicate player-match keys": "playing_xi_duplicate_player_match_keys",
            "Keys with conflicting XI attributes": "playing_xi_keys_with_conflicting_attributes",
            "Positive extras with missing type": "unclassified_extra_rows_with_positive_extra_runs",
        }
        for label, key in audit_metrics.items():
            st.write(f"**{label}:** {number(summary_value(summary, key))}")

st.divider()
st.caption(
    "PPR is a custom index, not an official cricket statistic or probability. "
    "Interpret rankings alongside sample sizes, assumptions, and sensitivity results."
)
