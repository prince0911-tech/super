#!/usr/bin/env python3
"""SupeRR Selector Season 2: pressure-performance analysis for IPL and T20I.

Expected input files (CSV) in --input-dir:
    ball_by_ball.csv
    matches.csv
    competitions.csv
    teams.csv
    playing_eleven.csv
    competition_team_player_mapping.csv

Example:
    python supeer_selector_analysis.py --input-dir ./data --output-dir ./outputs

Dependencies:
    pandas, numpy, matplotlib

The script writes cleaned match data, pressure-scored deliveries, player-level
metrics, ratings, sensitivity tables, charts, and submission-ready Top 10 files.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LOGGER = logging.getLogger("supeer_selector")

INPUT_FILES = {
    "balls": "ball_by_ball.csv",
    "matches": "matches.csv",
    "competitions": "competitions.csv",
    "teams": "teams.csv",
    "playing_eleven": "playing_eleven.csv",
    "player_mapping": "competition_team_player_mapping.csv",
}

BATTER_WEIGHTS = {
    "sr_change": 0.50,
    "dot_reduction": 0.20,
    "boundary_change": 0.15,
    "dismissal_improvement": 0.15,
}

BOWLER_WEIGHTS = {
    "economy_improvement": 0.40,
    "wicket_improvement": 0.25,
    "dot_improvement": 0.20,
    "boundary_improvement": 0.15,
}

BATTER_SCENARIOS = {
    "Baseline": BATTER_WEIGHTS,
    "Equal Weights": {
        "sr_change": 0.25,
        "dot_reduction": 0.25,
        "boundary_change": 0.25,
        "dismissal_improvement": 0.25,
    },
    "Strike Rate Focus": {
        "sr_change": 0.65,
        "dot_reduction": 0.10,
        "boundary_change": 0.15,
        "dismissal_improvement": 0.10,
    },
    "Dismissal Focus": {
        "sr_change": 0.30,
        "dot_reduction": 0.15,
        "boundary_change": 0.15,
        "dismissal_improvement": 0.40,
    },
}

BOWLER_SCENARIOS = {
    "Baseline": BOWLER_WEIGHTS,
    "Equal Weights": {
        "economy_improvement": 0.25,
        "wicket_improvement": 0.25,
        "dot_improvement": 0.25,
        "boundary_improvement": 0.25,
    },
    "Economy Focus": {
        "economy_improvement": 0.55,
        "wicket_improvement": 0.20,
        "dot_improvement": 0.15,
        "boundary_improvement": 0.10,
    },
    "Wicket Focus": {
        "economy_improvement": 0.25,
        "wicket_improvement": 0.45,
        "dot_improvement": 0.15,
        "boundary_improvement": 0.15,
    },
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse IPL and T20I player performance under pressure."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path.cwd(),
        help="Folder containing the six source CSV files (default: current folder).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("supeer_selector_outputs"),
        help="Folder where generated files will be saved.",
    )
    parser.add_argument(
        "--pressure-quantile",
        type=float,
        default=0.80,
        help="Within-format/innings pressure percentile cutoff (default: 0.80).",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def save_csv(df: pd.DataFrame, output_dir: Path, filename: str) -> None:
    df.to_csv(output_dir / filename, index=False)


def robust_standardize_by_format(
    data: pd.DataFrame,
    metrics: Iterable[str],
    format_column: str = "match_format",
) -> pd.DataFrame:
    """Robust z-scores using median and scaled MAD within each format."""
    result = data.copy()

    for metric in metrics:
        z_column = f"{metric}_z"
        result[z_column] = np.nan

        for _, index in result.groupby(format_column, dropna=False).groups.items():
            values = pd.to_numeric(result.loc[index, metric], errors="coerce")
            valid = values.dropna()
            if valid.empty:
                continue

            median = valid.median()
            mad = (valid - median).abs().median()
            scale = 1.4826 * mad

            if not np.isfinite(scale) or scale <= 1e-12:
                scale = valid.std(ddof=0)
            if not np.isfinite(scale) or scale <= 1e-12:
                scale = 1.0

            z_values = ((values - median) / scale).clip(-3, 3)
            result.loc[index, z_column] = z_values

    return result


def build_overall_ranking(
    format_ratings: pd.DataFrame,
    weights: Dict[str, float],
    exposure_column: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build format ratings and combine formats into one player ranking."""
    per_format = format_ratings.copy()

    per_format["raw_score"] = 0.0
    for metric, weight in weights.items():
        per_format["raw_score"] += (
            per_format[f"{metric}_z"].fillna(0) * weight
        )

    per_format["adjusted_score"] = (
        per_format["raw_score"] * per_format["reliability"]
    )
    per_format["format_PPR"] = 50 + 10 * per_format["adjusted_score"]

    # Reliability is already applied to the format rating. Exposure is used
    # only to weight the combination of format ratings, avoiding double penalty.
    per_format["aggregation_weight"] = np.sqrt(
        per_format[exposure_column].clip(lower=1)
    )
    per_format["weighted_PPR"] = (
        per_format["format_PPR"] * per_format["aggregation_weight"]
    )

    overall = (
        per_format.groupby(["player_id", "player_name"], as_index=False)
        .agg(
            weighted_score=("weighted_PPR", "sum"),
            total_weight=("aggregation_weight", "sum"),
            pressure_matches=("pressure_matches", "sum"),
            normal_matches=("normal_matches", "sum"),
            formats_covered=("match_format", "nunique"),
            mean_reliability=("reliability", "mean"),
        )
    )
    overall["PPR"] = safe_divide(
        overall["weighted_score"], overall["total_weight"]
    )
    overall = overall.dropna(subset=["PPR"]).sort_values(
        ["PPR", "pressure_matches", "player_name"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    overall.insert(0, "Rank", np.arange(1, len(overall) + 1))

    return overall, per_format


def build_rank_stability(
    scenario_rankings: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    tables: List[pd.DataFrame] = []
    for scenario, ranking in scenario_rankings.items():
        table = ranking[["player_id", "player_name", "Rank"]].copy()
        table = table.rename(columns={"Rank": scenario})
        tables.append(table)

    if not tables:
        return pd.DataFrame()

    result = tables[0]
    for table in tables[1:]:
        result = result.merge(
            table,
            on=["player_id", "player_name"],
            how="outer",
            validate="one_to_one",
        )

    rank_columns = list(scenario_rankings.keys())
    result["top10_count"] = result[rank_columns].le(10).sum(axis=1)
    result["average_rank"] = result[rank_columns].mean(axis=1)
    return result.sort_values(
        ["top10_count", "average_rank", "player_name"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def build_format_rank_stability(
    scenario_rankings: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Build alternative-weight rank stability separately for each format."""
    formats = sorted(
        set().union(*(set(ranking["match_format"].dropna()) for ranking in scenario_rankings.values()))
    )
    tables = []
    for match_format in formats:
        per_format_scenarios = {
            scenario: ranking.loc[ranking.match_format.eq(match_format)]
            for scenario, ranking in scenario_rankings.items()
        }
        stability = build_rank_stability(per_format_scenarios)
        stability.insert(2, "match_format", match_format)
        tables.append(stability)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


# ---------------------------------------------------------------------------
# Data loading and source validation
# ---------------------------------------------------------------------------


def load_source_data(input_dir: Path):
    loaded: Dict[str, pd.DataFrame] = {}
    for key, filename in INPUT_FILES.items():
        path = input_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Could not find {filename} in {input_dir.resolve()}"
            )
        loaded[key] = pd.read_csv(path)
        LOGGER.info("Loaded %-18s shape=%s", filename, loaded[key].shape)
    return loaded


def create_data_audit(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in data.items():
        rows.append(
            {
                "dataset": INPUT_FILES[name],
                "rows": len(df),
                "columns": len(df.columns),
                "exact_duplicate_rows": int(df.duplicated().sum()),
                "missing_cells": int(df.isna().sum().sum()),
            }
        )
    return pd.DataFrame(rows)


def validate_and_enrich_supporting_data(
    matches: pd.DataFrame,
    competitions: pd.DataFrame,
    teams: pd.DataFrame,
    playing_eleven: pd.DataFrame,
    player_mapping: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    require_columns(
        matches,
        ["match_id", "comp_id", "match_date", "result", "revised_target",
         "first_inn_revised_overs", "second_inn_revised_overs"],
        "matches.csv",
    )
    require_columns(
        competitions,
        ["comp_id", "comp_name", "match_format", "season"],
        "competitions.csv",
    )
    require_columns(teams, ["comp_id", "team_id", "team_name"], "teams.csv")
    require_columns(
        playing_eleven,
        ["match_id", "team_id", "player_id", "is_played"],
        "playing_eleven.csv",
    )
    require_columns(
        player_mapping,
        ["comp_id", "team_id", "player_id", "player_name"],
        "competition_team_player_mapping.csv",
    )

    if competitions["comp_id"].duplicated().any():
        raise ValueError("competitions.csv contains duplicate comp_id values.")
    if matches["match_id"].duplicated().any():
        raise ValueError("matches.csv contains duplicate match_id values.")
    if teams.duplicated(["comp_id", "team_id"]).any():
        raise ValueError("teams.csv contains duplicate (comp_id, team_id) keys.")
    if player_mapping.duplicated(["comp_id", "team_id", "player_id"]).any():
        raise ValueError(
            "Player mapping contains duplicate (comp_id, team_id, player_id) keys."
        )

    match_info = matches.merge(
        competitions[["comp_id", "comp_name", "match_format", "season"]],
        on="comp_id",
        how="left",
        validate="many_to_one",
        indicator="competition_join",
    )

    unmatched_competitions = int(match_info["competition_join"].eq("left_only").sum())
    match_info = match_info.drop(columns="competition_join")
    match_info["match_date"] = pd.to_datetime(
        match_info["match_date"], errors="coerce"
    )

    xi_context = playing_eleven.merge(
        matches[["match_id", "comp_id"]],
        on="match_id",
        how="left",
        validate="many_to_one",
    )
    xi_named = xi_context.merge(
        player_mapping[["comp_id", "team_id", "player_id", "player_name"]],
        on=["comp_id", "team_id", "player_id"],
        how="left",
        validate="many_to_one",
        indicator="player_join",
    )
    unmatched_xi_players = int(
        (xi_named["player_join"].eq("left_only") & as_bool(xi_named["is_played"])).sum()
    )
    unmatched_unplayed_xi_players = int(
        (xi_named["player_join"].eq("left_only") & ~as_bool(xi_named["is_played"])).sum()
    )
    xi_named = xi_named.drop(columns="player_join")
    xi_named = xi_named.merge(
        teams,
        on=["comp_id", "team_id"],
        how="left",
        validate="many_to_one",
        indicator="team_join",
    )
    unmatched_xi_teams = int(xi_named["team_join"].eq("left_only").sum())
    xi_named = xi_named.drop(columns="team_join")

    stats = {
        "unmatched_competition_matches": unmatched_competitions,
        "unmatched_playing_xi_players": unmatched_xi_players,
        "unmatched_unplayed_xi_players": unmatched_unplayed_xi_players,
        "unmatched_playing_xi_teams": unmatched_xi_teams,
        # Distinguish repeated keys from repeated rows: this source has multiple
        # batting-order values for some players, so collapsing them would lose data.
        "playing_xi_duplicate_player_match_key_rows": int(
            playing_eleven.duplicated(
                ["match_id", "team_id", "player_id"], keep=False
            ).sum()
        ),
        "playing_xi_duplicate_player_match_keys": int(
            playing_eleven.loc[
                playing_eleven.duplicated(
                    ["match_id", "team_id", "player_id"], keep=False
                ), ["match_id", "team_id", "player_id"]
            ].drop_duplicates().shape[0]
        ),
        "playing_xi_keys_with_conflicting_attributes": int(
            playing_eleven.groupby(["match_id", "team_id", "player_id"])
            .filter(lambda group: len(group.drop_duplicates()) > 1)
            .groupby(["match_id", "team_id", "player_id"])
            .ngroups
        ),
    }
    return match_info, xi_named, stats


# ---------------------------------------------------------------------------
# Competition selection and delivery context
# ---------------------------------------------------------------------------


def select_eligible_matches(match_info: pd.DataFrame) -> pd.DataFrame:
    is_ipl = match_info["comp_name"].astype("string").str.match(
        r"^IPL(?:\s|$)", case=False, na=False
    )
    is_t20i = match_info["match_format"].astype("string").str.upper().eq("T20I")

    selected = match_info.loc[is_ipl | is_t20i].copy()
    incomplete = selected["result"].fillna("").astype(str).str.contains(
        r"no\s*result|abandon", case=False, regex=True
    )
    selected = selected.loc[~incomplete].copy()

    selected["competition_type"] = np.where(
        selected["comp_name"].astype("string").str.match(
            r"^IPL(?:\s|$)", case=False, na=False
        ),
        "IPL",
        "T20I",
    )

    if selected["match_id"].nunique() != len(selected):
        raise ValueError("Eligible match selection contains duplicate match IDs.")
    return selected


def build_delivery_dataset(
    balls: pd.DataFrame,
    eligible_matches: pd.DataFrame,
    player_mapping: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(
        balls,
        ["match_id", "innings", "innings_ball_num", "batting_team_id",
         "bowling_team_id", "striker_id", "striker_name", "bowler_id",
         "bowler_name", "runs", "is_four", "is_six", "is_wicket",
         "dismissal_type", "dismissed_player_id", "extras_type",
         "extras1_runs", "extras_2_runs"],
        "ball_by_ball.csv",
    )

    match_columns = [
        "match_id", "comp_id", "comp_name", "match_format", "season",
        "match_date", "stage", "result", "revised_target",
        "first_inn_revised_overs", "second_inn_revised_overs",
        "competition_type",
    ]
    eligible_ids = set(eligible_matches["match_id"])
    deliveries = balls.loc[balls["match_id"].isin(eligible_ids)].copy()

    deliveries = deliveries.merge(
        eligible_matches[match_columns],
        on="match_id",
        how="left",
        validate="many_to_one",
    )

    # Innings 3 and 4 are Super Over innings in this supplied dataset.
    deliveries = deliveries.loc[deliveries["innings"].isin([1, 2])].copy()

    # Use competition/team/player mapping as the canonical player-name source.
    id_names = player_mapping[["player_id", "player_name"]].drop_duplicates()
    conflicting_ids = id_names.groupby("player_id")["player_name"].nunique()
    if conflicting_ids.gt(1).any():
        LOGGER.warning(
            "%s player IDs have multiple names in the mapping; delivery names retained.",
            int(conflicting_ids.gt(1).sum()),
        )
    else:
        name_map = id_names.drop_duplicates("player_id").set_index("player_id")["player_name"]
        deliveries["striker_name"] = (
            deliveries["striker_id"].map(name_map).fillna(deliveries["striker_name"])
        )
        deliveries["bowler_name"] = (
            deliveries["bowler_id"].map(name_map).fillna(deliveries["bowler_name"])
        )

    if deliveries.duplicated(["match_id", "innings", "innings_ball_num"]).any():
        raise ValueError(
            "Duplicate delivery sequence keys found for match/innings/innings_ball_num."
        )

    deliveries["match_date"] = pd.to_datetime(
        deliveries["match_date"], errors="coerce"
    )
    deliveries = deliveries.sort_values(
        ["match_id", "innings", "innings_ball_num"]
    ).reset_index(drop=True)
    return deliveries


# ---------------------------------------------------------------------------
# Match state and pressure features
# ---------------------------------------------------------------------------


def add_match_state_and_pressure(
    deliveries: pd.DataFrame,
    pressure_quantile: float,
) -> pd.DataFrame:
    if not 0.50 <= pressure_quantile < 1.0:
        raise ValueError("pressure_quantile must be between 0.50 and 1.00.")

    df = deliveries.copy()

    numeric_columns = [
        "runs", "extras1_runs", "extras_2_runs",
        "first_inn_revised_overs", "second_inn_revised_overs",
        "revised_target",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df[["runs", "extras1_runs", "extras_2_runs"]] = df[
        ["runs", "extras1_runs", "extras_2_runs"]
    ].fillna(0)

    df["is_four_flag"] = as_bool(df["is_four"])
    df["is_six_flag"] = as_bool(df["is_six"])
    df["is_wicket_flag"] = as_bool(df["is_wicket"])

    df["extras_clean"] = (
        df["extras_type"].fillna("").astype(str).str.strip().str.lower()
    )
    df["is_wide"] = df["extras_clean"].eq("wide")
    df["is_no_ball"] = df["extras_clean"].isin({"overstep", "other"})
    df["is_legal_ball"] = ~df["extras_clean"].isin({"wide", "overstep", "other"})

    df["total_runs_delivery"] = (
        df["runs"] + df["extras1_runs"] + df["extras_2_runs"]
    )

    # Byes, leg-byes and penalty runs are not charged to the bowler.
    uncharged_types = {"byes", "legbyes", "penalty"}
    uncharged_extras = df["extras_clean"].isin(uncharged_types)
    extras_component = df["extras1_runs"] + df["extras_2_runs"]
    df["bowler_runs"] = (
        df["total_runs_delivery"]
        - extras_component.where(uncharged_extras, 0)
    ).clip(lower=0)

    # A wide is not a ball faced; a no-ball is treated as a ball faced.
    df["batter_ball_faced"] = ~df["is_wide"]
    df["batter_dot"] = df["is_legal_ball"] & df["total_runs_delivery"].eq(0)
    df["bowler_dot"] = df["is_legal_ball"] & df["total_runs_delivery"].eq(0)

    # Legal-ball limits are stored in the revised-overs columns in this source.
    df["innings_ball_limit"] = np.where(
        df["innings"].eq(1),
        df["first_inn_revised_overs"].fillna(120),
        df["second_inn_revised_overs"].fillna(120),
    )
    df["innings_ball_limit"] = pd.to_numeric(
        df["innings_ball_limit"], errors="coerce"
    ).fillna(120).clip(lower=1)

    groups = ["match_id", "innings"]
    df = df.sort_values(groups + ["innings_ball_num"]).reset_index(drop=True)

    df["score_after"] = df.groupby(groups)["total_runs_delivery"].cumsum()
    df["score_before"] = df["score_after"] - df["total_runs_delivery"]

    df["wickets_this_delivery"] = df["is_wicket_flag"].astype(int)
    df["wickets_after"] = df.groupby(groups)["wickets_this_delivery"].cumsum()
    df["wickets_before"] = df["wickets_after"] - df["wickets_this_delivery"]
    df["wickets_in_hand"] = (10 - df["wickets_before"]).clip(0, 10)

    df["legal_balls_this_delivery"] = df["is_legal_ball"].astype(int)
    df["legal_balls_after"] = df.groupby(groups)["legal_balls_this_delivery"].cumsum()
    df["legal_balls_before"] = (
        df["legal_balls_after"] - df["legal_balls_this_delivery"]
    )
    df["balls_remaining"] = (
        df["innings_ball_limit"] - df["legal_balls_before"]
    ).clip(lower=0)
    df["innings_progress"] = (
        df["legal_balls_before"] / df["innings_ball_limit"]
    ).clip(0, 1)

    # Targets are required only in the second innings.
    first_innings_totals = (
        df.loc[df["innings"].eq(1)]
        .groupby("match_id")["total_runs_delivery"]
        .sum()
    )
    normal_targets = first_innings_totals + 1
    revised_targets = (
        eligible_target_series(df)
    )
    target_by_match = revised_targets.combine_first(normal_targets)

    df["target_runs"] = np.nan
    second_innings_mask = df["innings"].eq(2)
    df.loc[second_innings_mask, "target_runs"] = (
        df.loc[second_innings_mask, "match_id"].map(target_by_match)
    )
    df["runs_required"] = (
        df["target_runs"] - df["score_before"]
    ).clip(lower=0)

    df["current_run_rate"] = safe_divide(
        df["score_before"] * 6,
        df["legal_balls_before"],
    )
    df["required_run_rate"] = np.nan
    valid_rrr = second_innings_mask & df["balls_remaining"].gt(0)
    df.loc[valid_rrr, "required_run_rate"] = (
        df.loc[valid_rrr, "runs_required"] * 6
        / df.loc[valid_rrr, "balls_remaining"]
    )

    # Context percentiles put IPL and T20I on comparable contextual scales.
    context_groups = ["match_format", "innings"]
    df["crr_percentile"] = (
        df.groupby(context_groups)["current_run_rate"]
        .rank(pct=True)
        .fillna(0.5)
    )
    df["rrr_percentile"] = (
        df.groupby(context_groups)["required_run_rate"]
        .rank(pct=True)
        .fillna(0.5)
    )

    wickets_lost_fraction = (df["wickets_before"] / 9).clip(0, 1)
    wickets_in_hand_fraction = (df["wickets_in_hand"] / 10).clip(0, 1)
    progress = df["innings_progress"]

    first_innings = df["innings"].eq(1)
    second_innings = df["innings"].eq(2)

    df["batting_pressure_score"] = np.nan
    df.loc[first_innings, "batting_pressure_score"] = 100 * (
        0.65 * wickets_lost_fraction[first_innings]
        + 0.35 * progress[first_innings]
    )
    df.loc[second_innings, "batting_pressure_score"] = 100 * (
        0.50 * df.loc[second_innings, "rrr_percentile"]
        + 0.30 * wickets_lost_fraction[second_innings]
        + 0.20 * progress[second_innings]
    )

    df["bowling_pressure_score"] = np.nan
    df.loc[first_innings, "bowling_pressure_score"] = 100 * (
        0.40 * progress[first_innings]
        + 0.35 * df.loc[first_innings, "crr_percentile"]
        + 0.25 * wickets_in_hand_fraction[first_innings]
    )
    df.loc[second_innings, "bowling_pressure_score"] = 100 * (
        0.40 * progress[second_innings]
        + 0.35 * (1 - df.loc[second_innings, "rrr_percentile"])
        + 0.25 * wickets_in_hand_fraction[second_innings]
    )

    # The pressure cutoff is calculated separately by format and innings.
    df["batting_pressure_percentile"] = (
        df.groupby(context_groups)["batting_pressure_score"].rank(pct=True)
    )
    df["bowling_pressure_percentile"] = (
        df.groupby(context_groups)["bowling_pressure_score"].rank(pct=True)
    )
    df["is_batting_pressure"] = (
        df["batting_pressure_percentile"] >= pressure_quantile
    )
    df["is_bowling_pressure"] = (
        df["bowling_pressure_percentile"] >= pressure_quantile
    )

    return df


def eligible_target_series(df: pd.DataFrame) -> pd.Series:
    """Return one revised target per match, taking the first non-null value."""
    revised = df[["match_id", "revised_target"]].dropna(subset=["revised_target"])
    if revised.empty:
        return pd.Series(dtype=float)
    return revised.drop_duplicates("match_id").set_index("match_id")["revised_target"]


# ---------------------------------------------------------------------------
# Player performance metrics
# ---------------------------------------------------------------------------


def calculate_player_match_metrics(
    pressure_deliveries: pd.DataFrame,
    player_mapping: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pressure_deliveries.copy()

    credited_dismissals = {
        "caught", "bowled", "lbw", "stump", "hit wicket"
    }
    dismissal_clean = (
        df["dismissal_type"].fillna("").astype(str).str.strip().str.lower()
    )
    df["bowler_wicket"] = (
        df["is_wicket_flag"] & dismissal_clean.isin(credited_dismissals)
    )

    # Batter delivery-level records.
    bat = df.loc[df["batter_ball_faced"]].copy()
    bat["player_id"] = bat["striker_id"]
    bat["player_name"] = bat["striker_name"]
    bat["pressure"] = bat["is_batting_pressure"]
    bat["balls_faced"] = 1
    bat["dot_balls"] = bat["batter_dot"].astype(int)
    bat["fours"] = bat["is_four_flag"].astype(int)
    bat["sixes"] = bat["is_six_flag"].astype(int)

    bat_keys = [
        "player_id", "player_name", "match_id", "match_format", "pressure"
    ]
    batter_stats = (
        bat.groupby(bat_keys, as_index=False)
        .agg(
            runs=("runs", "sum"),
            balls=("balls_faced", "sum"),
            dots=("dot_balls", "sum"),
            fours=("fours", "sum"),
            sixes=("sixes", "sum"),
        )
    )

    # Count every dismissed batter, including a non-striker dismissed run out.
    dismissals = df.loc[
        df["is_wicket_flag"] & df["dismissed_player_id"].notna()
    ].copy()
    dismissals["player_id"] = dismissals["dismissed_player_id"].astype(int)

    id_names = player_mapping[["player_id", "player_name"]].drop_duplicates()
    id_name_counts = id_names.groupby("player_id")["player_name"].nunique()
    if id_name_counts.gt(1).any():
        # The supplied source has one name per player ID; fail rather than
        # silently assigning a potentially incorrect name if that changes.
        raise ValueError("A player_id maps to multiple player names.")
    name_map = id_names.drop_duplicates("player_id").set_index("player_id")["player_name"]
    dismissals["player_name"] = (
        dismissals["player_id"].map(name_map)
        .fillna(dismissals["striker_name"])
    )
    dismissals["pressure"] = dismissals["is_batting_pressure"]
    dismissal_stats = (
        dismissals.groupby(bat_keys, as_index=False)
        .agg(outs=("is_wicket_flag", "sum"))
    )

    batter_stats = batter_stats.merge(
        dismissal_stats,
        on=bat_keys,
        how="outer",
        validate="one_to_one",
    )
    batter_stats[["runs", "balls", "dots", "fours", "sixes", "outs"]] = (
        batter_stats[["runs", "balls", "dots", "fours", "sixes", "outs"]]
        .fillna(0)
    )
    batter_stats["strike_rate"] = safe_divide(
        100 * batter_stats["runs"], batter_stats["balls"]
    )
    batter_stats["dot_percentage"] = safe_divide(
        100 * batter_stats["dots"], batter_stats["balls"]
    )
    batter_stats["boundary_percentage"] = safe_divide(
        100 * (batter_stats["fours"] + batter_stats["sixes"]),
        batter_stats["balls"],
    )
    batter_stats["dismissal_rate"] = safe_divide(
        100 * batter_stats["outs"], batter_stats["balls"]
    )

    # Bowler delivery-level records.
    bowl = df.copy()
    bowl["player_id"] = bowl["bowler_id"]
    bowl["player_name"] = bowl["bowler_name"]
    bowl["pressure"] = bowl["is_bowling_pressure"]
    bowl["legal_balls"] = bowl["is_legal_ball"].astype(int)
    bowl["wickets"] = bowl["bowler_wicket"].astype(int)
    bowl["dots"] = bowl["bowler_dot"].astype(int)
    bowl["boundaries"] = (
        bowl["is_four_flag"].astype(int) + bowl["is_six_flag"].astype(int)
    )
    bowl["wides"] = bowl["is_wide"].astype(int)
    bowl["no_balls"] = bowl["is_no_ball"].astype(int)

    bowl_keys = [
        "player_id", "player_name", "match_id", "match_format", "pressure"
    ]
    bowler_stats = (
        bowl.groupby(bowl_keys, as_index=False)
        .agg(
            runs_conceded=("bowler_runs", "sum"),
            legal_balls=("legal_balls", "sum"),
            wickets=("wickets", "sum"),
            dots=("dots", "sum"),
            boundaries=("boundaries", "sum"),
            wides=("wides", "sum"),
            no_balls=("no_balls", "sum"),
        )
    )
    bowler_stats["overs"] = bowler_stats["legal_balls"] / 6
    bowler_stats["economy"] = safe_divide(
        6 * bowler_stats["runs_conceded"], bowler_stats["legal_balls"]
    )
    bowler_stats["wickets_per_24_balls"] = safe_divide(
        24 * bowler_stats["wickets"], bowler_stats["legal_balls"]
    )
    bowler_stats["dot_percentage"] = safe_divide(
        100 * bowler_stats["dots"], bowler_stats["legal_balls"]
    )
    bowler_stats["boundary_percentage"] = safe_divide(
        100 * bowler_stats["boundaries"], bowler_stats["legal_balls"]
    )

    return batter_stats, bowler_stats


def build_pressure_comparisons(
    batter_match_stats: pd.DataFrame,
    bowler_match_stats: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Batter summaries by player, format and pressure state.
    batter_summary = (
        batter_match_stats.groupby(
            ["player_id", "player_name", "match_format", "pressure"],
            as_index=False,
        )
        .agg(
            runs=("runs", "sum"),
            balls=("balls", "sum"),
            outs=("outs", "sum"),
            dots=("dots", "sum"),
            fours=("fours", "sum"),
            sixes=("sixes", "sum"),
            matches=("match_id", "nunique"),
        )
    )
    batter_summary["strike_rate"] = safe_divide(
        100 * batter_summary["runs"], batter_summary["balls"]
    )
    batter_summary["dot_percentage"] = safe_divide(
        100 * batter_summary["dots"], batter_summary["balls"]
    )
    batter_summary["boundary_percentage"] = safe_divide(
        100 * (batter_summary["fours"] + batter_summary["sixes"]),
        batter_summary["balls"],
    )
    batter_summary["dismissal_rate"] = safe_divide(
        100 * batter_summary["outs"], batter_summary["balls"]
    )

    batter_pressure = batter_summary.loc[batter_summary["pressure"]].drop(
        columns="pressure"
    )
    batter_normal = batter_summary.loc[~batter_summary["pressure"]].drop(
        columns="pressure"
    )
    identifiers = ["player_id", "player_name", "match_format"]
    batter_pressure = batter_pressure.rename(
        columns={c: f"pressure_{c}" for c in batter_pressure.columns if c not in identifiers}
    )
    batter_normal = batter_normal.rename(
        columns={c: f"normal_{c}" for c in batter_normal.columns if c not in identifiers}
    )
    batter_comparison = batter_pressure.merge(
        batter_normal,
        on=identifiers,
        how="outer",
        validate="one_to_one",
    )
    batter_comparison["sr_change"] = (
        batter_comparison["pressure_strike_rate"]
        - batter_comparison["normal_strike_rate"]
    )
    batter_comparison["dot_reduction"] = (
        batter_comparison["normal_dot_percentage"]
        - batter_comparison["pressure_dot_percentage"]
    )
    batter_comparison["boundary_change"] = (
        batter_comparison["pressure_boundary_percentage"]
        - batter_comparison["normal_boundary_percentage"]
    )
    batter_comparison["dismissal_improvement"] = (
        batter_comparison["normal_dismissal_rate"]
        - batter_comparison["pressure_dismissal_rate"]
    )

    # Bowler summaries by player, format and pressure state.
    bowler_summary = (
        bowler_match_stats.groupby(
            ["player_id", "player_name", "match_format", "pressure"],
            as_index=False,
        )
        .agg(
            runs_conceded=("runs_conceded", "sum"),
            legal_balls=("legal_balls", "sum"),
            wickets=("wickets", "sum"),
            dots=("dots", "sum"),
            boundaries=("boundaries", "sum"),
            wides=("wides", "sum"),
            no_balls=("no_balls", "sum"),
            matches=("match_id", "nunique"),
        )
    )
    bowler_summary["economy"] = safe_divide(
        6 * bowler_summary["runs_conceded"], bowler_summary["legal_balls"]
    )
    bowler_summary["wickets_per_24_balls"] = safe_divide(
        24 * bowler_summary["wickets"], bowler_summary["legal_balls"]
    )
    bowler_summary["dot_percentage"] = safe_divide(
        100 * bowler_summary["dots"], bowler_summary["legal_balls"]
    )
    bowler_summary["boundary_percentage"] = safe_divide(
        100 * bowler_summary["boundaries"], bowler_summary["legal_balls"]
    )

    bowler_pressure = bowler_summary.loc[bowler_summary["pressure"]].drop(
        columns="pressure"
    )
    bowler_normal = bowler_summary.loc[~bowler_summary["pressure"]].drop(
        columns="pressure"
    )
    bowler_pressure = bowler_pressure.rename(
        columns={c: f"pressure_{c}" for c in bowler_pressure.columns if c not in identifiers}
    )
    bowler_normal = bowler_normal.rename(
        columns={c: f"normal_{c}" for c in bowler_normal.columns if c not in identifiers}
    )
    bowler_comparison = bowler_pressure.merge(
        bowler_normal,
        on=identifiers,
        how="outer",
        validate="one_to_one",
    )
    bowler_comparison["economy_improvement"] = (
        bowler_comparison["normal_economy"]
        - bowler_comparison["pressure_economy"]
    )
    bowler_comparison["wicket_improvement"] = (
        bowler_comparison["pressure_wickets_per_24_balls"]
        - bowler_comparison["normal_wickets_per_24_balls"]
    )
    bowler_comparison["dot_improvement"] = (
        bowler_comparison["pressure_dot_percentage"]
        - bowler_comparison["normal_dot_percentage"]
    )
    bowler_comparison["boundary_improvement"] = (
        bowler_comparison["normal_boundary_percentage"]
        - bowler_comparison["pressure_boundary_percentage"]
    )

    return batter_comparison, bowler_comparison


# ---------------------------------------------------------------------------
# Eligibility, reliability-adjusted ratings and consistency
# ---------------------------------------------------------------------------


def create_eligible_ratings(
    comparison: pd.DataFrame,
    role: str,
) -> pd.DataFrame:
    data = comparison.copy()

    if role == "batter":
        numeric_counts = [
            "pressure_balls", "normal_balls",
            "pressure_matches", "normal_matches",
        ]
        data[numeric_counts] = data[numeric_counts].fillna(0)
        eligible = data.loc[
            data["pressure_balls"].ge(30)
            & data["normal_balls"].ge(40)
            & data["pressure_matches"].ge(3)
            & data["normal_matches"].ge(3)
        ].copy()
        metrics = list(BATTER_WEIGHTS)
        pressure_exposure = "pressure_balls"
        normal_exposure = "normal_balls"
        pressure_matches = "pressure_matches"
        normal_matches = "normal_matches"
        pressure_constant = 60
        normal_constant = 60
    elif role == "bowler":
        numeric_counts = [
            "pressure_legal_balls", "normal_legal_balls",
            "pressure_matches", "normal_matches",
        ]
        data[numeric_counts] = data[numeric_counts].fillna(0)
        eligible = data.loc[
            data["pressure_legal_balls"].ge(24)
            & data["normal_legal_balls"].ge(48)
            & data["pressure_matches"].ge(3)
            & data["normal_matches"].ge(3)
        ].copy()
        metrics = list(BOWLER_WEIGHTS)
        pressure_exposure = "pressure_legal_balls"
        normal_exposure = "normal_legal_balls"
        pressure_matches = "pressure_matches"
        normal_matches = "normal_matches"
        pressure_constant = 48
        normal_constant = 48
    else:
        raise ValueError("role must be 'batter' or 'bowler'.")

    if eligible.empty:
        raise RuntimeError(
            f"No eligible {role}s remain. Review eligibility thresholds and pressure samples."
        )

    eligible = robust_standardize_by_format(eligible, metrics)

    pressure_confidence = (
        eligible[pressure_exposure]
        / (eligible[pressure_exposure] + pressure_constant)
    )
    normal_confidence = (
        eligible[normal_exposure]
        / (eligible[normal_exposure] + normal_constant)
    )
    match_confidence = np.sqrt(
        (eligible[pressure_matches] / (eligible[pressure_matches] + 3))
        * (eligible[normal_matches] / (eligible[normal_matches] + 3))
    )
    eligible["reliability"] = (
        np.sqrt(pressure_confidence * normal_confidence) * match_confidence
    ).clip(0, 1)

    return eligible


def calculate_consistency(
    batter_match_stats: pd.DataFrame,
    bowler_match_stats: pd.DataFrame,
    batter_comparison: pd.DataFrame,
    bowler_comparison: pd.DataFrame,
    by_format: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    batter_pressure_matches = batter_match_stats.loc[
        batter_match_stats["pressure"]
    ].copy()
    batter_pressure_matches["match_strike_rate"] = safe_divide(
        100 * batter_pressure_matches["runs"],
        batter_pressure_matches["balls"],
    )
    batter_baselines = batter_comparison[
        ["player_id", "match_format", "normal_strike_rate"]
    ]
    batter_pressure_matches = batter_pressure_matches.merge(
        batter_baselines,
        on=["player_id", "match_format"],
        how="left",
        validate="many_to_one",
    )
    batter_pressure_matches["performed_at_or_above_baseline"] = (
        batter_pressure_matches["match_strike_rate"]
        >= batter_pressure_matches["normal_strike_rate"]
    )
    consistency_keys = ["player_id", "player_name"]
    if by_format:
        consistency_keys.append("match_format")
    batter_consistency = (
        batter_pressure_matches.groupby(
            consistency_keys, as_index=False
        )
        .agg(
            pressure_matches_checked=("match_id", "nunique"),
            positive_matches=("performed_at_or_above_baseline", "sum"),
            consistency_pct=("performed_at_or_above_baseline", "mean"),
        )
    )
    batter_consistency["consistency_pct"] *= 100

    bowler_pressure_matches = bowler_match_stats.loc[
        bowler_match_stats["pressure"]
    ].copy()
    bowler_pressure_matches["match_economy"] = safe_divide(
        6 * bowler_pressure_matches["runs_conceded"],
        bowler_pressure_matches["legal_balls"],
    )
    bowler_baselines = bowler_comparison[
        ["player_id", "match_format", "normal_economy"]
    ]
    bowler_pressure_matches = bowler_pressure_matches.merge(
        bowler_baselines,
        on=["player_id", "match_format"],
        how="left",
        validate="many_to_one",
    )
    bowler_pressure_matches["performed_at_or_below_baseline"] = (
        bowler_pressure_matches["match_economy"]
        <= bowler_pressure_matches["normal_economy"]
    )
    bowler_consistency = (
        bowler_pressure_matches.groupby(
            consistency_keys, as_index=False
        )
        .agg(
            pressure_matches_checked=("match_id", "nunique"),
            positive_matches=("performed_at_or_below_baseline", "sum"),
            consistency_pct=("performed_at_or_below_baseline", "mean"),
        )
    )
    bowler_consistency["consistency_pct"] *= 100

    return batter_consistency, bowler_consistency


# ---------------------------------------------------------------------------
# Evidence tables, charts and report notes
# ---------------------------------------------------------------------------


def create_top10_evidence(
    overall: pd.DataFrame,
    per_format: pd.DataFrame,
    consistency: pd.DataFrame,
    role: str,
) -> pd.DataFrame:
    top = overall.head(10).copy()
    top = top.merge(
        consistency,
        on=["player_id", "player_name"],
        how="left",
        validate="one_to_one",
    )

    if role == "batter":
        fields = [
            "player_id", "match_format", "pressure_balls", "normal_balls",
            "pressure_strike_rate", "normal_strike_rate", "sr_change",
            "reliability", "format_PPR",
        ]
    else:
        fields = [
            "player_id", "match_format", "pressure_legal_balls",
            "normal_legal_balls", "pressure_economy", "normal_economy",
            "economy_improvement", "reliability", "format_PPR",
        ]

    return top.merge(
        per_format[fields],
        on="player_id",
        how="left",
        validate="one_to_many",
    )


def save_ranking_chart(
    ranking: pd.DataFrame,
    title: str,
    xlabel: str,
    output_path: Path,
) -> None:
    chart_data = ranking.head(10).sort_values("PPR", ascending=True)
    height = max(5, 0.48 * len(chart_data) + 1.5)
    fig, ax = plt.subplots(figsize=(10, height))
    ax.barh(chart_data["player_name"], chart_data["PPR"])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Player")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_methodology_notes(
    output_path: Path,
    pressure_quantile: float,
    eligible_matches: pd.DataFrame,
    pressure_deliveries: pd.DataFrame,
    batter_eligible_count: int,
    bowler_eligible_count: int,
) -> None:
    competition_counts = eligible_matches.groupby("competition_type")["match_id"].nunique()
    text = f"""SUPERR SELECTOR SEASON 2 — ANALYSIS NOTES

DATA SCOPE
- Completed IPL and T20I matches from the supplied source data.
- Eligible matches: {len(eligible_matches)}.
- IPL matches: {int(competition_counts.get('IPL', 0))}.
- T20I matches: {int(competition_counts.get('T20I', 0))}.
- Delivery rows after excluding innings 3 and 4: {len(pressure_deliveries)}.

PRESSURE DEFINITION
Pressure is a custom contextual index, not an official cricket statistic.
The top {100 * (1 - pressure_quantile):.0f}% of deliveries are classified as high
pressure separately by match format and innings.

Batting pressure:
- First innings: 65% wickets-lost fraction + 35% innings progress.
- Second innings: 50% required-run-rate percentile + 30% wickets-lost
  fraction + 20% innings progress.

Bowling pressure:
- First innings: 40% innings progress + 35% current-run-rate percentile
  + 25% wickets-in-hand fraction.
- Second innings: 40% innings progress + 35% inverse required-run-rate
  percentile + 25% wickets-in-hand fraction.

PERFORMANCE DIFFERENCES
Batters:
- Strike-rate change: 50%.
- Dot-ball-rate reduction: 20%.
- Boundary-rate change: 15%.
- Dismissal-rate improvement: 15%.

Bowlers:
- Economy improvement: 40%.
- Wickets per 24 legal balls improvement: 25%.
- Dot-ball-rate improvement: 20%.
- Boundary-concession-rate improvement: 15%.

Ratings are robust-standardized within format using median and scaled MAD,
clipped to [-3, 3], then reliability-adjusted for ball and match sample sizes.
The combined PPR is a custom index centered around 50, not a probability.

ELIGIBILITY
Batters: at least 30 pressure balls, 40 normal balls, 3 pressure matches,
and 3 normal matches, within a format.
Bowlers: at least 24 pressure legal balls, 48 normal legal balls,
and 3 pressure matches and 3 normal matches, within a format.

CONSISTENCY
The report includes the percentage of pressure matches where the batter's
strike rate was at least their normal baseline, or the bowler's economy was
at or below their normal baseline. This is a diagnostic and is not an extra
component in the PPR formula.

IMPORTANT ASSUMPTIONS AND LIMITATIONS
- The source stores revised-overs values as legal-ball limits (for example,
  114 means 114 balls); missing limits default to 120 balls.
- Wide, Overstep and other are treated as illegal deliveries; other extra types
  are treated as legal unless identified otherwise in the source.
- A wide is excluded from batter balls faced. A no-ball is included in balls
  faced and excluded from legal balls.
- Byes, leg-byes and penalty extras are excluded from bowler runs conceded.
- Extras with a missing type label are handled using the numeric run fields;
  inspect the data-quality audit for such records before final submission.
- Bowler wickets include caught, bowled, LBW, stump and hit wicket dismissals.
- Super Over innings (innings 3 and 4) and abandoned/no-result matches are
  excluded from the main ranking.
- Completed revised-target matches are retained and use the supplied revised
  target where available.
- The pressure threshold, weights and sample cutoffs are modelling choices;
  rankings should be interpreted alongside the sensitivity results.
- Match-level consistency is a simple directional diagnostic, not a measure
  of statistical uncertainty.

ELIGIBLE PLAYER POOLS
- Batter player-format records: {batter_eligible_count}.
- Bowler player-format records: {bowler_eligible_count}.
"""
    output_path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------


def run_analysis(input_dir: Path, output_dir: Path, pressure_quantile: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    source = load_source_data(input_dir)
    balls = source["balls"]
    matches = source["matches"]
    competitions = source["competitions"]
    teams = source["teams"]
    playing_eleven = source["playing_eleven"]
    player_mapping = source["player_mapping"]

    require_columns(
        balls,
        ["match_id", "innings", "innings_ball_num", "batting_team_id",
         "bowling_team_id", "striker_id", "striker_name", "non_striker_id",
         "non_striker_name", "bowler_id", "bowler_name", "runs", "is_four",
         "is_six", "is_wicket", "dismissal_type", "dismissed_player_id",
         "extras_type", "extras1_runs", "extras_2_runs"],
        "ball_by_ball.csv",
    )

    audit = create_data_audit(source)
    save_csv(audit, output_dir, "data_quality_audit.csv")
    schema_rows = []
    for dataset_name, dataset in source.items():
        for column in dataset.columns:
            schema_rows.append(
                {
                    "dataset": INPUT_FILES[dataset_name],
                    "column": column,
                    "dtype": str(dataset[column].dtype),
                    "missing_values": int(dataset[column].isna().sum()),
                }
            )
    save_csv(pd.DataFrame(schema_rows), output_dir, "dataset_schema.csv")

    match_info, playing_xi_enriched, supporting_stats = (
        validate_and_enrich_supporting_data(
            matches, competitions, teams, playing_eleven, player_mapping
        )
    )
    save_csv(playing_xi_enriched, output_dir, "playing_xi_enriched.csv")

    eligible_matches = select_eligible_matches(match_info)
    save_csv(eligible_matches, output_dir, "eligible_matches.csv")

    deliveries = build_delivery_dataset(
        balls, eligible_matches, player_mapping
    )
    pressure_deliveries = add_match_state_and_pressure(
        deliveries, pressure_quantile
    )

    # Preserve the full delivery-level calculations for reproducibility.
    save_csv(
        pressure_deliveries,
        output_dir,
        "pressure_scored_deliveries.csv",
    )

    batter_match_stats, bowler_match_stats = calculate_player_match_metrics(
        pressure_deliveries, player_mapping
    )
    save_csv(batter_match_stats, output_dir, "batter_match_metrics.csv")
    save_csv(bowler_match_stats, output_dir, "bowler_match_metrics.csv")

    batter_comparison, bowler_comparison = build_pressure_comparisons(
        batter_match_stats, bowler_match_stats
    )
    save_csv(batter_comparison, output_dir, "batter_pressure_comparison.csv")
    save_csv(bowler_comparison, output_dir, "bowler_pressure_comparison.csv")

    batter_eligible = create_eligible_ratings(batter_comparison, "batter")
    bowler_eligible = create_eligible_ratings(bowler_comparison, "bowler")

    batter_ranking, batter_format_ratings = build_overall_ranking(
        batter_eligible, BATTER_WEIGHTS, "pressure_balls"
    )
    bowler_ranking, bowler_format_ratings = build_overall_ranking(
        bowler_eligible, BOWLER_WEIGHTS, "pressure_legal_balls"
    )

    save_csv(batter_eligible, output_dir, "all_eligible_batter_ratings.csv")
    save_csv(bowler_eligible, output_dir, "all_eligible_bowler_ratings.csv")
    save_csv(batter_ranking, output_dir, "all_batter_rankings.csv")
    save_csv(bowler_ranking, output_dir, "all_bowler_rankings.csv")

    # Keep format-specific PPRs available to the dashboard. The displayed
    # format rank uses the same per-format rating that feeds the overall PPR.
    batter_format_ratings = batter_format_ratings.sort_values(
        ["match_format", "format_PPR", "pressure_matches", "player_name"],
        ascending=[True, False, False, True],
    ).copy()
    batter_format_ratings["Rank"] = batter_format_ratings.groupby(
        "match_format"
    ).cumcount() + 1
    bowler_format_ratings = bowler_format_ratings.sort_values(
        ["match_format", "format_PPR", "pressure_matches", "player_name"],
        ascending=[True, False, False, True],
    ).copy()
    bowler_format_ratings["Rank"] = bowler_format_ratings.groupby(
        "match_format"
    ).cumcount() + 1
    save_csv(batter_format_ratings, output_dir, "all_batter_format_rankings.csv")
    save_csv(bowler_format_ratings, output_dir, "all_bowler_format_rankings.csv")

    # Alternative weighting scenarios use the same eligible pool and standardized metrics.
    batter_scenarios: Dict[str, pd.DataFrame] = {}
    batter_format_scenarios: Dict[str, pd.DataFrame] = {}
    for scenario, weights in BATTER_SCENARIOS.items():
        batter_scenarios[scenario], scenario_format_ratings = build_overall_ranking(
            batter_eligible, weights, "pressure_balls"
        )
        scenario_format_ratings = scenario_format_ratings.sort_values(
            ["match_format", "format_PPR", "pressure_matches", "player_name"],
            ascending=[True, False, False, True],
        ).copy()
        scenario_format_ratings["Rank"] = scenario_format_ratings.groupby(
            "match_format"
        ).cumcount() + 1
        batter_format_scenarios[scenario] = scenario_format_ratings

    bowler_scenarios: Dict[str, pd.DataFrame] = {}
    bowler_format_scenarios: Dict[str, pd.DataFrame] = {}
    for scenario, weights in BOWLER_SCENARIOS.items():
        bowler_scenarios[scenario], scenario_format_ratings = build_overall_ranking(
            bowler_eligible, weights, "pressure_legal_balls"
        )
        scenario_format_ratings = scenario_format_ratings.sort_values(
            ["match_format", "format_PPR", "pressure_matches", "player_name"],
            ascending=[True, False, False, True],
        ).copy()
        scenario_format_ratings["Rank"] = scenario_format_ratings.groupby(
            "match_format"
        ).cumcount() + 1
        bowler_format_scenarios[scenario] = scenario_format_ratings

    batter_stability = build_rank_stability(batter_scenarios)
    bowler_stability = build_rank_stability(bowler_scenarios)
    save_csv(batter_stability, output_dir, "batter_sensitivity_analysis.csv")
    save_csv(bowler_stability, output_dir, "bowler_sensitivity_analysis.csv")
    save_csv(build_format_rank_stability(batter_format_scenarios), output_dir, "batter_format_sensitivity_analysis.csv")
    save_csv(build_format_rank_stability(bowler_format_scenarios), output_dir, "bowler_format_sensitivity_analysis.csv")

    batter_consistency, bowler_consistency = calculate_consistency(
        batter_match_stats,
        bowler_match_stats,
        batter_comparison,
        bowler_comparison,
    )
    save_csv(batter_consistency, output_dir, "batter_match_consistency.csv")
    save_csv(bowler_consistency, output_dir, "bowler_match_consistency.csv")
    batter_format_consistency, bowler_format_consistency = calculate_consistency(
        batter_match_stats,
        bowler_match_stats,
        batter_comparison,
        bowler_comparison,
        by_format=True,
    )
    save_csv(batter_format_consistency, output_dir, "batter_format_match_consistency.csv")
    save_csv(bowler_format_consistency, output_dir, "bowler_format_match_consistency.csv")

    if len(batter_ranking) < 10 or len(bowler_ranking) < 10:
        raise RuntimeError(
            "Fewer than 10 eligible players in one role; review data and eligibility thresholds."
        )

    batter_top10 = batter_ranking.head(10).copy()
    bowler_top10 = bowler_ranking.head(10).copy()
    batter_top10_evidence = create_top10_evidence(
        batter_ranking, batter_format_ratings, batter_consistency, "batter"
    )
    bowler_top10_evidence = create_top10_evidence(
        bowler_ranking, bowler_format_ratings, bowler_consistency, "bowler"
    )

    # Final rank integrity checks.
    assert batter_top10["player_id"].nunique() == 10
    assert bowler_top10["player_id"].nunique() == 10
    assert batter_top10["PPR"].notna().all()
    assert bowler_top10["PPR"].notna().all()

    save_csv(batter_top10, output_dir, "SupeRR_Top10_Batters.csv")
    save_csv(bowler_top10, output_dir, "SupeRR_Top10_Bowlers.csv")
    save_csv(batter_top10_evidence, output_dir, "SupeRR_Batter_Top10_Evidence.csv")
    save_csv(bowler_top10_evidence, output_dir, "SupeRR_Bowler_Top10_Evidence.csv")

    combined_top10 = pd.concat(
        [
            batter_top10.assign(role="Batter"),
            bowler_top10.assign(role="Bowler"),
        ],
        ignore_index=True,
        sort=False,
    )
    save_csv(combined_top10, output_dir, "SupeRR_Top10_Rankings.csv")

    # Delivery-level pressure summary for the methodology report.
    pressure_summary = (
        pressure_deliveries.groupby(
            ["match_format", "innings"], as_index=False
        )
        .agg(
            deliveries=("match_id", "size"),
            batting_pressure_deliveries=("is_batting_pressure", "sum"),
            bowling_pressure_deliveries=("is_bowling_pressure", "sum"),
            batting_pressure_score_mean=("batting_pressure_score", "mean"),
            bowling_pressure_score_mean=("bowling_pressure_score", "mean"),
        )
    )
    pressure_summary["batting_pressure_pct"] = (
        100 * pressure_summary["batting_pressure_deliveries"]
        / pressure_summary["deliveries"]
    )
    pressure_summary["bowling_pressure_pct"] = (
        100 * pressure_summary["bowling_pressure_deliveries"]
        / pressure_summary["deliveries"]
    )
    save_csv(pressure_summary, output_dir, "pressure_distribution_summary.csv")

    save_ranking_chart(
        batter_ranking,
        "Top 10 Batters Under Pressure",
        "Pressure Performance Rating (PPR)",
        output_dir / "SupeRR_Batter_Rankings.png",
    )
    save_ranking_chart(
        bowler_ranking,
        "Top 10 Bowlers Under Pressure",
        "Pressure Performance Rating (PPR)",
        output_dir / "SupeRR_Bowler_Rankings.png",
    )

    write_methodology_notes(
        output_dir / "Methodology_Summary.txt",
        pressure_quantile,
        eligible_matches,
        pressure_deliveries,
        len(batter_eligible),
        len(bowler_eligible),
    )

    # A plain-text list is convenient when entering rankings manually.
    entry_lines = ["TOP 10 BATTERS", ""]
    entry_lines.extend(
        f"Rank {row.Rank}: {row.player_name}"
        for row in batter_top10.itertuples(index=False)
    )
    entry_lines.extend(["", "TOP 10 BOWLERS", ""])
    entry_lines.extend(
        f"Rank {row.Rank}: {row.player_name}"
        for row in bowler_top10.itertuples(index=False)
    )
    (output_dir / "Unstop_Entry_Order.txt").write_text(
        "\n".join(entry_lines) + "\n", encoding="utf-8"
    )

    # Record audit and scope details in a machine-readable summary.
    match_counts = eligible_matches.groupby("competition_type")["match_id"].nunique()
    summary = pd.DataFrame(
        [
            {"metric": "source_delivery_rows", "value": len(balls)},
            {"metric": "source_match_rows", "value": len(matches)},
            {"metric": "eligible_matches", "value": len(eligible_matches)},
            {"metric": "eligible_ipl_matches", "value": int(match_counts.get("IPL", 0))},
            {"metric": "eligible_t20i_matches", "value": int(match_counts.get("T20I", 0))},
            {"metric": "analysis_delivery_rows", "value": len(pressure_deliveries)},
            {"metric": "eligible_batter_format_records", "value": len(batter_eligible)},
            {"metric": "eligible_bowler_format_records", "value": len(bowler_eligible)},
            {"metric": "unmatched_competition_matches", "value": supporting_stats["unmatched_competition_matches"]},
            {"metric": "unmatched_playing_xi_players", "value": supporting_stats["unmatched_playing_xi_players"]},
            {"metric": "unmatched_unplayed_xi_players", "value": supporting_stats["unmatched_unplayed_xi_players"]},
            {"metric": "unmatched_playing_xi_teams", "value": supporting_stats["unmatched_playing_xi_teams"]},
            {"metric": "playing_xi_duplicate_player_match_key_rows", "value": supporting_stats["playing_xi_duplicate_player_match_key_rows"]},
            {"metric": "playing_xi_duplicate_player_match_keys", "value": supporting_stats["playing_xi_duplicate_player_match_keys"]},
            {"metric": "playing_xi_keys_with_conflicting_attributes", "value": supporting_stats["playing_xi_keys_with_conflicting_attributes"]},
            {"metric": "unclassified_extra_rows_with_positive_extra_runs", "value": int((balls["extras_type"].isna() & ((pd.to_numeric(balls["extras1_runs"], errors="coerce").fillna(0) + pd.to_numeric(balls["extras_2_runs"], errors="coerce").fillna(0)) > 0)).sum())},
        ]
    )
    save_csv(summary, output_dir, "analysis_summary.csv")

    LOGGER.info("Analysis complete. Outputs saved to %s", output_dir.resolve())
    print("\nTOP 10 BATTERS")
    print(batter_top10[["Rank", "player_name", "PPR", "pressure_matches", "normal_matches"]].to_string(index=False))
    print("\nTOP 10 BOWLERS")
    print(bowler_top10[["Rank", "player_name", "PPR", "pressure_matches", "normal_matches"]].to_string(index=False))
    print("\nGenerated output directory:", output_dir.resolve())


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    run_analysis(
        input_dir=args.input_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        pressure_quantile=args.pressure_quantile,
    )


if __name__ == "__main__":
    main()
