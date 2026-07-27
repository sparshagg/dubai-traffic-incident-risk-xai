from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import shap


ROOT = Path(__file__).resolve().parents[2]
SHAP_SCRIPT = ROOT / "scripts" / "modeling" / "04_generate_shap_explanations.py"
DASHBOARD_DIR = ROOT / "data" / "dashboard"
AUDIT_DIR = ROOT / "data" / "audit"

TOP_N_PER_WINDOW = 100
REPLAY_DAYS = 7
DATA_CUTOFF = "2026-06-01"
LEGACY_OUTPUT_PATTERNS = [
    "*forecast_top_zones.csv",
    "*forecast_window_summary.csv",
    "*forecast_model_metrics.csv",
    "*forecast_model_metadata.json",
]


def load_final_model_module():
    spec = importlib.util.spec_from_file_location("final_shap_explanations", SHAP_SCRIPT)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not import {SHAP_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sx = load_final_model_module()
fg = sx.fg
ng = sx.ng


def ensure_dirs() -> None:
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def remove_legacy_forecast_files() -> None:
    for pattern in LEGACY_OUTPUT_PATTERNS:
        for path in DASHBOARD_DIR.glob(pattern):
            path.unlink()


def h3_feature(cell: str, inside_neighbor_count: int) -> dict:
    boundary = list(h3.cell_to_boundary(cell))
    coordinates = [[float(lng), float(lat)] for lat, lng in boundary]
    coordinates.append(coordinates[0])
    lat_center, lng_center = h3.cell_to_latlng(cell)
    return {
        "type": "Feature",
        "properties": {
            "h3_cell_res8": cell,
            "inside_neighbor_count": int(inside_neighbor_count),
            "centroid_latitude": float(lat_center),
            "centroid_longitude": float(lng_center),
        },
        "geometry": {"type": "Polygon", "coordinates": [coordinates]},
    }


def write_cell_geojson(context) -> Path:
    features = [
        h3_feature(cell, context.state.inside_neighbor_count[idx])
        for idx, cell in enumerate(context.inside_cells)
    ]
    geojson = {
        "type": "FeatureCollection",
        "name": "inside_dubai_h3_res8_cells",
        "features": features,
    }
    output_path = DASHBOARD_DIR / "inside_dubai_h3_cells.geojson"
    output_path.write_text(json.dumps(geojson), encoding="utf-8")
    return output_path


def top_indices_by_window(scores: np.ndarray, n_windows: int, n_cells: int, top_n: int) -> tuple[np.ndarray, np.ndarray]:
    score_matrix = scores.reshape(n_windows, n_cells)
    selected_indices: list[np.ndarray] = []
    selected_ranks: list[np.ndarray] = []
    for offset in range(n_windows):
        row = score_matrix[offset]
        if top_n >= n_cells:
            candidate_cells = np.arange(n_cells, dtype=np.int32)
        else:
            candidate_cells = np.argpartition(row, -top_n)[-top_n:].astype(np.int32)
        ordered_cells = candidate_cells[np.argsort(row[candidate_cells])[::-1]]
        selected_indices.append((offset * n_cells + ordered_cells).astype(np.int64))
        selected_ranks.append(np.arange(1, len(ordered_cells) + 1, dtype=np.int16))
    return np.concatenate(selected_indices), np.concatenate(selected_ranks)


def time_window_label(hour_block: int) -> str:
    start_hour = int(hour_block) * fg.WINDOW_HOURS
    end_hour = (start_hour + fg.WINDOW_HOURS) % 24
    return f"{start_hour:02d}:00-{end_hour:02d}:00"


def unique_columns(columns: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for column in columns:
        if column not in seen:
            seen.add(column)
            result.append(column)
    return result


def write_risk_scores(context) -> tuple[Path, pd.DataFrame]:
    n_windows = len(context.test_windows)
    n_cells = len(context.inside_cells)
    top_indices, ranks = top_indices_by_window(context.test_scores, n_windows, n_cells, TOP_N_PER_WINDOW)
    frame = sx.candidate_frame_for_indices(top_indices, context)
    frame.insert(3, "risk_rank", ranks)
    frame["raw_xgboost_score"] = context.test_scores[top_indices].astype(np.float32)
    frame["sigmoid_calibrated_risk"] = context.calibrated_test_scores[top_indices].astype(np.float32)

    base_columns = [
        "window_index",
        "window_start",
        "h3_cell_res8",
        "risk_rank",
        "sigmoid_calibrated_risk",
        "raw_xgboost_score",
        "actual_label",
        "incident_count",
    ]
    keep_columns = unique_columns(base_columns + ng.NEIGHBOR_FEATURE_COLUMNS)
    output = frame[keep_columns].copy()
    output_path = DASHBOARD_DIR / "risk_scores_top100.csv"
    output.to_csv(output_path, index=False)
    return output_path, output


def write_window_summary(context, top_scores: pd.DataFrame) -> tuple[Path, pd.DataFrame]:
    n_windows = len(context.test_windows)
    n_cells = len(context.inside_cells)
    labels = context.y_test.reshape(n_windows, n_cells)
    incidents = context.incident_counts[:, context.test_windows].T.astype(np.int16)
    wf = fg.window_features(context.test_windows)
    summary = wf[["window_index", "window_start", "hour_block", "day_of_week", "is_weekend", "month", "year"]].copy()
    summary["date"] = pd.to_datetime(summary["window_start"]).dt.strftime("%Y-%m-%d")
    summary["time_window"] = summary["hour_block"].map(time_window_label)
    summary["positive_cell_windows"] = labels.sum(axis=1).astype(np.int32)
    summary["incident_count"] = incidents.sum(axis=1).astype(np.int32)
    top20 = top_scores[top_scores["risk_rank"] <= 20].groupby("window_index", as_index=False).agg(
        top20_mean_calibrated_risk=("sigmoid_calibrated_risk", "mean"),
        top20_max_calibrated_risk=("sigmoid_calibrated_risk", "max"),
        top20_positive_cells=("actual_label", "sum"),
        top20_incidents=("incident_count", "sum"),
    )
    summary = summary.merge(top20, on="window_index", how="left")
    fill_columns = [
        "top20_mean_calibrated_risk",
        "top20_max_calibrated_risk",
        "top20_positive_cells",
        "top20_incidents",
    ]
    summary[fill_columns] = summary[fill_columns].fillna(0)
    output_path = DASHBOARD_DIR / "window_summary.csv"
    summary.to_csv(output_path, index=False)
    return output_path, summary


def replay_window_summary(window_summary: pd.DataFrame) -> pd.DataFrame:
    cutoff_date = pd.Timestamp(DATA_CUTOFF).date()
    starts = pd.to_datetime(window_summary["window_start"])
    candidate = window_summary.copy()
    candidate["date"] = starts.dt.date.astype(str)
    full_counts = candidate.groupby("date")["window_index"].nunique()
    full_dates = [date for date, count in full_counts.items() if count == 8 and pd.Timestamp(date).date() < cutoff_date]
    if not full_dates:
        raise SystemExit("No complete replay date was found before the data cutoff")
    selected_dates = sorted(full_dates)[-REPLAY_DAYS:]
    replay = candidate[candidate["date"].isin(selected_dates)].copy()
    replay = replay.sort_values("window_index")
    if replay.groupby("date")["window_index"].nunique().min() != 8:
        raise SystemExit("Replay date selection includes an incomplete day")
    return replay


def reason_label(feature_name: str) -> str:
    if feature_name == "hist_cell_hour_risk":
        return "This zone has often had incidents in this time block."
    if feature_name == "prev_7d_incident_count":
        return "This zone had recent incidents in the past 7 days."
    if feature_name == "neighbor_prev_7d_incident_count":
        return "Nearby zones had recent incidents in the past 7 days."
    if feature_name == "hist_cell_risk":
        return "This zone has high historical incident risk."
    if feature_name == "prev_24h_incident_count":
        return "This zone had incidents in the previous 24 hours."
    if feature_name == "prev_3h_incident_count":
        return "This zone had an incident in the previous 3-hour window."
    if feature_name == "neighbor_prev_24h_incident_count":
        return "Nearby zones had incidents in the previous 24 hours."
    if feature_name == "neighbor_prev_3h_incident_count":
        return "Nearby zones had incidents in the previous 3-hour window."
    if feature_name == "neighbor_prev_7d_active_cell_count":
        return "Several nearby zones were active in the past 7 days."
    if feature_name == "neighbor_prev_24h_active_cell_count":
        return "Several nearby zones were active in the previous 24 hours."
    if feature_name in {"prev_7d_severity_weight_sum", "neighbor_prev_7d_severity_weight_sum"}:
        return "Recent known-severity incidents add to the model score."
    if feature_name in {"prev_24h_severity_weight_sum", "neighbor_prev_24h_severity_weight_sum"}:
        return "Known-severity incidents in the last day add to the model score."
    if feature_name == "hist_hour_risk":
        return "This time block is historically active."
    if feature_name == "hist_global_risk":
        return "The citywide historical baseline anchors the prediction."
    if feature_name == "hour_block":
        return "This 3-hour block has a distinct historical pattern."
    if feature_name.startswith("day_of_week_"):
        return "The selected weekday pattern affects the prediction."
    if feature_name == "is_weekend":
        return "The weekend or weekday pattern affects the prediction."
    if feature_name == "month":
        return "The selected month affects the prediction."
    if feature_name == "year":
        return "The model accounts for the year in the historical record."
    if feature_name == "inside_neighbor_count":
        return "The number of adjacent Dubai zones affects the neighborhood features."
    return "A model feature increases the predicted risk for this zone and time."


def feature_group(feature_name: str) -> str:
    if feature_name.startswith("day_of_week_"):
        return "day_of_week"
    if feature_name.startswith("geo_scope_"):
        return "geo_scope"
    return feature_name


def feature_value_text(feature_name: str, row: pd.Series) -> str:
    group = feature_group(feature_name)
    if feature_name.startswith("day_of_week_"):
        return feature_name.replace("day_of_week_", "")
    if feature_name.startswith("geo_scope_"):
        return feature_name.replace("geo_scope_", "").replace("_", " ")
    if group in {"hist_cell_hour_risk", "hist_cell_risk", "hist_hour_risk", "hist_global_risk"}:
        return f"{float(row[group]) * 100:.2f}%"
    if group == "hour_block":
        return str(row.get("time_window", time_window_label(int(row[group]))))
    if group == "is_weekend":
        return "weekend" if int(row[group]) == 1 else "weekday"
    if group in row:
        value = row[group]
        if pd.isna(value):
            return "not available"
        if isinstance(value, (np.integer, int)):
            return f"{int(value):,}"
        if isinstance(value, (np.floating, float)):
            return f"{float(value):.3f}"
        return str(value)
    return "active"


def shap_strength(value: float) -> str:
    absolute = abs(value)
    if absolute >= 0.45:
        return "strong"
    if absolute >= 0.18:
        return "moderate"
    return "small"


def write_replay_outputs(context, top_scores: pd.DataFrame, window_summary: pd.DataFrame) -> tuple[dict[str, Path], dict, pd.DataFrame]:
    replay_windows = replay_window_summary(window_summary)
    replay_window_ids = set(replay_windows["window_index"].astype(int))
    replay = top_scores[top_scores["window_index"].isin(replay_window_ids)].copy()
    replay = replay.merge(
        replay_windows[["window_index", "date", "time_window", "incident_count", "positive_cell_windows"]],
        on="window_index",
        how="left",
        suffixes=("", "_window"),
    )
    replay = replay.sort_values(["window_index", "risk_rank"]).reset_index(drop=True)
    replay["rank"] = replay["risk_rank"].astype(np.int16)
    replay["map_zone"] = replay["h3_cell_res8"]
    replay["predicted_incident_risk"] = replay["sigmoid_calibrated_risk"]
    replay = replay.rename(columns={"date": "replay_date"})

    transformed = sx.transformed_frame(context.pipeline, replay)
    explainer = shap.TreeExplainer(
        context.pipeline.named_steps["model"],
        feature_perturbation="tree_path_dependent",
        feature_names=list(transformed.columns),
    )
    explanation = explainer(transformed, check_additivity=False)
    feature_names = list(transformed.columns)

    contributor_rows: list[dict[str, object]] = []
    reason_rows: list[dict[str, object]] = []
    for row_idx, shap_row in enumerate(np.asarray(explanation.values)):
        replay_row = replay.iloc[row_idx]
        selected = [idx for idx in np.argsort(-shap_row) if shap_row[idx] > 0][:5]
        if not selected:
            selected = np.argsort(-np.abs(shap_row))[:5].tolist()

        reason_parts: list[str] = []
        for contributor_rank, feature_idx in enumerate(selected, start=1):
            feature_name = feature_names[feature_idx]
            grouped = feature_group(feature_name)
            shap_value = float(shap_row[feature_idx])
            value_text = feature_value_text(feature_name, replay_row)
            reason = reason_label(feature_name)
            detail = f"{reason} Value: {value_text}. SHAP: {shap_value:+.3f}."
            if len(reason_parts) < 3:
                reason_parts.append(detail)
            contributor_rows.append(
                {
                    "replay_date": replay_row["replay_date"],
                    "time_window": replay_row["time_window"],
                    "window_index": int(replay_row["window_index"]),
                    "h3_cell_res8": replay_row["h3_cell_res8"],
                    "rank": int(replay_row["rank"]),
                    "contributor_rank": contributor_rank,
                    "feature": grouped,
                    "raw_transformed_feature": feature_name,
                    "reason": reason,
                    "feature_value": value_text,
                    "shap_value": shap_value,
                    "direction": "increases risk" if shap_value >= 0 else "reduces risk",
                    "strength": shap_strength(shap_value),
                }
            )

        reason_rows.append(
            {
                "main_reason": reason_parts[0] if reason_parts else "The model ranked this zone-time window as high risk.",
                "reason_2": reason_parts[1] if len(reason_parts) > 1 else "",
                "reason_3": reason_parts[2] if len(reason_parts) > 2 else "",
                "other_reasons": " ".join(reason_parts[1:3]),
            }
        )

    replay = pd.concat([replay, pd.DataFrame(reason_rows)], axis=1)
    replay_columns = unique_columns(
        [
            "replay_date",
            "time_window",
            "window_index",
            "window_start",
            "rank",
            "map_zone",
            "h3_cell_res8",
            "predicted_incident_risk",
            "raw_xgboost_score",
            "actual_label",
            "incident_count",
            "incident_count_window",
            "positive_cell_windows",
            "main_reason",
            "reason_2",
            "reason_3",
            "other_reasons",
        ]
        + ng.NEIGHBOR_FEATURE_COLUMNS
    )
    replay_path = DASHBOARD_DIR / "rolling_replay_top_zones.csv"
    replay[replay_columns].to_csv(replay_path, index=False)

    contributor_path = DASHBOARD_DIR / "rolling_replay_shap_contributors.csv"
    contributors = pd.DataFrame(contributor_rows)
    contributors.to_csv(contributor_path, index=False)

    replay_summary_path = DASHBOARD_DIR / "rolling_replay_window_summary.csv"
    replay_windows.to_csv(replay_summary_path, index=False)

    default_date = replay_windows["date"].max()
    metadata = {
        "replay_scope": "Rolling replay using the final neighbor-lag XGBoost model.",
        "data_cutoff": DATA_CUTOFF,
        "default_replay_date": default_date,
        "replay_start_date": replay_windows["date"].min(),
        "replay_end_date": replay_windows["date"].max(),
        "replay_days": int(replay_windows["date"].nunique()),
        "replay_windows": int(replay_windows["window_index"].nunique()),
        "replay_rows_written": int(len(replay)),
        "shap_contributor_rows": int(len(contributors)),
        "top_n_per_window": TOP_N_PER_WINDOW,
        "model": "Default neighbor-feature XGBoost with sigmoid-calibrated risk display.",
        "warning": "Forecasts after 2026-06-01 require updated incident data so recent 3-hour, 24-hour, 7-day, and neighboring lag features can be computed.",
    }
    replay_metadata_path = DASHBOARD_DIR / "rolling_replay_metadata.json"
    replay_metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    outputs = {
        "rolling_replay_rows": replay_path,
        "rolling_replay_window_summary": replay_summary_path,
        "rolling_replay_shap_contributors": contributor_path,
        "rolling_replay_metadata": replay_metadata_path,
    }
    return outputs, metadata, replay


def write_metadata(context, top_scores: pd.DataFrame, replay_metadata: dict, outputs: dict[str, Path]) -> Path:
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dashboard_scope": "Rolling replay and historical-check dashboard for final thesis report. Not a live traffic-control system.",
        "historical_model": "Default neighbor-feature XGBoost with sigmoid-calibrated risk display.",
        "replay_model": replay_metadata["model"],
        "h3_resolution": fg.H3_RESOLUTION,
        "time_window_hours": fg.WINDOW_HOURS,
        "inside_dubai_cells": len(context.inside_cells),
        "data_cutoff": DATA_CUTOFF,
        "default_replay_date": replay_metadata["default_replay_date"],
        "replay_start_date": replay_metadata["replay_start_date"],
        "replay_end_date": replay_metadata["replay_end_date"],
        "replay_windows": replay_metadata["replay_windows"],
        "test_windows": len(context.test_windows),
        "test_candidates": int(len(context.y_test)),
        "test_positive_cell_windows": int(context.y_test.sum()),
        "top_n_per_window": TOP_N_PER_WINDOW,
        "risk_rows": int(len(top_scores)),
        "calibrated_threshold": float(context.calibrated_threshold),
        "raw_threshold": float(context.raw_threshold),
        "selected_calibration": "sigmoid",
        "warning": replay_metadata["warning"],
        "outputs": {key: str(path.relative_to(ROOT)) for key, path in outputs.items()},
    }
    output_path = DASHBOARD_DIR / "dashboard_metadata.json"
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_path


def write_audit(context, top_scores: pd.DataFrame, replay_top: pd.DataFrame, replay_metadata: dict, outputs: dict[str, Path]) -> None:
    lines = [
        "# Streamlit dashboard data audit",
        "",
        "- Dashboard type: rolling replay plus historical thesis demonstration.",
        "- Live deployment claim: none.",
        "- Map used by the app: MapTiler vector style through PyDeck when `MAPTILER_API_KEY` is available; Plotly/Esri label-free fallback only if the key is unavailable.",
        "- Replay model path: default neighbor-feature XGBoost.",
        "- Calibration path: sigmoid calibration fitted on validation predictions only.",
        "- Post-cutoff dates are not scored in the active dashboard.",
        "- Warning shown in app: forecasts after 2026-06-01 require updated incident counts to compute recent lag features.",
        f"- H3 resolution: `{fg.H3_RESOLUTION}`",
        f"- Time window hours: `{fg.WINDOW_HOURS}`",
        f"- Inside-Dubai cells: `{len(context.inside_cells)}`",
        f"- Replay date range: `{replay_metadata['replay_start_date']}` to `{replay_metadata['replay_end_date']}`",
        f"- Default replay date: `{replay_metadata['default_replay_date']}`",
        f"- Replay windows: `{replay_metadata['replay_windows']}`",
        f"- Replay rows written: `{replay_metadata['replay_rows_written']}`",
        f"- Replay SHAP contributor rows: `{replay_metadata['shap_contributor_rows']}`",
        f"- Test windows: `{len(context.test_windows)}`",
        f"- Test candidate rows: `{len(context.y_test)}`",
        f"- Test positive cell/windows: `{int(context.y_test.sum())}`",
        f"- Top-risk rows retained per test window: `{TOP_N_PER_WINDOW}`",
        f"- Historical dashboard risk rows written: `{len(top_scores)}`",
        "- The dashboard data files are generated and ignored by Git; screenshots and source code are committed.",
        "",
        "## Generated files",
        "",
    ]
    for key, path in outputs.items():
        lines.append(f"- {key}: `{path.relative_to(ROOT)}`")
    lines.extend(
        [
            "",
            "## Data checks",
            "",
            f"- Unique windows in historical risk table: `{top_scores['window_index'].nunique()}`",
            f"- Maximum historical risk rank: `{int(top_scores['risk_rank'].max())}`",
            f"- Minimum calibrated historical risk: `{float(top_scores['sigmoid_calibrated_risk'].min()):.8f}`",
            f"- Maximum calibrated historical risk: `{float(top_scores['sigmoid_calibrated_risk'].max()):.8f}`",
            f"- Replay dates: `{', '.join(sorted(replay_top['replay_date'].unique()))}`",
            f"- Replay windows per date: `{int(replay_top.groupby('replay_date')['time_window'].nunique().min())}`",
            f"- Replay maximum rank: `{int(replay_top['rank'].max())}`",
            f"- Replay minimum predicted risk: `{float(replay_top['predicted_incident_risk'].min()):.8f}`",
            f"- Replay maximum predicted risk: `{float(replay_top['predicted_incident_risk'].max()):.8f}`",
            "",
            "## Replay model features",
            "",
            "The rolling replay uses the final neighbor-lag feature set, including recent same-cell and nearby-cell lag features. It is therefore not used for dates after the data cutoff unless updated incident data is available.",
        ]
    )
    (AUDIT_DIR / "dashboard_data_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    remove_legacy_forecast_files()
    context = sx.build_final_model_context()
    if len(context.inside_cells) != fg.EXPECTED_INSIDE_DUBAI_CELLS:
        raise SystemExit(f"Expected {fg.EXPECTED_INSIDE_DUBAI_CELLS} cells, got {len(context.inside_cells)}")
    if len(context.y_test) != fg.EXPECTED_TEST_CANDIDATES:
        raise SystemExit(f"Expected {fg.EXPECTED_TEST_CANDIDATES} test rows, got {len(context.y_test)}")
    if int(context.y_test.sum()) != fg.EXPECTED_TEST_POSITIVE_CELL_WINDOWS:
        raise SystemExit(f"Expected {fg.EXPECTED_TEST_POSITIVE_CELL_WINDOWS} positives, got {int(context.y_test.sum())}")

    cell_path = write_cell_geojson(context)
    risk_path, top_scores = write_risk_scores(context)
    window_path, window_summary = write_window_summary(context, top_scores)
    replay_outputs, replay_metadata, replay_top = write_replay_outputs(context, top_scores, window_summary)
    outputs = {
        "cell_geojson": cell_path,
        "risk_scores": risk_path,
        "window_summary": window_path,
        **replay_outputs,
    }
    metadata_path = write_metadata(context, top_scores, replay_metadata, outputs)
    outputs["metadata"] = metadata_path
    write_audit(context, top_scores, replay_top, replay_metadata, outputs)
    print(f"Wrote dashboard data to {DASHBOARD_DIR}")
    print(f"Wrote audit to {AUDIT_DIR / 'dashboard_data_audit.md'}")


if __name__ == "__main__":
    main()
