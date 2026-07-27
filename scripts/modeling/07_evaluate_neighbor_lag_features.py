from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/thesis-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/thesis-cache")

import matplotlib

matplotlib.use("Agg")

import h3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[2]
FULL_GRID_SCRIPT = ROOT / "scripts" / "modeling" / "02_run_full_grid_evaluation.py"


def load_full_grid_module():
    spec = importlib.util.spec_from_file_location("full_grid_evaluation", FULL_GRID_SCRIPT)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not import {FULL_GRID_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fg = load_full_grid_module()


TABLE_DIR = ROOT / "reports" / "modeling" / "tables"
FIGURE_DIR = ROOT / "reports" / "modeling" / "figures"
AUDIT_DIR = ROOT / "data" / "audit"

NEIGHBOR_FEATURES = [
    "inside_neighbor_count",
    "neighbor_prev_3h_incident_count",
    "neighbor_prev_24h_incident_count",
    "neighbor_prev_7d_incident_count",
    "neighbor_prev_24h_severity_weight_sum",
    "neighbor_prev_7d_severity_weight_sum",
    "neighbor_prev_24h_active_cell_count",
    "neighbor_prev_7d_active_cell_count",
]
CURRENT_FEATURES = fg.FEATURE_COLUMNS
NEIGHBOR_FEATURE_COLUMNS = fg.NUMERIC_FEATURES + NEIGHBOR_FEATURES + fg.CATEGORICAL_FEATURES


@dataclass(frozen=True)
class NeighborState:
    neighbor_indices: list[np.ndarray]
    inside_neighbor_count: np.ndarray
    neighbor_count_cumulative: np.ndarray
    neighbor_severity_cumulative: np.ndarray
    neighbor_active_24h: np.ndarray
    neighbor_active_7d: np.ndarray
    cell_to_index: dict[str, int]


@dataclass
class ExperimentModel:
    name: str
    feature_set_id: str
    notes: str
    feature_columns: list[str]
    train_rows_used: int
    estimator: Pipeline | None


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def frame_to_markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for record in frame.astype(str).to_dict(orient="records"):
        rows.append("| " + " | ".join(record[column] for column in columns) + " |")
    return "\n".join(rows)


def lag_matrix(cumulative: np.ndarray, window_indices: np.ndarray, length: int) -> np.ndarray:
    start = np.maximum(window_indices - length, 0)
    end = window_indices
    return (cumulative[:, end] - cumulative[:, start]).astype(np.int32)


def lag_rows(cumulative: np.ndarray, cell_index: np.ndarray, window_index: np.ndarray, length: int) -> np.ndarray:
    start = np.maximum(window_index - length, 0)
    end = window_index
    return (cumulative[cell_index, end] - cumulative[cell_index, start]).astype(np.int32)


def build_neighbor_state(
    inside_cells: list[str],
    labels: np.ndarray,
    incident_counts: np.ndarray,
    severity_weights: np.ndarray,
) -> NeighborState:
    inside_set = set(inside_cells)
    cell_to_index = {cell: idx for idx, cell in enumerate(inside_cells)}
    neighbor_indices: list[np.ndarray] = []
    for cell in inside_cells:
        neighbors = sorted(
            neighbor
            for neighbor in h3.grid_disk(cell, 1)
            if neighbor != cell and neighbor in inside_set
        )
        if len(neighbors) != len(set(neighbors)):
            raise SystemExit(f"Duplicate neighbors detected for {cell}")
        for neighbor in neighbors:
            if h3.get_resolution(neighbor) != fg.H3_RESOLUTION:
                raise SystemExit(f"Neighbor {neighbor} is not H3 resolution {fg.H3_RESOLUTION}")
        neighbor_indices.append(np.asarray([cell_to_index[neighbor] for neighbor in neighbors], dtype=np.int32))

    inside_neighbor_count = np.asarray([len(indices) for indices in neighbor_indices], dtype=np.int8)
    if inside_neighbor_count.min() < 0 or inside_neighbor_count.max() > 6:
        raise SystemExit("Inside-neighbor count outside expected ring-1 range")

    n_cells, n_windows = labels.shape
    neighbor_incident = np.zeros((n_cells, n_windows), dtype=np.int32)
    neighbor_severity = np.zeros((n_cells, n_windows), dtype=np.int32)
    for idx, neighbors in enumerate(neighbor_indices):
        if len(neighbors):
            neighbor_incident[idx] = incident_counts[neighbors].astype(np.int32).sum(axis=0)
            neighbor_severity[idx] = severity_weights[neighbors].astype(np.int32).sum(axis=0)

    neighbor_count_cumulative = np.concatenate(
        [np.zeros((n_cells, 1), dtype=np.int32), np.cumsum(neighbor_incident, axis=1, dtype=np.int32)],
        axis=1,
    )
    neighbor_severity_cumulative = np.concatenate(
        [np.zeros((n_cells, 1), dtype=np.int32), np.cumsum(neighbor_severity, axis=1, dtype=np.int32)],
        axis=1,
    )

    label_cumulative = np.concatenate(
        [np.zeros((n_cells, 1), dtype=np.int32), np.cumsum(labels.astype(np.int32), axis=1, dtype=np.int32)],
        axis=1,
    )
    all_windows = np.arange(n_windows, dtype=np.int32)
    active_by_cell_24h = (lag_matrix(label_cumulative, all_windows, 8) > 0).astype(np.int8)
    active_by_cell_7d = (lag_matrix(label_cumulative, all_windows, 56) > 0).astype(np.int8)
    neighbor_active_24h = np.zeros((n_cells, n_windows), dtype=np.int8)
    neighbor_active_7d = np.zeros((n_cells, n_windows), dtype=np.int8)
    for idx, neighbors in enumerate(neighbor_indices):
        if len(neighbors):
            neighbor_active_24h[idx] = active_by_cell_24h[neighbors].sum(axis=0).astype(np.int8)
            neighbor_active_7d[idx] = active_by_cell_7d[neighbors].sum(axis=0).astype(np.int8)

    return NeighborState(
        neighbor_indices=neighbor_indices,
        inside_neighbor_count=inside_neighbor_count,
        neighbor_count_cumulative=neighbor_count_cumulative,
        neighbor_severity_cumulative=neighbor_severity_cumulative,
        neighbor_active_24h=neighbor_active_24h,
        neighbor_active_7d=neighbor_active_7d,
        cell_to_index=cell_to_index,
    )


def add_neighbor_features(frame: pd.DataFrame, state: NeighborState) -> pd.DataFrame:
    output = frame.copy()
    cell_index = output["h3_cell_res8"].astype(str).map(state.cell_to_index)
    if cell_index.isna().any():
        raise SystemExit("Frame contains cells outside the inside-Dubai neighbor universe")
    cell_index_array = cell_index.to_numpy(dtype=np.int32)
    window_index = output["window_index"].to_numpy(dtype=np.int32)
    output["inside_neighbor_count"] = state.inside_neighbor_count[cell_index_array].astype(np.int8)
    output["neighbor_prev_3h_incident_count"] = lag_rows(
        state.neighbor_count_cumulative, cell_index_array, window_index, 1
    )
    output["neighbor_prev_24h_incident_count"] = lag_rows(
        state.neighbor_count_cumulative, cell_index_array, window_index, 8
    )
    output["neighbor_prev_7d_incident_count"] = lag_rows(
        state.neighbor_count_cumulative, cell_index_array, window_index, 56
    )
    output["neighbor_prev_24h_severity_weight_sum"] = lag_rows(
        state.neighbor_severity_cumulative, cell_index_array, window_index, 8
    )
    output["neighbor_prev_7d_severity_weight_sum"] = lag_rows(
        state.neighbor_severity_cumulative, cell_index_array, window_index, 56
    )
    output["neighbor_prev_24h_active_cell_count"] = state.neighbor_active_24h[cell_index_array, window_index]
    output["neighbor_prev_7d_active_cell_count"] = state.neighbor_active_7d[cell_index_array, window_index]
    return output


def validate_neighbor_state(
    state: NeighborState,
    labels: np.ndarray,
    incident_counts: np.ndarray,
    severity_weights: np.ndarray,
) -> dict[str, int]:
    if any(len(indices) != len(set(indices.tolist())) for indices in state.neighbor_indices):
        raise SystemExit("Duplicate neighbor indices detected")
    if any(idx in neighbors for idx, neighbors in enumerate(state.neighbor_indices)):
        raise SystemExit("A neighbor list includes its center cell")

    cells_with_neighbors = np.flatnonzero(state.inside_neighbor_count > 0)
    if len(cells_with_neighbors) == 0:
        raise SystemExit("No inside-Dubai cells have inside-Dubai neighbors")

    cell = int(cells_with_neighbors[0])
    neighbors = state.neighbor_indices[cell]
    for window in [0, 1, 8, 56, fg.TRAIN_END_EXCLUSIVE, fg.VALIDATION_END_EXCLUSIVE]:
        got_prev = int(lag_rows(state.neighbor_count_cumulative, np.array([cell]), np.array([window]), 1)[0])
        expected_prev = 0 if window == 0 else int(incident_counts[neighbors, window - 1].sum())
        if got_prev != expected_prev:
            raise SystemExit(f"Bad neighbor previous-window count for cell={cell}, window={window}: {got_prev} != {expected_prev}")

        start_24 = max(window - 8, 0)
        got_24 = int(lag_rows(state.neighbor_count_cumulative, np.array([cell]), np.array([window]), 8)[0])
        expected_24 = int(incident_counts[neighbors, start_24:window].sum())
        if got_24 != expected_24:
            raise SystemExit(f"Bad neighbor 24h count for cell={cell}, window={window}: {got_24} != {expected_24}")

        got_weight_24 = int(lag_rows(state.neighbor_severity_cumulative, np.array([cell]), np.array([window]), 8)[0])
        expected_weight_24 = int(severity_weights[neighbors, start_24:window].sum())
        if got_weight_24 != expected_weight_24:
            raise SystemExit(
                f"Bad neighbor 24h severity sum for cell={cell}, window={window}: {got_weight_24} != {expected_weight_24}"
            )

        start_7d = max(window - 56, 0)
        got_active_7d = int(state.neighbor_active_7d[cell, window])
        expected_active_7d = int((labels[neighbors, start_7d:window].sum(axis=1) > 0).sum())
        if got_active_7d != expected_active_7d:
            raise SystemExit(f"Bad neighbor 7d active-cell count for cell={cell}, window={window}: {got_active_7d} != {expected_active_7d}")

    zero_window_counts = int(lag_matrix(state.neighbor_count_cumulative, np.array([0], dtype=np.int32), 1).sum())
    zero_window_weights = int(lag_matrix(state.neighbor_severity_cumulative, np.array([0], dtype=np.int32), 8).sum())
    if zero_window_counts != 0 or zero_window_weights != 0:
        raise SystemExit("Window 0 neighbor lag features are not zero")

    return {
        "min_inside_neighbor_count": int(state.inside_neighbor_count.min()),
        "max_inside_neighbor_count": int(state.inside_neighbor_count.max()),
        "mean_inside_neighbor_count_x1000": int(round(float(state.inside_neighbor_count.mean()) * 1000)),
        "cells_with_zero_inside_neighbors": int((state.inside_neighbor_count == 0).sum()),
    }


def build_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    categorical = [column for column in fg.CATEGORICAL_FEATURES if column in feature_columns]
    numeric = [column for column in feature_columns if column not in categorical]
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def train_xgboost(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    sample_indices: np.ndarray,
    seed: int,
) -> tuple[Pipeline, int]:
    if fg.XGBClassifier is None:
        raise SystemExit(fg.XGB_IMPORT_ERROR)
    y_train = train_df["risk_label"].astype(int)
    y_sample = y_train.iloc[sample_indices]
    pos = int(y_sample.sum())
    neg = int(len(y_sample) - pos)
    model = fg.XGBClassifier(
        n_estimators=160,
        max_depth=5,
        learning_rate=0.07,
        subsample=0.85,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
        scale_pos_weight=neg / max(pos, 1),
    )
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(feature_columns)),
            ("model", model),
        ]
    )
    pipeline.fit(train_df.iloc[sample_indices][feature_columns], y_train.iloc[sample_indices])
    return pipeline, len(sample_indices)


def score_models(
    split_name: str,
    window_indices: np.ndarray,
    inside_cells: list[str],
    labels: np.ndarray,
    incident_counts: np.ndarray,
    count_cumulative: np.ndarray,
    severity_cumulative: np.ndarray,
    priors: dict[str, np.ndarray | float],
    state: NeighborState,
    models: list[ExperimentModel],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    y_parts: list[np.ndarray] = []
    incident_parts: list[np.ndarray] = []
    score_parts: dict[str, list[np.ndarray]] = {model.name: [] for model in models}
    for start in range(0, len(window_indices), fg.CHUNK_WINDOWS):
        windows = window_indices[start : start + fg.CHUNK_WINDOWS]
        chunk = fg.make_candidate_chunk(
            inside_cells,
            windows,
            labels,
            incident_counts,
            count_cumulative,
            severity_cumulative,
            priors,
        )
        chunk_frame = add_neighbor_features(chunk.frame, state)
        y_parts.append(chunk.labels)
        incident_parts.append(chunk.incident_counts)
        for model in models:
            if model.estimator is None:
                scores = chunk_frame["hist_cell_hour_risk"].to_numpy(dtype=np.float32)
            else:
                scores = model.estimator.predict_proba(chunk_frame[model.feature_columns])[:, 1].astype(np.float32)
            score_parts[model.name].append(scores)
    y = np.concatenate(y_parts).astype(np.uint8)
    incidents = np.concatenate(incident_parts).astype(np.int16)
    scores = {name: np.concatenate(parts).astype(np.float32) for name, parts in score_parts.items()}
    expected = len(window_indices) * len(inside_cells)
    if len(y) != expected:
        raise SystemExit(f"{split_name} produced {len(y)} rows, expected {expected}")
    return y, incidents, scores


def evaluate(
    models: list[ExperimentModel],
    y_val: np.ndarray,
    val_scores: dict[str, np.ndarray],
    y_test: np.ndarray,
    test_scores: dict[str, np.ndarray],
) -> list[fg.EvaluationResult]:
    results: list[fg.EvaluationResult] = []
    for model in models:
        threshold, validation_f1 = fg.threshold_by_validation_f1(y_val, val_scores[model.name])
        prediction = (test_scores[model.name] >= threshold).astype(np.uint8)
        tn, fp, fn, tp = confusion_matrix(y_test, prediction, labels=[0, 1]).ravel()
        results.append(
            fg.EvaluationResult(
                name=model.name,
                notes=model.notes,
                train_rows_used=model.train_rows_used,
                threshold=threshold,
                validation_f1=validation_f1,
                validation_pr_auc=float(average_precision_score(y_val, val_scores[model.name])),
                test_roc_auc=float(roc_auc_score(y_test, test_scores[model.name])),
                test_pr_auc=float(average_precision_score(y_test, test_scores[model.name])),
                test_precision=float(precision_score(y_test, prediction, zero_division=0)),
                test_recall=float(recall_score(y_test, prediction, zero_division=0)),
                test_f1=float(f1_score(y_test, prediction, zero_division=0)),
                tn=int(tn),
                fp=int(fp),
                fn=int(fn),
                tp=int(tp),
                validation_scores=val_scores[model.name],
                test_scores=test_scores[model.name],
            )
        )
    return results


def metrics_frame(results: list[fg.EvaluationResult], validation_rows: int, test_rows: int) -> pd.DataFrame:
    rows = []
    for result in results:
        rows.append(
            {
                "model": result.name,
                "notes": result.notes,
                "train_rows_used": result.train_rows_used,
                "validation_rows": validation_rows,
                "test_rows": test_rows,
                "threshold": result.threshold,
                "validation_f1": result.validation_f1,
                "validation_pr_auc": result.validation_pr_auc,
                "test_roc_auc": result.test_roc_auc,
                "test_pr_auc": result.test_pr_auc,
                "test_precision": result.test_precision,
                "test_recall": result.test_recall,
                "test_f1": result.test_f1,
                "test_tn": result.tn,
                "test_fp": result.fp,
                "test_fn": result.fn,
                "test_tp": result.tp,
            }
        )
    return pd.DataFrame(rows)


def feature_sets_frame(models: list[ExperimentModel]) -> pd.DataFrame:
    rows = []
    for model in models:
        rows.append(
            {
                "model": model.name,
                "feature_set_id": model.feature_set_id,
                "feature_count": len(model.feature_columns),
                "uses_neighbor_features": any(column in NEIGHBOR_FEATURES for column in model.feature_columns),
                "features": "|".join(model.feature_columns),
                "notes": model.notes,
            }
        )
    return pd.DataFrame(rows)


def plot_model_comparison(metrics: pd.DataFrame) -> None:
    plot_df = metrics.melt(
        id_vars=["model"],
        value_vars=["test_roc_auc", "test_pr_auc", "test_f1", "test_recall"],
        var_name="metric",
        value_name="value",
    )
    label_map = {
        "test_roc_auc": "ROC-AUC",
        "test_pr_auc": "PR-AUC",
        "test_f1": "F1",
        "test_recall": "Recall",
    }
    plot_df["metric"] = plot_df["metric"].map(label_map)
    plt.figure(figsize=(9, 5.5))
    sns.barplot(data=plot_df, x="model", y="value", hue="metric")
    plt.ylim(0, 1)
    plt.xlabel("")
    plt.ylabel("Score")
    plt.title("Neighbor lag feature experiment")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "neighbor_lag_feature_model_comparison.png", dpi=200)
    plt.close()


def plot_top_k(top_k: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 5.5))
    sns.lineplot(data=top_k, x="k", y="precision_at_k", hue="model", marker="o", linewidth=2)
    plt.xlabel("Top-k cells per 3-hour window")
    plt.ylabel("Precision at k")
    plt.title("Neighbor lag feature top-k hotspot precision")
    plt.ylim(0, max(0.05, float(top_k["precision_at_k"].max()) * 1.15))
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "neighbor_lag_feature_topk_comparison.png", dpi=200)
    plt.close()


def write_audit(
    neighbor_stats: dict[str, int],
    metrics: pd.DataFrame,
    top_k: pd.DataFrame,
    feature_sets: pd.DataFrame,
) -> None:
    current = metrics.loc[metrics["model"] == "XGBoost current features"].iloc[0]
    neighbor = metrics.loc[metrics["model"] == "XGBoost + neighbor lags"].iloc[0]
    audit = [
        "# Neighboring H3 lag feature experiment",
        "",
        f"- H3 resolution: `{fg.H3_RESOLUTION}`",
        f"- Time window hours: `{fg.WINDOW_HOURS}`",
        f"- Inside-Dubai cells: `{fg.EXPECTED_INSIDE_DUBAI_CELLS}`",
        "- Neighbor definition: H3 ring-1 cells from `h3.grid_disk(cell, 1)`, excluding the center cell and excluding cells outside the inside-Dubai universe.",
        f"- Inside-neighbor count range: `{neighbor_stats['min_inside_neighbor_count']}` to `{neighbor_stats['max_inside_neighbor_count']}`",
        f"- Mean inside-neighbor count: `{neighbor_stats['mean_inside_neighbor_count_x1000'] / 1000:.3f}`",
        f"- Cells with zero inside-Dubai neighbors: `{neighbor_stats['cells_with_zero_inside_neighbors']}`",
        "- Neighbor lag features use only windows before the target window.",
        "- Unknown-severity incidents remain in incident counts, but unknown severity contributes zero to neighbor severity-weighted sums.",
        f"- PR-AUC change vs current XGBoost: `{neighbor['test_pr_auc'] - current['test_pr_auc']:.6f}`",
        f"- F1 change vs current XGBoost: `{neighbor['test_f1'] - current['test_f1']:.6f}`",
        "",
        "## Feature sets",
        "",
        frame_to_markdown(feature_sets[["model", "feature_set_id", "feature_count", "uses_neighbor_features"]]),
        "",
        "## Test metrics",
        "",
        frame_to_markdown(
            metrics[
                [
                    "model",
                    "test_roc_auc",
                    "test_pr_auc",
                    "test_precision",
                    "test_recall",
                    "test_f1",
                    "threshold",
                ]
            ].round(6)
        ),
        "",
        "## Top-k hotspot metrics",
        "",
        frame_to_markdown(
            top_k[
                [
                    "model",
                    "k",
                    "recall_at_k",
                    "precision_at_k",
                    "positive_window_hit_rate_at_k",
                    "incident_recall_at_k",
                ]
            ].round(6)
        ),
        "",
    ]
    (AUDIT_DIR / "neighbor_lag_feature_audit.md").write_text("\n".join(audit), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    fg.validate_inputs([fg.MODEL_SAMPLE_PATH, fg.POSITIVE_COUNTS_PATH, fg.CELL_SCOPE_PATH])
    inside_cells = fg.load_inside_cells(fg.CELL_SCOPE_PATH)
    positive = fg.load_inside_positive_counts(fg.POSITIVE_COUNTS_PATH, inside_cells)
    labels, incident_counts, severity_weights = fg.build_dense_matrices(positive, inside_cells)
    priors = fg.historical_priors(labels)
    state = build_neighbor_state(inside_cells, labels, incident_counts, severity_weights)
    neighbor_stats = validate_neighbor_state(state, labels, incident_counts, severity_weights)

    count_cumulative = np.concatenate(
        [np.zeros((labels.shape[0], 1), dtype=np.int32), np.cumsum(incident_counts.astype(np.int32), axis=1)],
        axis=1,
    )
    severity_cumulative = np.concatenate(
        [np.zeros((labels.shape[0], 1), dtype=np.int32), np.cumsum(severity_weights.astype(np.int32), axis=1)],
        axis=1,
    )

    train_df = fg.load_training_sample(fg.MODEL_SAMPLE_PATH, inside_cells, priors)
    train_with_neighbors = add_neighbor_features(train_df, state)
    y_train = train_df["risk_label"].astype(int)
    sample_indices = fg.sample_training_rows(y_train, fg.XGB_TRAIN_CAP, fg.RANDOM_SEED + 1)

    current_estimator, current_rows = train_xgboost(train_df, CURRENT_FEATURES, sample_indices, fg.RANDOM_SEED)
    neighbor_estimator, neighbor_rows = train_xgboost(
        train_with_neighbors, NEIGHBOR_FEATURE_COLUMNS, sample_indices, fg.RANDOM_SEED
    )
    models = [
        ExperimentModel(
            name="Historical risk",
            feature_set_id="historical_score_only",
            notes="Train-period full-grid empirical risk by inside-Dubai H3 cell and hour block.",
            feature_columns=["hist_cell_hour_risk"],
            train_rows_used=len(train_df),
            estimator=None,
        ),
        ExperimentModel(
            name="XGBoost current features",
            feature_set_id="current_xgboost_features",
            notes="Current full-grid XGBoost feature set without neighbor lags.",
            feature_columns=CURRENT_FEATURES,
            train_rows_used=current_rows,
            estimator=current_estimator,
        ),
        ExperimentModel(
            name="XGBoost + neighbor lags",
            feature_set_id="neighbor_lag_xgboost_features",
            notes="Current XGBoost features plus ring-1 inside-Dubai neighbor lag features.",
            feature_columns=NEIGHBOR_FEATURE_COLUMNS,
            train_rows_used=neighbor_rows,
            estimator=neighbor_estimator,
        ),
    ]

    validation_windows = np.arange(fg.TRAIN_END_EXCLUSIVE, fg.VALIDATION_END_EXCLUSIVE, dtype=np.int32)
    test_windows = np.arange(fg.VALIDATION_END_EXCLUSIVE, fg.WINDOW_COUNT, dtype=np.int32)
    y_val, _, val_scores = score_models(
        "validation",
        validation_windows,
        inside_cells,
        labels,
        incident_counts,
        count_cumulative,
        severity_cumulative,
        priors,
        state,
        models,
    )
    y_test, test_incidents, test_scores = score_models(
        "test",
        test_windows,
        inside_cells,
        labels,
        incident_counts,
        count_cumulative,
        severity_cumulative,
        priors,
        state,
        models,
    )
    if len(y_val) != fg.EXPECTED_VALIDATION_CANDIDATES:
        raise SystemExit(f"Expected {fg.EXPECTED_VALIDATION_CANDIDATES} validation candidates, got {len(y_val)}")
    if len(y_test) != fg.EXPECTED_TEST_CANDIDATES:
        raise SystemExit(f"Expected {fg.EXPECTED_TEST_CANDIDATES} test candidates, got {len(y_test)}")
    if int(y_test.sum()) != fg.EXPECTED_TEST_POSITIVE_CELL_WINDOWS:
        raise SystemExit(f"Expected {fg.EXPECTED_TEST_POSITIVE_CELL_WINDOWS} test positives, got {int(y_test.sum())}")

    results = evaluate(models, y_val, val_scores, y_test, test_scores)
    metrics = metrics_frame(results, len(y_val), len(y_test))
    top_k = fg.full_grid_top_k_metrics(results, y_test, test_incidents, len(test_windows), len(inside_cells))
    feature_sets = feature_sets_frame(models)

    metrics.to_csv(TABLE_DIR / "neighbor_lag_feature_metrics.csv", index=False)
    top_k.to_csv(TABLE_DIR / "neighbor_lag_feature_topk.csv", index=False)
    feature_sets.to_csv(TABLE_DIR / "neighbor_lag_feature_feature_sets.csv", index=False)
    plot_model_comparison(metrics)
    plot_top_k(top_k)
    write_audit(neighbor_stats, metrics, top_k, feature_sets)

    print(f"Wrote {TABLE_DIR / 'neighbor_lag_feature_metrics.csv'}")
    print(f"Wrote {TABLE_DIR / 'neighbor_lag_feature_topk.csv'}")
    print(f"Wrote {TABLE_DIR / 'neighbor_lag_feature_feature_sets.csv'}")
    print(f"Wrote {FIGURE_DIR / 'neighbor_lag_feature_model_comparison.png'}")
    print(f"Wrote {FIGURE_DIR / 'neighbor_lag_feature_topk_comparison.png'}")
    print(f"Wrote {AUDIT_DIR / 'neighbor_lag_feature_audit.md'}")


if __name__ == "__main__":
    main()
