from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/thesis-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/thesis-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier

    XGB_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - handled at runtime.
    XGBClassifier = None
    XGB_IMPORT_ERROR = f"xgboost could not be imported: {exc}"


ROOT = Path(__file__).resolve().parents[2]
MODEL_SAMPLE_PATH = ROOT / "data" / "processed" / "grid_time_model_sample_res8_3h.csv"
POSITIVE_COUNTS_PATH = ROOT / "data" / "processed" / "grid_time_incident_counts_res8_3h.csv"
CELL_SCOPE_PATH = ROOT / "data" / "audit" / "h3_cell_scope_summary_res8.csv"
REPORT_DIR = ROOT / "reports" / "modeling"
TABLE_DIR = REPORT_DIR / "tables"
FIGURE_DIR = REPORT_DIR / "figures"
AUDIT_DIR = ROOT / "data" / "audit"

H3_RESOLUTION = 8
WINDOW_HOURS = 3
START_WINDOW = pd.Timestamp("2018-08-13T06:00:00")
WINDOW_COUNT = 22_795
TRAIN_END_EXCLUSIVE = 15_956
VALIDATION_END_EXCLUSIVE = 19_375
EXPECTED_INSIDE_DUBAI_CELLS = 1_305
EXPECTED_VALIDATION_CANDIDATES = 4_461_795
EXPECTED_TEST_CANDIDATES = 4_463_100
EXPECTED_TEST_POSITIVE_CELL_WINDOWS = 108_224

RANDOM_SEED = 42
RF_TRAIN_CAP = 500_000
XGB_TRAIN_CAP = 1_000_000
CHUNK_WINDOWS = 256
TOP_K_VALUES = [5, 10, 20]

LEAKAGE_COLUMNS = {
    "incident_count",
    "severity_weight_sum",
    "minor_count",
    "moderate_count",
    "severe_count",
    "unknown_count",
}

NUMERIC_FEATURES = [
    "hour_block",
    "is_weekend",
    "month",
    "year",
    "prev_3h_incident_count",
    "prev_24h_incident_count",
    "prev_7d_incident_count",
    "prev_24h_severity_weight_sum",
    "prev_7d_severity_weight_sum",
    "hist_cell_hour_risk",
    "hist_cell_risk",
    "hist_hour_risk",
    "hist_global_risk",
]
CATEGORICAL_FEATURES = ["day_of_week", "geo_scope"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass
class CandidateChunk:
    frame: pd.DataFrame
    labels: np.ndarray
    incident_counts: np.ndarray


@dataclass
class FittedModel:
    name: str
    notes: str
    train_rows_used: int
    estimator: Pipeline | None


@dataclass
class EvaluationResult:
    name: str
    notes: str
    train_rows_used: int
    threshold: float
    validation_f1: float
    validation_pr_auc: float
    test_roc_auc: float
    test_pr_auc: float
    test_precision: float
    test_recall: float
    test_f1: float
    tn: int
    fp: int
    fn: int
    tp: int
    validation_scores: np.ndarray
    test_scores: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inside-Dubai full-grid evaluation for H3 grid-time risk.")
    parser.add_argument("--model-sample", type=Path, default=MODEL_SAMPLE_PATH)
    parser.add_argument("--positive-counts", type=Path, default=POSITIVE_COUNTS_PATH)
    parser.add_argument("--cell-scope", type=Path, default=CELL_SCOPE_PATH)
    parser.add_argument("--rf-train-cap", type=int, default=RF_TRAIN_CAP)
    parser.add_argument("--xgb-train-cap", type=int, default=XGB_TRAIN_CAP)
    parser.add_argument("--chunk-windows", type=int, default=CHUNK_WINDOWS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def validate_inputs(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"Missing required input files: {missing}")


def uae_weekend_flag(window_start: pd.Timestamp) -> int:
    weekday = int(window_start.weekday())
    if window_start < pd.Timestamp("2022-01-01"):
        return int(weekday in {4, 5})
    return int(weekday in {5, 6})


def window_features(window_indices: np.ndarray) -> pd.DataFrame:
    starts = START_WINDOW + pd.to_timedelta(window_indices * WINDOW_HOURS, unit="h")
    return pd.DataFrame(
        {
            "window_index": window_indices.astype(np.int32),
            "window_start": starts.astype(str),
            "hour_block": (starts.hour // WINDOW_HOURS).astype(np.int8),
            "day_of_week": pd.Categorical(starts.day_name()),
            "is_weekend": np.array([uae_weekend_flag(ts) for ts in starts], dtype=np.int8),
            "month": starts.month.astype(np.int8),
            "year": starts.year.astype(np.int16),
        }
    )


def load_inside_cells(path: Path) -> list[str]:
    scope = pd.read_csv(path, dtype={"h3_cell_res8": "string", "geo_scope": "string"})
    inside = sorted(scope.loc[scope["geo_scope"] == "inside_dubai", "h3_cell_res8"].astype(str).tolist())
    if len(inside) != EXPECTED_INSIDE_DUBAI_CELLS:
        raise SystemExit(f"Expected {EXPECTED_INSIDE_DUBAI_CELLS} inside-Dubai cells, got {len(inside)}")
    if len(set(inside)) != len(inside):
        raise SystemExit("Inside-Dubai cell list contains duplicates")
    return inside


def load_inside_positive_counts(path: Path, inside_cells: list[str]) -> pd.DataFrame:
    usecols = [
        "h3_cell_res8",
        "window_start",
        "window_index",
        "geo_scope",
        "incident_count",
        "severity_weight_sum",
    ]
    dtype = {
        "h3_cell_res8": "string",
        "window_start": "string",
        "window_index": "int32",
        "geo_scope": "category",
        "incident_count": "int16",
        "severity_weight_sum": "int16",
    }
    positive = pd.read_csv(path, usecols=usecols, dtype=dtype)
    positive = positive[positive["geo_scope"].astype(str) == "inside_dubai"].copy()
    inside_set = set(inside_cells)
    unknown_cells = set(positive["h3_cell_res8"].astype(str)).difference(inside_set)
    if unknown_cells:
        raise SystemExit(f"Positive inside-Dubai rows contain cells outside the inside universe: {len(unknown_cells)}")
    if positive.duplicated(["h3_cell_res8", "window_index"]).any():
        raise SystemExit("Positive count table has duplicate inside-Dubai cell/window rows")
    if int(positive["window_index"].max()) >= WINDOW_COUNT:
        raise SystemExit("Positive count table contains window index beyond configured window count")
    return positive


def build_dense_matrices(
    positive: pd.DataFrame, inside_cells: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cell_to_index = {cell: idx for idx, cell in enumerate(inside_cells)}
    labels = np.zeros((len(inside_cells), WINDOW_COUNT), dtype=np.uint8)
    incident_counts = np.zeros((len(inside_cells), WINDOW_COUNT), dtype=np.int16)
    severity_weights = np.zeros((len(inside_cells), WINDOW_COUNT), dtype=np.int16)
    cell_idx = positive["h3_cell_res8"].astype(str).map(cell_to_index).to_numpy(dtype=np.int32)
    window_idx = positive["window_index"].to_numpy(dtype=np.int32)
    incident_values = positive["incident_count"].to_numpy(dtype=np.int16)
    severity_values = positive["severity_weight_sum"].to_numpy(dtype=np.int16)
    labels[cell_idx, window_idx] = 1
    incident_counts[cell_idx, window_idx] = incident_values
    severity_weights[cell_idx, window_idx] = severity_values
    return labels, incident_counts, severity_weights


def historical_priors(labels: np.ndarray) -> dict[str, np.ndarray | float]:
    train_labels = labels[:, :TRAIN_END_EXCLUSIVE]
    global_risk = float(train_labels.mean())
    cell_risk = train_labels.mean(axis=1).astype(np.float32)
    train_windows = np.arange(TRAIN_END_EXCLUSIVE, dtype=np.int32)
    train_window_frame = window_features(train_windows)
    hour_blocks = train_window_frame["hour_block"].to_numpy(dtype=np.int8)
    hour_risk = np.zeros(8, dtype=np.float32)
    cell_hour_risk = np.zeros((labels.shape[0], 8), dtype=np.float32)
    for hour in range(8):
        mask = hour_blocks == hour
        hour_slice = train_labels[:, mask]
        hour_risk[hour] = np.float32(hour_slice.mean())
        cell_hour_risk[:, hour] = hour_slice.mean(axis=1).astype(np.float32)
    return {
        "global": global_risk,
        "cell": cell_risk,
        "hour": hour_risk,
        "cell_hour": cell_hour_risk,
    }


def lag_matrix(cumulative: np.ndarray, window_indices: np.ndarray, length: int) -> np.ndarray:
    start = np.maximum(window_indices - length, 0)
    end = window_indices
    return (cumulative[:, end] - cumulative[:, start]).astype(np.int16)


def make_candidate_chunk(
    inside_cells: list[str],
    window_indices: np.ndarray,
    labels: np.ndarray,
    incident_counts: np.ndarray,
    count_cumulative: np.ndarray,
    severity_cumulative: np.ndarray,
    priors: dict[str, np.ndarray | float],
) -> CandidateChunk:
    n_cells = len(inside_cells)
    n_windows = len(window_indices)
    repeated_windows = np.repeat(window_indices, n_cells)
    tiled_cells = np.tile(np.asarray(inside_cells, dtype=object), n_windows)
    tiled_cell_index = np.tile(np.arange(n_cells, dtype=np.int32), n_windows)

    wf = window_features(window_indices)
    repeated_hour = np.repeat(wf["hour_block"].to_numpy(dtype=np.int8), n_cells)

    hist_cell = np.asarray(priors["cell"], dtype=np.float32)[tiled_cell_index]
    hist_hour = np.asarray(priors["hour"], dtype=np.float32)[repeated_hour]
    hist_cell_hour = np.asarray(priors["cell_hour"], dtype=np.float32)[tiled_cell_index, repeated_hour]
    hist_global = np.full(n_cells * n_windows, float(priors["global"]), dtype=np.float32)

    frame = pd.DataFrame(
        {
            "h3_cell_res8": tiled_cells,
            "window_index": repeated_windows.astype(np.int32),
            "window_start": np.repeat(wf["window_start"].astype(str).to_numpy(), n_cells),
            "geo_scope": pd.Categorical(["inside_dubai"] * (n_cells * n_windows)),
            "hour_block": repeated_hour,
            "day_of_week": pd.Categorical(np.repeat(wf["day_of_week"].astype(str).to_numpy(), n_cells)),
            "is_weekend": np.repeat(wf["is_weekend"].to_numpy(dtype=np.int8), n_cells),
            "month": np.repeat(wf["month"].to_numpy(dtype=np.int8), n_cells),
            "year": np.repeat(wf["year"].to_numpy(dtype=np.int16), n_cells),
            "prev_3h_incident_count": lag_matrix(count_cumulative, window_indices, 1).T.ravel(),
            "prev_24h_incident_count": lag_matrix(count_cumulative, window_indices, 8).T.ravel(),
            "prev_7d_incident_count": lag_matrix(count_cumulative, window_indices, 56).T.ravel(),
            "prev_24h_severity_weight_sum": lag_matrix(severity_cumulative, window_indices, 8).T.ravel(),
            "prev_7d_severity_weight_sum": lag_matrix(severity_cumulative, window_indices, 56).T.ravel(),
            "hist_cell_hour_risk": hist_cell_hour,
            "hist_cell_risk": hist_cell,
            "hist_hour_risk": hist_hour,
            "hist_global_risk": hist_global,
        }
    )
    y = labels[:, window_indices].T.ravel().astype(np.uint8)
    counts = incident_counts[:, window_indices].T.ravel().astype(np.int16)
    return CandidateChunk(frame=frame, labels=y, incident_counts=counts)


def load_training_sample(path: Path, inside_cells: list[str], priors: dict[str, np.ndarray | float]) -> pd.DataFrame:
    usecols = [
        "h3_cell_res8",
        "window_index",
        "geo_scope",
        "risk_label",
        "hour_block",
        "day_of_week",
        "is_weekend",
        "month",
        "year",
        "prev_3h_incident_count",
        "prev_24h_incident_count",
        "prev_7d_incident_count",
        "prev_24h_severity_weight_sum",
        "prev_7d_severity_weight_sum",
    ]
    dtype = {
        "h3_cell_res8": "string",
        "window_index": "int32",
        "geo_scope": "category",
        "risk_label": "int8",
        "hour_block": "int8",
        "day_of_week": "category",
        "is_weekend": "int8",
        "month": "int8",
        "year": "int16",
        "prev_3h_incident_count": "int16",
        "prev_24h_incident_count": "int16",
        "prev_7d_incident_count": "int16",
        "prev_24h_severity_weight_sum": "int16",
        "prev_7d_severity_weight_sum": "int16",
    }
    train = pd.read_csv(path, usecols=usecols, dtype=dtype)
    train = train[(train["geo_scope"].astype(str) == "inside_dubai") & (train["window_index"] < TRAIN_END_EXCLUSIVE)].copy()
    if train.empty:
        raise SystemExit("No inside-Dubai training rows found in sampled model table")

    cell_to_index = {cell: idx for idx, cell in enumerate(inside_cells)}
    cell_index = train["h3_cell_res8"].astype(str).map(cell_to_index)
    if cell_index.isna().any():
        raise SystemExit("Training sample contains inside-Dubai cells missing from the inside cell universe")
    cell_index = cell_index.to_numpy(dtype=np.int32)
    hour = train["hour_block"].to_numpy(dtype=np.int8)
    train["hist_cell_hour_risk"] = np.asarray(priors["cell_hour"], dtype=np.float32)[cell_index, hour]
    train["hist_cell_risk"] = np.asarray(priors["cell"], dtype=np.float32)[cell_index]
    train["hist_hour_risk"] = np.asarray(priors["hour"], dtype=np.float32)[hour]
    train["hist_global_risk"] = np.float32(float(priors["global"]))
    train["geo_scope"] = pd.Categorical(["inside_dubai"] * len(train))

    missing = sorted(set(FEATURE_COLUMNS).difference(train.columns))
    if missing:
        raise SystemExit(f"Training sample is missing feature columns: {missing}")
    return train


def sample_training_rows(y: pd.Series, cap: int, seed: int) -> np.ndarray:
    if cap <= 0 or len(y) <= cap:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    y_array = y.to_numpy()
    positives = np.flatnonzero(y_array == 1)
    negatives = np.flatnonzero(y_array == 0)
    pos_target = min(len(positives), max(1, int(cap * len(positives) / len(y))))
    neg_target = cap - pos_target
    pos_sample = rng.choice(positives, size=pos_target, replace=False)
    neg_sample = rng.choice(negatives, size=neg_target, replace=False)
    selected = np.concatenate([pos_sample, neg_sample])
    rng.shuffle(selected)
    return selected


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    numeric_step = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("num", numeric_step, NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def fit_pipeline(
    estimator: object,
    train_df: pd.DataFrame,
    train_y: pd.Series,
    scale_numeric: bool,
    sample_indices: np.ndarray | None = None,
) -> tuple[Pipeline, int]:
    if sample_indices is None:
        sample_indices = np.arange(len(train_df))
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(scale_numeric=scale_numeric)),
            ("model", estimator),
        ]
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        pipeline.fit(train_df.iloc[sample_indices][FEATURE_COLUMNS], train_y.iloc[sample_indices])
    return pipeline, len(sample_indices)


def train_models(train_df: pd.DataFrame, seed: int, rf_cap: int, xgb_cap: int) -> tuple[list[FittedModel], list[dict[str, str]]]:
    y_train = train_df["risk_label"].astype(int)
    models: list[FittedModel] = [
        FittedModel(
            name="Historical risk",
            notes="Train-period full-grid empirical risk by inside-Dubai H3 cell and hour block.",
            train_rows_used=len(train_df),
            estimator=None,
        )
    ]
    skipped: list[dict[str, str]] = []

    logistic = LogisticRegression(max_iter=300, class_weight="balanced", solver="lbfgs", random_state=seed)
    estimator, rows_used = fit_pipeline(logistic, train_df, y_train, scale_numeric=True)
    models.append(
        FittedModel(
            name="Logistic Regression",
            notes="Sampled inside-Dubai training rows with class_weight=balanced.",
            train_rows_used=rows_used,
            estimator=estimator,
        )
    )

    rf_indices = sample_training_rows(y_train, rf_cap, seed)
    random_forest = RandomForestClassifier(
        n_estimators=120,
        max_depth=18,
        min_samples_leaf=20,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )
    estimator, rows_used = fit_pipeline(random_forest, train_df, y_train, scale_numeric=False, sample_indices=rf_indices)
    models.append(
        FittedModel(
            name="Random Forest",
            notes=f"Sampled inside-Dubai training rows with deterministic stratified cap of {rows_used:,} rows.",
            train_rows_used=rows_used,
            estimator=estimator,
        )
    )

    if XGBClassifier is None:
        skipped.append({"model": "XGBoost", "reason": XGB_IMPORT_ERROR})
    else:
        xgb_indices = sample_training_rows(y_train, xgb_cap, seed + 1)
        y_xgb = y_train.iloc[xgb_indices]
        pos = int(y_xgb.sum())
        neg = int(len(y_xgb) - pos)
        xgb = XGBClassifier(
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
        estimator, rows_used = fit_pipeline(xgb, train_df, y_train, scale_numeric=False, sample_indices=xgb_indices)
        models.append(
            FittedModel(
                name="XGBoost",
                notes=f"Sampled inside-Dubai training rows, CPU histogram trees, deterministic cap of {rows_used:,} rows.",
                train_rows_used=rows_used,
                estimator=estimator,
            )
        )
    return models, skipped


def score_split(
    split_name: str,
    window_indices: np.ndarray,
    inside_cells: list[str],
    labels: np.ndarray,
    incident_counts: np.ndarray,
    count_cumulative: np.ndarray,
    severity_cumulative: np.ndarray,
    priors: dict[str, np.ndarray | float],
    models: list[FittedModel],
    chunk_windows: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    total_candidates = len(window_indices) * len(inside_cells)
    y_parts: list[np.ndarray] = []
    incident_parts: list[np.ndarray] = []
    score_parts: dict[str, list[np.ndarray]] = {model.name: [] for model in models}
    for start in range(0, len(window_indices), chunk_windows):
        windows = window_indices[start : start + chunk_windows]
        chunk = make_candidate_chunk(
            inside_cells,
            windows,
            labels,
            incident_counts,
            count_cumulative,
            severity_cumulative,
            priors,
        )
        y_parts.append(chunk.labels)
        incident_parts.append(chunk.incident_counts)
        for model in models:
            if model.estimator is None:
                scores = chunk.frame["hist_cell_hour_risk"].to_numpy(dtype=np.float32)
            else:
                scores = model.estimator.predict_proba(chunk.frame[FEATURE_COLUMNS])[:, 1].astype(np.float32)
            score_parts[model.name].append(scores)
    y = np.concatenate(y_parts).astype(np.uint8)
    incidents = np.concatenate(incident_parts).astype(np.int16)
    scores_by_model = {name: np.concatenate(parts).astype(np.float32) for name, parts in score_parts.items()}
    if len(y) != total_candidates:
        raise SystemExit(f"{split_name} candidate construction produced {len(y)} rows, expected {total_candidates}")
    return y, incidents, scores_by_model


def threshold_by_validation_f1(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5, 0.0
    f1_values = (2 * precision[:-1] * recall[:-1]) / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best_index = int(np.nanargmax(f1_values))
    return float(thresholds[best_index]), float(f1_values[best_index])


def evaluate_models(
    models: list[FittedModel],
    y_val: np.ndarray,
    val_scores_by_model: dict[str, np.ndarray],
    y_test: np.ndarray,
    test_scores_by_model: dict[str, np.ndarray],
) -> list[EvaluationResult]:
    results: list[EvaluationResult] = []
    for model in models:
        val_scores = val_scores_by_model[model.name]
        test_scores = test_scores_by_model[model.name]
        threshold, validation_f1 = threshold_by_validation_f1(y_val, val_scores)
        test_pred = (test_scores >= threshold).astype(np.uint8)
        tn, fp, fn, tp = confusion_matrix(y_test, test_pred, labels=[0, 1]).ravel()
        results.append(
            EvaluationResult(
                name=model.name,
                notes=model.notes,
                train_rows_used=model.train_rows_used,
                threshold=threshold,
                validation_f1=validation_f1,
                validation_pr_auc=float(average_precision_score(y_val, val_scores)),
                test_roc_auc=float(roc_auc_score(y_test, test_scores)),
                test_pr_auc=float(average_precision_score(y_test, test_scores)),
                test_precision=float(precision_score(y_test, test_pred, zero_division=0)),
                test_recall=float(recall_score(y_test, test_pred, zero_division=0)),
                test_f1=float(f1_score(y_test, test_pred, zero_division=0)),
                tn=int(tn),
                fp=int(fp),
                fn=int(fn),
                tp=int(tp),
                validation_scores=val_scores,
                test_scores=test_scores,
            )
        )
    return results


def results_to_frames(results: list[EvaluationResult], validation_rows: int, test_rows: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_rows = []
    confusion_rows = []
    for result in results:
        metrics_rows.append(
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
        confusion_rows.extend(
            [
                {"model": result.name, "actual": 0, "predicted": 0, "count": result.tn},
                {"model": result.name, "actual": 0, "predicted": 1, "count": result.fp},
                {"model": result.name, "actual": 1, "predicted": 0, "count": result.fn},
                {"model": result.name, "actual": 1, "predicted": 1, "count": result.tp},
            ]
        )
    return pd.DataFrame(metrics_rows), pd.DataFrame(confusion_rows)


def full_grid_top_k_metrics(
    results: list[EvaluationResult],
    y_test: np.ndarray,
    incident_counts: np.ndarray,
    n_test_windows: int,
    n_cells: int,
) -> pd.DataFrame:
    y_matrix = y_test.reshape(n_test_windows, n_cells)
    incident_matrix = incident_counts.reshape(n_test_windows, n_cells)
    positive_by_window = y_matrix.sum(axis=1)
    incident_by_window = incident_matrix.sum(axis=1)
    positive_window_mask = positive_by_window > 0
    total_positive_cell_windows = int(y_matrix.sum())
    total_incidents = int(incident_matrix.sum())
    rows = []
    for result in results:
        score_matrix = result.test_scores.reshape(n_test_windows, n_cells)
        for k in TOP_K_VALUES:
            top_indices = np.argpartition(score_matrix, -k, axis=1)[:, -k:]
            captured_labels = np.take_along_axis(y_matrix, top_indices, axis=1)
            captured_incidents = np.take_along_axis(incident_matrix, top_indices, axis=1)
            captured_by_window = captured_labels.sum(axis=1)
            incident_captured_by_window = captured_incidents.sum(axis=1)
            positive_captured = int(captured_by_window.sum())
            incidents_captured = int(incident_captured_by_window.sum())
            window_recall = np.divide(
                captured_by_window[positive_window_mask],
                positive_by_window[positive_window_mask],
                out=np.zeros(int(positive_window_mask.sum()), dtype=float),
                where=positive_by_window[positive_window_mask] > 0,
            )
            rows.append(
                {
                    "model": result.name,
                    "k": k,
                    "evaluated_windows": n_test_windows,
                    "windows_with_positives": int(positive_window_mask.sum()),
                    "total_positive_cell_windows": total_positive_cell_windows,
                    "positive_cell_windows_captured": positive_captured,
                    "recall_at_k": positive_captured / max(total_positive_cell_windows, 1),
                    "mean_window_recall_at_k": float(window_recall.mean()),
                    "precision_at_k": positive_captured / max(n_test_windows * k, 1),
                    "positive_window_hit_rate_at_k": float((captured_by_window[positive_window_mask] > 0).mean()),
                    "total_incidents": total_incidents,
                    "incidents_captured": incidents_captured,
                    "incident_recall_at_k": incidents_captured / max(total_incidents, 1),
                }
            )
    return pd.DataFrame(rows)


def split_summary(
    labels: np.ndarray,
    train_rows_used: int,
    validation_rows: int,
    test_rows: int,
    inside_cells: int,
) -> pd.DataFrame:
    split_defs = [
        ("train", 0, TRAIN_END_EXCLUSIVE - 1, train_rows_used),
        ("validation", TRAIN_END_EXCLUSIVE, VALIDATION_END_EXCLUSIVE - 1, 0),
        ("test", VALIDATION_END_EXCLUSIVE, WINDOW_COUNT - 1, 0),
    ]
    rows = []
    for split, start, end, sampled_rows in split_defs:
        window_indices = np.arange(start, end + 1)
        candidate_rows = len(window_indices) * inside_cells
        positive_cell_windows = int(labels[:, window_indices].sum())
        rows.append(
            {
                "split": split,
                "window_start_index": start,
                "window_end_index": end,
                "windows": len(window_indices),
                "inside_dubai_cells": inside_cells,
                "full_grid_candidate_rows": candidate_rows,
                "positive_cell_windows": positive_cell_windows,
                "negative_cell_windows": candidate_rows - positive_cell_windows,
                "positive_rate": positive_cell_windows / candidate_rows,
                "sampled_training_rows_used": sampled_rows if split == "train" else "",
                "min_window_start": str(START_WINDOW + pd.Timedelta(hours=WINDOW_HOURS * start)),
                "max_window_start": str(START_WINDOW + pd.Timedelta(hours=WINDOW_HOURS * end)),
            }
        )
    return pd.DataFrame(rows)


def plot_curves(y_test: np.ndarray, results: list[EvaluationResult]) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 6))
    for result in results:
        fpr, tpr, _ = roc_curve(y_test, result.test_scores)
        plt.plot(fpr, tpr, label=f"{result.name} ({result.test_roc_auc:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], color="0.5", linestyle="--", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Inside-Dubai full-grid ROC curves")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "final_full_grid_roc_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))
    for result in results:
        precision, recall, _ = precision_recall_curve(y_test, result.test_scores)
        plt.plot(recall, precision, label=f"{result.name} ({result.test_pr_auc:.3f})", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Inside-Dubai full-grid precision-recall curves")
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "final_full_grid_pr_curve.png", dpi=200)
    plt.close()


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
    plt.title("Inside-Dubai full-grid model comparison")
    plt.xticks(rotation=15, ha="right")
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "final_full_grid_model_comparison.png", dpi=200)
    plt.close()


def plot_top_k(top_k: pd.DataFrame) -> None:
    plot_df = top_k[top_k["k"].isin(TOP_K_VALUES)].copy()
    plt.figure(figsize=(9, 5.5))
    sns.lineplot(data=plot_df, x="k", y="recall_at_k", hue="model", marker="o", linewidth=2)
    plt.xlabel("Top-k cells per 3-hour window")
    plt.ylabel("Cell-window recall")
    plt.title("Inside-Dubai full-grid hotspot recall")
    plt.ylim(0, max(0.05, float(plot_df["recall_at_k"].max()) * 1.15))
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "final_full_grid_topk_comparison.png", dpi=200)
    plt.close()


def frame_to_markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for record in frame.astype(str).to_dict(orient="records"):
        rows.append("| " + " | ".join(record[column] for column in columns) + " |")
    return "\n".join(rows)


def write_audit(
    inside_cell_count: int,
    positive: pd.DataFrame,
    train_df: pd.DataFrame,
    split_frame: pd.DataFrame,
    metrics: pd.DataFrame,
    top_k: pd.DataFrame,
    skipped_models: list[dict[str, str]],
) -> None:
    audit = [
        "# Inside-Dubai full-grid evaluation audit",
        "",
        f"- H3 resolution: `{H3_RESOLUTION}`",
        f"- Time window hours: `{WINDOW_HOURS}`",
        f"- Inside-Dubai H3 cells: `{inside_cell_count}`",
        f"- Training source: sampled inside-Dubai model table rows from windows `0` to `{TRAIN_END_EXCLUSIVE - 1}`",
        f"- Validation candidate windows: `{TRAIN_END_EXCLUSIVE}` to `{VALIDATION_END_EXCLUSIVE - 1}`",
        f"- Test candidate windows: `{VALIDATION_END_EXCLUSIVE}` to `{WINDOW_COUNT - 1}`",
        f"- Inside-Dubai positive grid-time rows: `{len(positive)}`",
        f"- Inside-Dubai training rows used before model caps: `{len(train_df)}`",
        f"- Historical priors: training-period inside-Dubai full-grid denominators",
        f"- Leakage columns excluded from model features: `{', '.join(sorted(LEAKAGE_COLUMNS))}`",
        f"- Feature columns used: `{', '.join(FEATURE_COLUMNS)}`",
        "",
        "## Split summary",
        "",
        frame_to_markdown(split_frame),
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
        "## Full-grid top-k hotspot metrics",
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
    ]
    if skipped_models:
        audit.extend(["", "## Skipped models", ""])
        for skipped in skipped_models:
            audit.append(f"- {skipped['model']}: {skipped['reason']}")
    (AUDIT_DIR / "final_full_grid_evaluation_audit.md").write_text("\n".join(audit) + "\n", encoding="utf-8")


def validate_no_leakage() -> None:
    overlap = LEAKAGE_COLUMNS.intersection(FEATURE_COLUMNS)
    if overlap:
        raise SystemExit(f"Leakage columns present in model features: {sorted(overlap)}")


def main() -> None:
    args = parse_args()
    ensure_dirs()
    validate_inputs([args.model_sample, args.positive_counts, args.cell_scope])
    validate_no_leakage()

    inside_cells = load_inside_cells(args.cell_scope)
    positive = load_inside_positive_counts(args.positive_counts, inside_cells)
    labels, incident_counts, severity_weights = build_dense_matrices(positive, inside_cells)
    priors = historical_priors(labels)
    count_cumulative = np.concatenate(
        [np.zeros((labels.shape[0], 1), dtype=np.int32), np.cumsum(incident_counts.astype(np.int32), axis=1)],
        axis=1,
    )
    severity_cumulative = np.concatenate(
        [np.zeros((labels.shape[0], 1), dtype=np.int32), np.cumsum(severity_weights.astype(np.int32), axis=1)],
        axis=1,
    )

    train_df = load_training_sample(args.model_sample, inside_cells, priors)
    models, skipped_models = train_models(train_df, args.seed, args.rf_train_cap, args.xgb_train_cap)

    validation_windows = np.arange(TRAIN_END_EXCLUSIVE, VALIDATION_END_EXCLUSIVE, dtype=np.int32)
    test_windows = np.arange(VALIDATION_END_EXCLUSIVE, WINDOW_COUNT, dtype=np.int32)
    validation_candidates = len(validation_windows) * len(inside_cells)
    test_candidates = len(test_windows) * len(inside_cells)
    if validation_candidates != EXPECTED_VALIDATION_CANDIDATES:
        raise SystemExit(f"Expected {EXPECTED_VALIDATION_CANDIDATES} validation candidates, got {validation_candidates}")
    if test_candidates != EXPECTED_TEST_CANDIDATES:
        raise SystemExit(f"Expected {EXPECTED_TEST_CANDIDATES} test candidates, got {test_candidates}")

    y_val, val_incidents, val_scores_by_model = score_split(
        "validation",
        validation_windows,
        inside_cells,
        labels,
        incident_counts,
        count_cumulative,
        severity_cumulative,
        priors,
        models,
        args.chunk_windows,
    )
    y_test, test_incidents, test_scores_by_model = score_split(
        "test",
        test_windows,
        inside_cells,
        labels,
        incident_counts,
        count_cumulative,
        severity_cumulative,
        priors,
        models,
        args.chunk_windows,
    )
    if int(y_test.sum()) != EXPECTED_TEST_POSITIVE_CELL_WINDOWS:
        raise SystemExit(f"Expected {EXPECTED_TEST_POSITIVE_CELL_WINDOWS} test positives, got {int(y_test.sum())}")

    results = evaluate_models(models, y_val, val_scores_by_model, y_test, test_scores_by_model)
    metrics, confusion = results_to_frames(results, len(y_val), len(y_test))
    top_k = full_grid_top_k_metrics(results, y_test, test_incidents, len(test_windows), len(inside_cells))
    split_frame = split_summary(labels, len(train_df), len(y_val), len(y_test), len(inside_cells))

    metrics.to_csv(TABLE_DIR / "final_full_grid_metrics.csv", index=False)
    confusion.to_csv(TABLE_DIR / "final_full_grid_confusion_matrix.csv", index=False)
    top_k.to_csv(TABLE_DIR / "final_full_grid_topk_hotspot_metrics.csv", index=False)
    split_frame.to_csv(TABLE_DIR / "final_full_grid_split_summary.csv", index=False)
    (TABLE_DIR / "final_full_grid_skipped_models.json").write_text(
        json.dumps(skipped_models, indent=2), encoding="utf-8"
    )

    plot_curves(y_test, results)
    plot_model_comparison(metrics)
    plot_top_k(top_k)
    write_audit(len(inside_cells), positive, train_df, split_frame, metrics, top_k, skipped_models)

    print(f"Wrote metrics to {TABLE_DIR / 'final_full_grid_metrics.csv'}")
    print(f"Wrote figures to {FIGURE_DIR}")
    print(f"Wrote audit to {AUDIT_DIR / 'final_full_grid_evaluation_audit.md'}")


if __name__ == "__main__":
    main()
