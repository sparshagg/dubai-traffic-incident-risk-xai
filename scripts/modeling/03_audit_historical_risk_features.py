from __future__ import annotations

import importlib.util
import os
import sys
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
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
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

HISTORICAL_FEATURES = [
    "hist_cell_hour_risk",
    "hist_cell_risk",
    "hist_hour_risk",
    "hist_global_risk",
]
TEMPORAL_NUMERIC_FEATURES = ["hour_block", "is_weekend", "month", "year"]
LAG_FEATURES = [
    "prev_3h_incident_count",
    "prev_24h_incident_count",
    "prev_7d_incident_count",
    "prev_24h_severity_weight_sum",
    "prev_7d_severity_weight_sum",
]
DAY_FEATURE = ["day_of_week"]
SCOPE_FEATURE = ["geo_scope"]
LEAKAGE_COLUMNS = fg.LEAKAGE_COLUMNS


@dataclass(frozen=True)
class FeatureSet:
    feature_set_id: str
    model_type: str
    description: str
    numeric_features: list[str]
    categorical_features: list[str]
    train_history_mode: str

    @property
    def features(self) -> list[str]:
        return self.numeric_features + self.categorical_features

    @property
    def has_historical_features(self) -> bool:
        return bool(set(self.features).intersection(HISTORICAL_FEATURES))


@dataclass
class AuditResult:
    feature_set_id: str
    model_type: str
    description: str
    train_history_mode: str
    feature_count: int
    has_historical_features: bool
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


FEATURE_SETS = [
    FeatureSet(
        feature_set_id="all_features_train_period_hist",
        model_type="XGBoost",
        description="All current features with train-period full-grid historical-risk priors.",
        numeric_features=TEMPORAL_NUMERIC_FEATURES + LAG_FEATURES + HISTORICAL_FEATURES,
        categorical_features=DAY_FEATURE + SCOPE_FEATURE,
        train_history_mode="train_period",
    ),
    FeatureSet(
        feature_set_id="all_features_expanding_train_hist",
        model_type="XGBoost",
        description="All current features, but training-row historical-risk priors use only earlier training windows.",
        numeric_features=TEMPORAL_NUMERIC_FEATURES + LAG_FEATURES + HISTORICAL_FEATURES,
        categorical_features=DAY_FEATURE + SCOPE_FEATURE,
        train_history_mode="expanding_train_only",
    ),
    FeatureSet(
        feature_set_id="no_historical_risk",
        model_type="XGBoost",
        description="Temporal and lag features only, with all historical-risk prior columns removed.",
        numeric_features=TEMPORAL_NUMERIC_FEATURES + LAG_FEATURES,
        categorical_features=DAY_FEATURE + SCOPE_FEATURE,
        train_history_mode="not_used",
    ),
    FeatureSet(
        feature_set_id="temporal_lag_only",
        model_type="XGBoost",
        description="Temporal and lag features only, excluding both historical-risk priors and constant geo scope.",
        numeric_features=TEMPORAL_NUMERIC_FEATURES + LAG_FEATURES,
        categorical_features=DAY_FEATURE,
        train_history_mode="not_used",
    ),
    FeatureSet(
        feature_set_id="temporal_only",
        model_type="XGBoost",
        description="Temporal features only.",
        numeric_features=TEMPORAL_NUMERIC_FEATURES,
        categorical_features=DAY_FEATURE,
        train_history_mode="not_used",
    ),
    FeatureSet(
        feature_set_id="historical_score_only",
        model_type="Historical score",
        description="Train-period full-grid historical cell/hour score without a learned model.",
        numeric_features=["hist_cell_hour_risk"],
        categorical_features=[],
        train_history_mode="train_period",
    ),
]


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def validate_feature_sets() -> None:
    for feature_set in FEATURE_SETS:
        overlap = set(feature_set.features).intersection(LEAKAGE_COLUMNS)
        if overlap:
            raise SystemExit(f"{feature_set.feature_set_id} includes leakage columns: {sorted(overlap)}")
    for feature_set_id in ["no_historical_risk", "temporal_lag_only", "temporal_only"]:
        feature_set = next(item for item in FEATURE_SETS if item.feature_set_id == feature_set_id)
        if feature_set.has_historical_features:
            raise SystemExit(f"{feature_set_id} unexpectedly includes historical-risk features")


def safe_divide(numerator: np.ndarray, denominator: np.ndarray | float) -> np.ndarray:
    numerator = numerator.astype(np.float32)
    denominator_array = np.asarray(denominator, dtype=np.float32)
    return np.divide(
        numerator,
        denominator_array,
        out=np.zeros_like(numerator, dtype=np.float32),
        where=denominator_array > 0,
    ).astype(np.float32)


def add_expanding_training_history(
    train_df: pd.DataFrame,
    inside_cells: list[str],
    labels: np.ndarray,
    count_cumulative: np.ndarray,
) -> pd.DataFrame:
    train = train_df.copy()
    cell_to_index = {cell: idx for idx, cell in enumerate(inside_cells)}
    cell_index = train["h3_cell_res8"].astype(str).map(cell_to_index)
    if cell_index.isna().any():
        raise SystemExit("Training rows contain cells outside the inside-Dubai universe")
    cell_index_array = cell_index.to_numpy(dtype=np.int32)
    window_index = train["window_index"].to_numpy(dtype=np.int32)
    hour_block = train["hour_block"].to_numpy(dtype=np.int8)
    n_cells = len(inside_cells)

    cell_counts = count_cumulative[cell_index_array, window_index]
    train["hist_cell_risk"] = safe_divide(cell_counts, window_index)

    positives_by_window = labels.sum(axis=0).astype(np.int32)
    cumulative_positive_windows = np.concatenate([[0], np.cumsum(positives_by_window, dtype=np.int64)])
    global_counts = cumulative_positive_windows[window_index]
    train["hist_global_risk"] = safe_divide(global_counts, n_cells * window_index)

    window_features = fg.window_features(np.arange(fg.WINDOW_COUNT, dtype=np.int32))
    all_hour_blocks = window_features["hour_block"].to_numpy(dtype=np.int8)
    hist_hour = np.zeros(len(train), dtype=np.float32)
    hist_cell_hour = np.zeros(len(train), dtype=np.float32)
    hour_denominator_by_window = {}

    for hour in range(8):
        hour_windows = all_hour_blocks == hour
        prior_hour_window_counts = np.concatenate([[0], np.cumsum(hour_windows.astype(np.int32))])
        hour_denominator_by_window[hour] = prior_hour_window_counts
        rows_for_hour = np.flatnonzero(hour_block == hour)
        if len(rows_for_hour) == 0:
            continue

        masked_labels = np.where(hour_windows, labels, 0).astype(np.int16)
        cumulative_cell_hour = np.concatenate(
            [
                np.zeros((labels.shape[0], 1), dtype=np.int32),
                np.cumsum(masked_labels, axis=1, dtype=np.int32),
            ],
            axis=1,
        )
        row_windows = window_index[rows_for_hour]
        row_cells = cell_index_array[rows_for_hour]
        cell_hour_counts = cumulative_cell_hour[row_cells, row_windows]
        cell_hour_denoms = prior_hour_window_counts[row_windows]
        hist_cell_hour[rows_for_hour] = safe_divide(cell_hour_counts, cell_hour_denoms)

        total_hour_counts_by_window = np.concatenate([[0], np.cumsum((positives_by_window * hour_windows), dtype=np.int64)])
        hour_counts = total_hour_counts_by_window[row_windows]
        hist_hour[rows_for_hour] = safe_divide(hour_counts, n_cells * cell_hour_denoms)

    train["hist_hour_risk"] = hist_hour
    train["hist_cell_hour_risk"] = hist_cell_hour

    zero_cell_hour = train["hist_cell_hour_risk"] == 0
    train.loc[zero_cell_hour, "hist_cell_hour_risk"] = train.loc[zero_cell_hour, "hist_cell_risk"]
    zero_cell = train["hist_cell_hour_risk"] == 0
    train.loc[zero_cell, "hist_cell_hour_risk"] = train.loc[zero_cell, "hist_hour_risk"]
    zero_hour = train["hist_cell_hour_risk"] == 0
    train.loc[zero_hour, "hist_cell_hour_risk"] = train.loc[zero_hour, "hist_global_risk"]

    leakage_violations = int((window_index <= 0).sum() and (train.loc[window_index <= 0, HISTORICAL_FEATURES].to_numpy() != 0).sum())
    if leakage_violations:
        raise SystemExit(f"Expanding history has {leakage_violations} non-zero values at window 0")
    return train


def build_preprocessor(feature_set: FeatureSet) -> ColumnTransformer:
    transformers = []
    if feature_set.numeric_features:
        transformers.append(("num", "passthrough", feature_set.numeric_features))
    if feature_set.categorical_features:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), feature_set.categorical_features)
        )
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def train_xgboost(
    train_df: pd.DataFrame,
    feature_set: FeatureSet,
    sample_indices: np.ndarray,
    seed: int,
) -> tuple[Pipeline, int]:
    if fg.XGBClassifier is None:
        raise SystemExit(fg.XGB_IMPORT_ERROR)
    y_train = train_df["risk_label"].astype(int)
    y_sample = y_train.iloc[sample_indices]
    pos = int(y_sample.sum())
    neg = int(len(y_sample) - pos)
    xgb = fg.XGBClassifier(
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
            ("preprocess", build_preprocessor(feature_set)),
            ("model", xgb),
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.fit(train_df.iloc[sample_indices][feature_set.features], y_train.iloc[sample_indices])
    return pipeline, len(sample_indices)


def threshold_by_validation_f1(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5, 0.0
    f1_values = (2 * precision[:-1] * recall[:-1]) / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best_index = int(np.nanargmax(f1_values))
    return float(thresholds[best_index]), float(f1_values[best_index])


def score_models_on_full_grid(
    models: dict[str, Pipeline | None],
    feature_sets: dict[str, FeatureSet],
    window_indices: np.ndarray,
    inside_cells: list[str],
    labels: np.ndarray,
    incident_counts: np.ndarray,
    count_cumulative: np.ndarray,
    severity_cumulative: np.ndarray,
    priors: dict[str, np.ndarray | float],
    chunk_windows: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    y_parts: list[np.ndarray] = []
    incident_parts: list[np.ndarray] = []
    score_parts: dict[str, list[np.ndarray]] = {feature_set_id: [] for feature_set_id in models}
    expected_rows = len(window_indices) * len(inside_cells)

    for start in range(0, len(window_indices), chunk_windows):
        windows = window_indices[start : start + chunk_windows]
        chunk = fg.make_candidate_chunk(
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
        for feature_set_id, model in models.items():
            feature_set = feature_sets[feature_set_id]
            if model is None:
                scores = chunk.frame["hist_cell_hour_risk"].to_numpy(dtype=np.float32)
            else:
                scores = model.predict_proba(chunk.frame[feature_set.features])[:, 1].astype(np.float32)
            score_parts[feature_set_id].append(scores)

    y = np.concatenate(y_parts).astype(np.uint8)
    incidents = np.concatenate(incident_parts).astype(np.int16)
    if len(y) != expected_rows:
        raise SystemExit(f"Expected {expected_rows} full-grid rows, got {len(y)}")
    scores = {feature_set_id: np.concatenate(parts).astype(np.float32) for feature_set_id, parts in score_parts.items()}
    return y, incidents, scores


def evaluate_results(
    feature_sets: dict[str, FeatureSet],
    train_rows_used: dict[str, int],
    y_val: np.ndarray,
    val_scores: dict[str, np.ndarray],
    y_test: np.ndarray,
    test_scores: dict[str, np.ndarray],
) -> list[AuditResult]:
    results = []
    for feature_set_id, feature_set in feature_sets.items():
        threshold, validation_f1 = threshold_by_validation_f1(y_val, val_scores[feature_set_id])
        prediction = (test_scores[feature_set_id] >= threshold).astype(np.uint8)
        tn, fp, fn, tp = confusion_matrix(y_test, prediction, labels=[0, 1]).ravel()
        results.append(
            AuditResult(
                feature_set_id=feature_set_id,
                model_type=feature_set.model_type,
                description=feature_set.description,
                train_history_mode=feature_set.train_history_mode,
                feature_count=len(feature_set.features),
                has_historical_features=feature_set.has_historical_features,
                train_rows_used=train_rows_used.get(feature_set_id, 0),
                threshold=threshold,
                validation_f1=validation_f1,
                validation_pr_auc=float(average_precision_score(y_val, val_scores[feature_set_id])),
                test_roc_auc=float(roc_auc_score(y_test, test_scores[feature_set_id])),
                test_pr_auc=float(average_precision_score(y_test, test_scores[feature_set_id])),
                test_precision=float(precision_score(y_test, prediction, zero_division=0)),
                test_recall=float(recall_score(y_test, prediction, zero_division=0)),
                test_f1=float(f1_score(y_test, prediction, zero_division=0)),
                tn=int(tn),
                fp=int(fp),
                fn=int(fn),
                tp=int(tp),
                validation_scores=val_scores[feature_set_id],
                test_scores=test_scores[feature_set_id],
            )
        )
    return results


def metrics_frame(results: list[AuditResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature_set_id": result.feature_set_id,
                "model_type": result.model_type,
                "description": result.description,
                "train_history_mode": result.train_history_mode,
                "feature_count": result.feature_count,
                "has_historical_features": result.has_historical_features,
                "train_rows_used": result.train_rows_used,
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
            for result in results
        ]
    )


def feature_sets_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feature_set_id": feature_set.feature_set_id,
                "model_type": feature_set.model_type,
                "train_history_mode": feature_set.train_history_mode,
                "has_historical_features": feature_set.has_historical_features,
                "feature_count": len(feature_set.features),
                "features": "|".join(feature_set.features),
                "description": feature_set.description,
            }
            for feature_set in FEATURE_SETS
        ]
    )


def top_k_metrics(results: list[AuditResult], y_test: np.ndarray, incidents: np.ndarray, n_windows: int, n_cells: int) -> pd.DataFrame:
    y_matrix = y_test.reshape(n_windows, n_cells)
    incident_matrix = incidents.reshape(n_windows, n_cells)
    positive_by_window = y_matrix.sum(axis=1)
    positive_window_mask = positive_by_window > 0
    total_positive_cell_windows = int(y_matrix.sum())
    total_incidents = int(incident_matrix.sum())
    rows = []
    for result in results:
        score_matrix = result.test_scores.reshape(n_windows, n_cells)
        for k in fg.TOP_K_VALUES:
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
                    "feature_set_id": result.feature_set_id,
                    "model_type": result.model_type,
                    "k": k,
                    "evaluated_windows": n_windows,
                    "windows_with_positives": int(positive_window_mask.sum()),
                    "total_positive_cell_windows": total_positive_cell_windows,
                    "positive_cell_windows_captured": positive_captured,
                    "recall_at_k": positive_captured / max(total_positive_cell_windows, 1),
                    "mean_window_recall_at_k": float(window_recall.mean()),
                    "precision_at_k": positive_captured / max(n_windows * k, 1),
                    "positive_window_hit_rate_at_k": float((captured_by_window[positive_window_mask] > 0).mean()),
                    "total_incidents": total_incidents,
                    "incidents_captured": incidents_captured,
                    "incident_recall_at_k": incidents_captured / max(total_incidents, 1),
                }
            )
    return pd.DataFrame(rows)


def plot_metric_comparison(metrics: pd.DataFrame) -> None:
    plot_df = metrics.melt(
        id_vars=["feature_set_id"],
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
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(11, 6))
    sns.barplot(data=plot_df, x="feature_set_id", y="value", hue="metric")
    plt.ylim(0, 1)
    plt.xlabel("")
    plt.ylabel("Score")
    plt.title("Historical-risk audit feature-set comparison")
    plt.xticks(rotation=25, ha="right")
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "historical_risk_audit_model_comparison.png", dpi=200)
    plt.close()


def plot_top_k_comparison(top_k: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(11, 6))
    sns.lineplot(data=top_k, x="k", y="recall_at_k", hue="feature_set_id", marker="o", linewidth=2)
    plt.xlabel("Top-k cells per 3-hour window")
    plt.ylabel("Cell-window recall")
    plt.title("Historical-risk audit top-k hotspot recall")
    plt.ylim(0, max(0.05, float(top_k["recall_at_k"].max()) * 1.15))
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "historical_risk_audit_topk_comparison.png", dpi=200)
    plt.close()


def frame_to_markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for record in frame.astype(str).to_dict(orient="records"):
        rows.append("| " + " | ".join(record[column] for column in columns) + " |")
    return "\n".join(rows)


def write_audit(metrics: pd.DataFrame, top_k: pd.DataFrame, feature_sets: pd.DataFrame) -> None:
    no_hist = metrics.loc[metrics["feature_set_id"] == "no_historical_risk"].iloc[0]
    all_hist = metrics.loc[metrics["feature_set_id"] == "all_features_train_period_hist"].iloc[0]
    expanding = metrics.loc[metrics["feature_set_id"] == "all_features_expanding_train_hist"].iloc[0]
    lines = [
        "# Historical-risk feature audit",
        "",
        f"- H3 resolution: `{fg.H3_RESOLUTION}`",
        f"- Time window hours: `{fg.WINDOW_HOURS}`",
        f"- Inside-Dubai H3 cells: `{fg.EXPECTED_INSIDE_DUBAI_CELLS}`",
        f"- Validation candidate rows: `{fg.EXPECTED_VALIDATION_CANDIDATES}`",
        f"- Test candidate rows: `{fg.EXPECTED_TEST_CANDIDATES}`",
        f"- Test positive cell/windows: `{fg.EXPECTED_TEST_POSITIVE_CELL_WINDOWS}`",
        "- Model family audited: `XGBoost`, plus one historical-score-only baseline.",
        "- Validation/test historical-risk features use train-period full-grid denominators.",
        "- Expanding-history variant changes only training-row historical-risk features.",
        "",
        "## Main comparison",
        "",
        f"- Full feature PR-AUC: `{all_hist['test_pr_auc']:.6f}`; F1: `{all_hist['test_f1']:.6f}`.",
        f"- No-history PR-AUC: `{no_hist['test_pr_auc']:.6f}`; F1: `{no_hist['test_f1']:.6f}`.",
        f"- Expanding-history PR-AUC: `{expanding['test_pr_auc']:.6f}`; F1: `{expanding['test_f1']:.6f}`.",
        f"- PR-AUC drop after removing historical-risk features: `{all_hist['test_pr_auc'] - no_hist['test_pr_auc']:.6f}`.",
        f"- F1 drop after removing historical-risk features: `{all_hist['test_f1'] - no_hist['test_f1']:.6f}`.",
        f"- PR-AUC change with expanding training history: `{expanding['test_pr_auc'] - all_hist['test_pr_auc']:.6f}`.",
        f"- F1 change with expanding training history: `{expanding['test_f1'] - all_hist['test_f1']:.6f}`.",
        "",
        "## Feature sets",
        "",
        frame_to_markdown(feature_sets[["feature_set_id", "model_type", "train_history_mode", "has_historical_features", "feature_count"]]),
        "",
        "## Metrics",
        "",
        frame_to_markdown(
            metrics[
                [
                    "feature_set_id",
                    "test_roc_auc",
                    "test_pr_auc",
                    "test_precision",
                    "test_recall",
                    "test_f1",
                    "threshold",
                    "test_tn",
                    "test_fp",
                    "test_fn",
                    "test_tp",
                ]
            ].round(6)
        ),
        "",
        "## Top-k hotspot metrics",
        "",
        frame_to_markdown(
            top_k[
                [
                    "feature_set_id",
                    "k",
                    "recall_at_k",
                    "precision_at_k",
                    "positive_window_hit_rate_at_k",
                    "incident_recall_at_k",
                ]
            ].round(6)
        ),
    ]
    (AUDIT_DIR / "historical_risk_feature_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    validate_feature_sets()
    fg.validate_inputs([fg.MODEL_SAMPLE_PATH, fg.POSITIVE_COUNTS_PATH, fg.CELL_SCOPE_PATH])
    if fg.XGBClassifier is None:
        raise SystemExit(fg.XGB_IMPORT_ERROR)

    inside_cells = fg.load_inside_cells(fg.CELL_SCOPE_PATH)
    positive = fg.load_inside_positive_counts(fg.POSITIVE_COUNTS_PATH, inside_cells)
    labels, incident_counts, severity_weights = fg.build_dense_matrices(positive, inside_cells)
    priors = fg.historical_priors(labels)
    count_cumulative = np.concatenate(
        [np.zeros((labels.shape[0], 1), dtype=np.int32), np.cumsum(incident_counts.astype(np.int32), axis=1)],
        axis=1,
    )
    severity_cumulative = np.concatenate(
        [np.zeros((labels.shape[0], 1), dtype=np.int32), np.cumsum(severity_weights.astype(np.int32), axis=1)],
        axis=1,
    )

    train_period_df = fg.load_training_sample(fg.MODEL_SAMPLE_PATH, inside_cells, priors)
    expanding_df = add_expanding_training_history(train_period_df, inside_cells, labels, count_cumulative)
    y_train = train_period_df["risk_label"].astype(int)
    sample_indices = fg.sample_training_rows(y_train, fg.XGB_TRAIN_CAP, fg.RANDOM_SEED + 1)

    feature_set_by_id = {feature_set.feature_set_id: feature_set for feature_set in FEATURE_SETS}
    models: dict[str, Pipeline | None] = {}
    train_rows_used: dict[str, int] = {}
    for feature_set in FEATURE_SETS:
        if feature_set.model_type == "Historical score":
            models[feature_set.feature_set_id] = None
            train_rows_used[feature_set.feature_set_id] = len(train_period_df)
            continue
        train_df = expanding_df if feature_set.train_history_mode == "expanding_train_only" else train_period_df
        model, rows_used = train_xgboost(train_df, feature_set, sample_indices, fg.RANDOM_SEED)
        models[feature_set.feature_set_id] = model
        train_rows_used[feature_set.feature_set_id] = rows_used

    validation_windows = np.arange(fg.TRAIN_END_EXCLUSIVE, fg.VALIDATION_END_EXCLUSIVE, dtype=np.int32)
    test_windows = np.arange(fg.VALIDATION_END_EXCLUSIVE, fg.WINDOW_COUNT, dtype=np.int32)
    validation_candidates = len(validation_windows) * len(inside_cells)
    test_candidates = len(test_windows) * len(inside_cells)
    if validation_candidates != fg.EXPECTED_VALIDATION_CANDIDATES:
        raise SystemExit(f"Expected {fg.EXPECTED_VALIDATION_CANDIDATES} validation rows, got {validation_candidates}")
    if test_candidates != fg.EXPECTED_TEST_CANDIDATES:
        raise SystemExit(f"Expected {fg.EXPECTED_TEST_CANDIDATES} test rows, got {test_candidates}")

    y_val, val_incidents, val_scores = score_models_on_full_grid(
        models,
        feature_set_by_id,
        validation_windows,
        inside_cells,
        labels,
        incident_counts,
        count_cumulative,
        severity_cumulative,
        priors,
        fg.CHUNK_WINDOWS,
    )
    y_test, test_incidents, test_scores = score_models_on_full_grid(
        models,
        feature_set_by_id,
        test_windows,
        inside_cells,
        labels,
        incident_counts,
        count_cumulative,
        severity_cumulative,
        priors,
        fg.CHUNK_WINDOWS,
    )
    if int(y_test.sum()) != fg.EXPECTED_TEST_POSITIVE_CELL_WINDOWS:
        raise SystemExit(f"Expected {fg.EXPECTED_TEST_POSITIVE_CELL_WINDOWS} test positives, got {int(y_test.sum())}")

    results = evaluate_results(feature_set_by_id, train_rows_used, y_val, val_scores, y_test, test_scores)
    metrics = metrics_frame(results)
    feature_sets = feature_sets_frame()
    top_k = top_k_metrics(results, y_test, test_incidents, len(test_windows), len(inside_cells))

    metrics.to_csv(TABLE_DIR / "historical_risk_audit_metrics.csv", index=False)
    feature_sets.to_csv(TABLE_DIR / "historical_risk_audit_feature_sets.csv", index=False)
    top_k.to_csv(TABLE_DIR / "historical_risk_audit_topk_hotspot_metrics.csv", index=False)
    plot_metric_comparison(metrics)
    plot_top_k_comparison(top_k)
    write_audit(metrics, top_k, feature_sets)

    print(f"Wrote metrics to {TABLE_DIR / 'historical_risk_audit_metrics.csv'}")
    print(f"Wrote top-k metrics to {TABLE_DIR / 'historical_risk_audit_topk_hotspot_metrics.csv'}")
    print(f"Wrote audit to {AUDIT_DIR / 'historical_risk_feature_audit.md'}")


if __name__ == "__main__":
    main()
