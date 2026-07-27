from __future__ import annotations

import argparse
import csv
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
except Exception as exc:  # pragma: no cover - handled at runtime in the report.
    XGBClassifier = None
    if "libomp" in str(exc):
        XGB_IMPORT_ERROR = "xgboost could not load because libomp.dylib, the OpenMP runtime, is missing"
    else:
        XGB_IMPORT_ERROR = f"xgboost could not be imported: {exc}"


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data" / "processed" / "grid_time_model_sample_res8_3h.csv"
REPORT_DIR = ROOT / "reports" / "modeling"
TABLE_DIR = REPORT_DIR / "tables"
FIGURE_DIR = REPORT_DIR / "figures"
AUDIT_DIR = ROOT / "data" / "audit"

EXPECTED_ROWS = 4_069_044
INCLUDED_SCOPES = {"inside_dubai", "peripheral_observed"}
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

RANDOM_SEED = 42
RF_TRAIN_CAP = 500_000
XGB_TRAIN_CAP = 1_000_000
TOP_K_VALUES = [5, 10, 20]


@dataclass
class ModelResult:
    name: str
    status: str
    notes: str
    train_rows: int
    validation_rows: int
    test_rows: int
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
    parser = argparse.ArgumentParser(description="Train first baseline models for H3 grid-time incident risk.")
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--rf-train-cap", type=int, default=RF_TRAIN_CAP)
    parser.add_argument("--xgb-train-cap", type=int, default=XGB_TRAIN_CAP)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def load_model_sample(path: Path) -> pd.DataFrame:
    usecols = [
        "h3_cell_res8",
        "window_start",
        "window_index",
        "geo_scope",
        "risk_label",
        "incident_count",
        "severity_weight_sum",
        "minor_count",
        "moderate_count",
        "severe_count",
        "unknown_count",
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
        "window_start": "string",
        "window_index": "int32",
        "geo_scope": "category",
        "risk_label": "int8",
        "incident_count": "int16",
        "severity_weight_sum": "int16",
        "minor_count": "int16",
        "moderate_count": "int16",
        "severe_count": "int16",
        "unknown_count": "int16",
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
    df = pd.read_csv(path, usecols=usecols, dtype=dtype)
    if len(df) != EXPECTED_ROWS:
        raise SystemExit(f"Expected {EXPECTED_ROWS} model-sample rows, got {len(df)}")
    return df


def add_train_only_historical_features(df: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    train = df.loc[train_mask, ["h3_cell_res8", "hour_block", "risk_label"]].copy()
    global_rate = float(train["risk_label"].mean())

    train["cell_hour_key"] = train["h3_cell_res8"].astype(str) + "|" + train["hour_block"].astype(str)
    cell_hour_rate = train.groupby("cell_hour_key", observed=True)["risk_label"].mean()
    cell_rate = train.groupby("h3_cell_res8", observed=True)["risk_label"].mean()
    hour_rate = train.groupby("hour_block", observed=True)["risk_label"].mean()

    df = df.copy()
    df["cell_hour_key"] = df["h3_cell_res8"].astype(str) + "|" + df["hour_block"].astype(str)
    df["hist_cell_hour_risk"] = df["cell_hour_key"].map(cell_hour_rate).astype("float32")
    df["hist_cell_risk"] = df["h3_cell_res8"].map(cell_rate).astype("float32")
    df["hist_hour_risk"] = df["hour_block"].map(hour_rate).astype("float32")
    df["hist_global_risk"] = np.float32(global_rate)
    df["hist_cell_hour_risk"] = (
        df["hist_cell_hour_risk"]
        .fillna(df["hist_cell_risk"])
        .fillna(df["hist_hour_risk"])
        .fillna(global_rate)
        .astype("float32")
    )
    df["hist_cell_risk"] = df["hist_cell_risk"].fillna(global_rate).astype("float32")
    df["hist_hour_risk"] = df["hist_hour_risk"].fillna(global_rate).astype("float32")
    df = df.drop(columns=["cell_hour_key"])
    return df


def chronological_masks(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, dict[str, int]]:
    min_window = int(df["window_index"].min())
    max_window = int(df["window_index"].max())
    window_count = max_window - min_window + 1
    train_cut = min_window + int(window_count * 0.70)
    val_cut = min_window + int(window_count * 0.85)
    train_mask = df["window_index"] < train_cut
    val_mask = (df["window_index"] >= train_cut) & (df["window_index"] < val_cut)
    test_mask = df["window_index"] >= val_cut
    if train_mask.sum() == 0 or val_mask.sum() == 0 or test_mask.sum() == 0:
        raise SystemExit("Chronological split produced an empty split")
    split_meta = {
        "min_window": min_window,
        "max_window": max_window,
        "window_count": window_count,
        "train_cut": train_cut,
        "val_cut": val_cut,
    }
    return train_mask, val_mask, test_mask, split_meta


def split_summary(df: pd.DataFrame, masks: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for split, mask in masks.items():
        part = df.loc[mask]
        positives = int(part["risk_label"].sum())
        rows.append(
            {
                "split": split,
                "rows": int(len(part)),
                "positives": positives,
                "negatives": int(len(part) - positives),
                "positive_rate": positives / max(len(part), 1),
                "min_window_index": int(part["window_index"].min()),
                "max_window_index": int(part["window_index"].max()),
                "min_window_start": str(part["window_start"].min()),
                "max_window_start": str(part["window_start"].max()),
            }
        )
    return pd.DataFrame(rows)


def validate_no_leakage() -> None:
    overlap = LEAKAGE_COLUMNS.intersection(FEATURE_COLUMNS)
    if overlap:
        raise SystemExit(f"Leakage columns present in model features: {sorted(overlap)}")


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


def threshold_by_validation_f1(y_true: pd.Series, scores: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5, 0.0
    f1_values = (2 * precision[:-1] * recall[:-1]) / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best_index = int(np.nanargmax(f1_values))
    return float(thresholds[best_index]), float(f1_values[best_index])


def score_metrics(
    name: str,
    status: str,
    notes: str,
    train_rows: int,
    y_val: pd.Series,
    val_scores: np.ndarray,
    y_test: pd.Series,
    test_scores: np.ndarray,
) -> ModelResult:
    threshold, validation_f1 = threshold_by_validation_f1(y_val, val_scores)
    test_pred = (test_scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, test_pred, labels=[0, 1]).ravel()
    return ModelResult(
        name=name,
        status=status,
        notes=notes,
        train_rows=train_rows,
        validation_rows=len(y_val),
        test_rows=len(y_test),
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


def historical_scores(df: pd.DataFrame, mask: pd.Series) -> np.ndarray:
    return df.loc[mask, "hist_cell_hour_risk"].to_numpy(dtype=np.float32)


def fit_predict_pipeline(
    estimator: object,
    train_df: pd.DataFrame,
    train_y: pd.Series,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scale_numeric: bool,
    sample_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
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
    val_scores = pipeline.predict_proba(val_df[FEATURE_COLUMNS])[:, 1]
    test_scores = pipeline.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]
    return val_scores, test_scores, len(sample_indices)


def top_k_hotspot_recall(frame: pd.DataFrame, results: list[ModelResult]) -> pd.DataFrame:
    rows = []
    base = frame[["window_index", "risk_label"]].reset_index(drop=True)
    positive_by_window = base.groupby("window_index", observed=True)["risk_label"].sum()
    positive_by_window = positive_by_window[positive_by_window > 0]
    total_positives = int(base["risk_label"].sum())
    for result in results:
        scored = base.copy()
        scored["score"] = result.test_scores
        scored = scored.sort_values(["window_index", "score"], ascending=[True, False])
        for k in TOP_K_VALUES:
            top = scored.groupby("window_index", observed=True).head(k)
            captured_by_window = top.groupby("window_index", observed=True)["risk_label"].sum()
            captured_by_window = captured_by_window.reindex(positive_by_window.index, fill_value=0)
            positives_captured = int(captured_by_window.sum())
            window_recall = (captured_by_window / positive_by_window).replace([np.inf, -np.inf], 0)
            rows.append(
                {
                    "model": result.name,
                    "k": k,
                    "metric_name": f"sampled_candidate_hotspot_recall_at_{k}",
                    "windows_with_positives": int(len(positive_by_window)),
                    "total_positive_hotspots": total_positives,
                    "positives_captured": positives_captured,
                    "weighted_recall": positives_captured / max(total_positives, 1),
                    "mean_window_recall": float(window_recall.mean()),
                }
            )
    return pd.DataFrame(rows)


def results_to_frames(results: list[ModelResult]) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_rows = []
    confusion_rows = []
    for result in results:
        metrics_rows.append(
            {
                "model": result.name,
                "status": result.status,
                "notes": result.notes,
                "train_rows_used": result.train_rows,
                "validation_rows": result.validation_rows,
                "test_rows": result.test_rows,
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


def plot_curves(y_test: pd.Series, results: list[ModelResult]) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 6))
    for result in results:
        fpr, tpr, _ = roc_curve(y_test, result.test_scores)
        plt.plot(fpr, tpr, label=f"{result.name} ({result.test_roc_auc:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], color="0.5", linestyle="--", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Initial baseline ROC curves")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "initial_baseline_roc_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))
    for result in results:
        precision, recall, _ = precision_recall_curve(y_test, result.test_scores)
        plt.plot(recall, precision, label=f"{result.name} ({result.test_pr_auc:.3f})", linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Initial baseline precision-recall curves")
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "initial_baseline_pr_curve.png", dpi=200)
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
    plt.title("Initial baseline model comparison")
    plt.xticks(rotation=15, ha="right")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "initial_baseline_model_comparison.png", dpi=200)
    plt.close()


def frame_to_markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for record in frame.astype(str).to_dict(orient="records"):
        rows.append("| " + " | ".join(record[column] for column in columns) + " |")
    return "\n".join(rows)


def write_audit(
    raw_rows: int,
    filtered_df: pd.DataFrame,
    split_meta: dict[str, int],
    split_frame: pd.DataFrame,
    metrics: pd.DataFrame,
    top_k: pd.DataFrame,
    skipped_models: list[dict[str, str]],
) -> None:
    audit = [
        "# Initial baseline modeling audit",
        "",
        f"- Input file: `{INPUT_PATH}`",
        f"- Raw model-sample rows: `{raw_rows}`",
        f"- Included geo scopes: `{', '.join(sorted(INCLUDED_SCOPES))}`",
        f"- Rows after geo-scope filter: `{len(filtered_df)}`",
        f"- Excluded outside-UAE flagged rows: `{raw_rows - len(filtered_df)}`",
        f"- Minimum window index: `{split_meta['min_window']}`",
        f"- Maximum window index: `{split_meta['max_window']}`",
        f"- Window count: `{split_meta['window_count']}`",
        f"- Train/validation cut window index: `{split_meta['train_cut']}`",
        f"- Validation/test cut window index: `{split_meta['val_cut']}`",
        f"- Leakage columns excluded from features: `{', '.join(sorted(LEAKAGE_COLUMNS))}`",
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
            ]
        ),
        "",
        "## Sampled-candidate top-k hotspot recall",
        "",
        frame_to_markdown(top_k),
    ]
    if skipped_models:
        audit.extend(["", "## Skipped models", ""])
        for skipped in skipped_models:
            audit.append(f"- {skipped['model']}: {skipped['reason']}")
    (AUDIT_DIR / "initial_baseline_modeling_audit.md").write_text("\n".join(audit) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dirs()
    validate_no_leakage()

    df = load_model_sample(args.input)
    raw_rows = len(df)
    df = df[df["geo_scope"].astype(str).isin(INCLUDED_SCOPES)].copy()
    if len(df) == raw_rows:
        raise SystemExit("Geo-scope filter did not exclude any rows; expected outside_uae_flagged records")

    train_mask, val_mask, test_mask, split_meta = chronological_masks(df)
    df = add_train_only_historical_features(df, train_mask)
    train_mask, val_mask, test_mask, split_meta = chronological_masks(df)
    masks = {"train": train_mask, "validation": val_mask, "test": test_mask}
    split_frame = split_summary(df, masks)

    if set(FEATURE_COLUMNS).difference(df.columns):
        missing = sorted(set(FEATURE_COLUMNS).difference(df.columns))
        raise SystemExit(f"Missing model feature columns: {missing}")

    train_df = df.loc[train_mask].reset_index(drop=True)
    val_df = df.loc[val_mask].reset_index(drop=True)
    test_df = df.loc[test_mask].reset_index(drop=True)
    y_train = train_df["risk_label"].astype(int)
    y_val = val_df["risk_label"].astype(int)
    y_test = test_df["risk_label"].astype(int)

    results: list[ModelResult] = []
    skipped_models: list[dict[str, str]] = []

    val_scores = historical_scores(df, val_mask)
    test_scores = historical_scores(df, test_mask)
    results.append(
        score_metrics(
            "Historical risk",
            "trained",
            "Train-only empirical risk by H3 cell and hour block with fallback rates.",
            len(train_df),
            y_val,
            val_scores,
            y_test,
            test_scores,
        )
    )

    logistic = LogisticRegression(max_iter=300, class_weight="balanced", solver="lbfgs", random_state=args.seed)
    val_scores, test_scores, rows_used = fit_predict_pipeline(
        logistic, train_df, y_train, val_df, test_df, scale_numeric=True
    )
    results.append(
        score_metrics(
            "Logistic Regression",
            "trained",
            "Full training split, class_weight=balanced.",
            rows_used,
            y_val,
            val_scores,
            y_test,
            test_scores,
        )
    )

    rf_indices = sample_training_rows(y_train, args.rf_train_cap, args.seed)
    random_forest = RandomForestClassifier(
        n_estimators=120,
        max_depth=18,
        min_samples_leaf=20,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=args.seed,
    )
    val_scores, test_scores, rows_used = fit_predict_pipeline(
        random_forest,
        train_df,
        y_train,
        val_df,
        test_df,
        scale_numeric=False,
        sample_indices=rf_indices,
    )
    results.append(
        score_metrics(
            "Random Forest",
            "trained",
            f"Deterministic stratified training cap of {rows_used:,} rows.",
            rows_used,
            y_val,
            val_scores,
            y_test,
            test_scores,
        )
    )

    if XGBClassifier is None:
        skipped_models.append({"model": "XGBoost", "reason": XGB_IMPORT_ERROR})
    else:
        xgb_indices = sample_training_rows(y_train, args.xgb_train_cap, args.seed + 1)
        y_xgb = y_train.iloc[xgb_indices]
        pos = int(y_xgb.sum())
        neg = int(len(y_xgb) - pos)
        scale_pos_weight = neg / max(pos, 1)
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
            random_state=args.seed,
            scale_pos_weight=scale_pos_weight,
        )
        val_scores, test_scores, rows_used = fit_predict_pipeline(
            xgb,
            train_df,
            y_train,
            val_df,
            test_df,
            scale_numeric=False,
            sample_indices=xgb_indices,
        )
        results.append(
            score_metrics(
                "XGBoost",
                "trained",
                f"CPU hist tree method with deterministic stratified training cap of {rows_used:,} rows.",
                rows_used,
                y_val,
                val_scores,
                y_test,
                test_scores,
            )
        )

    metrics, confusion = results_to_frames(results)
    top_k = top_k_hotspot_recall(test_df, results)

    metrics.to_csv(TABLE_DIR / "initial_baseline_metrics.csv", index=False)
    split_frame.to_csv(TABLE_DIR / "initial_baseline_split_summary.csv", index=False)
    confusion.to_csv(TABLE_DIR / "initial_baseline_confusion_matrix.csv", index=False)
    top_k.to_csv(TABLE_DIR / "initial_baseline_topk_hotspot_recall.csv", index=False)
    (TABLE_DIR / "initial_baseline_skipped_models.json").write_text(
        json.dumps(skipped_models, indent=2), encoding="utf-8"
    )

    plot_curves(y_test, results)
    plot_model_comparison(metrics)
    write_audit(
        raw_rows,
        df,
        split_meta,
        split_frame,
        metrics,
        top_k,
        skipped_models,
    )

    print(f"Wrote metrics to {TABLE_DIR / 'initial_baseline_metrics.csv'}")
    print(f"Wrote plots to {FIGURE_DIR}")
    print(f"Wrote audit to {AUDIT_DIR / 'initial_baseline_modeling_audit.md'}")


if __name__ == "__main__":
    main()
