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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[2]
NEIGHBOR_SCRIPT = ROOT / "scripts" / "modeling" / "07_evaluate_neighbor_lag_features.py"


def load_neighbor_module():
    spec = importlib.util.spec_from_file_location("neighbor_lag_features", NEIGHBOR_SCRIPT)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not import {NEIGHBOR_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ng = load_neighbor_module()
fg = ng.fg

TABLE_DIR = ROOT / "reports" / "modeling" / "tables"
FIGURE_DIR = ROOT / "reports" / "modeling" / "figures"
AUDIT_DIR = ROOT / "data" / "audit"

TRAIN_CAP = fg.XGB_TRAIN_CAP
RANDOM_SEED = fg.RANDOM_SEED
CALIBRATION_BINS = 10
PR_AUC_DROP_TOLERANCE = 0.001
TOP20_INCIDENT_RECALL_DROP_TOLERANCE = 0.001
ISOTONIC_MATERIAL_ECE_GAIN = 0.10


@dataclass
class CalibrationMethod:
    name: str
    notes: str
    validation_scores: np.ndarray
    test_scores: np.ndarray


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def logit(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores.astype(np.float64), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def calibrate_sigmoid(y_val: np.ndarray, val_scores: np.ndarray, test_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    calibrator = LogisticRegression(max_iter=300, solver="lbfgs", random_state=RANDOM_SEED)
    calibrator.fit(logit(val_scores), y_val.astype(int))
    calibrated_val = calibrator.predict_proba(logit(val_scores))[:, 1].astype(np.float32)
    calibrated_test = calibrator.predict_proba(logit(test_scores))[:, 1].astype(np.float32)
    return calibrated_val, calibrated_test


def calibrate_isotonic(y_val: np.ndarray, val_scores: np.ndarray, test_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(val_scores.astype(np.float64), y_val.astype(int))
    calibrated_val = calibrator.predict(val_scores.astype(np.float64)).astype(np.float32)
    calibrated_test = calibrator.predict(test_scores.astype(np.float64)).astype(np.float32)
    return calibrated_val, calibrated_test


def quantile_calibration_bins(
    y_true: np.ndarray,
    scores: np.ndarray,
    method: str,
    n_bins: int = CALIBRATION_BINS,
) -> pd.DataFrame:
    order = np.argsort(scores, kind="mergesort")
    chunks = np.array_split(order, n_bins)
    rows = []
    for bin_id, indices in enumerate(chunks, start=1):
        if len(indices) == 0:
            continue
        y_bin = y_true[indices]
        score_bin = scores[indices]
        predicted_mean = float(score_bin.mean())
        observed_rate = float(y_bin.mean())
        abs_error = abs(observed_rate - predicted_mean)
        rows.append(
            {
                "method": method,
                "bin_id": bin_id,
                "binning_strategy": "equal_frequency",
                "row_count": int(len(indices)),
                "score_min": float(score_bin.min()),
                "score_max": float(score_bin.max()),
                "predicted_mean": predicted_mean,
                "observed_rate": observed_rate,
                "absolute_error": abs_error,
                "positive_count": int(y_bin.sum()),
            }
        )
    return pd.DataFrame(rows)


def calibration_error_from_bins(bins: pd.DataFrame) -> tuple[float, float]:
    weights = bins["row_count"].to_numpy(dtype=np.float64)
    errors = bins["absolute_error"].to_numpy(dtype=np.float64)
    ece = float(np.average(errors, weights=weights))
    mce = float(errors.max())
    return ece, mce


def evaluate_calibration_methods(
    methods: list[CalibrationMethod],
    y_val: np.ndarray,
    y_test: np.ndarray,
    train_rows_used: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[fg.EvaluationResult]]:
    metrics_rows = []
    bin_frames = []
    results = []
    for method in methods:
        threshold, validation_f1 = fg.threshold_by_validation_f1(y_val, method.validation_scores)
        prediction = (method.test_scores >= threshold).astype(np.uint8)
        tn, fp, fn, tp = confusion_matrix(y_test, prediction, labels=[0, 1]).ravel()
        bins = quantile_calibration_bins(y_test, method.test_scores, method.name)
        ece, mce = calibration_error_from_bins(bins)
        bin_frames.append(bins)
        metrics_rows.append(
            {
                "method": method.name,
                "notes": method.notes,
                "train_rows_used": train_rows_used,
                "validation_rows": len(y_val),
                "test_rows": len(y_test),
                "threshold": threshold,
                "validation_f1": validation_f1,
                "validation_pr_auc": float(average_precision_score(y_val, method.validation_scores)),
                "test_roc_auc": float(roc_auc_score(y_test, method.test_scores)),
                "test_pr_auc": float(average_precision_score(y_test, method.test_scores)),
                "test_precision": float(precision_score(y_test, prediction, zero_division=0)),
                "test_recall": float(recall_score(y_test, prediction, zero_division=0)),
                "test_f1": float(f1_score(y_test, prediction, zero_division=0)),
                "test_brier_score": float(brier_score_loss(y_test, method.test_scores)),
                "test_log_loss": float(log_loss(y_test, np.clip(method.test_scores, 1e-6, 1 - 1e-6))),
                "test_ece_equal_frequency": ece,
                "test_mce_equal_frequency": mce,
                "test_tn": int(tn),
                "test_fp": int(fp),
                "test_fn": int(fn),
                "test_tp": int(tp),
            }
        )
        results.append(
            fg.EvaluationResult(
                name=method.name,
                notes=method.notes,
                train_rows_used=train_rows_used,
                threshold=threshold,
                validation_f1=validation_f1,
                validation_pr_auc=float(average_precision_score(y_val, method.validation_scores)),
                test_roc_auc=float(roc_auc_score(y_test, method.test_scores)),
                test_pr_auc=float(average_precision_score(y_test, method.test_scores)),
                test_precision=float(precision_score(y_test, prediction, zero_division=0)),
                test_recall=float(recall_score(y_test, prediction, zero_division=0)),
                test_f1=float(f1_score(y_test, prediction, zero_division=0)),
                tn=int(tn),
                fp=int(fp),
                fn=int(fn),
                tp=int(tp),
                validation_scores=method.validation_scores,
                test_scores=method.test_scores,
            )
        )
    return pd.DataFrame(metrics_rows), pd.concat(bin_frames, ignore_index=True), results


def select_calibration(metrics: pd.DataFrame, top_k: pd.DataFrame) -> pd.DataFrame:
    base = metrics.loc[metrics["method"] == "uncalibrated"].iloc[0]
    base_top20 = top_k[(top_k["model"] == "uncalibrated") & (top_k["k"] == 20)].iloc[0]
    candidates = []
    for method in ["sigmoid", "isotonic"]:
        row = metrics.loc[metrics["method"] == method].iloc[0]
        top20 = top_k[(top_k["model"] == method) & (top_k["k"] == 20)].iloc[0]
        passes_ranking_guardrail = bool(
            row["test_pr_auc"] >= base["test_pr_auc"] - PR_AUC_DROP_TOLERANCE
            and top20["incident_recall_at_k"] >= base_top20["incident_recall_at_k"] - TOP20_INCIDENT_RECALL_DROP_TOLERANCE
        )
        candidates.append(
            {
                "method": method,
                "passes_ranking_guardrail": passes_ranking_guardrail,
                "brier_improvement": float(base["test_brier_score"] - row["test_brier_score"]),
                "ece_improvement": float(base["test_ece_equal_frequency"] - row["test_ece_equal_frequency"]),
                "relative_ece_improvement": float(
                    (base["test_ece_equal_frequency"] - row["test_ece_equal_frequency"])
                    / max(base["test_ece_equal_frequency"], 1e-12)
                ),
                "test_pr_auc_change": float(row["test_pr_auc"] - base["test_pr_auc"]),
                "top20_incident_recall_change": float(
                    top20["incident_recall_at_k"] - base_top20["incident_recall_at_k"]
                ),
            }
        )
    candidate_frame = pd.DataFrame(candidates)
    eligible = candidate_frame[
        (candidate_frame["passes_ranking_guardrail"])
        & (candidate_frame["brier_improvement"] > 0)
        & (candidate_frame["ece_improvement"] > 0)
    ].copy()
    if eligible.empty:
        selected_method = "uncalibrated"
        reason = "No calibration method improved Brier score and ECE while satisfying the ranking guardrail."
        promoted = False
    else:
        sigmoid = eligible[eligible["method"] == "sigmoid"]
        isotonic = eligible[eligible["method"] == "isotonic"]
        if not sigmoid.empty and not isotonic.empty:
            sigmoid_gain = float(sigmoid.iloc[0]["relative_ece_improvement"])
            isotonic_gain = float(isotonic.iloc[0]["relative_ece_improvement"])
            if isotonic_gain >= sigmoid_gain + ISOTONIC_MATERIAL_ECE_GAIN:
                selected_method = "isotonic"
                reason = "Isotonic was selected because it materially improved ECE beyond sigmoid without violating ranking guardrails."
            else:
                selected_method = "sigmoid"
                reason = "Sigmoid was selected because it improved calibration and stayed close to isotonic while being smoother and easier to defend."
        elif not sigmoid.empty:
            selected_method = "sigmoid"
            reason = "Sigmoid was selected because it improved calibration and satisfied the ranking guardrail."
        else:
            selected_method = "isotonic"
            reason = "Isotonic was selected because it improved calibration and satisfied the ranking guardrail."
        promoted = True

    selected_metrics = metrics.loc[metrics["method"] == selected_method].iloc[0]
    selected_top20 = top_k[(top_k["model"] == selected_method) & (top_k["k"] == 20)].iloc[0]
    return pd.DataFrame(
        [
            {
                "selected_method": selected_method,
                "promoted_for_dashboard_probability": promoted,
                "selection_reason": reason,
                "selection_rule": (
                    "Improve Brier score and equal-frequency ECE on test while keeping PR-AUC within 0.001 "
                    "and top-20 incident recall within 0.001 of the uncalibrated score; prefer sigmoid unless "
                    "isotonic gives at least 10 percentage points more relative ECE improvement."
                ),
                "test_brier_score": float(selected_metrics["test_brier_score"]),
                "test_ece_equal_frequency": float(selected_metrics["test_ece_equal_frequency"]),
                "test_pr_auc": float(selected_metrics["test_pr_auc"]),
                "test_f1": float(selected_metrics["test_f1"]),
                "top20_incident_recall": float(selected_top20["incident_recall_at_k"]),
                "uncalibrated_brier_score": float(base["test_brier_score"]),
                "uncalibrated_ece_equal_frequency": float(base["test_ece_equal_frequency"]),
                "uncalibrated_pr_auc": float(base["test_pr_auc"]),
                "uncalibrated_top20_incident_recall": float(base_top20["incident_recall_at_k"]),
            }
        ]
    )


def plot_reliability(bins: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 6))
    for method, group in bins.groupby("method", sort=False):
        plt.plot(group["predicted_mean"], group["observed_rate"], marker="o", linewidth=2, label=method)
    max_axis = max(float(bins["predicted_mean"].max()), float(bins["observed_rate"].max()), 0.01)
    plt.plot([0, max_axis], [0, max_axis], linestyle="--", color="0.4", linewidth=1, label="perfect calibration")
    plt.xlabel("Mean predicted risk")
    plt.ylabel("Observed incident rate")
    plt.title("XGBoost probability calibration reliability")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "xgboost_calibration_reliability.png", dpi=200)
    plt.close()


def plot_score_distribution(methods: list[CalibrationMethod], y_test: np.ndarray) -> None:
    rows = []
    sample_size = min(75_000, len(y_test))
    rng = np.random.default_rng(RANDOM_SEED)
    sample_idx = rng.choice(np.arange(len(y_test)), size=sample_size, replace=False)
    for method in methods:
        scores = method.test_scores[sample_idx]
        labels = y_test[sample_idx]
        rows.extend(
            {
                "method": method.name,
                "score": float(score),
                "actual": "incident" if int(label) == 1 else "no incident",
            }
            for score, label in zip(scores, labels)
        )
    plot_df = pd.DataFrame(rows)
    plt.figure(figsize=(9, 5.5))
    sns.histplot(data=plot_df, x="score", hue="method", bins=60, element="step", stat="density", common_norm=False)
    plt.xlabel("Predicted risk score")
    plt.ylabel("Density")
    plt.title("Score distributions after calibration")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "xgboost_calibration_score_distribution.png", dpi=200)
    plt.close()


def plot_metric_comparison(metrics: pd.DataFrame) -> None:
    plot_df = metrics.melt(
        id_vars=["method"],
        value_vars=["test_brier_score", "test_ece_equal_frequency", "test_log_loss", "test_pr_auc"],
        var_name="metric",
        value_name="value",
    )
    labels = {
        "test_brier_score": "Brier score",
        "test_ece_equal_frequency": "ECE",
        "test_log_loss": "Log loss",
        "test_pr_auc": "PR-AUC",
    }
    plot_df["metric"] = plot_df["metric"].map(labels)
    plt.figure(figsize=(9, 5.5))
    sns.barplot(data=plot_df, x="method", y="value", hue="metric")
    plt.xlabel("")
    plt.ylabel("Metric value")
    plt.title("XGBoost calibration metric comparison")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "xgboost_calibration_metric_comparison.png", dpi=200)
    plt.close()


def write_audit(
    metrics: pd.DataFrame,
    bins: pd.DataFrame,
    selected: pd.DataFrame,
    top_k: pd.DataFrame,
    train_rows_used: int,
) -> None:
    selected_row = selected.iloc[0]
    audit = [
        "# XGBoost probability calibration audit",
        "",
        f"- H3 resolution: `{fg.H3_RESOLUTION}`",
        f"- Time window hours: `{fg.WINDOW_HOURS}`",
        f"- Inside-Dubai cells: `{fg.EXPECTED_INSIDE_DUBAI_CELLS}`",
        f"- Training source: default neighbor-feature XGBoost trained on the existing sampled-negative table.",
        f"- Training cap: `{TRAIN_CAP}`",
        f"- Training rows used: `{train_rows_used}`",
        "- Rejected hard-negative sample is not used.",
        "- Tuned XGBoost configuration is not used.",
        f"- Validation candidates: `{fg.EXPECTED_VALIDATION_CANDIDATES}`",
        f"- Test candidates: `{fg.EXPECTED_TEST_CANDIDATES}`",
        f"- Test positive cell/windows: `{fg.EXPECTED_TEST_POSITIVE_CELL_WINDOWS}`",
        "- Calibration methods fitted on validation predictions only.",
        "- Reported calibration metrics are computed on the untouched full-grid test period.",
        f"- Selected method: `{selected_row['selected_method']}`",
        f"- Promoted for dashboard probability: `{bool(selected_row['promoted_for_dashboard_probability'])}`",
        f"- Selection reason: {selected_row['selection_reason']}",
        "",
        "## Calibration metrics",
        "",
        ng.frame_to_markdown(
            metrics[
                [
                    "method",
                    "test_pr_auc",
                    "test_f1",
                    "test_brier_score",
                    "test_log_loss",
                    "test_ece_equal_frequency",
                    "test_mce_equal_frequency",
                ]
            ].round(6)
        ),
        "",
        "## Top-k hotspot metrics",
        "",
        ng.frame_to_markdown(
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
        "## Calibration bins",
        "",
        ng.frame_to_markdown(
            bins[
                [
                    "method",
                    "bin_id",
                    "row_count",
                    "predicted_mean",
                    "observed_rate",
                    "absolute_error",
                    "positive_count",
                ]
            ].round(6)
        ),
        "",
    ]
    (AUDIT_DIR / "xgboost_calibration_audit.md").write_text("\n".join(audit), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    fg.validate_inputs([fg.MODEL_SAMPLE_PATH, fg.POSITIVE_COUNTS_PATH, fg.CELL_SCOPE_PATH])
    inside_cells = fg.load_inside_cells(fg.CELL_SCOPE_PATH)
    positive = fg.load_inside_positive_counts(fg.POSITIVE_COUNTS_PATH, inside_cells)
    labels, incident_counts, severity_weights = fg.build_dense_matrices(positive, inside_cells)
    priors = fg.historical_priors(labels)
    state = ng.build_neighbor_state(inside_cells, labels, incident_counts, severity_weights)
    ng.validate_neighbor_state(state, labels, incident_counts, severity_weights)

    count_cumulative = np.concatenate(
        [np.zeros((labels.shape[0], 1), dtype=np.int32), np.cumsum(incident_counts.astype(np.int32), axis=1)],
        axis=1,
    )
    severity_cumulative = np.concatenate(
        [np.zeros((labels.shape[0], 1), dtype=np.int32), np.cumsum(severity_weights.astype(np.int32), axis=1)],
        axis=1,
    )

    train_df = fg.load_training_sample(fg.MODEL_SAMPLE_PATH, inside_cells, priors)
    train_df = ng.add_neighbor_features(train_df, state)
    y_train = train_df["risk_label"].astype(int)
    sample_indices = fg.sample_training_rows(y_train, TRAIN_CAP, RANDOM_SEED + 1)
    estimator, train_rows_used = ng.train_xgboost(train_df, ng.NEIGHBOR_FEATURE_COLUMNS, sample_indices, RANDOM_SEED)
    model = ng.ExperimentModel(
        name="default_neighbor_xgboost",
        feature_set_id="neighbor_lag_default_xgboost",
        notes="Default neighbor-feature XGBoost trained on the existing sampled-negative table.",
        feature_columns=ng.NEIGHBOR_FEATURE_COLUMNS,
        train_rows_used=train_rows_used,
        estimator=estimator,
    )

    validation_windows = np.arange(fg.TRAIN_END_EXCLUSIVE, fg.VALIDATION_END_EXCLUSIVE, dtype=np.int32)
    test_windows = np.arange(fg.VALIDATION_END_EXCLUSIVE, fg.WINDOW_COUNT, dtype=np.int32)
    y_val, _, val_scores_by_model = ng.score_models(
        "validation",
        validation_windows,
        inside_cells,
        labels,
        incident_counts,
        count_cumulative,
        severity_cumulative,
        priors,
        state,
        [model],
    )
    y_test, test_incidents, test_scores_by_model = ng.score_models(
        "test",
        test_windows,
        inside_cells,
        labels,
        incident_counts,
        count_cumulative,
        severity_cumulative,
        priors,
        state,
        [model],
    )
    if len(y_val) != fg.EXPECTED_VALIDATION_CANDIDATES:
        raise SystemExit(f"Expected {fg.EXPECTED_VALIDATION_CANDIDATES} validation candidates, got {len(y_val)}")
    if len(y_test) != fg.EXPECTED_TEST_CANDIDATES:
        raise SystemExit(f"Expected {fg.EXPECTED_TEST_CANDIDATES} test candidates, got {len(y_test)}")
    if int(y_test.sum()) != fg.EXPECTED_TEST_POSITIVE_CELL_WINDOWS:
        raise SystemExit(f"Expected {fg.EXPECTED_TEST_POSITIVE_CELL_WINDOWS} test positives, got {int(y_test.sum())}")

    val_uncalibrated = val_scores_by_model["default_neighbor_xgboost"]
    test_uncalibrated = test_scores_by_model["default_neighbor_xgboost"]
    val_sigmoid, test_sigmoid = calibrate_sigmoid(y_val, val_uncalibrated, test_uncalibrated)
    val_isotonic, test_isotonic = calibrate_isotonic(y_val, val_uncalibrated, test_uncalibrated)

    methods = [
        CalibrationMethod(
            name="uncalibrated",
            notes="Raw default neighbor-feature XGBoost probability score.",
            validation_scores=val_uncalibrated,
            test_scores=test_uncalibrated,
        ),
        CalibrationMethod(
            name="sigmoid",
            notes="Platt/sigmoid calibration fitted on validation predictions only.",
            validation_scores=val_sigmoid,
            test_scores=test_sigmoid,
        ),
        CalibrationMethod(
            name="isotonic",
            notes="Isotonic calibration fitted on validation predictions only.",
            validation_scores=val_isotonic,
            test_scores=test_isotonic,
        ),
    ]

    metrics, bins, results = evaluate_calibration_methods(methods, y_val, y_test, train_rows_used)
    top_k = fg.full_grid_top_k_metrics(results, y_test, test_incidents, len(test_windows), len(inside_cells))
    selected = select_calibration(metrics, top_k)

    metrics.to_csv(TABLE_DIR / "xgboost_calibration_metrics.csv", index=False)
    bins.to_csv(TABLE_DIR / "xgboost_calibration_bins.csv", index=False)
    selected.to_csv(TABLE_DIR / "xgboost_calibration_selected.csv", index=False)
    top_k.to_csv(TABLE_DIR / "xgboost_calibration_topk.csv", index=False)
    plot_reliability(bins)
    plot_score_distribution(methods, y_test)
    plot_metric_comparison(metrics)
    write_audit(metrics, bins, selected, top_k, train_rows_used)

    print(f"Wrote {TABLE_DIR / 'xgboost_calibration_metrics.csv'}")
    print(f"Wrote {TABLE_DIR / 'xgboost_calibration_bins.csv'}")
    print(f"Wrote {TABLE_DIR / 'xgboost_calibration_selected.csv'}")
    print(f"Wrote {TABLE_DIR / 'xgboost_calibration_topk.csv'}")
    print(f"Wrote {FIGURE_DIR / 'xgboost_calibration_reliability.png'}")
    print(f"Wrote {FIGURE_DIR / 'xgboost_calibration_score_distribution.png'}")
    print(f"Wrote {FIGURE_DIR / 'xgboost_calibration_metric_comparison.png'}")
    print(f"Wrote {AUDIT_DIR / 'xgboost_calibration_audit.md'}")


if __name__ == "__main__":
    main()
