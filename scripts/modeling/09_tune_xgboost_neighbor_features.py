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
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


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
SELECTION_PR_TIE = 0.001

DEFAULT_CONFIG = {
    "n_estimators": 160,
    "max_depth": 5,
    "learning_rate": 0.07,
    "min_child_weight": 1,
}

CANDIDATE_CONFIGS = [
    {"candidate_id": "xgb_tune_01", "max_depth": 4, "learning_rate": 0.04, "n_estimators": 120, "min_child_weight": 5},
    {"candidate_id": "xgb_tune_02", "max_depth": 4, "learning_rate": 0.04, "n_estimators": 180, "min_child_weight": 10},
    {"candidate_id": "xgb_tune_03", "max_depth": 4, "learning_rate": 0.07, "n_estimators": 120, "min_child_weight": 5},
    {"candidate_id": "xgb_tune_04", "max_depth": 4, "learning_rate": 0.07, "n_estimators": 180, "min_child_weight": 10},
    {"candidate_id": "xgb_tune_05", "max_depth": 4, "learning_rate": 0.10, "n_estimators": 120, "min_child_weight": 5},
    {"candidate_id": "xgb_tune_06", "max_depth": 4, "learning_rate": 0.10, "n_estimators": 180, "min_child_weight": 10},
    {"candidate_id": "xgb_tune_07", "max_depth": 5, "learning_rate": 0.04, "n_estimators": 180, "min_child_weight": 5},
    {"candidate_id": "xgb_tune_08", "max_depth": 5, "learning_rate": 0.04, "n_estimators": 240, "min_child_weight": 10},
    {"candidate_id": "xgb_tune_09", "max_depth": 5, "learning_rate": 0.07, "n_estimators": 180, "min_child_weight": 5},
    {"candidate_id": "xgb_tune_10", "max_depth": 5, "learning_rate": 0.07, "n_estimators": 240, "min_child_weight": 10},
    {"candidate_id": "xgb_tune_11", "max_depth": 5, "learning_rate": 0.10, "n_estimators": 180, "min_child_weight": 5},
    {"candidate_id": "xgb_tune_12", "max_depth": 5, "learning_rate": 0.10, "n_estimators": 240, "min_child_weight": 10},
    {"candidate_id": "xgb_tune_13", "max_depth": 6, "learning_rate": 0.04, "n_estimators": 180, "min_child_weight": 5},
    {"candidate_id": "xgb_tune_14", "max_depth": 6, "learning_rate": 0.04, "n_estimators": 240, "min_child_weight": 10},
    {"candidate_id": "xgb_tune_15", "max_depth": 6, "learning_rate": 0.07, "n_estimators": 180, "min_child_weight": 5},
    {"candidate_id": "xgb_tune_16", "max_depth": 6, "learning_rate": 0.07, "n_estimators": 240, "min_child_weight": 10},
    {"candidate_id": "xgb_tune_17", "max_depth": 6, "learning_rate": 0.10, "n_estimators": 180, "min_child_weight": 5},
    {"candidate_id": "xgb_tune_18", "max_depth": 6, "learning_rate": 0.10, "n_estimators": 240, "min_child_weight": 10},
]


@dataclass
class ValidationCandidate:
    candidate_id: str
    max_depth: int
    learning_rate: float
    n_estimators: int
    min_child_weight: int
    train_rows_used: int
    threshold: float
    validation_pr_auc: float
    validation_roc_auc: float
    validation_precision: float
    validation_recall: float
    validation_f1: float
    validation_top20_positive_cell_windows_captured: int
    validation_top20_recall: float
    validation_top20_precision: float
    validation_top20_incidents_captured: int
    validation_top20_incident_recall: float


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def train_xgboost_with_config(
    train_df: pd.DataFrame,
    sample_indices: np.ndarray,
    config: dict[str, int | float | str],
    seed: int,
) -> tuple[Pipeline, int]:
    if fg.XGBClassifier is None:
        raise SystemExit(fg.XGB_IMPORT_ERROR)
    y_train = train_df["risk_label"].astype(int)
    y_sample = y_train.iloc[sample_indices]
    pos = int(y_sample.sum())
    neg = int(len(y_sample) - pos)
    model = fg.XGBClassifier(
        n_estimators=int(config["n_estimators"]),
        max_depth=int(config["max_depth"]),
        learning_rate=float(config["learning_rate"]),
        min_child_weight=int(config["min_child_weight"]),
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
            ("preprocess", ng.build_preprocessor(ng.NEIGHBOR_FEATURE_COLUMNS)),
            ("model", model),
        ]
    )
    pipeline.fit(train_df.iloc[sample_indices][ng.NEIGHBOR_FEATURE_COLUMNS], y_train.iloc[sample_indices])
    return pipeline, len(sample_indices)


def top_k_stats(
    scores: np.ndarray,
    y: np.ndarray,
    incident_counts: np.ndarray,
    n_windows: int,
    n_cells: int,
    k: int = 20,
) -> dict[str, float | int]:
    y_matrix = y.reshape(n_windows, n_cells)
    incident_matrix = incident_counts.reshape(n_windows, n_cells)
    score_matrix = scores.reshape(n_windows, n_cells)
    top_indices = np.argpartition(score_matrix, -k, axis=1)[:, -k:]
    captured_labels = np.take_along_axis(y_matrix, top_indices, axis=1)
    captured_incidents = np.take_along_axis(incident_matrix, top_indices, axis=1)
    positive_captured = int(captured_labels.sum())
    incidents_captured = int(captured_incidents.sum())
    total_positive = int(y_matrix.sum())
    total_incidents = int(incident_matrix.sum())
    return {
        "positive_cell_windows_captured": positive_captured,
        "recall": positive_captured / max(total_positive, 1),
        "precision": positive_captured / max(n_windows * k, 1),
        "incidents_captured": incidents_captured,
        "incident_recall": incidents_captured / max(total_incidents, 1),
    }


def score_validation_candidate(
    config: dict[str, int | float | str],
    train_df: pd.DataFrame,
    sample_indices: np.ndarray,
    inside_cells: list[str],
    labels: np.ndarray,
    incident_counts: np.ndarray,
    count_cumulative: np.ndarray,
    severity_cumulative: np.ndarray,
    priors: dict[str, np.ndarray | float],
    state: ng.NeighborState,
    validation_windows: np.ndarray,
) -> ValidationCandidate:
    candidate_id = str(config["candidate_id"])
    estimator, train_rows = train_xgboost_with_config(train_df, sample_indices, config, RANDOM_SEED)
    model = ng.ExperimentModel(
        name=candidate_id,
        feature_set_id=candidate_id,
        notes="Tuning candidate for neighbor-feature XGBoost.",
        feature_columns=ng.NEIGHBOR_FEATURE_COLUMNS,
        train_rows_used=train_rows,
        estimator=estimator,
    )
    y_val, val_incidents, val_scores_by_model = ng.score_models(
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
    scores = val_scores_by_model[candidate_id]
    threshold, validation_f1 = fg.threshold_by_validation_f1(y_val, scores)
    prediction = (scores >= threshold).astype(np.uint8)
    top20 = top_k_stats(scores, y_val, val_incidents, len(validation_windows), len(inside_cells), 20)
    return ValidationCandidate(
        candidate_id=candidate_id,
        max_depth=int(config["max_depth"]),
        learning_rate=float(config["learning_rate"]),
        n_estimators=int(config["n_estimators"]),
        min_child_weight=int(config["min_child_weight"]),
        train_rows_used=train_rows,
        threshold=threshold,
        validation_pr_auc=float(average_precision_score(y_val, scores)),
        validation_roc_auc=float(roc_auc_score(y_val, scores)),
        validation_precision=float(precision_score(y_val, prediction, zero_division=0)),
        validation_recall=float(recall_score(y_val, prediction, zero_division=0)),
        validation_f1=float(f1_score(y_val, prediction, zero_division=0)),
        validation_top20_positive_cell_windows_captured=int(top20["positive_cell_windows_captured"]),
        validation_top20_recall=float(top20["recall"]),
        validation_top20_precision=float(top20["precision"]),
        validation_top20_incidents_captured=int(top20["incidents_captured"]),
        validation_top20_incident_recall=float(top20["incident_recall"]),
    )


def select_candidate(candidates: pd.DataFrame) -> pd.Series:
    max_pr_auc = float(candidates["validation_pr_auc"].max())
    eligible = candidates[candidates["validation_pr_auc"] >= max_pr_auc - SELECTION_PR_TIE].copy()
    eligible = eligible.sort_values(
        by=[
            "validation_top20_incident_recall",
            "validation_f1",
            "validation_pr_auc",
            "n_estimators",
            "max_depth",
            "candidate_id",
        ],
        ascending=[False, False, False, True, True, True],
    )
    return eligible.iloc[0]


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


def selected_config_frame(
    selected: pd.Series,
    metrics: pd.DataFrame,
    top_k: pd.DataFrame,
) -> pd.DataFrame:
    default = metrics.loc[metrics["model"] == "XGBoost neighbor default"].iloc[0]
    tuned = metrics.loc[metrics["model"] == "XGBoost neighbor tuned"].iloc[0]
    default_top20 = top_k[(top_k["model"] == "XGBoost neighbor default") & (top_k["k"] == 20)].iloc[0]
    tuned_top20 = top_k[(top_k["model"] == "XGBoost neighbor tuned") & (top_k["k"] == 20)].iloc[0]
    promoted = bool(
        (tuned["test_pr_auc"] >= default["test_pr_auc"])
        and (tuned_top20["incident_recall_at_k"] >= default_top20["incident_recall_at_k"] - 0.001)
    )
    return pd.DataFrame(
        [
            {
                "selected_candidate_id": selected["candidate_id"],
                "max_depth": int(selected["max_depth"]),
                "learning_rate": float(selected["learning_rate"]),
                "n_estimators": int(selected["n_estimators"]),
                "min_child_weight": int(selected["min_child_weight"]),
                "selection_rule": "highest validation PR-AUC; tie within 0.001 by validation top-20 incident recall, then validation F1",
                "selected_validation_pr_auc": float(selected["validation_pr_auc"]),
                "selected_validation_f1": float(selected["validation_f1"]),
                "selected_validation_top20_incident_recall": float(selected["validation_top20_incident_recall"]),
                "default_test_pr_auc": float(default["test_pr_auc"]),
                "tuned_test_pr_auc": float(tuned["test_pr_auc"]),
                "test_pr_auc_change": float(tuned["test_pr_auc"] - default["test_pr_auc"]),
                "default_test_f1": float(default["test_f1"]),
                "tuned_test_f1": float(tuned["test_f1"]),
                "test_f1_change": float(tuned["test_f1"] - default["test_f1"]),
                "default_top20_incident_recall": float(default_top20["incident_recall_at_k"]),
                "tuned_top20_incident_recall": float(tuned_top20["incident_recall_at_k"]),
                "top20_incident_recall_change": float(tuned_top20["incident_recall_at_k"] - default_top20["incident_recall_at_k"]),
                "promoted_to_current_final_candidate": promoted,
            }
        ]
    )


def plot_validation_candidates(candidates: pd.DataFrame, selected_id: str) -> None:
    plot_df = candidates.sort_values("validation_pr_auc", ascending=False).copy()
    plot_df["selected"] = np.where(plot_df["candidate_id"] == selected_id, "selected", "candidate")
    plt.figure(figsize=(10, 5.5))
    sns.barplot(data=plot_df, x="candidate_id", y="validation_pr_auc", hue="selected", dodge=False)
    plt.xlabel("Candidate")
    plt.ylabel("Validation PR-AUC")
    plt.title("XGBoost tuning validation PR-AUC")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "xgboost_tuning_validation_pr_auc.png", dpi=200)
    plt.close()


def plot_test_comparison(metrics: pd.DataFrame) -> None:
    plot_df = metrics.melt(
        id_vars=["model"],
        value_vars=["test_roc_auc", "test_pr_auc", "test_f1", "test_recall"],
        var_name="metric",
        value_name="value",
    )
    labels = {
        "test_roc_auc": "ROC-AUC",
        "test_pr_auc": "PR-AUC",
        "test_f1": "F1",
        "test_recall": "Recall",
    }
    plot_df["metric"] = plot_df["metric"].map(labels)
    plt.figure(figsize=(9, 5.5))
    sns.barplot(data=plot_df, x="model", y="value", hue="metric")
    plt.ylim(0, 1)
    plt.xlabel("")
    plt.ylabel("Score")
    plt.title("XGBoost tuning test comparison")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "xgboost_tuning_test_comparison.png", dpi=200)
    plt.close()


def plot_top_k(top_k: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 5.5))
    sns.lineplot(data=top_k, x="k", y="precision_at_k", hue="model", marker="o", linewidth=2)
    plt.xlabel("Top-k cells per 3-hour window")
    plt.ylabel("Precision at k")
    plt.title("XGBoost tuning top-k hotspot precision")
    plt.ylim(0, max(0.05, float(top_k["precision_at_k"].max()) * 1.15))
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "xgboost_tuning_topk_comparison.png", dpi=200)
    plt.close()


def write_audit(
    candidates: pd.DataFrame,
    metrics: pd.DataFrame,
    top_k: pd.DataFrame,
    selected_config: pd.DataFrame,
) -> None:
    selected = selected_config.iloc[0]
    default = metrics.loc[metrics["model"] == "XGBoost neighbor default"].iloc[0]
    tuned = metrics.loc[metrics["model"] == "XGBoost neighbor tuned"].iloc[0]
    audit = [
        "# XGBoost neighbor-feature tuning experiment",
        "",
        f"- H3 resolution: `{fg.H3_RESOLUTION}`",
        f"- Time window hours: `{fg.WINDOW_HOURS}`",
        f"- Inside-Dubai cells: `{fg.EXPECTED_INSIDE_DUBAI_CELLS}`",
        f"- Training source: existing sampled-negative training table with cap `{TRAIN_CAP}`",
        "- Rejected hard-negative sample is not used.",
        f"- Validation candidates: `{fg.EXPECTED_VALIDATION_CANDIDATES}`",
        f"- Test candidates: `{fg.EXPECTED_TEST_CANDIDATES}`",
        f"- Test positive cell/windows: `{fg.EXPECTED_TEST_POSITIVE_CELL_WINDOWS}`",
        "- Candidate count: `18`",
        "- Selection rule: highest validation PR-AUC; tie within 0.001 by validation top-20 incident recall, then validation F1.",
        f"- Selected candidate: `{selected['selected_candidate_id']}`",
        f"- Selected hyperparameters: max_depth `{int(selected['max_depth'])}`, learning_rate `{selected['learning_rate']}`, n_estimators `{int(selected['n_estimators'])}`, min_child_weight `{int(selected['min_child_weight'])}`",
        f"- Test PR-AUC change vs default neighbor XGBoost: `{tuned['test_pr_auc'] - default['test_pr_auc']:.6f}`",
        f"- Test F1 change vs default neighbor XGBoost: `{tuned['test_f1'] - default['test_f1']:.6f}`",
        f"- Promoted to current final candidate: `{bool(selected['promoted_to_current_final_candidate'])}`",
        "",
        "## Selected configuration",
        "",
        ng.frame_to_markdown(selected_config.round(6)),
        "",
        "## Top validation candidates",
        "",
        ng.frame_to_markdown(
            candidates.sort_values("validation_pr_auc", ascending=False)
            .head(8)
            [
                [
                    "candidate_id",
                    "max_depth",
                    "learning_rate",
                    "n_estimators",
                    "min_child_weight",
                    "validation_pr_auc",
                    "validation_f1",
                    "validation_top20_incident_recall",
                ]
            ]
            .round(6)
        ),
        "",
        "## Test metrics",
        "",
        ng.frame_to_markdown(
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
    ]
    (AUDIT_DIR / "xgboost_tuning_audit.md").write_text("\n".join(audit), encoding="utf-8")


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
    validation_windows = np.arange(fg.TRAIN_END_EXCLUSIVE, fg.VALIDATION_END_EXCLUSIVE, dtype=np.int32)
    test_windows = np.arange(fg.VALIDATION_END_EXCLUSIVE, fg.WINDOW_COUNT, dtype=np.int32)

    candidate_rows = []
    for idx, config in enumerate(CANDIDATE_CONFIGS, start=1):
        print(f"Training/scoring tuning candidate {idx:02d}/{len(CANDIDATE_CONFIGS)}: {config['candidate_id']}", flush=True)
        candidate = score_validation_candidate(
            config,
            train_df,
            sample_indices,
            inside_cells,
            labels,
            incident_counts,
            count_cumulative,
            severity_cumulative,
            priors,
            state,
            validation_windows,
        )
        candidate_rows.append(candidate.__dict__)
        print(
            f"{candidate.candidate_id}: val PR-AUC={candidate.validation_pr_auc:.6f}, "
            f"val F1={candidate.validation_f1:.6f}, "
            f"val top20 incident recall={candidate.validation_top20_incident_recall:.6f}",
            flush=True,
        )

    candidates = pd.DataFrame(candidate_rows)
    selected = select_candidate(candidates)
    selected_config = {
        "candidate_id": str(selected["candidate_id"]),
        "max_depth": int(selected["max_depth"]),
        "learning_rate": float(selected["learning_rate"]),
        "n_estimators": int(selected["n_estimators"]),
        "min_child_weight": int(selected["min_child_weight"]),
    }
    print(f"Selected tuning candidate: {selected_config}", flush=True)

    default_estimator, default_rows = train_xgboost_with_config(train_df, sample_indices, DEFAULT_CONFIG, RANDOM_SEED)
    tuned_estimator, tuned_rows = train_xgboost_with_config(train_df, sample_indices, selected_config, RANDOM_SEED)
    models = [
        ng.ExperimentModel(
            name="Historical risk",
            feature_set_id="historical_score_only",
            notes="Train-period full-grid empirical risk by inside-Dubai H3 cell and hour block.",
            feature_columns=["hist_cell_hour_risk"],
            train_rows_used=len(train_df),
            estimator=None,
        ),
        ng.ExperimentModel(
            name="XGBoost neighbor default",
            feature_set_id="neighbor_lag_default_xgboost",
            notes="Neighbor-feature XGBoost with the current default hyperparameters.",
            feature_columns=ng.NEIGHBOR_FEATURE_COLUMNS,
            train_rows_used=default_rows,
            estimator=default_estimator,
        ),
        ng.ExperimentModel(
            name="XGBoost neighbor tuned",
            feature_set_id=str(selected["candidate_id"]),
            notes="Neighbor-feature XGBoost selected by full-grid validation PR-AUC.",
            feature_columns=ng.NEIGHBOR_FEATURE_COLUMNS,
            train_rows_used=tuned_rows,
            estimator=tuned_estimator,
        ),
    ]

    y_val, _, val_scores = ng.score_models(
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
    y_test, test_incidents, test_scores = ng.score_models(
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

    results = ng.evaluate(models, y_val, val_scores, y_test, test_scores)
    metrics = metrics_frame(results, len(y_val), len(y_test))
    top_k = fg.full_grid_top_k_metrics(results, y_test, test_incidents, len(test_windows), len(inside_cells))
    selected_config_table = selected_config_frame(selected, metrics, top_k)

    candidates.to_csv(TABLE_DIR / "xgboost_tuning_validation_candidates.csv", index=False)
    metrics.to_csv(TABLE_DIR / "xgboost_tuning_test_metrics.csv", index=False)
    top_k.to_csv(TABLE_DIR / "xgboost_tuning_topk.csv", index=False)
    selected_config_table.to_csv(TABLE_DIR / "xgboost_tuning_selected_config.csv", index=False)
    plot_validation_candidates(candidates, str(selected["candidate_id"]))
    plot_test_comparison(metrics)
    plot_top_k(top_k)
    write_audit(candidates, metrics, top_k, selected_config_table)

    print(f"Wrote {TABLE_DIR / 'xgboost_tuning_validation_candidates.csv'}")
    print(f"Wrote {TABLE_DIR / 'xgboost_tuning_test_metrics.csv'}")
    print(f"Wrote {TABLE_DIR / 'xgboost_tuning_topk.csv'}")
    print(f"Wrote {TABLE_DIR / 'xgboost_tuning_selected_config.csv'}")
    print(f"Wrote {FIGURE_DIR / 'xgboost_tuning_validation_pr_auc.png'}")
    print(f"Wrote {FIGURE_DIR / 'xgboost_tuning_test_comparison.png'}")
    print(f"Wrote {FIGURE_DIR / 'xgboost_tuning_topk_comparison.png'}")
    print(f"Wrote {AUDIT_DIR / 'xgboost_tuning_audit.md'}")


if __name__ == "__main__":
    main()
