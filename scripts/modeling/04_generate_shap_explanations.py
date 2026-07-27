from __future__ import annotations

import importlib.util
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/thesis-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/thesis-cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/thesis-numba")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.special import expit
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[2]
NEIGHBOR_SCRIPT = ROOT / "scripts" / "modeling" / "07_evaluate_neighbor_lag_features.py"
TABLE_DIR = ROOT / "reports" / "modeling" / "tables"
FIGURE_DIR = ROOT / "reports" / "modeling" / "figures"
AUDIT_DIR = ROOT / "data" / "audit"

BACKGROUND_SAMPLE_SIZE = 2_000
GLOBAL_SAMPLE_SIZE = 10_000
LOCAL_TOP_FEATURES = 8
RANDOM_SEED = 42


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


@dataclass
class FinalModelContext:
    inside_cells: list[str]
    labels: np.ndarray
    incident_counts: np.ndarray
    severity_weights: np.ndarray
    count_cumulative: np.ndarray
    severity_cumulative: np.ndarray
    priors: dict[str, np.ndarray | float]
    state: ng.NeighborState
    train_df: pd.DataFrame
    pipeline: object
    train_rows_used: int
    validation_windows: np.ndarray
    test_windows: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    val_scores: np.ndarray
    test_scores: np.ndarray
    raw_threshold: float
    calibrated_val_scores: np.ndarray
    calibrated_test_scores: np.ndarray
    calibrated_threshold: float


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def logit(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores.astype(np.float64), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def fit_sigmoid_calibrator(
    y_val: np.ndarray,
    val_scores: np.ndarray,
    test_scores: np.ndarray,
) -> tuple[LogisticRegression, np.ndarray, np.ndarray]:
    calibrator = LogisticRegression(max_iter=300, solver="lbfgs", random_state=RANDOM_SEED)
    calibrator.fit(logit(val_scores), y_val.astype(int))
    calibrated_val = calibrator.predict_proba(logit(val_scores))[:, 1].astype(np.float32)
    calibrated_test = calibrator.predict_proba(logit(test_scores))[:, 1].astype(np.float32)
    return calibrator, calibrated_val, calibrated_test


def train_final_xgboost_pipeline(train_df: pd.DataFrame):
    if fg.XGBClassifier is None:
        raise SystemExit(fg.XGB_IMPORT_ERROR)
    y_train = train_df["risk_label"].astype(int)
    sample_indices = fg.sample_training_rows(y_train, fg.XGB_TRAIN_CAP, fg.RANDOM_SEED + 1)
    return ng.train_xgboost(train_df, ng.NEIGHBOR_FEATURE_COLUMNS, sample_indices, fg.RANDOM_SEED)


def build_final_model_context() -> FinalModelContext:
    fg.validate_inputs([fg.MODEL_SAMPLE_PATH, fg.POSITIVE_COUNTS_PATH, fg.CELL_SCOPE_PATH])
    if fg.XGBClassifier is None:
        raise SystemExit(fg.XGB_IMPORT_ERROR)

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
    pipeline, train_rows_used = train_final_xgboost_pipeline(train_df)
    fitted_model = ng.ExperimentModel(
        name="Final neighbor XGBoost",
        feature_set_id="neighbor_lag_default_xgboost",
        notes="Default neighbor-feature XGBoost used for final SHAP/LIME explanation.",
        feature_columns=ng.NEIGHBOR_FEATURE_COLUMNS,
        train_rows_used=train_rows_used,
        estimator=pipeline,
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
        [fitted_model],
    )
    y_test, _, test_scores_by_model = ng.score_models(
        "test",
        test_windows,
        inside_cells,
        labels,
        incident_counts,
        count_cumulative,
        severity_cumulative,
        priors,
        state,
        [fitted_model],
    )
    if len(y_val) != fg.EXPECTED_VALIDATION_CANDIDATES:
        raise SystemExit(f"Expected {fg.EXPECTED_VALIDATION_CANDIDATES} validation candidates, got {len(y_val)}")
    if len(y_test) != fg.EXPECTED_TEST_CANDIDATES:
        raise SystemExit(f"Expected {fg.EXPECTED_TEST_CANDIDATES} test candidates, got {len(y_test)}")
    if int(y_test.sum()) != fg.EXPECTED_TEST_POSITIVE_CELL_WINDOWS:
        raise SystemExit(f"Expected {fg.EXPECTED_TEST_POSITIVE_CELL_WINDOWS} positives, got {int(y_test.sum())}")

    val_scores = val_scores_by_model["Final neighbor XGBoost"]
    test_scores = test_scores_by_model["Final neighbor XGBoost"]
    raw_threshold, _ = fg.threshold_by_validation_f1(y_val, val_scores)
    _, calibrated_val_scores, calibrated_test_scores = fit_sigmoid_calibrator(y_val, val_scores, test_scores)
    calibrated_threshold, _ = fg.threshold_by_validation_f1(y_val, calibrated_val_scores)

    return FinalModelContext(
        inside_cells=inside_cells,
        labels=labels,
        incident_counts=incident_counts,
        severity_weights=severity_weights,
        count_cumulative=count_cumulative,
        severity_cumulative=severity_cumulative,
        priors=priors,
        state=state,
        train_df=train_df,
        pipeline=pipeline,
        train_rows_used=train_rows_used,
        validation_windows=validation_windows,
        test_windows=test_windows,
        y_val=y_val,
        y_test=y_test,
        val_scores=val_scores,
        test_scores=test_scores,
        raw_threshold=raw_threshold,
        calibrated_val_scores=calibrated_val_scores,
        calibrated_test_scores=calibrated_test_scores,
        calibrated_threshold=calibrated_threshold,
    )


def feature_group(feature_name: str) -> str:
    if feature_name.startswith("day_of_week_"):
        return "day_of_week"
    if feature_name.startswith("geo_scope_"):
        return "geo_scope"
    if feature_name.startswith("neighbor_"):
        return feature_name
    return feature_name


def candidate_frame_for_indices(indices: np.ndarray, context: FinalModelContext) -> pd.DataFrame:
    n_cells = len(context.inside_cells)
    offsets = indices // n_cells
    split_windows = context.test_windows
    cell_indices = indices % n_cells
    window_indices = split_windows[offsets].astype(np.int32)
    wf = fg.window_features(window_indices)
    hour = wf["hour_block"].to_numpy(dtype=np.int8)

    cell_array = np.asarray(context.inside_cells, dtype=object)[cell_indices]
    hist_cell = np.asarray(context.priors["cell"], dtype=np.float32)[cell_indices]
    hist_hour = np.asarray(context.priors["hour"], dtype=np.float32)[hour]
    hist_cell_hour = np.asarray(context.priors["cell_hour"], dtype=np.float32)[cell_indices, hour]
    hist_global = np.full(len(indices), float(context.priors["global"]), dtype=np.float32)

    def lag(cumulative: np.ndarray, length: int) -> np.ndarray:
        start = np.maximum(window_indices - length, 0)
        end = window_indices
        return (cumulative[cell_indices, end] - cumulative[cell_indices, start]).astype(np.int16)

    frame = pd.DataFrame(
        {
            "candidate_index": indices.astype(np.int64),
            "h3_cell_res8": cell_array,
            "window_index": window_indices,
            "window_start": wf["window_start"].astype(str).to_numpy(),
            "geo_scope": pd.Categorical(["inside_dubai"] * len(indices)),
            "hour_block": hour,
            "day_of_week": pd.Categorical(wf["day_of_week"].astype(str).to_numpy()),
            "is_weekend": wf["is_weekend"].to_numpy(dtype=np.int8),
            "month": wf["month"].to_numpy(dtype=np.int8),
            "year": wf["year"].to_numpy(dtype=np.int16),
            "prev_3h_incident_count": lag(context.count_cumulative, 1),
            "prev_24h_incident_count": lag(context.count_cumulative, 8),
            "prev_7d_incident_count": lag(context.count_cumulative, 56),
            "prev_24h_severity_weight_sum": lag(context.severity_cumulative, 8),
            "prev_7d_severity_weight_sum": lag(context.severity_cumulative, 56),
            "hist_cell_hour_risk": hist_cell_hour,
            "hist_cell_risk": hist_cell,
            "hist_hour_risk": hist_hour,
            "hist_global_risk": hist_global,
            "actual_label": context.labels[cell_indices, window_indices].astype(np.uint8),
            "incident_count": context.incident_counts[cell_indices, window_indices].astype(np.int16),
        }
    )
    return ng.add_neighbor_features(frame, context.state)


def transformed_frame(pipeline, frame: pd.DataFrame) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocess"]
    matrix = preprocessor.transform(frame[ng.NEIGHBOR_FEATURE_COLUMNS])
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    feature_names = list(preprocessor.get_feature_names_out())
    return pd.DataFrame(np.asarray(matrix, dtype=np.float32), columns=feature_names)


def sample_global_indices(y_test: np.ndarray, sample_size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    size = min(sample_size, len(y_test))
    return np.sort(rng.choice(np.arange(len(y_test)), size=size, replace=False)).astype(np.int64)


def local_example_indices(y_test: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, int]:
    positives = np.flatnonzero(y_test == 1)
    negatives = np.flatnonzero(y_test == 0)
    false_positive_candidates = negatives[scores[negatives] >= threshold]
    false_negative_candidates = positives[scores[positives] < threshold]
    true_negative_candidates = negatives[scores[negatives] < threshold]

    return {
        "high_risk_true_positive": int(positives[np.argmax(scores[positives])]),
        "high_risk_false_positive": int(
            false_positive_candidates[np.argmax(scores[false_positive_candidates])]
            if len(false_positive_candidates)
            else negatives[np.argmax(scores[negatives])]
        ),
        "low_score_false_negative": int(
            false_negative_candidates[np.argmin(scores[false_negative_candidates])]
            if len(false_negative_candidates)
            else positives[np.argmin(scores[positives])]
        ),
        "low_risk_true_negative": int(
            true_negative_candidates[np.argmin(scores[true_negative_candidates])]
            if len(true_negative_candidates)
            else negatives[np.argmin(scores[negatives])]
        ),
    }


def mean_abs_importance(shap_values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    raw = pd.DataFrame(
        {
            "row_type": "raw_transformed_feature",
            "feature": feature_names,
            "feature_group": [feature_group(name) for name in feature_names],
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            "mean_shap": shap_values.mean(axis=0),
        }
    )
    raw["rank_within_type"] = raw["mean_abs_shap"].rank(ascending=False, method="first").astype(int)

    grouped = (
        raw.groupby("feature_group", as_index=False)
        .agg(mean_abs_shap=("mean_abs_shap", "sum"), mean_shap=("mean_shap", "sum"))
        .rename(columns={"feature_group": "feature"})
    )
    grouped["row_type"] = "grouped_feature"
    grouped["feature_group"] = grouped["feature"]
    grouped["rank_within_type"] = grouped["mean_abs_shap"].rank(ascending=False, method="first").astype(int)
    grouped = grouped[raw.columns]
    return pd.concat([raw, grouped], ignore_index=True).sort_values(["row_type", "rank_within_type"])


def contributor_text(shap_row: np.ndarray, value_row: pd.Series, feature_names: list[str], positive: bool) -> str:
    if positive:
        order = np.argsort(-shap_row)
        selected = [idx for idx in order if shap_row[idx] > 0][:LOCAL_TOP_FEATURES]
    else:
        order = np.argsort(shap_row)
        selected = [idx for idx in order if shap_row[idx] < 0][:LOCAL_TOP_FEATURES]
    return "; ".join(
        f"{feature_names[idx]}={value_row.iloc[idx]:.4g} ({shap_row[idx]:+.4f})"
        for idx in selected
    )


def local_examples_table(
    local_frame: pd.DataFrame,
    local_features: pd.DataFrame,
    local_shap_values: np.ndarray,
    local_base_values: np.ndarray,
    local_scores: np.ndarray,
    local_calibrated_scores: np.ndarray,
    raw_threshold: float,
    calibrated_threshold: float,
    example_ids: list[str],
) -> pd.DataFrame:
    feature_names = list(local_features.columns)
    rows = []
    for i, example_id in enumerate(example_ids):
        raw_margin = float(local_base_values[i] + local_shap_values[i].sum())
        rows.append(
            {
                "example_id": example_id,
                "h3_cell_res8": local_frame.iloc[i]["h3_cell_res8"],
                "window_index": int(local_frame.iloc[i]["window_index"]),
                "window_start": local_frame.iloc[i]["window_start"],
                "actual_label": int(local_frame.iloc[i]["actual_label"]),
                "incident_count": int(local_frame.iloc[i]["incident_count"]),
                "raw_xgboost_score": float(local_scores[i]),
                "sigmoid_calibrated_risk": float(local_calibrated_scores[i]),
                "raw_threshold": float(raw_threshold),
                "calibrated_threshold": float(calibrated_threshold),
                "predicted_label": int(local_scores[i] >= raw_threshold),
                "calibrated_predicted_label": int(local_calibrated_scores[i] >= calibrated_threshold),
                "raw_margin_from_shap": raw_margin,
                "probability_from_shap_margin": float(expit(raw_margin)),
                "top_positive_contributors": contributor_text(
                    local_shap_values[i],
                    local_features.iloc[i],
                    feature_names,
                    positive=True,
                ),
                "top_negative_contributors": contributor_text(
                    local_shap_values[i],
                    local_features.iloc[i],
                    feature_names,
                    positive=False,
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_feature_importance(importance: pd.DataFrame) -> None:
    grouped = (
        importance[importance["row_type"] == "grouped_feature"]
        .sort_values("mean_abs_shap", ascending=False)
        .head(15)
        .sort_values("mean_abs_shap", ascending=True)
    )
    plt.figure(figsize=(9, 6))
    plt.barh(grouped["feature"], grouped["mean_abs_shap"], color="#4477aa")
    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("")
    plt.title("SHAP feature importance for final XGBoost model")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "shap_feature_importance_bar.png", dpi=200)
    plt.close()


def plot_beeswarm(explanation, feature_frame: pd.DataFrame) -> None:
    shap.plots.beeswarm(
        shap.Explanation(
            values=explanation.values,
            base_values=explanation.base_values,
            data=feature_frame,
            feature_names=list(feature_frame.columns),
        ),
        max_display=20,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "shap_beeswarm.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_local_waterfalls(local_explanation, example_ids: list[str]) -> None:
    output_names = {
        "high_risk_true_positive": "shap_local_high_risk_tp.png",
        "high_risk_false_positive": "shap_local_false_positive.png",
        "low_score_false_negative": "shap_local_false_negative.png",
        "low_risk_true_negative": "shap_local_true_negative.png",
    }
    for idx, example_id in enumerate(example_ids):
        shap.plots.waterfall(local_explanation[idx], max_display=12, show=False)
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / output_names[example_id], dpi=200, bbox_inches="tight")
        plt.close()


def frame_to_markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for record in frame.astype(str).to_dict(orient="records"):
        rows.append("| " + " | ".join(record[column] for column in columns) + " |")
    return "\n".join(rows)


def write_audit(
    context: FinalModelContext,
    global_indices: np.ndarray,
    background_rows: int,
    importance: pd.DataFrame,
    local_examples: pd.DataFrame,
) -> None:
    top_grouped = (
        importance[importance["row_type"] == "grouped_feature"]
        .sort_values("mean_abs_shap", ascending=False)
        .head(12)
        [["feature", "mean_abs_shap", "mean_shap"]]
        .round(6)
    )
    lines = [
        "# Final SHAP explainability audit",
        "",
        f"- SHAP version: `{shap.__version__}`",
        "- Model: default neighbor-feature XGBoost.",
        f"- H3 resolution: `{fg.H3_RESOLUTION}`",
        f"- Time window hours: `{fg.WINDOW_HOURS}`",
        f"- Inside-Dubai H3 cells: `{fg.EXPECTED_INSIDE_DUBAI_CELLS}`",
        f"- Validation candidate rows: `{len(context.y_val)}`",
        f"- Test candidate rows: `{len(context.y_test)}`",
        f"- Test positive cell/windows: `{int(context.y_test.sum())}`",
        f"- Training rows used after deterministic cap: `{context.train_rows_used}`",
        f"- Raw validation-selected threshold: `{context.raw_threshold:.6f}`",
        f"- Sigmoid-calibrated validation-selected threshold: `{context.calibrated_threshold:.6f}`",
        f"- Background/reference sample rows selected: `{background_rows}`",
        "- Final feature path: current features plus ring-1 inside-Dubai neighbor lag features.",
        "- Rejected hard-negative sample is not used.",
        "- Tuned XGBoost configuration is not used.",
        "- SHAP values are reported on the XGBoost raw-margin scale, not as causal effects.",
        f"- Global explanation sample rows: `{len(global_indices)}`",
        "",
        "## Top grouped SHAP features",
        "",
        frame_to_markdown(top_grouped),
        "",
        "## Local examples",
        "",
        frame_to_markdown(
            local_examples[
                [
                    "example_id",
                    "h3_cell_res8",
                    "window_start",
                    "actual_label",
                    "incident_count",
                    "raw_xgboost_score",
                    "sigmoid_calibrated_risk",
                    "predicted_label",
                ]
            ].round(6)
        ),
    ]
    (AUDIT_DIR / "shap_explainability_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    context = build_final_model_context()

    background_reference = context.train_df.sample(
        n=min(BACKGROUND_SAMPLE_SIZE, len(context.train_df)),
        random_state=RANDOM_SEED,
    )

    global_indices = sample_global_indices(context.y_test, GLOBAL_SAMPLE_SIZE, RANDOM_SEED)
    global_frame = candidate_frame_for_indices(global_indices, context)
    global_transformed = transformed_frame(context.pipeline, global_frame)

    xgb_model = context.pipeline.named_steps["model"]
    explainer = shap.TreeExplainer(
        xgb_model,
        feature_perturbation="tree_path_dependent",
        feature_names=list(global_transformed.columns),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        global_explanation = explainer(global_transformed, check_additivity=False)

    importance = mean_abs_importance(global_explanation.values, list(global_transformed.columns))
    importance.to_csv(TABLE_DIR / "shap_global_feature_importance.csv", index=False)
    plot_feature_importance(importance)
    plot_beeswarm(global_explanation, global_transformed)

    local_index_map = local_example_indices(context.y_test, context.test_scores, context.raw_threshold)
    example_ids = list(local_index_map)
    local_indices = np.asarray([local_index_map[example_id] for example_id in example_ids], dtype=np.int64)
    local_frame = candidate_frame_for_indices(local_indices, context)
    local_transformed = transformed_frame(context.pipeline, local_frame)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        local_explanation = explainer(local_transformed, check_additivity=False)
    local_scores = context.test_scores[local_indices]
    local_calibrated_scores = context.calibrated_test_scores[local_indices]
    local_examples = local_examples_table(
        local_frame,
        local_transformed,
        local_explanation.values,
        np.asarray(local_explanation.base_values),
        local_scores,
        local_calibrated_scores,
        context.raw_threshold,
        context.calibrated_threshold,
        example_ids,
    )
    local_examples.to_csv(TABLE_DIR / "shap_local_examples.csv", index=False)
    plot_local_waterfalls(local_explanation, example_ids)
    write_audit(
        context,
        global_indices,
        len(background_reference),
        importance,
        local_examples,
    )

    print(f"Wrote SHAP importance to {TABLE_DIR / 'shap_global_feature_importance.csv'}")
    print(f"Wrote local examples to {TABLE_DIR / 'shap_local_examples.csv'}")
    print(f"Wrote SHAP audit to {AUDIT_DIR / 'shap_explainability_audit.md'}")


if __name__ == "__main__":
    main()
