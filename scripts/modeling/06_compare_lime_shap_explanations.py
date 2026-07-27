from __future__ import annotations

import importlib.util
import os
import sys
import warnings
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
from lime.lime_tabular import LimeTabularExplainer


ROOT = Path(__file__).resolve().parents[2]
SHAP_SCRIPT = ROOT / "scripts" / "modeling" / "04_generate_shap_explanations.py"
TABLE_DIR = ROOT / "reports" / "modeling" / "tables"
FIGURE_DIR = ROOT / "reports" / "modeling" / "figures"
AUDIT_DIR = ROOT / "data" / "audit"

BACKGROUND_SAMPLE_SIZE = 2_000
LIME_TOP_FEATURES = 8
SHAP_TOP_FEATURES = 8
RANDOM_SEED = 42


def load_shap_module():
    spec = importlib.util.spec_from_file_location("final_shap_explanations", SHAP_SCRIPT)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not import {SHAP_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


shap_mod = load_shap_module()
fg = shap_mod.fg


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def signed_shap_rows(example_id: str, shap_values: np.ndarray, features: pd.Series) -> list[dict[str, object]]:
    order = np.argsort(-np.abs(shap_values))[:SHAP_TOP_FEATURES]
    rows: list[dict[str, object]] = []
    for rank, idx in enumerate(order, 1):
        value = float(shap_values[idx])
        rows.append(
            {
                "example_id": example_id,
                "method": "SHAP",
                "rank": rank,
                "feature": features.index[idx],
                "feature_value": float(features.iloc[idx]),
                "contribution": value,
                "direction": "increases_risk" if value > 0 else "decreases_risk",
                "contribution_scale": "xgboost_raw_margin",
            }
        )
    return rows


def lime_rows(example_id: str, explanation) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rank, (feature_text, weight) in enumerate(explanation.as_list(label=1)[:LIME_TOP_FEATURES], 1):
        value = float(weight)
        rows.append(
            {
                "example_id": example_id,
                "method": "LIME",
                "rank": rank,
                "feature": feature_text,
                "feature_value": "",
                "contribution": value,
                "direction": "increases_risk" if value > 0 else "decreases_risk",
                "contribution_scale": "local_probability_surrogate",
            }
        )
    return rows


def token_set(feature_text: str) -> set[str]:
    clean = (
        feature_text.replace("<=", " ")
        .replace(">=", " ")
        .replace("<", " ")
        .replace(">", " ")
        .replace("=", " ")
        .replace("_", " ")
    )
    return {token for token in clean.lower().split() if len(token) >= 3 and not token.replace(".", "", 1).isdigit()}


def feature_overlap(shap_features: list[str], lime_features: list[str]) -> int:
    lime_tokens = [token_set(feature) for feature in lime_features]
    overlap = 0
    for shap_feature in shap_features:
        shap_tokens = token_set(shap_feature)
        if any(shap_tokens and shap_tokens.intersection(tokens) for tokens in lime_tokens):
            overlap += 1
    return overlap


def plot_comparison(contributors: pd.DataFrame) -> None:
    example_ids = list(dict.fromkeys(contributors["example_id"]))
    fig, axes = plt.subplots(len(example_ids), 2, figsize=(14, 3.2 * len(example_ids)))
    if len(example_ids) == 1:
        axes = np.asarray([axes])
    for row_idx, example_id in enumerate(example_ids):
        for col_idx, method in enumerate(["SHAP", "LIME"]):
            ax = axes[row_idx, col_idx]
            subset = contributors[(contributors["example_id"] == example_id) & (contributors["method"] == method)].copy()
            subset = subset.sort_values("contribution")
            colors = ["#444444" if value > 0 else "#999999" for value in subset["contribution"]]
            labels = subset["feature"].astype(str).str.slice(0, 52)
            ax.barh(labels, subset["contribution"], color=colors)
            ax.axvline(0, color="#222222", linewidth=0.8)
            ax.set_title(f"{example_id.replace('_', ' ').title()} - {method}", fontsize=10)
            ax.tick_params(axis="y", labelsize=7)
            ax.tick_params(axis="x", labelsize=8)
            ax.set_xlabel("Contribution", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "lime_shap_local_comparison.png", dpi=200)
    plt.close(fig)


def write_audit(
    summary: pd.DataFrame,
    contributors: pd.DataFrame,
    raw_threshold: float,
    calibrated_threshold: float,
    train_rows_used: int,
) -> None:
    lines = [
        "# Final LIME vs SHAP local explanation comparison audit",
        "",
        f"- Model: default neighbor-feature XGBoost, H3 resolution {fg.H3_RESOLUTION}, {fg.WINDOW_HOURS}-hour windows, inside-Dubai full-grid setup.",
        f"- Training rows used after deterministic cap: `{train_rows_used}`",
        f"- Raw validation-selected threshold: `{raw_threshold:.6f}`",
        f"- Sigmoid-calibrated validation-selected threshold: `{calibrated_threshold:.6f}`",
        f"- LIME background rows: `{BACKGROUND_SAMPLE_SIZE}` maximum, sampled deterministically from final training candidates.",
        f"- Local examples compared: `{len(summary)}`",
        "- SHAP explains the XGBoost raw-margin score.",
        "- LIME explains the local XGBoost class-probability surface using a model-agnostic surrogate.",
        "- Sigmoid-calibrated risk is included for dashboard display but is not treated as a separate explanation method.",
        "- Methods explain the trained model, not causal effects.",
        "- Rejected hard-negative sample and tuned XGBoost configuration are not used.",
        "",
        "## Example summary",
        "",
        "```csv",
        summary.to_csv(index=False).strip(),
        "```",
        "",
        "## Top contributor rows",
        "",
        "```csv",
        contributors.head(80).to_csv(index=False).strip(),
        "```",
    ]
    (AUDIT_DIR / "lime_shap_explanation_comparison_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    context = shap_mod.build_final_model_context()

    example_index_map = shap_mod.local_example_indices(context.y_test, context.test_scores, context.raw_threshold)
    example_ids = list(example_index_map)
    local_indices = np.asarray([example_index_map[example_id] for example_id in example_ids], dtype=np.int64)
    local_frame = shap_mod.candidate_frame_for_indices(local_indices, context)
    local_transformed = shap_mod.transformed_frame(context.pipeline, local_frame)

    train_background = context.train_df.sample(
        n=min(BACKGROUND_SAMPLE_SIZE, len(context.train_df)),
        random_state=RANDOM_SEED,
    )
    train_transformed = shap_mod.transformed_frame(context.pipeline, train_background)
    feature_names = list(train_transformed.columns)
    xgb_model = context.pipeline.named_steps["model"]

    shap_explainer = shap.TreeExplainer(
        xgb_model,
        feature_perturbation="tree_path_dependent",
        feature_names=feature_names,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_explanation = shap_explainer(local_transformed, check_additivity=False)

    lime_explainer = LimeTabularExplainer(
        training_data=train_transformed.to_numpy(dtype=np.float32),
        feature_names=feature_names,
        class_names=["no_incident", "incident"],
        mode="classification",
        discretize_continuous=True,
        random_state=RANDOM_SEED,
    )

    def predict_transformed(matrix: np.ndarray) -> np.ndarray:
        return xgb_model.predict_proba(pd.DataFrame(matrix, columns=feature_names))

    contributor_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for i, example_id in enumerate(example_ids):
        local_vector = local_transformed.iloc[i].to_numpy(dtype=np.float32)
        lime_explanation = lime_explainer.explain_instance(
            data_row=local_vector,
            predict_fn=predict_transformed,
            num_features=LIME_TOP_FEATURES,
            labels=(1,),
        )
        shap_rows = signed_shap_rows(example_id, shap_explanation.values[i], local_transformed.iloc[i])
        lime_contributors = lime_rows(example_id, lime_explanation)
        contributor_rows.extend(shap_rows)
        contributor_rows.extend(lime_contributors)
        top_shap = [row["feature"] for row in shap_rows[:5]]
        top_lime = [row["feature"] for row in lime_contributors[:5]]
        raw_score = float(context.test_scores[local_indices[i]])
        calibrated_risk = float(context.calibrated_test_scores[local_indices[i]])
        summary_rows.append(
            {
                "example_id": example_id,
                "candidate_index": int(local_frame.iloc[i]["candidate_index"]),
                "h3_cell_res8": local_frame.iloc[i]["h3_cell_res8"],
                "window_index": int(local_frame.iloc[i]["window_index"]),
                "window_start": local_frame.iloc[i]["window_start"],
                "actual_label": int(local_frame.iloc[i]["actual_label"]),
                "incident_count": int(local_frame.iloc[i]["incident_count"]),
                "raw_xgboost_score": raw_score,
                "sigmoid_calibrated_risk": calibrated_risk,
                "raw_threshold": float(context.raw_threshold),
                "calibrated_threshold": float(context.calibrated_threshold),
                "predicted_label": int(raw_score >= context.raw_threshold),
                "calibrated_predicted_label": int(calibrated_risk >= context.calibrated_threshold),
                "top5_feature_overlap": feature_overlap(top_shap, top_lime),
                "top_shap_features": "; ".join(top_shap),
                "top_lime_features": "; ".join(top_lime),
            }
        )

    contributors = pd.DataFrame(contributor_rows)
    summary = pd.DataFrame(summary_rows)
    contributors.to_csv(TABLE_DIR / "lime_shap_local_contributors.csv", index=False)
    summary.to_csv(TABLE_DIR / "lime_shap_local_comparison.csv", index=False)
    plot_comparison(contributors)
    write_audit(
        summary,
        contributors,
        context.raw_threshold,
        context.calibrated_threshold,
        context.train_rows_used,
    )

    print(f"Wrote {TABLE_DIR / 'lime_shap_local_contributors.csv'}")
    print(f"Wrote {TABLE_DIR / 'lime_shap_local_comparison.csv'}")
    print(f"Wrote {FIGURE_DIR / 'lime_shap_local_comparison.png'}")
    print(f"Wrote {AUDIT_DIR / 'lime_shap_explanation_comparison_audit.md'}")


if __name__ == "__main__":
    main()
