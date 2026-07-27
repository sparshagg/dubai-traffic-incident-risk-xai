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

HARD_NEGATIVE_FRACTION = 0.70
RANDOM_NEGATIVE_FRACTION = 1.0 - HARD_NEGATIVE_FRACTION
TRAIN_CAP = fg.XGB_TRAIN_CAP
MINING_CHUNK_WINDOWS = fg.CHUNK_WINDOWS
RANDOM_SEED = fg.RANDOM_SEED


@dataclass
class TrainingSampleSummary:
    sample_id: str
    row_count: int
    positive_rows: int
    negative_rows: int
    hard_negative_rows: int
    random_negative_rows: int
    positive_rate: float
    notes: str


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def pair_ids_to_frame(
    pair_ids: np.ndarray,
    inside_cells: list[str],
    labels: np.ndarray,
    count_cumulative: np.ndarray,
    severity_cumulative: np.ndarray,
    priors: dict[str, np.ndarray | float],
    state: ng.NeighborState,
    role: str,
) -> pd.DataFrame:
    n_cells = len(inside_cells)
    pair_ids = pair_ids.astype(np.int64, copy=False)
    window_index = (pair_ids // n_cells).astype(np.int32)
    cell_index = (pair_ids % n_cells).astype(np.int32)
    starts = fg.START_WINDOW + pd.to_timedelta(window_index * fg.WINDOW_HOURS, unit="h")
    hour_block = (starts.hour // fg.WINDOW_HOURS).astype(np.int8)

    frame = pd.DataFrame(
        {
            "h3_cell_res8": np.asarray(inside_cells, dtype=object)[cell_index],
            "window_index": window_index,
            "window_start": starts.astype(str),
            "geo_scope": pd.Categorical(["inside_dubai"] * len(pair_ids)),
            "risk_label": labels[cell_index, window_index].astype(np.int8),
            "hour_block": hour_block,
            "day_of_week": pd.Categorical(starts.day_name()),
            "is_weekend": np.array([fg.uae_weekend_flag(ts) for ts in starts], dtype=np.int8),
            "month": starts.month.astype(np.int8),
            "year": starts.year.astype(np.int16),
            "prev_3h_incident_count": ng.lag_rows(count_cumulative, cell_index, window_index, 1),
            "prev_24h_incident_count": ng.lag_rows(count_cumulative, cell_index, window_index, 8),
            "prev_7d_incident_count": ng.lag_rows(count_cumulative, cell_index, window_index, 56),
            "prev_24h_severity_weight_sum": ng.lag_rows(severity_cumulative, cell_index, window_index, 8),
            "prev_7d_severity_weight_sum": ng.lag_rows(severity_cumulative, cell_index, window_index, 56),
            "hist_cell_hour_risk": np.asarray(priors["cell_hour"], dtype=np.float32)[cell_index, hour_block],
            "hist_cell_risk": np.asarray(priors["cell"], dtype=np.float32)[cell_index],
            "hist_hour_risk": np.asarray(priors["hour"], dtype=np.float32)[hour_block],
            "hist_global_risk": np.full(len(pair_ids), float(priors["global"]), dtype=np.float32),
            "sample_role": role,
        }
    )
    return ng.add_neighbor_features(frame, state)


def train_positive_pair_ids(labels: np.ndarray, n_cells: int) -> np.ndarray:
    cell_index, window_index = np.nonzero(labels[:, : fg.TRAIN_END_EXCLUSIVE])
    return (window_index.astype(np.int64) * n_cells + cell_index.astype(np.int64)).astype(np.int64)


def update_top_hard(
    current_pair_ids: np.ndarray,
    current_scores: np.ndarray,
    new_pair_ids: np.ndarray,
    new_scores: np.ndarray,
    target: int,
) -> tuple[np.ndarray, np.ndarray]:
    if target <= 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float32)
    if len(new_pair_ids) == 0:
        return current_pair_ids, current_scores
    if len(current_pair_ids) == 0:
        combined_pair_ids = new_pair_ids
        combined_scores = new_scores
    else:
        combined_pair_ids = np.concatenate([current_pair_ids, new_pair_ids]).astype(np.int64, copy=False)
        combined_scores = np.concatenate([current_scores, new_scores]).astype(np.float32, copy=False)
    if len(combined_pair_ids) <= target:
        return combined_pair_ids, combined_scores
    keep = np.argpartition(combined_scores, -target)[-target:]
    kept_scores = combined_scores[keep]
    order = np.argsort(kept_scores)[::-1]
    kept = keep[order]
    return combined_pair_ids[kept], combined_scores[kept]


def mine_hard_negatives(
    miner: object,
    inside_cells: list[str],
    labels: np.ndarray,
    incident_counts: np.ndarray,
    count_cumulative: np.ndarray,
    severity_cumulative: np.ndarray,
    priors: dict[str, np.ndarray | float],
    state: ng.NeighborState,
    target: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    n_cells = len(inside_cells)
    top_pair_ids = np.asarray([], dtype=np.int64)
    top_scores = np.asarray([], dtype=np.float32)
    negatives_scored = 0
    train_windows = np.arange(fg.TRAIN_END_EXCLUSIVE, dtype=np.int32)
    for start in range(0, len(train_windows), MINING_CHUNK_WINDOWS):
        windows = train_windows[start : start + MINING_CHUNK_WINDOWS]
        chunk = fg.make_candidate_chunk(
            inside_cells,
            windows,
            labels,
            incident_counts,
            count_cumulative,
            severity_cumulative,
            priors,
        )
        chunk_frame = ng.add_neighbor_features(chunk.frame, state)
        scores = miner.predict_proba(chunk_frame[ng.NEIGHBOR_FEATURE_COLUMNS])[:, 1].astype(np.float32)
        negative_index = np.flatnonzero(chunk.labels == 0)
        negatives_scored += len(negative_index)
        if len(negative_index) == 0:
            continue
        local_scores = scores[negative_index]
        local_take = min(target, len(local_scores))
        if local_take < len(local_scores):
            local_keep = np.argpartition(local_scores, -local_take)[-local_take:]
        else:
            local_keep = np.arange(len(local_scores))
        chunk_windows = np.repeat(windows, n_cells)
        chunk_cells = np.tile(np.arange(n_cells, dtype=np.int32), len(windows))
        selected_rows = negative_index[local_keep]
        local_pair_ids = (chunk_windows[selected_rows].astype(np.int64) * n_cells) + chunk_cells[selected_rows].astype(np.int64)
        top_pair_ids, top_scores = update_top_hard(
            top_pair_ids,
            top_scores,
            local_pair_ids.astype(np.int64),
            local_scores[local_keep].astype(np.float32),
            target,
        )
    if len(top_pair_ids) != target:
        raise SystemExit(f"Expected {target} hard negatives, got {len(top_pair_ids)}")
    if len(set(top_pair_ids.tolist())) != len(top_pair_ids):
        raise SystemExit("Hard-negative mining produced duplicate cell/window pairs")
    return top_pair_ids, top_scores, negatives_scored


def sample_random_negative_pair_ids(
    labels: np.ndarray,
    n_cells: int,
    target: int,
    excluded_pair_ids: set[int],
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected: set[int] = set()
    max_pair_id = n_cells * fg.TRAIN_END_EXCLUSIVE
    while len(selected) < target:
        need = target - len(selected)
        candidates = rng.integers(0, max_pair_id, size=max(need * 3, 50_000), dtype=np.int64)
        for pair_id in candidates:
            pair = int(pair_id)
            if pair in selected or pair in excluded_pair_ids:
                continue
            window = pair // n_cells
            cell = pair % n_cells
            if labels[cell, window] != 0:
                continue
            selected.add(pair)
            if len(selected) == target:
                break
    return np.fromiter(sorted(selected), dtype=np.int64)


def validate_training_sample(
    hard_train: pd.DataFrame,
    positive_pair_ids: np.ndarray,
    hard_pair_ids: np.ndarray,
    random_pair_ids: np.ndarray,
    n_cells: int,
) -> None:
    expected_rows = len(positive_pair_ids) + len(hard_pair_ids) + len(random_pair_ids)
    if len(hard_train) != expected_rows:
        raise SystemExit(f"Hard training sample has {len(hard_train)} rows, expected {expected_rows}")
    if len(hard_train) != TRAIN_CAP:
        raise SystemExit(f"Hard training sample has {len(hard_train)} rows, expected cap {TRAIN_CAP}")
    pair_tuples = set(zip(hard_train["h3_cell_res8"].astype(str), hard_train["window_index"].astype(int)))
    if len(pair_tuples) != len(hard_train):
        raise SystemExit("Hard training sample contains duplicate cell/window rows")
    positive_rows = hard_train[hard_train["sample_role"] == "positive"]
    hard_rows = hard_train[hard_train["sample_role"] == "hard_negative"]
    random_rows = hard_train[hard_train["sample_role"] == "random_negative"]
    if len(positive_rows) != len(positive_pair_ids):
        raise SystemExit("Hard training sample does not include all training positives")
    if not (positive_rows["risk_label"] == 1).all():
        raise SystemExit("A positive sample row has risk_label != 1")
    if not (hard_rows["risk_label"] == 0).all():
        raise SystemExit("A hard-negative sample row has risk_label != 0")
    if not (random_rows["risk_label"] == 0).all():
        raise SystemExit("A random-negative sample row has risk_label != 0")
    if len(hard_pair_ids) != len(set(hard_pair_ids.tolist())):
        raise SystemExit("Hard-negative pair IDs contain duplicates")
    if len(random_pair_ids) != len(set(random_pair_ids.tolist())):
        raise SystemExit("Random-negative pair IDs contain duplicates")
    if set(hard_pair_ids.tolist()).intersection(set(random_pair_ids.tolist())):
        raise SystemExit("Hard and random negative samples overlap")
    leakage = set(ng.NEIGHBOR_FEATURE_COLUMNS).intersection(fg.LEAKAGE_COLUMNS)
    if leakage:
        raise SystemExit(f"Leakage columns found in model features: {sorted(leakage)}")


def build_sample_summary(
    random_train: pd.DataFrame,
    random_indices: np.ndarray,
    hard_train: pd.DataFrame,
    hard_negative_count: int,
    random_negative_count: int,
    negatives_scored: int,
) -> pd.DataFrame:
    random_y = random_train["risk_label"].astype(int).iloc[random_indices]
    rows = [
        TrainingSampleSummary(
            sample_id="current_random_negative_sample",
            row_count=len(random_indices),
            positive_rows=int(random_y.sum()),
            negative_rows=int(len(random_y) - int(random_y.sum())),
            hard_negative_rows=0,
            random_negative_rows=int(len(random_y) - int(random_y.sum())),
            positive_rate=float(random_y.mean()),
            notes="Existing sampled training table with deterministic 1,000,000-row cap.",
        ),
        TrainingSampleSummary(
            sample_id="hard_negative_sample",
            row_count=len(hard_train),
            positive_rows=int((hard_train["risk_label"] == 1).sum()),
            negative_rows=int((hard_train["risk_label"] == 0).sum()),
            hard_negative_rows=hard_negative_count,
            random_negative_rows=random_negative_count,
            positive_rate=float(hard_train["risk_label"].mean()),
            notes=f"All training positives plus 70/30 hard/random negatives; mined from {negatives_scored:,} training negatives.",
        ),
    ]
    return pd.DataFrame([row.__dict__ for row in rows])


def plot_model_comparison(metrics: pd.DataFrame) -> None:
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
    plt.title("Hard-negative training experiment")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "hard_negative_training_model_comparison.png", dpi=200)
    plt.close()


def plot_top_k(top_k: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 5.5))
    sns.lineplot(data=top_k, x="k", y="precision_at_k", hue="model", marker="o", linewidth=2)
    plt.xlabel("Top-k cells per 3-hour window")
    plt.ylabel("Precision at k")
    plt.title("Hard-negative training top-k hotspot precision")
    plt.ylim(0, max(0.05, float(top_k["precision_at_k"].max()) * 1.15))
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "hard_negative_training_topk_comparison.png", dpi=200)
    plt.close()


def write_audit(
    metrics: pd.DataFrame,
    top_k: pd.DataFrame,
    sample_summary: pd.DataFrame,
    neighbor_stats: dict[str, int],
) -> None:
    random = metrics.loc[metrics["model"] == "XGBoost neighbor random sample"].iloc[0]
    hard = metrics.loc[metrics["model"] == "XGBoost neighbor hard negatives"].iloc[0]
    top20 = top_k[top_k["k"] == 20]
    random_top20 = top20.loc[top20["model"] == "XGBoost neighbor random sample"].iloc[0]
    hard_top20 = top20.loc[top20["model"] == "XGBoost neighbor hard negatives"].iloc[0]
    audit = [
        "# Hard-negative training sample experiment",
        "",
        f"- H3 resolution: `{fg.H3_RESOLUTION}`",
        f"- Time window hours: `{fg.WINDOW_HOURS}`",
        f"- Inside-Dubai cells: `{fg.EXPECTED_INSIDE_DUBAI_CELLS}`",
        f"- Training windows: `0` to `{fg.TRAIN_END_EXCLUSIVE - 1}`",
        f"- Validation windows: `{fg.TRAIN_END_EXCLUSIVE}` to `{fg.VALIDATION_END_EXCLUSIVE - 1}`",
        f"- Test windows: `{fg.VALIDATION_END_EXCLUSIVE}` to `{fg.WINDOW_COUNT - 1}`",
        f"- Validation candidates: `{fg.EXPECTED_VALIDATION_CANDIDATES}`",
        f"- Test candidates: `{fg.EXPECTED_TEST_CANDIDATES}`",
        f"- Test positive cell/windows: `{fg.EXPECTED_TEST_POSITIVE_CELL_WINDOWS}`",
        f"- Hard-negative training cap: `{TRAIN_CAP}`",
        f"- Negative mix after positives: `{HARD_NEGATIVE_FRACTION:.0%}` hard negatives and `{RANDOM_NEGATIVE_FRACTION:.0%}` random negatives",
        f"- Inside-neighbor count range: `{neighbor_stats['min_inside_neighbor_count']}` to `{neighbor_stats['max_inside_neighbor_count']}`",
        "- Hard negatives are mined only from training-window full-grid negatives.",
        "- Validation and test remain full-grid and unchanged.",
        "- Unknown-severity incidents remain in incident counts, but unknown severity contributes zero to severity-weighted sums.",
        f"- PR-AUC change vs random-sample neighbor XGBoost: `{hard['test_pr_auc'] - random['test_pr_auc']:.6f}`",
        f"- F1 change vs random-sample neighbor XGBoost: `{hard['test_f1'] - random['test_f1']:.6f}`",
        f"- Top-20 precision change vs random-sample neighbor XGBoost: `{hard_top20['precision_at_k'] - random_top20['precision_at_k']:.6f}`",
        "",
        "## Training samples",
        "",
        ng.frame_to_markdown(sample_summary.round(6)),
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
    (AUDIT_DIR / "hard_negative_training_audit.md").write_text("\n".join(audit), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    fg.validate_inputs([fg.MODEL_SAMPLE_PATH, fg.POSITIVE_COUNTS_PATH, fg.CELL_SCOPE_PATH])
    inside_cells = fg.load_inside_cells(fg.CELL_SCOPE_PATH)
    positive = fg.load_inside_positive_counts(fg.POSITIVE_COUNTS_PATH, inside_cells)
    labels, incident_counts, severity_weights = fg.build_dense_matrices(positive, inside_cells)
    priors = fg.historical_priors(labels)
    state = ng.build_neighbor_state(inside_cells, labels, incident_counts, severity_weights)
    neighbor_stats = ng.validate_neighbor_state(state, labels, incident_counts, severity_weights)
    n_cells = len(inside_cells)

    count_cumulative = np.concatenate(
        [np.zeros((labels.shape[0], 1), dtype=np.int32), np.cumsum(incident_counts.astype(np.int32), axis=1)],
        axis=1,
    )
    severity_cumulative = np.concatenate(
        [np.zeros((labels.shape[0], 1), dtype=np.int32), np.cumsum(severity_weights.astype(np.int32), axis=1)],
        axis=1,
    )

    random_train = fg.load_training_sample(fg.MODEL_SAMPLE_PATH, inside_cells, priors)
    random_train = ng.add_neighbor_features(random_train, state)
    random_y = random_train["risk_label"].astype(int)
    random_indices = fg.sample_training_rows(random_y, TRAIN_CAP, RANDOM_SEED + 1)

    random_estimator, random_rows = ng.train_xgboost(
        random_train,
        ng.NEIGHBOR_FEATURE_COLUMNS,
        random_indices,
        RANDOM_SEED,
    )

    positive_pair_ids = train_positive_pair_ids(labels, n_cells)
    if len(positive_pair_ids) >= TRAIN_CAP:
        raise SystemExit(
            f"Training positives alone ({len(positive_pair_ids):,}) exceed or equal the training cap ({TRAIN_CAP:,})"
        )
    negative_budget = TRAIN_CAP - len(positive_pair_ids)
    hard_negative_target = int(round(negative_budget * HARD_NEGATIVE_FRACTION))
    random_negative_target = negative_budget - hard_negative_target

    hard_pair_ids, _hard_scores, negatives_scored = mine_hard_negatives(
        random_estimator,
        inside_cells,
        labels,
        incident_counts,
        count_cumulative,
        severity_cumulative,
        priors,
        state,
        hard_negative_target,
    )
    excluded = set(positive_pair_ids.tolist()).union(set(hard_pair_ids.tolist()))
    random_pair_ids = sample_random_negative_pair_ids(
        labels,
        n_cells,
        random_negative_target,
        excluded,
        RANDOM_SEED + 202,
    )

    positive_frame = pair_ids_to_frame(
        positive_pair_ids,
        inside_cells,
        labels,
        count_cumulative,
        severity_cumulative,
        priors,
        state,
        "positive",
    )
    hard_frame = pair_ids_to_frame(
        hard_pair_ids,
        inside_cells,
        labels,
        count_cumulative,
        severity_cumulative,
        priors,
        state,
        "hard_negative",
    )
    random_negative_frame = pair_ids_to_frame(
        random_pair_ids,
        inside_cells,
        labels,
        count_cumulative,
        severity_cumulative,
        priors,
        state,
        "random_negative",
    )
    hard_train = pd.concat([positive_frame, hard_frame, random_negative_frame], ignore_index=True)
    hard_train = hard_train.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    validate_training_sample(hard_train, positive_pair_ids, hard_pair_ids, random_pair_ids, n_cells)

    hard_indices = np.arange(len(hard_train), dtype=np.int64)
    hard_estimator, hard_rows = ng.train_xgboost(
        hard_train,
        ng.NEIGHBOR_FEATURE_COLUMNS,
        hard_indices,
        RANDOM_SEED,
    )

    models = [
        ng.ExperimentModel(
            name="Historical risk",
            feature_set_id="historical_score_only",
            notes="Train-period full-grid empirical risk by inside-Dubai H3 cell and hour block.",
            feature_columns=["hist_cell_hour_risk"],
            train_rows_used=len(random_train),
            estimator=None,
        ),
        ng.ExperimentModel(
            name="XGBoost neighbor random sample",
            feature_set_id="neighbor_lag_random_negative_training",
            notes="Neighbor-feature XGBoost trained on the existing sampled-negative training table.",
            feature_columns=ng.NEIGHBOR_FEATURE_COLUMNS,
            train_rows_used=random_rows,
            estimator=random_estimator,
        ),
        ng.ExperimentModel(
            name="XGBoost neighbor hard negatives",
            feature_set_id="neighbor_lag_hard_negative_training",
            notes="Neighbor-feature XGBoost trained on all positives plus 70/30 hard/random negatives from training windows.",
            feature_columns=ng.NEIGHBOR_FEATURE_COLUMNS,
            train_rows_used=hard_rows,
            estimator=hard_estimator,
        ),
    ]

    validation_windows = np.arange(fg.TRAIN_END_EXCLUSIVE, fg.VALIDATION_END_EXCLUSIVE, dtype=np.int32)
    test_windows = np.arange(fg.VALIDATION_END_EXCLUSIVE, fg.WINDOW_COUNT, dtype=np.int32)
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
    metrics = ng.metrics_frame(results, len(y_val), len(y_test))
    top_k = fg.full_grid_top_k_metrics(results, y_test, test_incidents, len(test_windows), len(inside_cells))
    sample_summary = build_sample_summary(
        random_train,
        random_indices,
        hard_train,
        len(hard_pair_ids),
        len(random_pair_ids),
        negatives_scored,
    )

    metrics.to_csv(TABLE_DIR / "hard_negative_training_metrics.csv", index=False)
    top_k.to_csv(TABLE_DIR / "hard_negative_training_topk.csv", index=False)
    sample_summary.to_csv(TABLE_DIR / "hard_negative_training_sample_summary.csv", index=False)
    plot_model_comparison(metrics)
    plot_top_k(top_k)
    write_audit(metrics, top_k, sample_summary, neighbor_stats)

    print(f"Wrote {TABLE_DIR / 'hard_negative_training_metrics.csv'}")
    print(f"Wrote {TABLE_DIR / 'hard_negative_training_topk.csv'}")
    print(f"Wrote {TABLE_DIR / 'hard_negative_training_sample_summary.csv'}")
    print(f"Wrote {FIGURE_DIR / 'hard_negative_training_model_comparison.png'}")
    print(f"Wrote {FIGURE_DIR / 'hard_negative_training_topk_comparison.png'}")
    print(f"Wrote {AUDIT_DIR / 'hard_negative_training_audit.md'}")


if __name__ == "__main__":
    main()
