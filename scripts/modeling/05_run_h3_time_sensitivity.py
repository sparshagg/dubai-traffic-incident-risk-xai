from __future__ import annotations

import json
import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/thesis-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/thesis-cache")

import h3
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from shapely.geometry import Point, shape
from shapely.ops import unary_union
from shapely.prepared import prep
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
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data" / "processed" / "traffic_incidents_eda_ready.csv"
DUBAI_GEOJSON = ROOT / "resources" / "geo" / "dubai.geojson"
TABLE_DIR = ROOT / "reports" / "modeling" / "tables"
FIGURE_DIR = ROOT / "reports" / "modeling" / "figures"
AUDIT_DIR = ROOT / "data" / "audit"

VALID_COORDINATE_STATUSES = {"as_provided_lon_lat", "swapped_lat_lon"}
EXPECTED_TOTAL_ROWS = 720_155
EXPECTED_MAP_USABLE_ROWS = 717_615

RANDOM_SEED = 42
NEGATIVE_RATIO = 5
XGB_TRAIN_CAP = 1_000_000
CHUNK_ROW_TARGET = 1_000_000
TOP_K_VALUES = [5, 10, 20]
REFERENCE_TOP_SHARE = 20 / 1_305

SETTINGS = [
    ("res7_3h", 7, 3),
    ("res8_1h", 8, 1),
    ("res8_3h", 8, 3),
    ("res8_6h", 8, 6),
    ("res9_3h", 9, 3),
]

NUMERIC_FEATURES = [
    "hour_block",
    "is_weekend",
    "month",
    "year",
    "prev_window_incident_count",
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


@dataclass(frozen=True)
class Setting:
    setting_id: str
    resolution: int
    window_hours: int


@dataclass
class SettingData:
    setting: Setting
    inside_cells: list[str]
    start_window: pd.Timestamp
    window_count: int
    train_end: int
    validation_end: int
    positive: pd.DataFrame
    cell_series: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    priors: dict[str, np.ndarray | float]


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def load_geojson_geometry(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("type") == "FeatureCollection":
        features = data.get("features") or []
    elif data.get("type") == "Feature":
        features = [data]
    else:
        features = [{"type": "Feature", "properties": {}, "geometry": data}]
    if not features:
        raise SystemExit(f"No features found in {path}")
    geometries = [shape(feature["geometry"]) for feature in features]
    return geometries[0] if len(geometries) == 1 else unary_union(geometries)


def load_incidents() -> pd.DataFrame:
    usecols = [
        "acci_id",
        "acci_time",
        "include_in_eda",
        "coordinate_status",
        "longitude",
        "latitude",
        "severity_weight",
    ]
    df = pd.read_csv(INPUT_PATH, usecols=usecols, dtype=str)
    if len(df) != EXPECTED_TOTAL_ROWS:
        raise SystemExit(f"Expected {EXPECTED_TOTAL_ROWS} EDA-ready rows, got {len(df)}")
    df = df[(df["include_in_eda"] == "true") & df["coordinate_status"].isin(VALID_COORDINATE_STATUSES)].copy()
    if len(df) != EXPECTED_MAP_USABLE_ROWS:
        raise SystemExit(f"Expected {EXPECTED_MAP_USABLE_ROWS} map-usable rows, got {len(df)}")
    df["longitude"] = df["longitude"].astype(float)
    df["latitude"] = df["latitude"].astype(float)
    df["severity_weight"] = df["severity_weight"].astype(np.int16)
    df["acci_dt"] = pd.to_datetime(df["acci_time"], errors="coerce")
    if df["acci_dt"].isna().any():
        raise SystemExit("Map-usable incident rows contain invalid timestamps")
    return df[["acci_id", "acci_dt", "longitude", "latitude", "severity_weight"]].copy()


def add_dubai_point_flag(df: pd.DataFrame, dubai_geometry: object) -> pd.DataFrame:
    dubai_prepared = prep(dubai_geometry)
    flagged = df.copy()
    flagged["point_inside_dubai"] = [
        dubai_prepared.covers(Point(lon, lat))
        for lon, lat in zip(flagged["longitude"], flagged["latitude"], strict=True)
    ]
    return flagged


def uae_weekend_flag(starts: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    weekdays = starts.weekday
    before_change = starts < pd.Timestamp("2022-01-01")
    old_weekend = np.isin(weekdays, [4, 5])
    new_weekend = np.isin(weekdays, [5, 6])
    return np.where(before_change, old_weekend, new_weekend).astype(np.int8)


def window_features(window_indices: np.ndarray, start_window: pd.Timestamp, window_hours: int) -> pd.DataFrame:
    starts = start_window + pd.to_timedelta(window_indices * window_hours, unit="h")
    return pd.DataFrame(
        {
            "window_index": window_indices.astype(np.int32),
            "window_start": starts.astype(str),
            "hour_block": (starts.hour // window_hours).astype(np.int16),
            "day_of_week": pd.Categorical(starts.day_name()),
            "is_weekend": uae_weekend_flag(starts),
            "month": starts.month.astype(np.int8),
            "year": starts.year.astype(np.int16),
        }
    )


def build_setting_data(base_df: pd.DataFrame, dubai_geometry: object, setting: Setting) -> SettingData:
    polygon_cells = set(h3.geo_to_cells(dubai_geometry.__geo_interface__, setting.resolution))
    observed_inside_cells = {
        h3.latlng_to_cell(lat, lon, setting.resolution)
        for lat, lon in zip(
            base_df.loc[base_df["point_inside_dubai"], "latitude"],
            base_df.loc[base_df["point_inside_dubai"], "longitude"],
            strict=True,
        )
    }
    inside_cells = sorted(polygon_cells | observed_inside_cells)
    if not inside_cells:
        raise SystemExit(f"{setting.setting_id} produced zero inside-Dubai H3 cells")
    if any(h3.get_resolution(cell) != setting.resolution for cell in inside_cells):
        raise SystemExit(f"{setting.setting_id} produced cells at the wrong H3 resolution")
    cell_to_index = {cell: idx for idx, cell in enumerate(inside_cells)}

    df = base_df.copy()
    df["h3_cell"] = [
        h3.latlng_to_cell(lat, lon, setting.resolution)
        for lat, lon in zip(df["latitude"], df["longitude"], strict=True)
    ]
    df = df[df["h3_cell"].isin(cell_to_index)].copy()
    if df.empty:
        raise SystemExit(f"{setting.setting_id} has no incidents inside the Dubai H3 cell universe")
    df["cell_index"] = df["h3_cell"].map(cell_to_index).astype(np.int32)

    start_window = base_df["acci_dt"].min().floor(f"{setting.window_hours}h")
    end_window = base_df["acci_dt"].max().floor(f"{setting.window_hours}h")
    df["window_start"] = df["acci_dt"].dt.floor(f"{setting.window_hours}h")
    df["window_index"] = ((df["window_start"] - start_window) / pd.Timedelta(hours=setting.window_hours)).astype(np.int32)
    window_count = int(((end_window - start_window) / pd.Timedelta(hours=setting.window_hours))) + 1
    train_end = int(window_count * 0.70)
    validation_end = int(window_count * 0.85)
    if not (0 < train_end < validation_end < window_count):
        raise SystemExit(f"{setting.setting_id} produced invalid chronological split")

    positive = (
        df.groupby(["cell_index", "h3_cell", "window_index"], as_index=False)
        .agg(incident_count=("acci_id", "count"), severity_weight_sum=("severity_weight", "sum"))
        .sort_values(["window_index", "cell_index"])
    )
    positive["incident_count"] = positive["incident_count"].astype(np.int16)
    positive["severity_weight_sum"] = positive["severity_weight_sum"].astype(np.int16)
    if positive.duplicated(["cell_index", "window_index"]).any():
        raise SystemExit(f"{setting.setting_id} has duplicate positive cell/window rows")
    if int(positive["window_index"].max()) >= window_count:
        raise SystemExit(f"{setting.setting_id} contains a positive row beyond the configured window count")

    cell_series: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for cell_index, group in positive.groupby("cell_index", sort=False):
        windows = group["window_index"].to_numpy(dtype=np.int32)
        counts = group["incident_count"].to_numpy(dtype=np.int32)
        weights = group["severity_weight_sum"].to_numpy(dtype=np.int32)
        cumulative_counts = np.concatenate([[0], np.cumsum(counts, dtype=np.int32)])
        cumulative_weights = np.concatenate([[0], np.cumsum(weights, dtype=np.int32)])
        cell_series[int(cell_index)] = (windows, cumulative_counts, cumulative_weights, counts)

    priors = historical_priors(positive, len(inside_cells), train_end, start_window, setting.window_hours)
    return SettingData(
        setting=setting,
        inside_cells=inside_cells,
        start_window=start_window,
        window_count=window_count,
        train_end=train_end,
        validation_end=validation_end,
        positive=positive,
        cell_series=cell_series,
        priors=priors,
    )


def historical_priors(
    positive: pd.DataFrame,
    n_cells: int,
    train_end: int,
    start_window: pd.Timestamp,
    window_hours: int,
) -> dict[str, np.ndarray | float]:
    train_positive = positive[positive["window_index"] < train_end].copy()
    hour_count = 24 // window_hours
    global_risk = len(train_positive) / max(n_cells * train_end, 1)
    cell_risk = np.zeros(n_cells, dtype=np.float32)
    cell_counts = train_positive.groupby("cell_index").size()
    if not cell_counts.empty:
        cell_risk[cell_counts.index.to_numpy(dtype=np.int32)] = (cell_counts.to_numpy(dtype=np.float32) / train_end)

    train_windows = np.arange(train_end, dtype=np.int32)
    train_wf = window_features(train_windows, start_window, window_hours)
    hour_by_window = train_wf["hour_block"].to_numpy(dtype=np.int16)
    hour_denominators = np.array([(hour_by_window == hour).sum() for hour in range(hour_count)], dtype=np.int32)
    hour_risk = np.zeros(hour_count, dtype=np.float32)
    cell_hour_risk = np.zeros((n_cells, hour_count), dtype=np.float32)
    if not train_positive.empty:
        train_positive["hour_block"] = hour_by_window[train_positive["window_index"].to_numpy(dtype=np.int32)]
        hour_counts = train_positive.groupby("hour_block").size()
        for hour, count in hour_counts.items():
            denominator = n_cells * max(int(hour_denominators[int(hour)]), 1)
            hour_risk[int(hour)] = np.float32(count / denominator)
        cell_hour_counts = train_positive.groupby(["cell_index", "hour_block"]).size().reset_index(name="count")
        for row in cell_hour_counts.itertuples(index=False):
            denominator = max(int(hour_denominators[int(row.hour_block)]), 1)
            cell_hour_risk[int(row.cell_index), int(row.hour_block)] = np.float32(row.count / denominator)
    return {
        "global": float(global_risk),
        "cell": cell_risk,
        "hour": hour_risk,
        "cell_hour": cell_hour_risk,
    }


def lag_lengths(window_hours: int) -> dict[str, int]:
    return {
        "prev_window": 1,
        "prev_24h": max(1, 24 // window_hours),
        "prev_7d": max(1, 168 // window_hours),
    }


def add_lag_features(frame: pd.DataFrame, data: SettingData) -> pd.DataFrame:
    frame = frame.copy()
    cell_index = frame["cell_index"].to_numpy(dtype=np.int32)
    window_index = frame["window_index"].to_numpy(dtype=np.int32)
    lengths = lag_lengths(data.setting.window_hours)
    count_outputs = {
        "prev_window_incident_count": np.zeros(len(frame), dtype=np.int16),
        "prev_24h_incident_count": np.zeros(len(frame), dtype=np.int16),
        "prev_7d_incident_count": np.zeros(len(frame), dtype=np.int16),
    }
    weight_outputs = {
        "prev_24h_severity_weight_sum": np.zeros(len(frame), dtype=np.int16),
        "prev_7d_severity_weight_sum": np.zeros(len(frame), dtype=np.int16),
    }
    for cell in np.unique(cell_index):
        positions = np.flatnonzero(cell_index == cell)
        series = data.cell_series.get(int(cell))
        if series is None:
            continue
        windows, cumulative_counts, cumulative_weights, _ = series
        row_windows = window_index[positions]
        for output, length in [
            ("prev_window_incident_count", lengths["prev_window"]),
            ("prev_24h_incident_count", lengths["prev_24h"]),
            ("prev_7d_incident_count", lengths["prev_7d"]),
        ]:
            start = np.searchsorted(windows, row_windows - length, side="left")
            end = np.searchsorted(windows, row_windows, side="left")
            count_outputs[output][positions] = (cumulative_counts[end] - cumulative_counts[start]).astype(np.int16)
        for output, length in [
            ("prev_24h_severity_weight_sum", lengths["prev_24h"]),
            ("prev_7d_severity_weight_sum", lengths["prev_7d"]),
        ]:
            start = np.searchsorted(windows, row_windows - length, side="left")
            end = np.searchsorted(windows, row_windows, side="left")
            weight_outputs[output][positions] = (cumulative_weights[end] - cumulative_weights[start]).astype(np.int16)

    for output, values in count_outputs.items():
        frame[output] = values
    for output, values in weight_outputs.items():
        frame[output] = values
    return frame


def base_candidate_frame(cell_indices: np.ndarray, window_indices: np.ndarray, data: SettingData) -> pd.DataFrame:
    wf = window_features(window_indices, data.start_window, data.setting.window_hours)
    frame = pd.DataFrame(
        {
            "cell_index": cell_indices.astype(np.int32),
            "h3_cell": np.asarray(data.inside_cells, dtype=object)[cell_indices],
            "window_index": window_indices.astype(np.int32),
            "window_start": wf["window_start"].astype(str).to_numpy(),
            "geo_scope": pd.Categorical(["inside_dubai"] * len(window_indices)),
            "hour_block": wf["hour_block"].to_numpy(dtype=np.int16),
            "day_of_week": pd.Categorical(wf["day_of_week"].astype(str).to_numpy()),
            "is_weekend": wf["is_weekend"].to_numpy(dtype=np.int8),
            "month": wf["month"].to_numpy(dtype=np.int8),
            "year": wf["year"].to_numpy(dtype=np.int16),
        }
    )
    priors = data.priors
    hour = frame["hour_block"].to_numpy(dtype=np.int16)
    frame["hist_cell_hour_risk"] = np.asarray(priors["cell_hour"], dtype=np.float32)[cell_indices, hour]
    frame["hist_cell_risk"] = np.asarray(priors["cell"], dtype=np.float32)[cell_indices]
    frame["hist_hour_risk"] = np.asarray(priors["hour"], dtype=np.float32)[hour]
    frame["hist_global_risk"] = np.float32(float(priors["global"]))
    return add_lag_features(frame, data)


def positive_encoded(positive: pd.DataFrame, n_cells: int, window_upper: int | None = None) -> np.ndarray:
    frame = positive if window_upper is None else positive[positive["window_index"] < window_upper]
    return (
        frame["window_index"].to_numpy(dtype=np.int64) * n_cells
        + frame["cell_index"].to_numpy(dtype=np.int64)
    )


def sample_negative_encoded(
    n_cells: int,
    train_end: int,
    positive_sorted: np.ndarray,
    target: int,
    seed: int,
) -> np.ndarray:
    max_candidates = n_cells * train_end
    target = min(target, max_candidates - len(positive_sorted))
    rng = np.random.default_rng(seed)
    selected: set[int] = set()
    while len(selected) < target:
        remaining = target - len(selected)
        batch_size = max(remaining * 3, 50_000)
        candidates = rng.integers(0, max_candidates, size=batch_size, dtype=np.int64)
        candidates = np.unique(candidates)
        candidates = candidates[~np.isin(candidates, positive_sorted, assume_unique=False)]
        for value in candidates:
            selected.add(int(value))
            if len(selected) >= target:
                break
    return np.asarray(sorted(selected), dtype=np.int64)


def build_training_frame(data: SettingData, seed: int) -> tuple[pd.DataFrame, dict[str, int]]:
    n_cells = len(data.inside_cells)
    train_positive = positive_encoded(data.positive, n_cells, data.train_end)
    train_positive = np.unique(train_positive)
    max_negative = n_cells * data.train_end - len(train_positive)
    source_negative_count = min(len(train_positive) * NEGATIVE_RATIO, max_negative)
    source_rows = len(train_positive) + source_negative_count
    if source_rows <= XGB_TRAIN_CAP:
        fit_positive_count = len(train_positive)
        fit_negative_count = source_negative_count
    else:
        fit_positive_count = min(len(train_positive), max(1, int(XGB_TRAIN_CAP * len(train_positive) / source_rows)))
        fit_negative_count = XGB_TRAIN_CAP - fit_positive_count

    rng = np.random.default_rng(seed)
    if fit_positive_count < len(train_positive):
        positive_fit = np.sort(rng.choice(train_positive, size=fit_positive_count, replace=False))
    else:
        positive_fit = train_positive
    negative_fit = sample_negative_encoded(
        n_cells,
        data.train_end,
        np.sort(train_positive),
        fit_negative_count,
        seed + 1000,
    )

    encoded = np.concatenate([positive_fit, negative_fit])
    labels = np.concatenate(
        [
            np.ones(len(positive_fit), dtype=np.int8),
            np.zeros(len(negative_fit), dtype=np.int8),
        ]
    )
    order = rng.permutation(len(encoded))
    encoded = encoded[order]
    labels = labels[order]
    window_indices = (encoded // n_cells).astype(np.int32)
    cell_indices = (encoded % n_cells).astype(np.int32)
    train = base_candidate_frame(cell_indices, window_indices, data)
    train["risk_label"] = labels
    info = {
        "train_positive_source_rows": int(len(train_positive)),
        "train_negative_source_rows": int(source_negative_count),
        "train_source_rows": int(source_rows),
        "train_positive_fit_rows": int(len(positive_fit)),
        "train_negative_fit_rows": int(len(negative_fit)),
        "train_fit_rows": int(len(train)),
    }
    return train, info


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def train_xgboost(train: pd.DataFrame, seed: int) -> Pipeline:
    y_train = train["risk_label"].astype(int)
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    model = XGBClassifier(
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
    pipeline = Pipeline(steps=[("preprocess", build_preprocessor()), ("model", model)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.fit(train[FEATURE_COLUMNS], y_train)
    return pipeline


def make_full_grid_chunk(data: SettingData, windows: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    n_cells = len(data.inside_cells)
    cell_indices = np.tile(np.arange(n_cells, dtype=np.int32), len(windows))
    window_indices = np.repeat(windows.astype(np.int32), n_cells)
    frame = base_candidate_frame(cell_indices, window_indices, data)
    y = np.zeros(len(frame), dtype=np.uint8)
    incidents = np.zeros(len(frame), dtype=np.int16)
    start = int(windows[0])
    end = int(windows[-1])
    positive = data.positive[(data.positive["window_index"] >= start) & (data.positive["window_index"] <= end)]
    if not positive.empty:
        offsets = (positive["window_index"].to_numpy(dtype=np.int32) - start) * n_cells
        positions = offsets + positive["cell_index"].to_numpy(dtype=np.int32)
        y[positions] = 1
        incidents[positions] = positive["incident_count"].to_numpy(dtype=np.int16)
    return frame, y, incidents


def score_split(data: SettingData, pipeline: Pipeline, start_window: int, end_window: int) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    windows = np.arange(start_window, end_window, dtype=np.int32)
    n_cells = len(data.inside_cells)
    chunk_windows = max(1, min(256, CHUNK_ROW_TARGET // n_cells))
    y_parts: list[np.ndarray] = []
    incident_parts: list[np.ndarray] = []
    historical_parts: list[np.ndarray] = []
    xgb_parts: list[np.ndarray] = []
    for start in range(0, len(windows), chunk_windows):
        chunk_windows_array = windows[start : start + chunk_windows]
        frame, y, incidents = make_full_grid_chunk(data, chunk_windows_array)
        y_parts.append(y)
        incident_parts.append(incidents)
        historical_parts.append(frame["hist_cell_hour_risk"].to_numpy(dtype=np.float32))
        xgb_parts.append(pipeline.predict_proba(frame[FEATURE_COLUMNS])[:, 1].astype(np.float32))
    return (
        np.concatenate(y_parts).astype(np.uint8),
        np.concatenate(incident_parts).astype(np.int16),
        {
            "Historical risk": np.concatenate(historical_parts).astype(np.float32),
            "XGBoost": np.concatenate(xgb_parts).astype(np.float32),
        },
    )


def threshold_by_validation_f1(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5, 0.0
    f1_values = (2 * precision[:-1] * recall[:-1]) / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best_index = int(np.nanargmax(f1_values))
    return float(thresholds[best_index]), float(f1_values[best_index])


def evaluate_setting(
    data: SettingData,
    train_info: dict[str, int],
    y_val: np.ndarray,
    val_scores: dict[str, np.ndarray],
    y_test: np.ndarray,
    test_scores: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    test_positive_rate = float(y_test.mean())
    for model_name, scores in test_scores.items():
        threshold, validation_f1 = threshold_by_validation_f1(y_val, val_scores[model_name])
        prediction = (scores >= threshold).astype(np.uint8)
        tn, fp, fn, tp = confusion_matrix(y_test, prediction, labels=[0, 1]).ravel()
        test_pr_auc = float(average_precision_score(y_test, scores))
        rows.append(
            {
                "setting_id": data.setting.setting_id,
                "h3_resolution": data.setting.resolution,
                "window_hours": data.setting.window_hours,
                "model": model_name,
                "train_rows_used": train_info["train_fit_rows"] if model_name == "XGBoost" else train_info["train_source_rows"],
                "threshold": threshold,
                "validation_f1": validation_f1,
                "validation_pr_auc": float(average_precision_score(y_val, val_scores[model_name])),
                "test_roc_auc": float(roc_auc_score(y_test, scores)),
                "test_pr_auc": test_pr_auc,
                "test_positive_rate": test_positive_rate,
                "test_pr_auc_lift_vs_base_rate": test_pr_auc / max(test_positive_rate, 1e-12),
                "test_precision": float(precision_score(y_test, prediction, zero_division=0)),
                "test_recall": float(recall_score(y_test, prediction, zero_division=0)),
                "test_f1": float(f1_score(y_test, prediction, zero_division=0)),
                "test_tn": int(tn),
                "test_fp": int(fp),
                "test_fn": int(fn),
                "test_tp": int(tp),
            }
        )
    return pd.DataFrame(rows)


def top_k_metrics(data: SettingData, y_test: np.ndarray, incidents: np.ndarray, test_scores: dict[str, np.ndarray]) -> pd.DataFrame:
    n_cells = len(data.inside_cells)
    n_windows = data.window_count - data.validation_end
    y_matrix = y_test.reshape(n_windows, n_cells)
    incident_matrix = incidents.reshape(n_windows, n_cells)
    total_positive = int(y_matrix.sum())
    total_incidents = int(incident_matrix.sum())
    positive_by_window = y_matrix.sum(axis=1)
    positive_window_mask = positive_by_window > 0
    normalized_k = max(1, min(n_cells, int(round(REFERENCE_TOP_SHARE * n_cells))))
    k_specs = [("fixed", k) for k in TOP_K_VALUES] + [("top_share_1p5pct", normalized_k)]
    rows = []
    for model_name, scores in test_scores.items():
        score_matrix = scores.reshape(n_windows, n_cells)
        for metric_type, k in k_specs:
            k = min(k, n_cells)
            top_indices = np.argpartition(score_matrix, -k, axis=1)[:, -k:]
            captured_labels = np.take_along_axis(y_matrix, top_indices, axis=1)
            captured_incidents = np.take_along_axis(incident_matrix, top_indices, axis=1)
            captured_by_window = captured_labels.sum(axis=1)
            positive_captured = int(captured_by_window.sum())
            incidents_captured = int(captured_incidents.sum())
            window_recall = np.divide(
                captured_by_window[positive_window_mask],
                positive_by_window[positive_window_mask],
                out=np.zeros(int(positive_window_mask.sum()), dtype=float),
                where=positive_by_window[positive_window_mask] > 0,
            )
            rows.append(
                {
                    "setting_id": data.setting.setting_id,
                    "h3_resolution": data.setting.resolution,
                    "window_hours": data.setting.window_hours,
                    "model": model_name,
                    "metric_type": metric_type,
                    "k": int(k),
                    "top_cell_share": float(k / n_cells),
                    "test_positive_rate": total_positive / max(n_windows * n_cells, 1),
                    "evaluated_windows": int(n_windows),
                    "windows_with_positives": int(positive_window_mask.sum()),
                    "total_positive_cell_windows": total_positive,
                    "positive_cell_windows_captured": positive_captured,
                    "recall_at_k": positive_captured / max(total_positive, 1),
                    "mean_window_recall_at_k": float(window_recall.mean()) if len(window_recall) else 0.0,
                    "precision_at_k": positive_captured / max(n_windows * k, 1),
                    "precision_lift_vs_base_rate": (
                        positive_captured / max(n_windows * k, 1)
                    )
                    / max(total_positive / max(n_windows * n_cells, 1), 1e-12),
                    "positive_window_hit_rate_at_k": float((captured_by_window[positive_window_mask] > 0).mean())
                    if positive_window_mask.any()
                    else 0.0,
                    "total_incidents": total_incidents,
                    "incidents_captured": incidents_captured,
                    "incident_recall_at_k": incidents_captured / max(total_incidents, 1),
                }
            )
    return pd.DataFrame(rows)


def population_rows(data: SettingData, train_info: dict[str, int]) -> list[dict[str, object]]:
    n_cells = len(data.inside_cells)
    rows = []
    splits = [
        ("train", 0, data.train_end),
        ("validation", data.train_end, data.validation_end),
        ("test", data.validation_end, data.window_count),
    ]
    for split, start, end in splits:
        positives = data.positive[(data.positive["window_index"] >= start) & (data.positive["window_index"] < end)]
        candidates = (end - start) * n_cells
        rows.append(
            {
                "setting_id": data.setting.setting_id,
                "h3_resolution": data.setting.resolution,
                "window_hours": data.setting.window_hours,
                "split": split,
                "inside_dubai_cells": n_cells,
                "window_start_index": start,
                "window_end_exclusive": end,
                "windows": end - start,
                "full_grid_candidate_rows": candidates,
                "positive_cell_windows": int(len(positives)),
                "positive_rate": len(positives) / max(candidates, 1),
                "incident_count": int(positives["incident_count"].sum()) if not positives.empty else 0,
                "train_positive_source_rows": train_info["train_positive_source_rows"] if split == "train" else 0,
                "train_negative_source_rows": train_info["train_negative_source_rows"] if split == "train" else 0,
                "train_fit_rows": train_info["train_fit_rows"] if split == "train" else 0,
                "min_window_start": str(data.start_window + pd.Timedelta(hours=data.setting.window_hours * start)),
                "max_window_start": str(data.start_window + pd.Timedelta(hours=data.setting.window_hours * (end - 1))),
            }
        )
    return rows


def plot_outputs(metrics: pd.DataFrame, top_k: pd.DataFrame, population: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    xgb = metrics[metrics["model"] == "XGBoost"].copy()
    order = [setting_id for setting_id, _, _ in SETTINGS]

    plt.figure(figsize=(9, 5.5))
    sns.barplot(data=xgb, x="setting_id", y="test_pr_auc", order=order, color="#4477aa")
    plt.xlabel("Sensitivity setting")
    plt.ylabel("Full-grid test PR-AUC")
    plt.title("XGBoost PR-AUC across H3/time-window settings")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "h3_time_sensitivity_pr_auc.png", dpi=200)
    plt.close()

    top_share = top_k[(top_k["model"] == "XGBoost") & (top_k["metric_type"] == "top_share_1p5pct")].copy()
    plt.figure(figsize=(9, 5.5))
    sns.barplot(data=top_share, x="setting_id", y="precision_at_k", order=order, color="#228833")
    plt.xlabel("Sensitivity setting")
    plt.ylabel("Precision at top ~1.5% cells")
    plt.title("Normalized hotspot precision across H3/time-window settings")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "h3_time_sensitivity_topk.png", dpi=200)
    plt.close()

    test_pop = population[population["split"] == "test"].copy()
    plot_df = test_pop.melt(
        id_vars=["setting_id"],
        value_vars=["inside_dubai_cells", "full_grid_candidate_rows", "positive_cell_windows"],
        var_name="measure",
        value_name="value",
    )
    plt.figure(figsize=(10, 5.5))
    sns.barplot(data=plot_df, x="setting_id", y="value", hue="measure", order=order)
    plt.yscale("log")
    plt.xlabel("Sensitivity setting")
    plt.ylabel("Count, log scale")
    plt.title("Full-grid population size across sensitivity settings")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "h3_time_sensitivity_population.png", dpi=200)
    plt.close()


def frame_to_markdown(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for record in frame.astype(str).to_dict(orient="records"):
        rows.append("| " + " | ".join(record[column] for column in columns) + " |")
    return "\n".join(rows)


def write_audit(metrics: pd.DataFrame, top_k: pd.DataFrame, population: pd.DataFrame) -> None:
    xgb = metrics[metrics["model"] == "XGBoost"].copy()
    top_share = top_k[(top_k["model"] == "XGBoost") & (top_k["metric_type"] == "top_share_1p5pct")].copy()
    merged = xgb.merge(
        top_share[["setting_id", "k", "precision_at_k", "recall_at_k"]],
        on="setting_id",
        how="left",
        suffixes=("", "_top_share"),
    )
    reference = merged.loc[merged["setting_id"] == "res8_3h"].iloc[0]
    best_pr = merged.sort_values("test_pr_auc", ascending=False).iloc[0]
    best_top_share = merged.sort_values("precision_at_k", ascending=False).iloc[0]
    lines = [
        "# H3/time-window sensitivity audit",
        "",
        "- Settings compared: `res7_3h`, `res8_1h`, `res8_3h`, `res8_6h`, and `res9_3h`.",
        "- Evaluation scope: inside-Dubai H3 cells only, using the same polygon-plus-observed-inside point rule as the main full-grid evaluation.",
        "- Models compared: historical-risk baseline and XGBoost.",
        "- Split: chronological 70/15/15 by window index for each setting.",
        "- XGBoost training: deterministic sampled training set with a 5:1 negative-to-positive source ratio and a 1,000,000-row fit cap.",
        "- Top-share metric: approximately the same share as top 20 of 1,305 cells, or about 1.5% of cells per test window.",
        "",
        "## Decision comparison",
        "",
        f"- Reference setting PR-AUC (`res8_3h`): `{reference['test_pr_auc']:.6f}`.",
        f"- Best XGBoost PR-AUC: `{best_pr['setting_id']}` with `{best_pr['test_pr_auc']:.6f}`.",
        f"- Reference normalized hotspot precision (`res8_3h`): `{reference['precision_at_k']:.6f}` at k `{int(reference['k'])}`.",
        f"- Best normalized hotspot precision: `{best_top_share['setting_id']}` with `{best_top_share['precision_at_k']:.6f}` at k `{int(best_top_share['k'])}`.",
        "",
        "## XGBoost metrics",
        "",
        frame_to_markdown(
            xgb[
                [
                    "setting_id",
                    "h3_resolution",
                    "window_hours",
                    "test_roc_auc",
                    "test_pr_auc",
                    "test_positive_rate",
                    "test_pr_auc_lift_vs_base_rate",
                    "test_precision",
                    "test_recall",
                    "test_f1",
                    "threshold",
                ]
            ].round(6)
        ),
        "",
        "## XGBoost normalized top-share metrics",
        "",
        frame_to_markdown(
            top_share[
                [
                    "setting_id",
                    "k",
                    "top_cell_share",
                    "test_positive_rate",
                    "precision_at_k",
                    "precision_lift_vs_base_rate",
                    "recall_at_k",
                    "positive_window_hit_rate_at_k",
                    "incident_recall_at_k",
                ]
            ].round(6)
        ),
        "",
        "## Population summary",
        "",
        frame_to_markdown(
            population[population["split"] == "test"][
                [
                    "setting_id",
                    "inside_dubai_cells",
                    "windows",
                    "full_grid_candidate_rows",
                    "positive_cell_windows",
                    "positive_rate",
                    "incident_count",
                ]
            ].round(6)
        ),
    ]
    (AUDIT_DIR / "h3_time_sensitivity_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_setting_outputs(data: SettingData, y_val: np.ndarray, y_test: np.ndarray, metrics: pd.DataFrame) -> None:
    n_cells = len(data.inside_cells)
    if n_cells <= 0:
        raise SystemExit(f"{data.setting.setting_id} has no inside-Dubai cells")
    if any(h3.get_resolution(cell) != data.setting.resolution for cell in data.inside_cells):
        raise SystemExit(f"{data.setting.setting_id} has an inside-Dubai cell at the wrong resolution")
    if y_val.sum() <= 0 or y_test.sum() <= 0:
        raise SystemExit(f"{data.setting.setting_id} has no validation or test positives")
    if metrics.isna().any().any():
        raise SystemExit(f"{data.setting.setting_id} produced NaN metrics")


def main() -> None:
    ensure_dirs()
    dubai_geometry = load_geojson_geometry(DUBAI_GEOJSON)
    base_df = add_dubai_point_flag(load_incidents(), dubai_geometry)
    all_metrics = []
    all_top_k = []
    all_population = []

    for setting_id, resolution, window_hours in SETTINGS:
        setting = Setting(setting_id=setting_id, resolution=resolution, window_hours=window_hours)
        print(f"Running sensitivity setting {setting.setting_id}")
        data = build_setting_data(base_df, dubai_geometry, setting)
        train, train_info = build_training_frame(data, RANDOM_SEED + resolution * 10 + window_hours)
        pipeline = train_xgboost(train, RANDOM_SEED)
        y_val, _, val_scores = score_split(data, pipeline, data.train_end, data.validation_end)
        y_test, test_incidents, test_scores = score_split(data, pipeline, data.validation_end, data.window_count)
        metrics = evaluate_setting(data, train_info, y_val, val_scores, y_test, test_scores)
        validate_setting_outputs(data, y_val, y_test, metrics)
        top_k = top_k_metrics(data, y_test, test_incidents, test_scores)
        all_metrics.append(metrics)
        all_top_k.append(top_k)
        all_population.extend(population_rows(data, train_info))
        print(
            f"  cells={len(data.inside_cells):,}, test_rows={len(y_test):,}, "
            f"test_pos={int(y_test.sum()):,}"
        )

    metrics_frame = pd.concat(all_metrics, ignore_index=True)
    top_k_frame = pd.concat(all_top_k, ignore_index=True)
    population_frame = pd.DataFrame(all_population)
    metrics_frame.to_csv(TABLE_DIR / "h3_time_sensitivity_metrics.csv", index=False)
    top_k_frame.to_csv(TABLE_DIR / "h3_time_sensitivity_topk.csv", index=False)
    population_frame.to_csv(TABLE_DIR / "h3_time_sensitivity_population.csv", index=False)
    plot_outputs(metrics_frame, top_k_frame, population_frame)
    write_audit(metrics_frame, top_k_frame, population_frame)
    print(f"Wrote sensitivity metrics to {TABLE_DIR / 'h3_time_sensitivity_metrics.csv'}")
    print(f"Wrote sensitivity audit to {AUDIT_DIR / 'h3_time_sensitivity_audit.md'}")


if __name__ == "__main__":
    main()
