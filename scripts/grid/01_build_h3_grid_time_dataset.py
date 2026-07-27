from __future__ import annotations

import argparse
import csv
import json
import random
from bisect import bisect_left
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import h3
import pandas as pd
from shapely.geometry import Point, shape
from shapely.ops import unary_union
from shapely.prepared import prep


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data" / "processed" / "traffic_incidents_eda_ready.csv"
DUBAI_GEOJSON = ROOT / "resources" / "geo" / "dubai.geojson"
UAE_GEOJSON = ROOT / "resources" / "geo" / "united_arab_emirates.geojson"
PROCESSED_DIR = ROOT / "data" / "processed"
AUDIT_DIR = ROOT / "data" / "audit"

H3_RESOLUTION = 8
WINDOW_HOURS = 3
NEGATIVE_RATIO = 5
RANDOM_SEED = 42

VALID_COORDINATE_STATUSES = {"as_provided_lon_lat", "swapped_lat_lon"}
EXPECTED_TOTAL_ROWS = 720_155
EXPECTED_MAP_USABLE_ROWS = 717_615


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the first H3 grid-time modeling dataset.")
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--dubai-geojson", type=Path, default=DUBAI_GEOJSON)
    parser.add_argument("--uae-geojson", type=Path, default=UAE_GEOJSON)
    parser.add_argument("--resolution", type=int, default=H3_RESOLUTION)
    parser.add_argument("--window-hours", type=int, default=WINDOW_HOURS)
    parser.add_argument("--negative-ratio", type=int, default=NEGATIVE_RATIO)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def ensure_dirs() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_geojson(path: Path) -> tuple[dict, object, int, list[str]]:
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
    geometry = geometries[0] if len(geometries) == 1 else unary_union(geometries)
    geometry_types = sorted({feature["geometry"]["type"] for feature in features})
    return data, geometry, len(features), geometry_types


def geometry_bbox(geometry: object) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = geometry.bounds
    return float(minx), float(miny), float(maxx), float(maxy)


def count_geojson_coordinates(data: dict) -> int:
    count = 0

    def walk(value: object) -> None:
        nonlocal count
        if isinstance(value, list) and value and isinstance(value[0], (int, float)) and len(value) >= 2:
            count += 1
        elif isinstance(value, list):
            for item in value:
                walk(item)

    if data.get("type") == "FeatureCollection":
        for feature in data.get("features", []):
            walk(feature.get("geometry", {}).get("coordinates"))
    elif data.get("type") == "Feature":
        walk(data.get("geometry", {}).get("coordinates"))
    else:
        walk(data.get("coordinates"))
    return count


def read_eda_rows(input_path: Path) -> pd.DataFrame:
    usecols = [
        "acci_id",
        "acci_time",
        "incident_type_code",
        "severity_code",
        "severity_weight",
        "is_severity_known",
        "include_in_eda",
        "longitude",
        "latitude",
        "coordinate_status",
    ]
    df = pd.read_csv(input_path, usecols=usecols, dtype=str)
    total_rows = len(df)
    if total_rows != EXPECTED_TOTAL_ROWS:
        raise SystemExit(f"Expected {EXPECTED_TOTAL_ROWS} EDA-ready rows, got {total_rows}")
    df = df[(df["include_in_eda"] == "true") & (df["coordinate_status"].isin(VALID_COORDINATE_STATUSES))].copy()
    if len(df) != EXPECTED_MAP_USABLE_ROWS:
        raise SystemExit(f"Expected {EXPECTED_MAP_USABLE_ROWS} map-usable rows, got {len(df)}")
    df["longitude"] = df["longitude"].astype(float)
    df["latitude"] = df["latitude"].astype(float)
    df["severity_weight"] = df["severity_weight"].astype(int)
    df["acci_dt"] = pd.to_datetime(df["acci_time"], errors="coerce")
    invalid_time = int(df["acci_dt"].isna().sum())
    if invalid_time:
        raise SystemExit(f"Map-usable rows contain {invalid_time} invalid timestamps")
    return df


def uae_weekend_flag(window_start: pd.Timestamp) -> int:
    # UAE public-sector weekend changed from Friday-Saturday to Saturday-Sunday in 2022.
    weekday = int(window_start.weekday())
    if window_start < pd.Timestamp("2022-01-01"):
        return int(weekday in {4, 5})
    return int(weekday in {5, 6})


def classify_point(point: Point, dubai_prepared: object, uae_prepared: object) -> str:
    if dubai_prepared.covers(point):
        return "inside_dubai"
    if uae_prepared.covers(point):
        return "peripheral_observed"
    return "outside_uae_flagged"


def build_cell_scope(dubai_cells: set[str], points_df: pd.DataFrame) -> dict[str, str]:
    scope_priority = {"inside_dubai": 3, "peripheral_observed": 2, "outside_uae_flagged": 1}
    cell_scope: dict[str, str] = {cell: "inside_dubai" for cell in dubai_cells}
    observed_scopes: dict[str, str] = {}
    for cell, scope in zip(points_df["h3_cell_res8"], points_df["geo_scope"], strict=True):
        current = observed_scopes.get(cell)
        if current is None or scope_priority[scope] > scope_priority[current]:
            observed_scopes[cell] = scope
    for cell, scope in observed_scopes.items():
        if cell not in cell_scope:
            cell_scope[cell] = scope
    return cell_scope


def write_preview(path: Path, frame: pd.DataFrame, n: int = 100) -> None:
    frame.head(n).to_csv(path, index=False)


def build_lag_lookup(positive_frame: pd.DataFrame) -> dict[str, tuple[list[int], list[int], list[int]]]:
    grouped: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for row in positive_frame.itertuples(index=False):
        grouped[row.h3_cell_res8].append((int(row.window_index), int(row.incident_count), int(row.severity_weight_sum)))

    lookup: dict[str, tuple[list[int], list[int], list[int]]] = {}
    for cell, rows in grouped.items():
        rows.sort()
        windows: list[int] = []
        cum_counts = [0]
        cum_weights = [0]
        for window_index, count, weight in rows:
            windows.append(window_index)
            cum_counts.append(cum_counts[-1] + count)
            cum_weights.append(cum_weights[-1] + weight)
        lookup[cell] = (windows, cum_counts, cum_weights)
    return lookup


def lag_sum(lookup: dict[str, tuple[list[int], list[int], list[int]]], cell: str, window_index: int, length: int, value_kind: str) -> int:
    series = lookup.get(cell)
    if series is None:
        return 0
    windows, cum_counts, cum_weights = series
    start = bisect_left(windows, window_index - length)
    end = bisect_left(windows, window_index)
    cumulative = cum_counts if value_kind == "count" else cum_weights
    return int(cumulative[end] - cumulative[start])


def sample_negative_pairs(
    cells: list[str],
    window_count: int,
    positive_pairs: set[tuple[str, int]],
    target_negative_rows: int,
    seed: int,
) -> list[tuple[str, int]]:
    max_negative_rows = len(cells) * window_count - len(positive_pairs)
    target = min(target_negative_rows, max_negative_rows)
    rng = random.Random(seed)
    negatives: set[tuple[str, int]] = set()
    while len(negatives) < target:
        pair = (cells[rng.randrange(len(cells))], rng.randrange(window_count))
        if pair in positive_pairs or pair in negatives:
            continue
        negatives.add(pair)
    return sorted(negatives)


def feature_values(window_start: pd.Timestamp, window_hours: int) -> dict[str, object]:
    return {
        "hour_block": int(window_start.hour // window_hours),
        "day_of_week": window_start.day_name(),
        "is_weekend": uae_weekend_flag(window_start),
        "month": int(window_start.month),
        "year": int(window_start.year),
    }


def build_model_sample(
    positive_frame: pd.DataFrame,
    negative_pairs: list[tuple[str, int]],
    cells: list[str],
    cell_scope: dict[str, str],
    start_window: pd.Timestamp,
    window_hours: int,
    lag_lookup: dict[str, tuple[list[int], list[int], list[int]]],
    output_path: Path,
) -> int:
    output_columns = [
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

    rows_written = 0
    positive_frame = positive_frame.sort_values(["window_index", "h3_cell_res8"])
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_columns, lineterminator="\n")
        writer.writeheader()

        for row in positive_frame.itertuples(index=False):
            window_start = pd.Timestamp(row.window_start)
            base = feature_values(window_start, window_hours)
            record = {
                "h3_cell_res8": row.h3_cell_res8,
                "window_start": window_start.isoformat(),
                "window_index": int(row.window_index),
                "geo_scope": row.geo_scope,
                "risk_label": 1,
                "incident_count": int(row.incident_count),
                "severity_weight_sum": int(row.severity_weight_sum),
                "minor_count": int(row.minor_count),
                "moderate_count": int(row.moderate_count),
                "severe_count": int(row.severe_count),
                "unknown_count": int(row.unknown_count),
                **base,
                "prev_3h_incident_count": lag_sum(lag_lookup, row.h3_cell_res8, int(row.window_index), 1, "count"),
                "prev_24h_incident_count": lag_sum(lag_lookup, row.h3_cell_res8, int(row.window_index), 8, "count"),
                "prev_7d_incident_count": lag_sum(lag_lookup, row.h3_cell_res8, int(row.window_index), 56, "count"),
                "prev_24h_severity_weight_sum": lag_sum(lag_lookup, row.h3_cell_res8, int(row.window_index), 8, "weight"),
                "prev_7d_severity_weight_sum": lag_sum(lag_lookup, row.h3_cell_res8, int(row.window_index), 56, "weight"),
            }
            writer.writerow(record)
            rows_written += 1

        for cell, window_index in negative_pairs:
            window_start = start_window + pd.Timedelta(hours=window_hours * window_index)
            base = feature_values(window_start, window_hours)
            record = {
                "h3_cell_res8": cell,
                "window_start": window_start.isoformat(),
                "window_index": int(window_index),
                "geo_scope": cell_scope[cell],
                "risk_label": 0,
                "incident_count": 0,
                "severity_weight_sum": 0,
                "minor_count": 0,
                "moderate_count": 0,
                "severe_count": 0,
                "unknown_count": 0,
                **base,
                "prev_3h_incident_count": lag_sum(lag_lookup, cell, int(window_index), 1, "count"),
                "prev_24h_incident_count": lag_sum(lag_lookup, cell, int(window_index), 8, "count"),
                "prev_7d_incident_count": lag_sum(lag_lookup, cell, int(window_index), 56, "count"),
                "prev_24h_severity_weight_sum": lag_sum(lag_lookup, cell, int(window_index), 8, "weight"),
                "prev_7d_severity_weight_sum": lag_sum(lag_lookup, cell, int(window_index), 56, "weight"),
            }
            writer.writerow(record)
            rows_written += 1
    return rows_written


def main() -> None:
    args = parse_args()
    ensure_dirs()

    dubai_data, dubai_geom, dubai_feature_count, dubai_geometry_types = load_geojson(args.dubai_geojson)
    uae_data, uae_geom, uae_feature_count, uae_geometry_types = load_geojson(args.uae_geojson)
    dubai_cells = set(h3.geo_to_cells(dubai_geom.__geo_interface__, args.resolution))
    if not dubai_cells:
        raise SystemExit("Dubai polygon produced zero H3 cells")
    if any(h3.get_resolution(cell) != args.resolution for cell in dubai_cells):
        raise SystemExit("Dubai H3 fill produced cells at the wrong resolution")

    dubai_prepared = prep(dubai_geom)
    uae_prepared = prep(uae_geom)
    df = read_eda_rows(args.input)
    df["h3_cell_res8"] = [h3.latlng_to_cell(lat, lon, args.resolution) for lat, lon in zip(df["latitude"], df["longitude"], strict=True)]
    if df["h3_cell_res8"].isna().any():
        raise SystemExit("Some map-usable incidents did not receive an H3 cell")
    if any(h3.get_resolution(cell) != args.resolution for cell in df["h3_cell_res8"].unique()):
        raise SystemExit("Observed H3 cells are not at the requested resolution")

    df["geo_scope_point"] = [
        classify_point(Point(lon, lat), dubai_prepared, uae_prepared)
        for lon, lat in zip(df["longitude"], df["latitude"], strict=True)
    ]

    cell_scope = build_cell_scope(dubai_cells, df.rename(columns={"geo_scope_point": "geo_scope"}))
    df["geo_scope"] = [cell_scope[cell] for cell in df["h3_cell_res8"]]
    observed_cells = set(df["h3_cell_res8"].unique())
    universe_cells = sorted(set(cell_scope))

    point_output = PROCESSED_DIR / f"incident_points_h3_res{args.resolution}.csv"
    point_columns = [
        "acci_id",
        "acci_time",
        "h3_cell_res8",
        "longitude",
        "latitude",
        "incident_type_code",
        "severity_code",
        "severity_weight",
        "is_severity_known",
        "geo_scope",
    ]
    df[point_columns].to_csv(point_output, index=False)

    start_window = df["acci_dt"].min().floor(f"{args.window_hours}h")
    end_window = df["acci_dt"].max().floor(f"{args.window_hours}h")
    df["window_start"] = df["acci_dt"].dt.floor(f"{args.window_hours}h")
    df["window_index"] = ((df["window_start"] - start_window) / pd.Timedelta(hours=args.window_hours)).astype(int)
    window_count = int(((end_window - start_window) / pd.Timedelta(hours=args.window_hours))) + 1

    for severity in ["minor", "moderate", "severe", "unknown"]:
        df[f"{severity}_count"] = (df["severity_code"] == severity).astype(int)

    positive_frame = (
        df.groupby(["h3_cell_res8", "window_start", "window_index", "geo_scope"], as_index=False)
        .agg(
            incident_count=("acci_id", "count"),
            severity_weight_sum=("severity_weight", "sum"),
            minor_count=("minor_count", "sum"),
            moderate_count=("moderate_count", "sum"),
            severe_count=("severe_count", "sum"),
            unknown_count=("unknown_count", "sum"),
        )
        .sort_values(["window_index", "h3_cell_res8"])
    )

    count_output = PROCESSED_DIR / f"grid_time_incident_counts_res{args.resolution}_{args.window_hours}h.csv"
    positive_frame.to_csv(count_output, index=False)

    positive_pairs = set(zip(positive_frame["h3_cell_res8"], positive_frame["window_index"], strict=True))
    negative_pairs = sample_negative_pairs(
        universe_cells,
        window_count,
        positive_pairs,
        len(positive_pairs) * args.negative_ratio,
        args.seed,
    )
    lag_lookup = build_lag_lookup(positive_frame)
    model_output = PROCESSED_DIR / f"grid_time_model_sample_res{args.resolution}_{args.window_hours}h.csv"
    model_rows = build_model_sample(
        positive_frame,
        negative_pairs,
        universe_cells,
        cell_scope,
        start_window,
        args.window_hours,
        lag_lookup,
        model_output,
    )

    positive_preview = positive_frame.head(100).copy()
    model_preview = pd.read_csv(model_output, nrows=100)
    cell_scope_rows = pd.DataFrame(
        [{"h3_cell_res8": cell, "geo_scope": scope} for cell, scope in sorted(cell_scope.items())]
    )
    scope_counts = Counter(cell_scope.values())
    point_scope_counts = Counter(df["geo_scope"])

    write_preview(AUDIT_DIR / f"incident_points_h3_res{args.resolution}_sample.csv", df[point_columns])
    write_preview(AUDIT_DIR / f"grid_time_incident_counts_res{args.resolution}_{args.window_hours}h_sample.csv", positive_preview)
    write_preview(AUDIT_DIR / f"grid_time_model_sample_res{args.resolution}_{args.window_hours}h_preview.csv", model_preview)
    cell_scope_rows.to_csv(AUDIT_DIR / f"h3_cell_scope_summary_res{args.resolution}.csv", index=False)

    dubai_bbox = geometry_bbox(dubai_geom)
    uae_bbox = geometry_bbox(uae_geom)
    geo_audit = [
        "# Geo boundary audit",
        "",
        f"- Dubai GeoJSON: `{args.dubai_geojson}`",
        "- Dubai source URL: `https://github.com/sbma44/uber-cities/blob/master/geojson/dubai.geojson`",
        f"- Dubai feature count: `{dubai_feature_count}`",
        f"- Dubai geometry types: `{', '.join(dubai_geometry_types)}`",
        f"- Dubai coordinate count: `{count_geojson_coordinates(dubai_data)}`",
        f"- Dubai bounds lon/lat: `{dubai_bbox[0]}`, `{dubai_bbox[1]}`, `{dubai_bbox[2]}`, `{dubai_bbox[3]}`",
        f"- Dubai H3 resolution {args.resolution} cells: `{len(dubai_cells)}`",
        "",
        f"- UAE GeoJSON: `{args.uae_geojson}`",
        "- UAE source URL: `https://github.com/glynnbird/countriesgeojson/blob/master/united%20arab%20emirates.geojson`",
        f"- UAE feature count: `{uae_feature_count}`",
        f"- UAE geometry types: `{', '.join(uae_geometry_types)}`",
        f"- UAE coordinate count: `{count_geojson_coordinates(uae_data)}`",
        f"- UAE bounds lon/lat: `{uae_bbox[0]}`, `{uae_bbox[1]}`, `{uae_bbox[2]}`, `{uae_bbox[3]}`",
        "",
        "## Cell universe",
        "",
        f"- Observed H3 cells: `{len(observed_cells)}`",
        f"- Study universe H3 cells: `{len(universe_cells)}`",
        f"- Universe inside Dubai cells: `{scope_counts['inside_dubai']}`",
        f"- Universe peripheral observed cells: `{scope_counts['peripheral_observed']}`",
        f"- Universe outside UAE flagged cells: `{scope_counts['outside_uae_flagged']}`",
        "",
        "## Incident point scope",
        "",
        f"- Incident points inside Dubai: `{point_scope_counts['inside_dubai']}`",
        f"- Incident points peripheral observed: `{point_scope_counts['peripheral_observed']}`",
        f"- Incident points outside UAE flagged: `{point_scope_counts['outside_uae_flagged']}`",
    ]
    (AUDIT_DIR / "geo_boundary_audit.md").write_text("\n".join(geo_audit) + "\n", encoding="utf-8")

    model_label_counts = Counter()
    duplicate_pairs = 0
    seen_pairs: set[tuple[str, str]] = set()
    aligned_windows = True
    lag_current_leak_violations = 0
    with model_output.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            model_label_counts[row["risk_label"]] += 1
            pair = (row["h3_cell_res8"], row["window_start"])
            if pair in seen_pairs:
                duplicate_pairs += 1
            seen_pairs.add(pair)
            dt = datetime.fromisoformat(row["window_start"])
            if dt.hour % args.window_hours != 0 or dt.minute != 0 or dt.second != 0:
                aligned_windows = False
            if row["risk_label"] == "1" and int(row["prev_3h_incident_count"]) > int(row["prev_24h_incident_count"]):
                lag_current_leak_violations += 1

    h3_audit = [
        "# H3 grid-time dataset audit",
        "",
        f"- Input file: `{args.input}`",
        f"- H3 resolution: `{args.resolution}`",
        f"- Time window hours: `{args.window_hours}`",
        f"- Negative sampling ratio: `{args.negative_ratio}`",
        f"- Random seed: `{args.seed}`",
        f"- Map-usable incident rows: `{len(df)}`",
        f"- Incident point output: `{point_output}`",
        f"- Incident point output rows: `{len(df)}`",
        f"- Positive grid-time output: `{count_output}`",
        f"- Positive grid-time rows: `{len(positive_frame)}`",
        f"- Model sample output: `{model_output}`",
        f"- Model sample rows: `{model_rows}`",
        f"- Positive model rows: `{model_label_counts['1']}`",
        f"- Negative model rows: `{model_label_counts['0']}`",
        f"- Negative to positive ratio: `{model_label_counts['0'] / max(model_label_counts['1'], 1):.2f}`",
        f"- Study universe cells: `{len(universe_cells)}`",
        f"- Window start: `{start_window.isoformat()}`",
        f"- Window end: `{end_window.isoformat()}`",
        f"- Window count: `{window_count}`",
        f"- Duplicate model cell/window rows: `{duplicate_pairs}`",
        f"- All model windows aligned to {args.window_hours}-hour boundary: `{str(aligned_windows).lower()}`",
        f"- Lag consistency violations detected: `{lag_current_leak_violations}`",
        "",
        "## Output previews",
        "",
        f"- Incident point sample: `data/audit/incident_points_h3_res{args.resolution}_sample.csv`",
        f"- Positive grid-time sample: `data/audit/grid_time_incident_counts_res{args.resolution}_{args.window_hours}h_sample.csv`",
        f"- Model sample preview: `data/audit/grid_time_model_sample_res{args.resolution}_{args.window_hours}h_preview.csv`",
        f"- Cell scope summary: `data/audit/h3_cell_scope_summary_res{args.resolution}.csv`",
    ]
    (AUDIT_DIR / "h3_grid_time_audit.md").write_text("\n".join(h3_audit) + "\n", encoding="utf-8")

    if model_label_counts["1"] != len(positive_frame):
        raise SystemExit("Model sample does not contain all positive grid-time rows")
    if duplicate_pairs:
        raise SystemExit("Model sample contains duplicate cell/window rows")
    if not aligned_windows:
        raise SystemExit("Some window_start values are not aligned to the configured window size")
    if lag_current_leak_violations:
        raise SystemExit("Lag consistency check failed")
    print(f"Wrote {point_output} with {len(df)} rows")
    print(f"Wrote {count_output} with {len(positive_frame)} positive grid-time rows")
    print(f"Wrote {model_output} with {model_rows} rows")
    print(f"Wrote audits under {AUDIT_DIR}")


if __name__ == "__main__":
    main()
