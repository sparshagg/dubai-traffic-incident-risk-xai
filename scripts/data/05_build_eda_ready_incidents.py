from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from traffic_incident_utils import (
    AUDIT_DIR,
    DEFAULT_RAW_PATH,
    MAPPINGS_DIR,
    PROCESSED_DIR,
    ensure_dirs,
    normalize_acci_name,
    normalize_coordinates,
)


OUTPUT_COLUMNS = [
    "acci_id",
    "acci_time",
    "acci_name",
    "incident_type_ar",
    "severity_ar",
    "incident_type_en",
    "severity_label_en",
    "incident_type_code",
    "severity_code",
    "severity_weight",
    "is_severity_known",
    "include_in_eda",
    "exclude_reason",
    "review_status",
    "acci_x",
    "acci_y",
    "longitude",
    "latitude",
    "coordinate_status",
    "load_timestamp",
]

MAPPING_MATCH_COLUMNS = [
    "incident_type_ar",
    "severity_ar",
    "incident_type_en",
    "severity_label_en",
    "incident_type_code",
    "severity_code",
    "severity_weight",
    "is_severity_known",
    "include_in_eda",
    "exclude_reason",
    "review_status",
]

VALID_COORDINATE_STATUSES = {"as_provided_lon_lat", "swapped_lat_lon"}
ANOMALY_STATUSES = {"missing_coordinate", "invalid_coordinate", "zero_coordinate", "out_of_bounds"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an EDA-ready incident CSV with category flags and normalized coordinates.")
    parser.add_argument("--input", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--mapping", type=Path, default=MAPPINGS_DIR / "acci_name_translation_map.csv")
    parser.add_argument("--output", type=Path, default=PROCESSED_DIR / "traffic_incidents_eda_ready.csv")
    parser.add_argument("--audit-output", type=Path, default=AUDIT_DIR / "coordinate_normalization_audit.md")
    parser.add_argument("--anomaly-sample-output", type=Path, default=AUDIT_DIR / "coordinate_anomaly_samples.csv")
    parser.add_argument("--sample-size", type=int, default=100)
    return parser.parse_args()


def load_mapping(path: Path) -> tuple[dict[str, dict[str, str]], int, int]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        mapping: dict[str, dict[str, str]] = {}
        duplicate_normalized_keys = 0
        mapping_rows = 0
        needs_review_rows = 0
        for row in reader:
            mapping_rows += 1
            if row.get("review_status") == "needs_review":
                needs_review_rows += 1
            key = row["acci_name_normalized"]
            if key in mapping:
                duplicate_normalized_keys += 1
                existing = mapping[key]
                if any(existing.get(col) != row.get(col) for col in MAPPING_MATCH_COLUMNS):
                    raise SystemExit(f"Conflicting duplicate normalized mapping key: {key}")
                continue
            mapping[key] = row
    if not mapping:
        raise SystemExit("Mapping file is empty")
    if needs_review_rows:
        raise SystemExit(f"Mapping still contains {needs_review_rows} needs_review rows")
    return mapping, mapping_rows, duplicate_normalized_keys


def main() -> None:
    args = parse_args()
    ensure_dirs()
    mapping, mapping_rows, duplicate_normalized_keys = load_mapping(args.mapping)

    rows = 0
    missing_mapping = 0
    excluded_category_rows = 0
    usable_map_rows = 0
    coordinate_counts: Counter[str] = Counter()
    anomaly_samples: list[dict[str, str]] = []

    with args.input.open("r", encoding="utf-8-sig", newline="") as src, args.output.open("w", encoding="utf-8", newline="") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for raw in reader:
            rows += 1
            name = (raw.get("acci_name") or "").strip()
            mapped = mapping.get(normalize_acci_name(name))
            if mapped is None:
                missing_mapping += 1
                mapped = {key: "" for key in next(iter(mapping.values())).keys()}

            normalized = normalize_coordinates(raw.get("acci_x", ""), raw.get("acci_y", ""))
            coordinate_counts[normalized.coordinate_status] += 1
            include_in_eda = mapped.get("include_in_eda", "")
            if include_in_eda == "false":
                excluded_category_rows += 1
            if include_in_eda == "true" and normalized.coordinate_status in VALID_COORDINATE_STATUSES:
                usable_map_rows += 1
            is_severity_known = mapped.get("is_severity_known") or ("false" if mapped.get("severity_code") == "unknown" else "true")

            output_row = {
                "acci_id": raw.get("acci_id", ""),
                "acci_time": raw.get("acci_time", ""),
                "acci_name": name,
                "incident_type_ar": mapped.get("incident_type_ar", ""),
                "severity_ar": mapped.get("severity_ar", ""),
                "incident_type_en": mapped.get("incident_type_en", ""),
                "severity_label_en": mapped.get("severity_label_en", ""),
                "incident_type_code": mapped.get("incident_type_code", ""),
                "severity_code": mapped.get("severity_code", ""),
                "severity_weight": mapped.get("severity_weight", ""),
                "is_severity_known": is_severity_known,
                "include_in_eda": include_in_eda,
                "exclude_reason": mapped.get("exclude_reason", ""),
                "review_status": mapped.get("review_status", ""),
                "acci_x": raw.get("acci_x", ""),
                "acci_y": raw.get("acci_y", ""),
                "longitude": normalized.longitude,
                "latitude": normalized.latitude,
                "coordinate_status": normalized.coordinate_status,
                "load_timestamp": raw.get("load_timestamp", ""),
            }
            writer.writerow(output_row)

            if normalized.coordinate_status in ANOMALY_STATUSES and len(anomaly_samples) < args.sample_size:
                anomaly_samples.append(output_row)

    with args.anomaly_sample_output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(anomaly_samples)

    valid_coordinate_rows = sum(coordinate_counts[status] for status in VALID_COORDINATE_STATUSES)
    invalid_coordinate_rows = rows - valid_coordinate_rows
    lines = [
        "# Coordinate normalization audit",
        "",
        f"- Source file: `{args.input.name}`",
        f"- Mapping file: `{args.mapping}`",
        f"- Output file: `{args.output}`",
        f"- Row count: `{rows}`",
        f"- Mapping rows: `{mapping_rows}`",
        f"- Normalized mapping keys used for joining: `{len(mapping)}`",
        f"- Duplicate normalized mapping keys collapsed for joining: `{duplicate_normalized_keys}`",
        f"- Rows missing mapping: `{missing_mapping}`",
        f"- Rows excluded from category-level EDA: `{excluded_category_rows}`",
        f"- Rows with valid normalized coordinates: `{valid_coordinate_rows}`",
        f"- Rows without usable normalized coordinates: `{invalid_coordinate_rows}`",
        f"- Rows usable for map-based EDA after category and coordinate filters: `{usable_map_rows}`",
        f"- Anomaly sample file: `{args.anomaly_sample_output}`",
        f"- Anomaly sample rows: `{len(anomaly_samples)}`",
        "",
        "## Coordinate status counts",
        "",
        "| Coordinate status | Rows |",
        "| --- | ---: |",
    ]
    for status in [
        "as_provided_lon_lat",
        "swapped_lat_lon",
        "zero_coordinate",
        "missing_coordinate",
        "invalid_coordinate",
        "out_of_bounds",
    ]:
        lines.append(f"| `{status}` | {coordinate_counts[status]} |")
    args.audit_output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if missing_mapping:
        raise SystemExit(f"{missing_mapping} rows did not join to the mapping")
    print(f"Wrote {args.output} with {rows} rows")
    print(f"Wrote {args.audit_output}")
    print(f"Wrote {args.anomaly_sample_output} with {len(anomaly_samples)} rows")


if __name__ == "__main__":
    main()
