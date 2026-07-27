from __future__ import annotations

import argparse
import csv
from pathlib import Path

from traffic_incident_utils import AUDIT_DIR, DEFAULT_RAW_PATH, MAPPINGS_DIR, PROCESSED_DIR, ensure_dirs, normalize_acci_name


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
    "load_timestamp",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Join the reviewed category mapping back to the full incident CSV.")
    parser.add_argument("--input", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--mapping", type=Path, default=MAPPINGS_DIR / "acci_name_translation_map.csv")
    parser.add_argument("--output", type=Path, default=PROCESSED_DIR / "traffic_incidents_cleaned.csv")
    parser.add_argument("--audit-output", type=Path, default=AUDIT_DIR / "cleaned_incidents_validation.md")
    return parser.parse_args()


MATCH_COLUMNS = [
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


def load_mapping(path: Path) -> tuple[dict[str, dict[str, str]], int, int]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        mapping: dict[str, dict[str, str]] = {}
        duplicate_normalized_keys = 0
        mapping_rows = 0
        for row in reader:
            mapping_rows += 1
            key = row["acci_name_normalized"]
            if key in mapping:
                duplicate_normalized_keys += 1
                existing = mapping[key]
                if any(existing.get(col) != row.get(col) for col in MATCH_COLUMNS):
                    raise SystemExit(f"Conflicting duplicate normalized mapping key: {key}")
                continue
            mapping[key] = row
    if not mapping:
        raise SystemExit("Mapping file is empty")
    return mapping, mapping_rows, duplicate_normalized_keys


def main() -> None:
    args = parse_args()
    ensure_dirs()
    mapping, mapping_rows, duplicate_normalized_keys = load_mapping(args.mapping)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    missing_mapping = 0
    needs_review_rows = 0
    excluded_eda_rows = 0
    blank_translation_rows = 0
    unknown_severity_rows = 0
    bad_unknown_severity_rows = 0
    bad_known_severity_flags = 0

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
            if mapped.get("review_status") == "needs_review":
                needs_review_rows += 1
            if mapped.get("include_in_eda") == "false":
                excluded_eda_rows += 1
            if not mapped.get("incident_type_en") and mapped.get("review_status") != "needs_review":
                blank_translation_rows += 1
            is_severity_known = mapped.get("is_severity_known") or ("false" if mapped.get("severity_code") == "unknown" else "true")
            if mapped.get("severity_code") == "unknown":
                unknown_severity_rows += 1
                if mapped.get("severity_weight") != "0" or is_severity_known.lower() != "false":
                    bad_unknown_severity_rows += 1
            elif is_severity_known.lower() != "true":
                bad_known_severity_flags += 1
            writer.writerow(
                {
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
                    "include_in_eda": mapped.get("include_in_eda", ""),
                    "exclude_reason": mapped.get("exclude_reason", ""),
                    "review_status": mapped.get("review_status", ""),
                    "acci_x": raw.get("acci_x", ""),
                    "acci_y": raw.get("acci_y", ""),
                    "load_timestamp": raw.get("load_timestamp", ""),
                }
            )

    lines = [
        "# Cleaned incident dataset validation",
        "",
        f"- Source file: `{args.input.name}`",
        f"- Mapping file: `{args.mapping}`",
        f"- Output file: `{args.output}`",
        f"- Cleaned row count: `{rows}`",
        f"- Mapping rows: `{mapping_rows}`",
        f"- Normalized mapping keys used for joining: `{len(mapping)}`",
        f"- Duplicate normalized mapping keys collapsed for joining: `{duplicate_normalized_keys}`",
        f"- Rows missing mapping: `{missing_mapping}`",
        f"- Rows mapped to `needs_review` categories: `{needs_review_rows}`",
        f"- Rows excluded from category-level EDA: `{excluded_eda_rows}`",
        f"- Rows with blank translation outside `needs_review`: `{blank_translation_rows}`",
        f"- Rows with unknown or unspecified severity: `{unknown_severity_rows}`",
        f"- Unknown-severity rows with nonzero weight or wrong flag: `{bad_unknown_severity_rows}`",
        f"- Known-severity rows with wrong known-severity flag: `{bad_known_severity_flags}`",
    ]
    args.audit_output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if missing_mapping:
        raise SystemExit(f"{missing_mapping} rows did not join to the mapping")
    print(f"Wrote {args.output} with {rows} rows")
    print(f"Wrote {args.audit_output}")


if __name__ == "__main__":
    main()
