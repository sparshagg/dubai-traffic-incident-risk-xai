from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

from traffic_incident_utils import AUDIT_DIR, DEFAULT_RAW_PATH, RAW_COLUMNS, ensure_dirs, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the raw Dubai traffic incident CSV without modifying it.")
    parser.add_argument("--input", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--output", type=Path, default=AUDIT_DIR / "raw_incidents_audit.md")
    return parser.parse_args()


def parse_dt(value: str) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            pass
    return None


def main() -> None:
    args = parse_args()
    ensure_dirs()

    seen_ids: set[str] = set()
    duplicate_ids = 0
    row_count = 0
    missing_by_col = Counter()
    unique_names = Counter()
    min_time: datetime | None = None
    max_time: datetime | None = None
    invalid_time = 0
    missing_coord = 0
    invalid_coord = 0
    lon_lat_dubai = 0
    lat_lon_swapped_dubai = 0
    zero_coord = 0
    other_out_of_bounds = 0
    min_x = max_x = min_y = max_y = None

    with args.input.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if fieldnames != RAW_COLUMNS:
            raise SystemExit(f"Unexpected columns: {fieldnames}")

        for row in reader:
            row_count += 1
            for col in RAW_COLUMNS:
                if not (row.get(col) or "").strip():
                    missing_by_col[col] += 1

            acci_id = (row.get("acci_id") or "").strip()
            if acci_id:
                if acci_id in seen_ids:
                    duplicate_ids += 1
                seen_ids.add(acci_id)

            name = (row.get("acci_name") or "").strip()
            unique_names[name] += 1

            dt = parse_dt((row.get("acci_time") or "").strip())
            if dt is None:
                invalid_time += 1
            else:
                min_time = dt if min_time is None or dt < min_time else min_time
                max_time = dt if max_time is None or dt > max_time else max_time

            x_raw = (row.get("acci_x") or "").strip()
            y_raw = (row.get("acci_y") or "").strip()
            if not x_raw or not y_raw:
                missing_coord += 1
                continue
            try:
                x = float(x_raw)
                y = float(y_raw)
            except ValueError:
                invalid_coord += 1
                continue
            min_x = x if min_x is None or x < min_x else min_x
            max_x = x if max_x is None or x > max_x else max_x
            min_y = y if min_y is None or y < min_y else min_y
            max_y = y if max_y is None or y > max_y else max_y
            if 54.5 <= x <= 56.5 and 24.5 <= y <= 26.5:
                lon_lat_dubai += 1
            elif 24.5 <= x <= 26.5 and 54.5 <= y <= 56.5:
                lat_lon_swapped_dubai += 1
            elif x == 0 or y == 0:
                zero_coord += 1
            else:
                other_out_of_bounds += 1

    sha = sha256_file(args.input)
    lines = [
        "# Raw traffic incidents audit",
        "",
        f"- Source file: `{args.input.name}`",
        f"- File size bytes: `{args.input.stat().st_size}`",
        f"- SHA-256: `{sha}`",
        f"- Columns: `{', '.join(RAW_COLUMNS)}`",
        f"- Row count: `{row_count}`",
        f"- Unique `acci_id`: `{len(seen_ids)}`",
        f"- Duplicate `acci_id` rows: `{duplicate_ids}`",
        f"- Unique `acci_name`: `{len(unique_names)}`",
        f"- Date range: `{min_time}` to `{max_time}`",
        f"- Invalid/blank `acci_time`: `{invalid_time}`",
        f"- Missing coordinate rows: `{missing_coord}`",
        f"- Invalid coordinate rows: `{invalid_coord}`",
        f"- Longitude range: `{min_x}` to `{max_x}`",
        f"- Latitude range: `{min_y}` to `{max_y}`",
        f"- Rows with `acci_x`/`acci_y` in expected longitude/latitude Dubai bounds: `{lon_lat_dubai}`",
        f"- Rows that appear latitude/longitude swapped for Dubai bounds: `{lat_lon_swapped_dubai}`",
        f"- Rows with zero coordinates: `{zero_coord}`",
        f"- Other out-of-bounds coordinate rows: `{other_out_of_bounds}`",
        "",
        "## Missing values by column",
        "",
        "| Column | Missing rows |",
        "| --- | ---: |",
    ]
    for col in RAW_COLUMNS:
        lines.append(f"| `{col}` | {missing_by_col[col]} |")
    lines.extend(["", "## Top 20 incident categories", "", "| Rank | Count | `acci_name` |", "| ---: | ---: | --- |"])
    for rank, (name, count) in enumerate(unique_names.most_common(20), 1):
        lines.append(f"| {rank} | {count} | {name} |")

    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
