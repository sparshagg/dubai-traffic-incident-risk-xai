from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from traffic_incident_utils import DEFAULT_RAW_PATH, MAPPINGS_DIR, ensure_dirs, normalize_acci_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract unique acci_name categories from the raw incident CSV.")
    parser.add_argument("--input", type=Path, default=DEFAULT_RAW_PATH)
    parser.add_argument("--output", type=Path, default=MAPPINGS_DIR / "acci_name_categories.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    counts = Counter()
    total = 0

    with args.input.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            counts[(row.get("acci_name") or "").strip()] += 1

    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "acci_name_ar", "acci_name_normalized", "count", "share"], lineterminator="\n")
        writer.writeheader()
        for rank, (name, count) in enumerate(counts.most_common(), 1):
            writer.writerow(
                {
                    "rank": rank,
                    "acci_name_ar": name,
                    "acci_name_normalized": normalize_acci_name(name),
                    "count": count,
                    "share": f"{count / total:.8f}",
                }
            )

    print(f"Wrote {args.output} with {len(counts)} unique categories from {total} rows")


if __name__ == "__main__":
    main()
