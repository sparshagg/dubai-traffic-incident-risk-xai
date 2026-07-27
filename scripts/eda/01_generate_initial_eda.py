from __future__ import annotations

import csv
import html
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data" / "processed" / "traffic_incidents_eda_ready.csv"
REPORT_DIR = ROOT / "reports" / "eda"
FIGURES_DIR = REPORT_DIR / "figures"
TABLES_DIR = REPORT_DIR / "tables"
MAPS_DIR = REPORT_DIR / "maps"

VALID_COORDINATE_STATUSES = {"as_provided_lon_lat", "swapped_lat_lon"}
EXPECTED_COUNTS = {
    "total_rows": 720155,
    "excluded_category_rows": 426,
    "valid_coordinate_rows": 717868,
    "map_eda_rows": 717615,
}

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
FONT_REGULAR = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FONT_ARABIC = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"


def ensure_dirs() -> None:
    for path in [REPORT_DIR, FIGURES_DIR, TABLES_DIR, MAPS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def font(size: int, bold: bool = False, arabic: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_ARABIC if arabic else (FONT_BOLD if bold else FONT_REGULAR)
    return ImageFont.truetype(path, size)


def parse_dt(value: str) -> datetime | None:
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def pct(part: int, whole: int) -> str:
    if not whole:
        return "0.00%"
    return f"{part / whole * 100:.2f}%"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def truncate(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: str, size: int, bold: bool = False) -> None:
    draw.text(xy, text, fill=fill, font=font(size, bold=bold))


def make_canvas(title: str, subtitle: str = "") -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (1600, 900), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1600, 86), fill="#0B2545")
    draw_text(draw, (54, 24), title, "white", 34, bold=True)
    if subtitle:
        draw_text(draw, (56, 62), subtitle, "#D7E3F3", 18)
    return image, draw


def horizontal_bar_chart(
    rows: list[tuple[str, int]],
    path: Path,
    title: str,
    subtitle: str,
    color: str = "#2E74B5",
) -> None:
    image, draw = make_canvas(title, subtitle)
    left, top, right, bottom = 520, 135, 1510, 820
    label_x = 54
    n = len(rows)
    max_value = max(value for _, value in rows) if rows else 1
    bar_gap = 7
    bar_h = max(18, int((bottom - top - bar_gap * (n - 1)) / max(n, 1)))
    axis_font = font(18)
    label_font = font(18)
    for i, (label, value) in enumerate(rows):
        y = top + i * (bar_h + bar_gap)
        draw_text(draw, (label_x, y + 3), truncate(label, 42), "#1F2933", 18)
        bar_w = int((right - left) * value / max_value)
        draw.rectangle((left, y, left + bar_w, y + bar_h), fill=color)
        draw.text((left + bar_w + 12, y + 2), f"{value:,}", fill="#111827", font=axis_font)
    draw.line((left, bottom + 10, right, bottom + 10), fill="#A0AEC0", width=2)
    draw.text((left, bottom + 24), "Incident records", fill="#4A5568", font=label_font)
    image.save(path)


def vertical_bar_chart(
    rows: list[tuple[str, int]],
    path: Path,
    title: str,
    subtitle: str,
    color: str = "#3B7A57",
    rotate_labels: bool = False,
) -> None:
    image, draw = make_canvas(title, subtitle)
    left, top, right, bottom = 105, 140, 1505, 760
    max_value = max(value for _, value in rows) if rows else 1
    draw.line((left, top, left, bottom), fill="#A0AEC0", width=2)
    draw.line((left, bottom, right, bottom), fill="#A0AEC0", width=2)
    bar_gap = 8
    bar_w = max(10, int((right - left - bar_gap * (len(rows) + 1)) / max(len(rows), 1)))
    for i, (label, value) in enumerate(rows):
        x0 = left + bar_gap + i * (bar_w + bar_gap)
        h = int((bottom - top) * value / max_value)
        y0 = bottom - h
        draw.rectangle((x0, y0, x0 + bar_w, bottom), fill=color)
        if len(rows) <= 12:
            draw.text((x0, y0 - 26), f"{value:,}", fill="#111827", font=font(16))
        label_text = truncate(label, 11)
        if rotate_labels:
            label_img = Image.new("RGBA", (160, 32), (255, 255, 255, 0))
            label_draw = ImageDraw.Draw(label_img)
            label_draw.text((0, 5), label_text, fill="#1F2933", font=font(15))
            rotated = label_img.rotate(60, expand=True)
            image.paste(rotated, (x0 - 18, bottom + 8), rotated)
        else:
            draw.text((x0, bottom + 12), label_text, fill="#1F2933", font=font(16))
    for frac in [0.25, 0.5, 0.75, 1.0]:
        y = bottom - int((bottom - top) * frac)
        draw.line((left - 5, y, right, y), fill="#EDF2F7", width=1)
        draw.text((28, y - 10), f"{int(max_value * frac):,}", fill="#4A5568", font=font(14))
    image.save(path)


def line_chart(rows: list[tuple[str, int]], path: Path, title: str, subtitle: str) -> None:
    image, draw = make_canvas(title, subtitle)
    left, top, right, bottom = 110, 135, 1510, 760
    max_value = max(value for _, value in rows) if rows else 1
    n = len(rows)
    draw.line((left, top, left, bottom), fill="#A0AEC0", width=2)
    draw.line((left, bottom, right, bottom), fill="#A0AEC0", width=2)
    points: list[tuple[int, int]] = []
    for i, (_, value) in enumerate(rows):
        x = left + int((right - left) * i / max(n - 1, 1))
        y = bottom - int((bottom - top) * value / max_value)
        points.append((x, y))
    if len(points) >= 2:
        draw.line(points, fill="#2E74B5", width=4)
    for x, y in points[:: max(1, len(points) // 24)]:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#0B2545")
    for frac in [0.25, 0.5, 0.75, 1.0]:
        y = bottom - int((bottom - top) * frac)
        draw.line((left - 5, y, right, y), fill="#EDF2F7", width=1)
        draw.text((26, y - 10), f"{int(max_value * frac):,}", fill="#4A5568", font=font(14))
    if rows:
        step = max(1, len(rows) // 8)
        for i in range(0, len(rows), step):
            label, _ = rows[i]
            x, _ = points[i]
            draw.text((x - 38, bottom + 12), label, fill="#1F2933", font=font(15))
    image.save(path)


def hotspot_static_map(rows: list[dict[str, object]], path: Path) -> None:
    image, draw = make_canvas("Dubai incident hotspot map", "Top 500 0.01-degree bins after coordinate normalization")
    left, top, right, bottom = 130, 135, 1450, 790
    lon_min, lon_max = 54.5, 56.5
    lat_min, lat_max = 24.5, 26.5
    draw.rectangle((left, top, right, bottom), outline="#718096", width=2)
    draw.text((left, bottom + 18), "Longitude", fill="#1F2933", font=font(18))
    draw.text((26, top + 280), "Latitude", fill="#1F2933", font=font(18))
    for i in range(5):
        lon = lon_min + (lon_max - lon_min) * i / 4
        x = left + int((right - left) * i / 4)
        draw.line((x, top, x, bottom), fill="#EDF2F7", width=1)
        draw.text((x - 22, bottom + 45), f"{lon:.1f}", fill="#4A5568", font=font(14))
        lat = lat_min + (lat_max - lat_min) * i / 4
        y = bottom - int((bottom - top) * i / 4)
        draw.line((left, y, right, y), fill="#EDF2F7", width=1)
        draw.text((left - 58, y - 8), f"{lat:.1f}", fill="#4A5568", font=font(14))
    max_count = max(int(row["count"]) for row in rows) if rows else 1
    for row in reversed(rows):
        lon = float(row["longitude"])
        lat = float(row["latitude"])
        count = int(row["count"])
        x = left + int((lon - lon_min) / (lon_max - lon_min) * (right - left))
        y = bottom - int((lat - lat_min) / (lat_max - lat_min) * (bottom - top))
        intensity = math.sqrt(count / max_count)
        radius = int(3 + 18 * intensity)
        red = 170 + int(65 * intensity)
        green = 120 - int(90 * intensity)
        blue = 40 - int(30 * intensity)
        fill = (max(0, min(255, red)), max(0, min(255, green)), max(0, min(255, blue)))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline="#7F1D1D")
    draw.text((1165, 815), "Larger/darker circles = more incidents", fill="#4A5568", font=font(17))
    image.save(path)


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def arabic_cell(value: object) -> str:
    return r"{\arabicfont " + latex_escape(value) + "}"


def format_int(value: int) -> str:
    return f"{value:,}"


def aggregate() -> dict[str, object]:
    if not INPUT_PATH.exists():
        raise SystemExit(f"Missing input file: {INPUT_PATH}")

    total_rows = 0
    category_rows = 0
    excluded_category_rows = 0
    valid_coordinate_rows = 0
    map_eda_rows = 0
    invalid_time_rows = 0
    min_time: datetime | None = None
    max_time: datetime | None = None

    type_counter: Counter[tuple[str, str]] = Counter()
    raw_counter: Counter[str] = Counter()
    severity_counter: Counter[str] = Counter()
    month_counter: Counter[str] = Counter()
    year_counter: Counter[str] = Counter()
    day_counter: Counter[str] = Counter()
    hour_counter: Counter[int] = Counter()
    coordinate_counter: Counter[str] = Counter()
    hotspot_counter: Counter[tuple[float, float]] = Counter()

    with INPUT_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            include = row["include_in_eda"] == "true"
            coordinate_status = row["coordinate_status"]
            raw_counter[row["acci_name"]] += 1
            coordinate_counter[coordinate_status] += 1
            if coordinate_status in VALID_COORDINATE_STATUSES:
                valid_coordinate_rows += 1

            dt = parse_dt(row["acci_time"])
            if dt is None:
                invalid_time_rows += 1
            else:
                min_time = dt if min_time is None or dt < min_time else min_time
                max_time = dt if max_time is None or dt > max_time else max_time

            if not include:
                excluded_category_rows += 1
                continue

            category_rows += 1
            type_counter[(row["incident_type_code"], row["incident_type_en"])] += 1
            severity_counter[row["severity_code"]] += 1

            if dt is not None:
                month_counter[dt.strftime("%Y-%m")] += 1
                year_counter[str(dt.year)] += 1
                day_counter[DAY_ORDER[dt.weekday()]] += 1
                hour_counter[dt.hour] += 1

            if coordinate_status in VALID_COORDINATE_STATUSES:
                map_eda_rows += 1
                lon = float(row["longitude"])
                lat = float(row["latitude"])
                lon_bin = math.floor(lon * 100) / 100 + 0.005
                lat_bin = math.floor(lat * 100) / 100 + 0.005
                hotspot_counter[(round(lat_bin, 3), round(lon_bin, 3))] += 1

    results = {
        "total_rows": total_rows,
        "category_rows": category_rows,
        "excluded_category_rows": excluded_category_rows,
        "valid_coordinate_rows": valid_coordinate_rows,
        "map_eda_rows": map_eda_rows,
        "invalid_time_rows": invalid_time_rows,
        "min_time": min_time,
        "max_time": max_time,
        "type_counter": type_counter,
        "raw_counter": raw_counter,
        "severity_counter": severity_counter,
        "month_counter": month_counter,
        "year_counter": year_counter,
        "day_counter": day_counter,
        "hour_counter": hour_counter,
        "coordinate_counter": coordinate_counter,
        "hotspot_counter": hotspot_counter,
    }
    for key, expected in EXPECTED_COUNTS.items():
        actual = results[key]
        if actual != expected:
            raise SystemExit(f"{key} expected {expected}, got {actual}")
    return results


def build_outputs(results: dict[str, object]) -> dict[str, list[dict[str, object]]]:
    total_rows = int(results["total_rows"])
    category_rows = int(results["category_rows"])
    map_eda_rows = int(results["map_eda_rows"])
    type_counter: Counter[tuple[str, str]] = results["type_counter"]  # type: ignore[assignment]
    raw_counter: Counter[str] = results["raw_counter"]  # type: ignore[assignment]
    severity_counter: Counter[str] = results["severity_counter"]  # type: ignore[assignment]
    month_counter: Counter[str] = results["month_counter"]  # type: ignore[assignment]
    year_counter: Counter[str] = results["year_counter"]  # type: ignore[assignment]
    day_counter: Counter[str] = results["day_counter"]  # type: ignore[assignment]
    hour_counter: Counter[int] = results["hour_counter"]  # type: ignore[assignment]
    coordinate_counter: Counter[str] = results["coordinate_counter"]  # type: ignore[assignment]
    hotspot_counter: Counter[tuple[float, float]] = results["hotspot_counter"]  # type: ignore[assignment]

    summary_rows = [
        {"metric": "Total raw records", "value": total_rows},
        {"metric": "Category/temporal EDA records", "value": category_rows},
        {"metric": "Rows excluded from category-level EDA", "value": results["excluded_category_rows"]},
        {"metric": "Rows with valid normalized coordinates", "value": results["valid_coordinate_rows"]},
        {"metric": "Map-EDA usable rows", "value": map_eda_rows},
        {"metric": "Invalid occurrence timestamps", "value": results["invalid_time_rows"]},
        {"metric": "Occurrence start", "value": results["min_time"]},
        {"metric": "Occurrence end", "value": results["max_time"]},
    ]

    top_types = [
        {
            "rank": rank,
            "incident_type_code": code,
            "incident_type_en": label,
            "count": count,
            "share_of_category_eda": pct(count, category_rows),
        }
        for rank, ((code, label), count) in enumerate(type_counter.most_common(20), 1)
    ]
    top_raw = [
        {
            "rank": rank,
            "acci_name": name,
            "count": count,
            "share_of_all_rows": pct(count, total_rows),
        }
        for rank, (name, count) in enumerate(raw_counter.most_common(20), 1)
    ]
    severity_order = ["minor", "moderate", "severe", "unknown"]
    severity_rows = [
        {
            "severity_code": severity,
            "count": severity_counter[severity],
            "share_of_category_eda": pct(severity_counter[severity], category_rows),
        }
        for severity in severity_order
    ]
    month_rows = [{"month": month, "count": month_counter[month]} for month in sorted(month_counter)]
    year_rows = [{"year": year, "count": year_counter[year]} for year in sorted(year_counter)]
    day_rows = [{"day_of_week": day, "count": day_counter[day]} for day in DAY_ORDER]
    hour_rows = [{"hour": hour, "count": hour_counter[hour]} for hour in range(24)]
    coord_statuses = [
        "as_provided_lon_lat",
        "swapped_lat_lon",
        "zero_coordinate",
        "missing_coordinate",
        "invalid_coordinate",
        "out_of_bounds",
    ]
    coordinate_rows = [
        {
            "coordinate_status": status,
            "count": coordinate_counter[status],
            "share_of_all_rows": pct(coordinate_counter[status], total_rows),
        }
        for status in coord_statuses
    ]
    coordinate_rows.extend(
        [
            {
                "coordinate_status": "valid_after_normalization",
                "count": results["valid_coordinate_rows"],
                "share_of_all_rows": pct(int(results["valid_coordinate_rows"]), total_rows),
            },
            {
                "coordinate_status": "map_eda_usable_after_filters",
                "count": map_eda_rows,
                "share_of_all_rows": pct(map_eda_rows, total_rows),
            },
        ]
    )
    hotspot_rows = [
        {
            "rank": rank,
            "latitude": f"{lat:.3f}",
            "longitude": f"{lon:.3f}",
            "count": count,
            "share_of_map_eda_rows": pct(count, map_eda_rows),
        }
        for rank, ((lat, lon), count) in enumerate(hotspot_counter.most_common(500), 1)
    ]

    outputs = {
        "summary": summary_rows,
        "top_types": top_types,
        "top_raw": top_raw,
        "severity": severity_rows,
        "monthly": month_rows,
        "yearly": year_rows,
        "day": day_rows,
        "hour": hour_rows,
        "coordinate": coordinate_rows,
        "hotspot": hotspot_rows,
    }

    write_csv(TABLES_DIR / "summary_metrics.csv", ["metric", "value"], summary_rows)
    write_csv(TABLES_DIR / "top_incident_types.csv", ["rank", "incident_type_code", "incident_type_en", "count", "share_of_category_eda"], top_types)
    write_csv(TABLES_DIR / "top_raw_categories.csv", ["rank", "acci_name", "count", "share_of_all_rows"], top_raw)
    write_csv(TABLES_DIR / "severity_distribution.csv", ["severity_code", "count", "share_of_category_eda"], severity_rows)
    write_csv(TABLES_DIR / "monthly_incidents.csv", ["month", "count"], month_rows)
    write_csv(TABLES_DIR / "yearly_incidents.csv", ["year", "count"], year_rows)
    write_csv(TABLES_DIR / "day_of_week_incidents.csv", ["day_of_week", "count"], day_rows)
    write_csv(TABLES_DIR / "hour_of_day_incidents.csv", ["hour", "count"], hour_rows)
    write_csv(TABLES_DIR / "coordinate_validation_summary.csv", ["coordinate_status", "count", "share_of_all_rows"], coordinate_rows)
    write_csv(TABLES_DIR / "hotspot_bins_top500.csv", ["rank", "latitude", "longitude", "count", "share_of_map_eda_rows"], hotspot_rows)
    return outputs


def build_figures(outputs: dict[str, list[dict[str, object]]]) -> None:
    horizontal_bar_chart(
        [(str(row["incident_type_en"]), int(row["count"])) for row in outputs["top_types"][:15]],
        FIGURES_DIR / "top_incident_types.png",
        "Top translated incident types",
        "Category/temporal EDA population, top 15 shown",
    )
    vertical_bar_chart(
        [(str(row["severity_code"]).title(), int(row["count"])) for row in outputs["severity"]],
        FIGURES_DIR / "severity_distribution.png",
        "Severity distribution",
        "Based on parsed severity labels in the cleaned category map",
        color="#C05621",
    )
    line_chart(
        [(str(row["month"]), int(row["count"])) for row in outputs["monthly"]],
        FIGURES_DIR / "monthly_incidents.png",
        "Monthly incident records",
        "Occurrence month from August 2018 to June 2026",
    )
    vertical_bar_chart(
        [(str(row["day_of_week"]), int(row["count"])) for row in outputs["day"]],
        FIGURES_DIR / "day_of_week_incidents.png",
        "Incidents by day of week",
        "Category/temporal EDA population",
        color="#3B7A57",
        rotate_labels=False,
    )
    vertical_bar_chart(
        [(str(row["hour"]), int(row["count"])) for row in outputs["hour"]],
        FIGURES_DIR / "hour_of_day_incidents.png",
        "Incidents by hour of day",
        "Hour extracted from occurrence timestamp",
        color="#805AD5",
        rotate_labels=False,
    )
    coordinate_fig_rows = [
        (str(row["coordinate_status"]), int(row["count"]))
        for row in outputs["coordinate"]
        if row["coordinate_status"]
        in {"as_provided_lon_lat", "swapped_lat_lon", "zero_coordinate", "missing_coordinate", "out_of_bounds"}
    ]
    horizontal_bar_chart(
        coordinate_fig_rows,
        FIGURES_DIR / "coordinate_validation_summary.png",
        "Coordinate validation summary",
        "Before and after coordinate-order normalization",
        color="#4A5568",
    )
    hotspot_static_map(outputs["hotspot"], FIGURES_DIR / "dubai_hotspot_static.png")


def build_map(outputs: dict[str, list[dict[str, object]]]) -> None:
    hotspot_data = [
        {
            "rank": int(row["rank"]),
            "lat": float(row["latitude"]),
            "lon": float(row["longitude"]),
            "count": int(row["count"]),
        }
        for row in outputs["hotspot"]
    ]
    max_count = max((row["count"] for row in hotspot_data), default=1)
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Dubai traffic incident hotspot map</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; color: #111827; }}
    header {{ padding: 14px 18px; border-bottom: 1px solid #d8dee9; }}
    h1 {{ margin: 0; font-size: 20px; }}
    p {{ margin: 6px 0 0; font-size: 13px; color: #4b5563; }}
    #map {{ height: calc(100vh - 86px); min-height: 560px; }}
    .legend {{ background: white; padding: 10px 12px; border: 1px solid #cbd5e1; border-radius: 4px; line-height: 1.35; }}
  </style>
</head>
<body>
  <header>
    <h1>Dubai traffic incident hotspot map</h1>
    <p>Top 500 aggregated 0.01-degree bins from normalized coordinates. This map uses {EXPECTED_COUNTS["map_eda_rows"]:,} map-EDA usable records, not raw point markers.</p>
  </header>
  <div id="map"></div>
  <script>
    const hotspotData = {json.dumps(hotspot_data, ensure_ascii=False)};
    const maxCount = {max_count};
    const map = L.map('map').setView([25.18, 55.30], 10);
    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 18,
      attribution: '&copy; OpenStreetMap contributors'
    }}).addTo(map);
    function colorFor(count) {{
      const t = Math.sqrt(count / maxCount);
      const r = Math.round(175 + 70 * t);
      const g = Math.round(120 - 85 * t);
      const b = Math.round(35 - 25 * t);
      return `rgb(${{r}}, ${{g}}, ${{b}})`;
    }}
    hotspotData.forEach((d) => {{
      const radius = 4 + 18 * Math.sqrt(d.count / maxCount);
      L.circleMarker([d.lat, d.lon], {{
        radius,
        color: '#7f1d1d',
        weight: 1,
        fillColor: colorFor(d.count),
        fillOpacity: 0.72
      }}).addTo(map).bindPopup(
        `<strong>Rank ${{d.rank}}</strong><br>Incidents: ${{d.count.toLocaleString()}}<br>Lat: ${{d.lat.toFixed(3)}}<br>Lon: ${{d.lon.toFixed(3)}}`
      );
    }});
    L.control({{position: 'bottomright'}}).onAdd = function() {{
      const div = L.DomUtil.create('div', 'legend');
      div.innerHTML = '<strong>Hotspot bins</strong><br>Larger and darker circles indicate more incident records.';
      return div;
    }}.addTo(map);
  </script>
</body>
</html>
"""
    (MAPS_DIR / "dubai_hotspot_map.html").write_text(html_text, encoding="utf-8")


def table_rows_latex(rows: list[dict[str, object]], columns: list[str], arabic_columns: set[str] | None = None) -> str:
    arabic_columns = arabic_columns or set()
    lines = []
    for row in rows:
        values = []
        for col in columns:
            value = row[col]
            if col in arabic_columns:
                values.append(arabic_cell(value))
            else:
                values.append(latex_escape(value))
        lines.append(" & ".join(values) + r" \\")
    return "\n".join(lines)


def build_report(results: dict[str, object], outputs: dict[str, list[dict[str, object]]]) -> None:
    summary = {str(row["metric"]): row["value"] for row in outputs["summary"]}
    top_type_rows = outputs["top_types"][:10]
    severity_rows = outputs["severity"]
    yearly_rows = outputs["yearly"]
    coordinate_rows = outputs["coordinate"][:6]
    top_raw_rows = outputs["top_raw"]
    tex = rf"""\documentclass[11pt,a4paper]{{article}}
\usepackage[margin=0.78in]{{geometry}}
\usepackage{{iftex}}
\ifPDFTeX
  \usepackage[T1]{{fontenc}}
  \usepackage[utf8]{{inputenc}}
  \usepackage{{lmodern}}
\else
  \usepackage{{fontspec}}
  \setmainfont[
    Path=/usr/local/texlive/2026/texmf-dist/fonts/opentype/public/lm/,
    UprightFont={{lmroman10-regular.otf}},
    BoldFont={{lmroman10-bold.otf}},
    ItalicFont={{lmroman10-italic.otf}},
    BoldItalicFont={{lmroman10-bolditalic.otf}}
  ]{{Latin Modern Roman}}
  \newfontfamily\arabicfont[
    Path=/System/Library/Fonts/Supplemental/,
    UprightFont={{Arial Unicode.ttf}}
  ]{{Arial Unicode}}
\fi
\usepackage{{array}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{longtable}}
\usepackage{{microtype}}
\usepackage{{tabularx}}
\usepackage[table]{{xcolor}}
\usepackage[hidelinks]{{hyperref}}
\graphicspath{{{{figures/}}}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{0.38em}}
\newcolumntype{{Y}}{{>{{\raggedright\arraybackslash}}X}}
\newcolumntype{{L}}[1]{{>{{\raggedright\arraybackslash}}p{{#1}}}}
\definecolor{{bitsblue}}{{HTML}}{{1F4D78}}
\definecolor{{lightgray}}{{HTML}}{{F2F4F7}}
\begin{{document}}
\begin{{center}}
{{\Large\bfseries Explainable AI for Non-Recurrent Traffic Incident Risk Prediction in Dubai}}\\[0.25em]
{{\large Initial EDA Results Packet}}\\[0.4em]
Sparsh Aggarwal \quad 2022A7TS0279U \quad Supervisor: Dr. Sujala D. Shetty
\end{{center}}

\section*{{Purpose}}
This packet records the first descriptive analysis from the cleaned Dubai traffic incident snapshot. It does not contain model training, H3 grid construction, or prediction results. The goal is to show that the dataset has been cleaned enough for exploratory analysis and map validation.

\section*{{Dataset summary}}
\begin{{tabularx}}{{\textwidth}}{{L{{0.44\textwidth}}Y}}
\toprule
\textbf{{Metric}} & \textbf{{Value}} \\
\midrule
Total raw records & {format_int(int(summary["Total raw records"]))} \\
Occurrence date range & {latex_escape(summary["Occurrence start"])} to {latex_escape(summary["Occurrence end"])} \\
Category/temporal EDA records & {format_int(int(summary["Category/temporal EDA records"]))} \\
Rows excluded from category-level EDA & {format_int(int(summary["Rows excluded from category-level EDA"]))} \\
Rows with valid normalized coordinates & {format_int(int(summary["Rows with valid normalized coordinates"]))} \\
Map-EDA usable rows & {format_int(int(summary["Map-EDA usable rows"]))} \\
\bottomrule
\end{{tabularx}}

\section*{{Category and severity patterns}}
\begin{{figure}}[h]
\centering
\includegraphics[width=0.96\textwidth]{{top_incident_types.png}}
\caption{{Top translated incident types.}}
\end{{figure}}

\begin{{figure}}[h]
\centering
\includegraphics[width=0.82\textwidth]{{severity_distribution.png}}
\caption{{Severity distribution from parsed category labels.}}
\end{{figure}}

\begin{{tabularx}}{{\textwidth}}{{r L{{0.47\textwidth}} r r}}
\toprule
\textbf{{Rank}} & \textbf{{Incident type}} & \textbf{{Count}} & \textbf{{Share}} \\
\midrule
{table_rows_latex(top_type_rows, ["rank", "incident_type_en", "count", "share_of_category_eda"])}
\bottomrule
\end{{tabularx}}

\section*{{Temporal patterns}}
\begin{{figure}}[h]
\centering
\includegraphics[width=0.96\textwidth]{{monthly_incidents.png}}
\caption{{Monthly incident records.}}
\end{{figure}}

\begin{{figure}}[h]
\centering
\includegraphics[width=0.96\textwidth]{{day_of_week_incidents.png}}
\caption{{Incidents by day of week.}}
\end{{figure}}

\begin{{figure}}[h]
\centering
\includegraphics[width=0.96\textwidth]{{hour_of_day_incidents.png}}
\caption{{Incidents by hour of day.}}
\end{{figure}}

\begin{{tabularx}}{{0.55\textwidth}}{{L{{0.22\textwidth}} r}}
\toprule
\textbf{{Year}} & \textbf{{Incident records}} \\
\midrule
{table_rows_latex(yearly_rows, ["year", "count"])}
\bottomrule
\end{{tabularx}}

\section*{{Coordinate validation and hotspot map}}
The raw coordinate fields mix longitude/latitude and latitude/longitude order. The EDA-ready file keeps the raw fields and adds normalized \texttt{{longitude}}, \texttt{{latitude}}, and \texttt{{coordinate\_status}} fields.

\begin{{figure}}[h]
\centering
\includegraphics[width=0.96\textwidth]{{coordinate_validation_summary.png}}
\caption{{Coordinate validation summary.}}
\end{{figure}}

\begin{{tabularx}}{{\textwidth}}{{L{{0.46\textwidth}} r r}}
\toprule
\textbf{{Coordinate status}} & \textbf{{Rows}} & \textbf{{Share of all rows}} \\
\midrule
{table_rows_latex(coordinate_rows, ["coordinate_status", "count", "share_of_all_rows"])}
\bottomrule
\end{{tabularx}}

\begin{{figure}}[h]
\centering
\includegraphics[width=0.96\textwidth]{{dubai_hotspot_static.png}}
\caption{{Static hotspot plot from the top 500 0.01-degree coordinate bins.}}
\end{{figure}}

An interactive version is saved at \texttt{{reports/eda/maps/dubai\_hotspot\_map.html}}. It uses aggregated bins rather than raw point markers.

\section*{{Top raw Arabic categories}}
This supporting table keeps the source labels visible for supervisor review. English modeling fields use the translated incident-type columns from the mapping table.

\small
\begin{{longtable}}{{r L{{0.54\textwidth}} r r}}
\toprule
\textbf{{Rank}} & \textbf{{Raw Arabic category}} & \textbf{{Count}} & \textbf{{Share}} \\
\midrule
\endfirsthead
\toprule
\textbf{{Rank}} & \textbf{{Raw Arabic category}} & \textbf{{Count}} & \textbf{{Share}} \\
\midrule
\endhead
{table_rows_latex(top_raw_rows, ["rank", "acci_name", "count", "share_of_all_rows"], arabic_columns={"acci_name"})}
\bottomrule
\end{{longtable}}
\normalsize

\section*{{Next step}}
The next technical step is to convert the normalized coordinates into grid cells, choose an H3 resolution, and build the first zone/time-window modeling table. Before that, the hotspot map should be visually checked to confirm that normalized points fall over Dubai.

\end{{document}}
"""
    (REPORT_DIR / "initial_eda_report.tex").write_text(tex, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    results = aggregate()
    outputs = build_outputs(results)
    build_figures(outputs)
    build_map(outputs)
    build_report(results, outputs)
    print(f"Wrote EDA outputs to {REPORT_DIR}")


if __name__ == "__main__":
    main()
