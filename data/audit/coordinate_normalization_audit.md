# Coordinate normalization audit

- Source file: `traffic_incidents_2026-06-01_10-44-37_1.csv`
- Mapping file: `data/mappings/acci_name_translation_map.csv`
- Output file: `data/processed/traffic_incidents_eda_ready.csv`
- Row count: `720155`
- Mapping rows: `233`
- Normalized mapping keys used for joining: `226`
- Duplicate normalized mapping keys collapsed for joining: `7`
- Rows missing mapping: `0`
- Rows excluded from category-level EDA: `426`
- Rows with valid normalized coordinates: `717868`
- Rows without usable normalized coordinates: `2287`
- Rows usable for map-based EDA after category and coordinate filters: `717615`
- Anomaly sample file: `data/audit/coordinate_anomaly_samples.csv`
- Anomaly sample rows: `100`

## Coordinate status counts

| Coordinate status | Rows |
| --- | ---: |
| `as_provided_lon_lat` | 31934 |
| `swapped_lat_lon` | 685934 |
| `zero_coordinate` | 2191 |
| `missing_coordinate` | 1 |
| `invalid_coordinate` | 0 |
| `out_of_bounds` | 95 |
