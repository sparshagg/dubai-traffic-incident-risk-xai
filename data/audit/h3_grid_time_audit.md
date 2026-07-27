# H3 grid-time dataset audit

- Input file: `data/processed/traffic_incidents_eda_ready.csv`
- H3 resolution: `8`
- Time window hours: `3`
- Negative sampling ratio: `5`
- Random seed: `42`
- Map-usable incident rows: `717615`
- Incident point output: `data/processed/incident_points_h3_res8.csv`
- Incident point output rows: `717615`
- Positive grid-time output: `data/processed/grid_time_incident_counts_res8_3h.csv`
- Positive grid-time rows: `678174`
- Model sample output: `data/processed/grid_time_model_sample_res8_3h.csv`
- Model sample rows: `4069044`
- Positive model rows: `678174`
- Negative model rows: `3390870`
- Negative to positive ratio: `5.00`
- Study universe cells: `3584`
- Window start: `2018-08-13T06:00:00`
- Window end: `2026-06-01T12:00:00`
- Window count: `22795`
- Duplicate model cell/window rows: `0`
- All model windows aligned to 3-hour boundary: `true`
- Lag consistency violations detected: `0`

## Output previews

- Incident point sample: `data/audit/incident_points_h3_res8_sample.csv`
- Positive grid-time sample: `data/audit/grid_time_incident_counts_res8_3h_sample.csv`
- Model sample preview: `data/audit/grid_time_model_sample_res8_3h_preview.csv`
- Cell scope summary: `data/audit/h3_cell_scope_summary_res8.csv`
