# Inside-Dubai full-grid evaluation audit

- H3 resolution: `8`
- Time window hours: `3`
- Inside-Dubai H3 cells: `1305`
- Training source: sampled inside-Dubai model table rows from windows `0` to `15955`
- Validation candidate windows: `15956` to `19374`
- Test candidate windows: `19375` to `22794`
- Inside-Dubai positive grid-time rows: `572937`
- Inside-Dubai training rows used before model caps: `1217062`
- Historical priors: training-period inside-Dubai full-grid denominators
- Leakage columns excluded from model features: `incident_count, minor_count, moderate_count, severe_count, severity_weight_sum, unknown_count`
- Feature columns used: `hour_block, is_weekend, month, year, prev_3h_incident_count, prev_24h_incident_count, prev_7d_incident_count, prev_24h_severity_weight_sum, prev_7d_severity_weight_sum, hist_cell_hour_risk, hist_cell_risk, hist_hour_risk, hist_global_risk, day_of_week, geo_scope`

## Split summary

| split | window_start_index | window_end_index | windows | inside_dubai_cells | full_grid_candidate_rows | positive_cell_windows | negative_cell_windows | positive_rate | sampled_training_rows_used | min_window_start | max_window_start |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0 | 15955 | 15956 | 1305 | 20822580 | 360278 | 20462302 | 0.017302274742130898 | 1217062 | 2018-08-13 06:00:00 | 2024-01-28 15:00:00 |
| validation | 15956 | 19374 | 3419 | 1305 | 4461795 | 104435 | 4357360 | 0.02340649895389636 |  | 2024-01-28 18:00:00 | 2025-03-31 00:00:00 |
| test | 19375 | 22794 | 3420 | 1305 | 4463100 | 108224 | 4354876 | 0.02424861643252448 |  | 2025-03-31 03:00:00 | 2026-06-01 12:00:00 |

## Test metrics

| model | test_roc_auc | test_pr_auc | test_precision | test_recall | test_f1 | threshold |
| --- | --- | --- | --- | --- | --- | --- |
| Historical risk | 0.799047 | 0.088238 | 0.106301 | 0.267861 | 0.152201 | 0.062688 |
| Logistic Regression | 0.799469 | 0.093207 | 0.108674 | 0.279199 | 0.156451 | 0.828054 |
| Random Forest | 0.806195 | 0.09535 | 0.113259 | 0.27213 | 0.159948 | 0.799127 |
| XGBoost | 0.806693 | 0.096701 | 0.116506 | 0.263029 | 0.161484 | 0.832275 |

## Full-grid top-k hotspot metrics

| model | k | recall_at_k | precision_at_k | positive_window_hit_rate_at_k | incident_recall_at_k |
| --- | --- | --- | --- | --- | --- |
| Historical risk | 5 | 0.027157 | 0.171871 | 0.615631 | 0.029862 |
| Historical risk | 10 | 0.045933 | 0.145351 | 0.782191 | 0.04968 |
| Historical risk | 20 | 0.081747 | 0.129342 | 0.910634 | 0.087105 |
| Logistic Regression | 5 | 0.027212 | 0.172222 | 0.611787 | 0.029975 |
| Logistic Regression | 10 | 0.046958 | 0.148596 | 0.78155 | 0.05086 |
| Logistic Regression | 20 | 0.083105 | 0.131491 | 0.910314 | 0.088701 |
| Random Forest | 5 | 0.026103 | 0.165205 | 0.600256 | 0.028665 |
| Random Forest | 10 | 0.046699 | 0.147778 | 0.778668 | 0.050738 |
| Random Forest | 20 | 0.082662 | 0.130789 | 0.912236 | 0.088345 |
| XGBoost | 5 | 0.026732 | 0.169181 | 0.609865 | 0.02922 |
| XGBoost | 10 | 0.047928 | 0.151667 | 0.795003 | 0.051692 |
| XGBoost | 20 | 0.083614 | 0.132295 | 0.918322 | 0.089169 |
