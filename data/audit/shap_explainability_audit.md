# Final SHAP explainability audit

- SHAP version: `0.52.0`
- Model: default neighbor-feature XGBoost.
- H3 resolution: `8`
- Time window hours: `3`
- Inside-Dubai H3 cells: `1305`
- Validation candidate rows: `4461795`
- Test candidate rows: `4463100`
- Test positive cell/windows: `108224`
- Training rows used after deterministic cap: `1000000`
- Raw validation-selected threshold: `0.826884`
- Sigmoid-calibrated validation-selected threshold: `0.079937`
- Background/reference sample rows selected: `2000`
- Final feature path: current features plus ring-1 inside-Dubai neighbor lag features.
- Rejected hard-negative sample is not used.
- Tuned XGBoost configuration is not used.
- SHAP values are reported on the XGBoost raw-margin scale, not as causal effects.
- Global explanation sample rows: `10000`

## Top grouped SHAP features

| feature | mean_abs_shap | mean_shap |
| --- | --- | --- |
| hist_cell_hour_risk | 2.173746 | -1.711507 |
| year | 0.25335 | 0.25335 |
| hist_cell_risk | 0.098837 | -0.075093 |
| prev_7d_incident_count | 0.093655 | -0.045876 |
| neighbor_prev_7d_incident_count | 0.065966 | -0.040875 |
| is_weekend | 0.049417 | 4.4e-05 |
| neighbor_prev_24h_incident_count | 0.036944 | -0.012146 |
| day_of_week | 0.03555 | 0.000274 |
| neighbor_prev_7d_severity_weight_sum | 0.034768 | -0.003763 |
| hist_hour_risk | 0.032014 | -0.008101 |
| hour_block | 0.029378 | -0.009921 |
| neighbor_prev_3h_incident_count | 0.024393 | -0.003182 |

## Local examples

| example_id | h3_cell_res8 | window_start | actual_label | incident_count | raw_xgboost_score | sigmoid_calibrated_risk | predicted_label |
| --- | --- | --- | --- | --- | --- | --- | --- |
| high_risk_true_positive | 8843acd9a7fffff | 2025-11-25 12:00:00 | 1 | 1 | 0.971437 | 0.350468 | 1 |
| high_risk_false_positive | 8843a13687fffff | 2025-05-15 21:00:00 | 0 | 0 | 0.973948 | 0.370748 | 1 |
| low_score_false_negative | 8843acd887fffff | 2025-04-12 09:00:00 | 1 | 1 | 6.9e-05 | 3e-06 | 0 |
| low_risk_true_negative | 8843acc329fffff | 2026-01-03 06:00:00 | 0 | 0 | 6e-05 | 2e-06 | 0 |
