# Initial baseline modeling audit

- Input file: `data/processed/grid_time_model_sample_res8_3h.csv`
- Raw model-sample rows: `4069044`
- Included geo scopes: `inside_dubai, peripheral_observed`
- Rows after geo-scope filter: `3902250`
- Excluded outside-UAE flagged rows: `166794`
- Minimum window index: `0`
- Maximum window index: `22794`
- Window count: `22795`
- Train/validation cut window index: `15956`
- Validation/test cut window index: `19375`
- Leakage columns excluded from features: `incident_count, minor_count, moderate_count, severe_count, severity_weight_sum, unknown_count`
- Feature columns used: `hour_block, is_weekend, month, year, prev_3h_incident_count, prev_24h_incident_count, prev_7d_incident_count, prev_24h_severity_weight_sum, prev_7d_severity_weight_sum, hist_cell_hour_risk, hist_cell_risk, hist_hour_risk, hist_global_risk, day_of_week, geo_scope`

## Split summary

| split | rows | positives | negatives | positive_rate | min_window_index | max_window_index | min_window_start | max_window_start |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 2685197 | 423682 | 2261515 | 0.15778432643861884 | 0 | 15955 | 2018-08-13T06:00:00 | 2024-01-28T15:00:00 |
| validation | 605660 | 122635 | 483025 | 0.20248159033120894 | 15956 | 19374 | 2024-01-28T18:00:00 | 2025-03-31T00:00:00 |
| test | 611393 | 128743 | 482650 | 0.21057323194737265 | 19375 | 22794 | 2025-03-31T03:00:00 | 2026-06-01T12:00:00 |

## Test metrics

| model | test_roc_auc | test_pr_auc | test_precision | test_recall | test_f1 | threshold |
| --- | --- | --- | --- | --- | --- | --- |
| Historical risk | 0.8850373042643308 | 0.6387003997603704 | 0.5693555806465103 | 0.7402732575751692 | 0.6436613154855572 | 0.24786324799060822 |
| Logistic Regression | 0.884696092442442 | 0.6441939100298966 | 0.5696207069620707 | 0.7404985125404876 | 0.6439158817589588 | 0.5337942566554547 |
| Random Forest | 0.890407122530187 | 0.6542151502086369 | 0.5740337932029034 | 0.7420442276473284 | 0.6473149098645852 | 0.6267421571266877 |
| XGBoost | 0.8898767592126371 | 0.65439361996585 | 0.5727240642926789 | 0.7436831517053354 | 0.6471025561307939 | 0.708366334438324 |

## Sampled-candidate top-k hotspot recall

| model | k | metric_name | windows_with_positives | total_positive_hotspots | positives_captured | weighted_recall | mean_window_recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Historical risk | 5 | sampled_candidate_hotspot_recall_at_5 | 3123 | 128743 | 11762 | 0.09136030696814584 | 0.10806522481316602 |
| Historical risk | 10 | sampled_candidate_hotspot_recall_at_10 | 3123 | 128743 | 22376 | 0.17380362427471785 | 0.20273285237161925 |
| Historical risk | 20 | sampled_candidate_hotspot_recall_at_20 | 3123 | 128743 | 41328 | 0.321011627816658 | 0.36553375480643957 |
| Logistic Regression | 5 | sampled_candidate_hotspot_recall_at_5 | 3123 | 128743 | 11827 | 0.09186518878696318 | 0.10862087482282139 |
| Logistic Regression | 10 | sampled_candidate_hotspot_recall_at_10 | 3123 | 128743 | 22519 | 0.17491436427611598 | 0.2045178989103515 |
| Logistic Regression | 20 | sampled_candidate_hotspot_recall_at_20 | 3123 | 128743 | 41500 | 0.32234762278337464 | 0.36695150977790864 |
| Random Forest | 5 | sampled_candidate_hotspot_recall_at_5 | 3123 | 128743 | 11860 | 0.09212151340267044 | 0.1090630590144508 |
| Random Forest | 10 | sampled_candidate_hotspot_recall_at_10 | 3123 | 128743 | 22547 | 0.1751318518288373 | 0.2046666053554551 |
| Random Forest | 20 | sampled_candidate_hotspot_recall_at_20 | 3123 | 128743 | 41679 | 0.3237379896382716 | 0.36869747514540097 |
| XGBoost | 5 | sampled_candidate_hotspot_recall_at_5 | 3123 | 128743 | 11887 | 0.09233123354279456 | 0.10966096831565558 |
| XGBoost | 10 | sampled_candidate_hotspot_recall_at_10 | 3123 | 128743 | 22552 | 0.17517068889182325 | 0.2043549772571328 |
| XGBoost | 20 | sampled_candidate_hotspot_recall_at_20 | 3123 | 128743 | 41704 | 0.3239321749532013 | 0.36892862098091 |
