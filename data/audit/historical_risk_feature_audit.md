# Historical-risk feature audit

- H3 resolution: `8`
- Time window hours: `3`
- Inside-Dubai H3 cells: `1305`
- Validation candidate rows: `4461795`
- Test candidate rows: `4463100`
- Test positive cell/windows: `108224`
- Model family audited: `XGBoost`, plus one historical-score-only baseline.
- Validation/test historical-risk features use train-period full-grid denominators.
- Expanding-history variant changes only training-row historical-risk features.

## Main comparison

- Full feature PR-AUC: `0.096701`; F1: `0.161484`.
- No-history PR-AUC: `0.084316`; F1: `0.149811`.
- Expanding-history PR-AUC: `0.096282`; F1: `0.161303`.
- PR-AUC drop after removing historical-risk features: `0.012385`.
- F1 drop after removing historical-risk features: `0.011673`.
- PR-AUC change with expanding training history: `-0.000419`.
- F1 change with expanding training history: `-0.000182`.

## Feature sets

| feature_set_id | model_type | train_history_mode | has_historical_features | feature_count |
| --- | --- | --- | --- | --- |
| all_features_train_period_hist | XGBoost | train_period | True | 15 |
| all_features_expanding_train_hist | XGBoost | expanding_train_only | True | 15 |
| no_historical_risk | XGBoost | not_used | False | 11 |
| temporal_lag_only | XGBoost | not_used | False | 10 |
| temporal_only | XGBoost | not_used | False | 5 |
| historical_score_only | Historical score | train_period | True | 1 |

## Metrics

| feature_set_id | test_roc_auc | test_pr_auc | test_precision | test_recall | test_f1 | threshold | test_tn | test_fp | test_fn | test_tp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_features_train_period_hist | 0.806693 | 0.096701 | 0.116506 | 0.263029 | 0.161484 | 0.832275 | 4139012 | 215864 | 79758 | 28466 |
| all_features_expanding_train_hist | 0.806573 | 0.096282 | 0.115958 | 0.264886 | 0.161303 | 0.827464 | 4136324 | 218552 | 79557 | 28667 |
| no_historical_risk | 0.775733 | 0.084316 | 0.102797 | 0.276076 | 0.149811 | 0.818001 | 4094103 | 260773 | 78346 | 29878 |
| temporal_lag_only | 0.776002 | 0.084999 | 0.103771 | 0.271308 | 0.150122 | 0.81926 | 4101288 | 253588 | 78862 | 29362 |
| temporal_only | 0.598289 | 0.03107 | 0.032859 | 0.479432 | 0.061503 | 0.624942 | 2827712 | 1527164 | 56338 | 51886 |
| historical_score_only | 0.799047 | 0.088238 | 0.106301 | 0.267861 | 0.152201 | 0.062688 | 4111159 | 243717 | 79235 | 28989 |

## Top-k hotspot metrics

| feature_set_id | k | recall_at_k | precision_at_k | positive_window_hit_rate_at_k | incident_recall_at_k |
| --- | --- | --- | --- | --- | --- |
| all_features_train_period_hist | 5 | 0.026732 | 0.169181 | 0.609865 | 0.02922 |
| all_features_train_period_hist | 10 | 0.047928 | 0.151667 | 0.795003 | 0.051692 |
| all_features_train_period_hist | 20 | 0.083614 | 0.132295 | 0.918322 | 0.089169 |
| all_features_expanding_train_hist | 5 | 0.027018 | 0.170994 | 0.61467 | 0.029636 |
| all_features_expanding_train_hist | 10 | 0.04754 | 0.150439 | 0.795964 | 0.051458 |
| all_features_expanding_train_hist | 20 | 0.083706 | 0.132442 | 0.919283 | 0.089256 |
| no_historical_risk | 5 | 0.022694 | 0.143626 | 0.552851 | 0.024849 |
| no_historical_risk | 10 | 0.04147 | 0.131228 | 0.748879 | 0.044745 |
| no_historical_risk | 20 | 0.07296 | 0.115439 | 0.888533 | 0.077694 |
| temporal_lag_only | 5 | 0.022851 | 0.14462 | 0.557976 | 0.025109 |
| temporal_lag_only | 10 | 0.040934 | 0.129532 | 0.745676 | 0.044355 |
| temporal_lag_only | 20 | 0.073736 | 0.116667 | 0.889174 | 0.078536 |
| temporal_only | 5 | 0.001599 | 0.010117 | 0.054132 | 0.001552 |
| temporal_only | 10 | 0.005664 | 0.017924 | 0.178091 | 0.005603 |
| temporal_only | 20 | 0.017362 | 0.027471 | 0.438181 | 0.017251 |
| historical_score_only | 5 | 0.027157 | 0.171871 | 0.615631 | 0.029862 |
| historical_score_only | 10 | 0.045933 | 0.145351 | 0.782191 | 0.04968 |
| historical_score_only | 20 | 0.081747 | 0.129342 | 0.910634 | 0.087105 |
