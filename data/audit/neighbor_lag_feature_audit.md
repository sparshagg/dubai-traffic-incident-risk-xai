# Neighboring H3 lag feature experiment

- H3 resolution: `8`
- Time window hours: `3`
- Inside-Dubai cells: `1305`
- Neighbor definition: H3 ring-1 cells from `h3.grid_disk(cell, 1)`, excluding the center cell and excluding cells outside the inside-Dubai universe.
- Inside-neighbor count range: `2` to `6`
- Mean inside-neighbor count: `5.681`
- Cells with zero inside-Dubai neighbors: `0`
- Neighbor lag features use only windows before the target window.
- Unknown-severity incidents remain in incident counts, but unknown severity contributes zero to neighbor severity-weighted sums.
- PR-AUC change vs current XGBoost: `0.002498`
- F1 change vs current XGBoost: `0.002980`

## Feature sets

| model | feature_set_id | feature_count | uses_neighbor_features |
| --- | --- | --- | --- |
| Historical risk | historical_score_only | 1 | False |
| XGBoost current features | current_xgboost_features | 15 | False |
| XGBoost + neighbor lags | neighbor_lag_xgboost_features | 23 | True |

## Test metrics

| model | test_roc_auc | test_pr_auc | test_precision | test_recall | test_f1 | threshold |
| --- | --- | --- | --- | --- | --- | --- |
| Historical risk | 0.799047 | 0.088238 | 0.106301 | 0.267861 | 0.152201 | 0.062688 |
| XGBoost current features | 0.806693 | 0.096701 | 0.116506 | 0.263029 | 0.161484 | 0.832275 |
| XGBoost + neighbor lags | 0.811954 | 0.0992 | 0.117866 | 0.272001 | 0.164464 | 0.826884 |

## Top-k hotspot metrics

| model | k | recall_at_k | precision_at_k | positive_window_hit_rate_at_k | incident_recall_at_k |
| --- | --- | --- | --- | --- | --- |
| Historical risk | 5 | 0.027157 | 0.171871 | 0.615631 | 0.029862 |
| Historical risk | 10 | 0.045933 | 0.145351 | 0.782191 | 0.04968 |
| Historical risk | 20 | 0.081747 | 0.129342 | 0.910634 | 0.087105 |
| XGBoost current features | 5 | 0.026732 | 0.169181 | 0.609865 | 0.02922 |
| XGBoost current features | 10 | 0.047928 | 0.151667 | 0.795003 | 0.051692 |
| XGBoost current features | 20 | 0.083614 | 0.132295 | 0.918322 | 0.089169 |
| XGBoost + neighbor lags | 5 | 0.027406 | 0.17345 | 0.618834 | 0.030157 |
| XGBoost + neighbor lags | 10 | 0.047198 | 0.149357 | 0.790199 | 0.050938 |
| XGBoost + neighbor lags | 20 | 0.084436 | 0.133596 | 0.921525 | 0.09034 |
