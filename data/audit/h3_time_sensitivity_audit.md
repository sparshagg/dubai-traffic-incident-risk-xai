# H3/time-window sensitivity audit

- Settings compared: `res7_3h`, `res8_1h`, `res8_3h`, `res8_6h`, and `res9_3h`.
- Evaluation scope: inside-Dubai H3 cells only, using the same polygon-plus-observed-inside point rule as the main full-grid evaluation.
- Models compared: historical-risk baseline and XGBoost.
- Split: chronological 70/15/15 by window index for each setting.
- XGBoost training: deterministic sampled training set with a 5:1 negative-to-positive source ratio and a 1,000,000-row fit cap.
- Top-share metric: approximately the same share as top 20 of 1,305 cells, or about 1.5% of cells per test window.

## Decision comparison

- Reference setting PR-AUC (`res8_3h`): `0.056416`.
- Best XGBoost PR-AUC: `res7_3h` with `0.219211`.
- Reference normalized hotspot precision (`res8_3h`): `0.106082` at k `20`.
- Best normalized hotspot precision: `res7_3h` with `0.375828` at k `3`.

## XGBoost metrics

| setting_id | h3_resolution | window_hours | test_roc_auc | test_pr_auc | test_positive_rate | test_pr_auc_lift_vs_base_rate | test_precision | test_recall | test_f1 | threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| res7_3h | 7 | 3 | 0.676705 | 0.219211 | 0.13089 | 1.674772 | 0.180218 | 0.84074 | 0.296812 | 0.999923 |
| res8_1h | 8 | 1 | 0.775826 | 0.024229 | 0.008388 | 2.888676 | 0.032902 | 0.14322 | 0.05351 | 0.999964 |
| res8_3h | 8 | 3 | 0.737118 | 0.056416 | 0.024249 | 2.326579 | 0.073236 | 0.179914 | 0.104098 | 0.99997 |
| res8_6h | 8 | 6 | 0.725374 | 0.096367 | 0.046506 | 2.072133 | 0.091249 | 0.544435 | 0.156302 | 0.999961 |
| res9_3h | 9 | 3 | 0.786579 | 0.013222 | 0.003772 | 3.505423 | 0.023324 | 0.089256 | 0.036984 | 0.999966 |

## XGBoost normalized top-share metrics

| setting_id | k | top_cell_share | test_positive_rate | precision_at_k | precision_lift_vs_base_rate | recall_at_k | positive_window_hit_rate_at_k | incident_recall_at_k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| res7_3h | 3 | 0.013889 | 0.13089 | 0.375828 | 2.871332 | 0.03988 | 0.71909 | 0.049265 |
| res8_1h | 20 | 0.015326 | 0.008388 | 0.041636 | 4.963979 | 0.076076 | 0.57295 | 0.077375 |
| res8_3h | 20 | 0.015326 | 0.024249 | 0.106082 | 4.37476 | 0.067046 | 0.842409 | 0.071059 |
| res8_6h | 20 | 0.015326 | 0.046506 | 0.212807 | 4.575881 | 0.070128 | 0.973197 | 0.079158 |
| res9_3h | 131 | 0.01527 | 0.003772 | 0.024398 | 6.468377 | 0.098771 | 0.933696 | 0.100315 |

## Population summary

| setting_id | inside_dubai_cells | windows | full_grid_candidate_rows | positive_cell_windows | positive_rate | incident_count |
| --- | --- | --- | --- | --- | --- | --- |
| res7_3h | 216 | 3420 | 738720 | 96691 | 0.13089 | 120653 |
| res8_1h | 1305 | 10258 | 13386690 | 112282 | 0.008388 | 115283 |
| res8_3h | 1305 | 3420 | 4463100 | 108224 | 0.024249 | 115298 |
| res8_6h | 1305 | 1710 | 2231550 | 103781 | 0.046506 | 115276 |
| res9_3h | 8579 | 3420 | 29340180 | 110670 | 0.003772 | 113393 |
