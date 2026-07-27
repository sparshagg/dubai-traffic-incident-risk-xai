# XGBoost probability calibration audit

- H3 resolution: `8`
- Time window hours: `3`
- Inside-Dubai cells: `1305`
- Training source: default neighbor-feature XGBoost trained on the existing sampled-negative table.
- Training cap: `1000000`
- Training rows used: `1000000`
- Rejected hard-negative sample is not used.
- Tuned XGBoost configuration is not used.
- Validation candidates: `4461795`
- Test candidates: `4463100`
- Test positive cell/windows: `108224`
- Calibration methods fitted on validation predictions only.
- Reported calibration metrics are computed on the untouched full-grid test period.
- Selected method: `sigmoid`
- Promoted for dashboard probability: `True`
- Selection reason: Sigmoid was selected because it improved calibration and satisfied the ranking guardrail.

## Calibration metrics

| method | test_pr_auc | test_f1 | test_brier_score | test_log_loss | test_ece_equal_frequency | test_mce_equal_frequency |
| --- | --- | --- | --- | --- | --- | --- |
| uncalibrated | 0.0992 | 0.164464 | 0.212706 | 0.59532 | 0.351782 | 0.741469 |
| sigmoid | 0.0992 | 0.164464 | 0.022674 | 0.098328 | 0.001324 | 0.005035 |
| isotonic | 0.097634 | 0.164464 | 0.022672 | 0.097902 | 0.001293 | 0.00271 |

## Top-k hotspot metrics

| model | k | recall_at_k | precision_at_k | positive_window_hit_rate_at_k | incident_recall_at_k |
| --- | --- | --- | --- | --- | --- |
| uncalibrated | 5 | 0.027406 | 0.17345 | 0.618834 | 0.030157 |
| uncalibrated | 10 | 0.047198 | 0.149357 | 0.790199 | 0.050938 |
| uncalibrated | 20 | 0.084436 | 0.133596 | 0.921525 | 0.09034 |
| sigmoid | 5 | 0.027406 | 0.17345 | 0.618834 | 0.030157 |
| sigmoid | 10 | 0.047198 | 0.149357 | 0.790199 | 0.050938 |
| sigmoid | 20 | 0.084436 | 0.133596 | 0.921525 | 0.09034 |
| isotonic | 5 | 0.027194 | 0.172105 | 0.61467 | 0.029922 |
| isotonic | 10 | 0.047346 | 0.149825 | 0.7918 | 0.051042 |
| isotonic | 20 | 0.084685 | 0.133991 | 0.920243 | 0.090548 |

## Calibration bins

| method | bin_id | row_count | predicted_mean | observed_rate | absolute_error | positive_count |
| --- | --- | --- | --- | --- | --- | --- |
| uncalibrated | 1 | 446310 | 8.9e-05 | 0.000361 | 0.000272 | 161 |
| uncalibrated | 2 | 446310 | 0.024388 | 0.001521 | 0.022867 | 679 |
| uncalibrated | 3 | 446310 | 0.109283 | 0.003448 | 0.105835 | 1539 |
| uncalibrated | 4 | 446310 | 0.206814 | 0.006592 | 0.200223 | 2942 |
| uncalibrated | 5 | 446310 | 0.306634 | 0.009274 | 0.29736 | 4139 |
| uncalibrated | 6 | 446310 | 0.401251 | 0.014409 | 0.386842 | 6431 |
| uncalibrated | 7 | 446310 | 0.508832 | 0.02149 | 0.487343 | 9591 |
| uncalibrated | 8 | 446310 | 0.628913 | 0.033038 | 0.595876 | 14745 |
| uncalibrated | 9 | 446310 | 0.733123 | 0.053389 | 0.679734 | 23828 |
| uncalibrated | 10 | 446310 | 0.840434 | 0.098965 | 0.741469 | 44169 |
| sigmoid | 1 | 446310 | 3e-06 | 0.000361 | 0.000357 | 161 |
| sigmoid | 2 | 446310 | 0.000641 | 0.001521 | 0.00088 | 679 |
| sigmoid | 3 | 446310 | 0.002891 | 0.003448 | 0.000558 | 1539 |
| sigmoid | 4 | 446310 | 0.005803 | 0.006592 | 0.000789 | 2942 |
| sigmoid | 5 | 446310 | 0.009446 | 0.009274 | 0.000172 | 4139 |
| sigmoid | 6 | 446310 | 0.013847 | 0.014409 | 0.000562 | 6431 |
| sigmoid | 7 | 446310 | 0.020685 | 0.02149 | 0.000804 | 9591 |
| sigmoid | 8 | 446310 | 0.032407 | 0.033038 | 0.00063 | 14745 |
| sigmoid | 9 | 446310 | 0.049942 | 0.053389 | 0.003447 | 23828 |
| sigmoid | 10 | 446310 | 0.093929 | 0.098965 | 0.005035 | 44169 |
| isotonic | 1 | 446310 | 0.000266 | 0.000376 | 0.00011 | 168 |
| isotonic | 2 | 446310 | 0.001215 | 0.001582 | 0.000367 | 706 |
| isotonic | 3 | 446310 | 0.002903 | 0.003374 | 0.000471 | 1506 |
| isotonic | 4 | 446310 | 0.005497 | 0.006621 | 0.001124 | 2955 |
| isotonic | 5 | 446310 | 0.008487 | 0.009254 | 0.000766 | 4130 |
| isotonic | 6 | 446310 | 0.013018 | 0.014423 | 0.001405 | 6437 |
| isotonic | 7 | 446310 | 0.019556 | 0.021472 | 0.001916 | 9583 |
| isotonic | 8 | 446310 | 0.031058 | 0.033203 | 0.002145 | 14819 |
| isotonic | 9 | 446310 | 0.050607 | 0.053317 | 0.00271 | 23796 |
| isotonic | 10 | 446310 | 0.096952 | 0.098864 | 0.001912 | 44124 |
