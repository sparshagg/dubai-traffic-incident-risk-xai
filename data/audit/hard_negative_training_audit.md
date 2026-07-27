# Hard-negative training sample experiment

- H3 resolution: `8`
- Time window hours: `3`
- Inside-Dubai cells: `1305`
- Training windows: `0` to `15955`
- Validation windows: `15956` to `19374`
- Test windows: `19375` to `22794`
- Validation candidates: `4461795`
- Test candidates: `4463100`
- Test positive cell/windows: `108224`
- Hard-negative training cap: `1000000`
- Negative mix after positives: `70%` hard negatives and `30%` random negatives
- Inside-neighbor count range: `2` to `6`
- Hard negatives are mined only from training-window full-grid negatives.
- Validation and test remain full-grid and unchanged.
- Unknown-severity incidents remain in incident counts, but unknown severity contributes zero to severity-weighted sums.
- PR-AUC change vs random-sample neighbor XGBoost: `-0.050515`
- F1 change vs random-sample neighbor XGBoost: `-0.062922`
- Top-20 precision change vs random-sample neighbor XGBoost: `-0.072588`

## Training samples

| sample_id | row_count | positive_rows | negative_rows | hard_negative_rows | random_negative_rows | positive_rate | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| current_random_negative_sample | 1000000 | 296022 | 703978 | 0 | 703978 | 0.296022 | Existing sampled training table with deterministic 1,000,000-row cap. |
| hard_negative_sample | 1000000 | 360278 | 639722 | 447805 | 191917 | 0.360278 | All training positives plus 70/30 hard/random negatives; mined from 20,462,302 training negatives. |

## Test metrics

| model | test_roc_auc | test_pr_auc | test_precision | test_recall | test_f1 | threshold |
| --- | --- | --- | --- | --- | --- | --- |
| Historical risk | 0.799047 | 0.088238 | 0.106301 | 0.267861 | 0.152201 | 0.062688 |
| XGBoost neighbor random sample | 0.811954 | 0.0992 | 0.117866 | 0.272001 | 0.164464 | 0.826884 |
| XGBoost neighbor hard negatives | 0.693431 | 0.048684 | 0.059894 | 0.333336 | 0.101542 | 0.868195 |

## Top-k hotspot metrics

| model | k | recall_at_k | precision_at_k | positive_window_hit_rate_at_k | incident_recall_at_k |
| --- | --- | --- | --- | --- | --- |
| Historical risk | 5 | 0.027157 | 0.171871 | 0.615631 | 0.029862 |
| Historical risk | 10 | 0.045933 | 0.145351 | 0.782191 | 0.04968 |
| Historical risk | 20 | 0.081747 | 0.129342 | 0.910634 | 0.087105 |
| XGBoost neighbor random sample | 5 | 0.027406 | 0.17345 | 0.618834 | 0.030157 |
| XGBoost neighbor random sample | 10 | 0.047198 | 0.149357 | 0.790199 | 0.050938 |
| XGBoost neighbor random sample | 20 | 0.084436 | 0.133596 | 0.921525 | 0.09034 |
| XGBoost neighbor hard negatives | 5 | 0.009785 | 0.06193 | 0.290839 | 0.009714 |
| XGBoost neighbor hard negatives | 10 | 0.019571 | 0.06193 | 0.495195 | 0.019419 |
| XGBoost neighbor hard negatives | 20 | 0.038559 | 0.061009 | 0.728059 | 0.038396 |
