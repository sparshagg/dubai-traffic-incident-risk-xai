# XGBoost neighbor-feature tuning experiment

- H3 resolution: `8`
- Time window hours: `3`
- Inside-Dubai cells: `1305`
- Training source: existing sampled-negative training table with cap `1000000`
- Rejected hard-negative sample is not used.
- Validation candidates: `4461795`
- Test candidates: `4463100`
- Test positive cell/windows: `108224`
- Candidate count: `18`
- Selection rule: highest validation PR-AUC; tie within 0.001 by validation top-20 incident recall, then validation F1.
- Selected candidate: `xgb_tune_12`
- Selected hyperparameters: max_depth `5`, learning_rate `0.1`, n_estimators `240`, min_child_weight `10`
- Test PR-AUC change vs default neighbor XGBoost: `-0.001570`
- Test F1 change vs default neighbor XGBoost: `-0.002420`
- Promoted to current final candidate: `False`

## Selected configuration

| selected_candidate_id | max_depth | learning_rate | n_estimators | min_child_weight | selection_rule | selected_validation_pr_auc | selected_validation_f1 | selected_validation_top20_incident_recall | default_test_pr_auc | tuned_test_pr_auc | test_pr_auc_change | default_test_f1 | tuned_test_f1 | test_f1_change | default_top20_incident_recall | tuned_top20_incident_recall | top20_incident_recall_change | promoted_to_current_final_candidate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xgb_tune_12 | 5 | 0.1 | 240 | 10 | highest validation PR-AUC; tie within 0.001 by validation top-20 incident recall, then validation F1 | 0.097663 | 0.165103 | 0.097664 | 0.0992 | 0.097629 | -0.00157 | 0.164464 | 0.162044 | -0.00242 | 0.09034 | 0.089533 | -0.000807 | False |

## Top validation candidates

| candidate_id | max_depth | learning_rate | n_estimators | min_child_weight | validation_pr_auc | validation_f1 | validation_top20_incident_recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| xgb_tune_06 | 4 | 0.1 | 180 | 10 | 0.098509 | 0.166138 | 0.097339 |
| xgb_tune_14 | 6 | 0.04 | 240 | 10 | 0.098412 | 0.166077 | 0.096987 |
| xgb_tune_09 | 5 | 0.07 | 180 | 5 | 0.098369 | 0.16633 | 0.097384 |
| xgb_tune_10 | 5 | 0.07 | 240 | 10 | 0.098328 | 0.166281 | 0.09733 |
| xgb_tune_04 | 4 | 0.07 | 180 | 10 | 0.098292 | 0.165649 | 0.097303 |
| xgb_tune_08 | 5 | 0.04 | 240 | 10 | 0.09829 | 0.165764 | 0.096599 |
| xgb_tune_05 | 4 | 0.1 | 120 | 5 | 0.098284 | 0.166041 | 0.097258 |
| xgb_tune_13 | 6 | 0.04 | 180 | 5 | 0.098264 | 0.165683 | 0.09733 |

## Test metrics

| model | test_roc_auc | test_pr_auc | test_precision | test_recall | test_f1 | threshold |
| --- | --- | --- | --- | --- | --- | --- |
| Historical risk | 0.799047 | 0.088238 | 0.106301 | 0.267861 | 0.152201 | 0.062688 |
| XGBoost neighbor default | 0.811954 | 0.0992 | 0.117866 | 0.272001 | 0.164464 | 0.826884 |
| XGBoost neighbor tuned | 0.808805 | 0.097629 | 0.115579 | 0.270984 | 0.162044 | 0.834232 |

## Top-k hotspot metrics

| model | k | recall_at_k | precision_at_k | positive_window_hit_rate_at_k | incident_recall_at_k |
| --- | --- | --- | --- | --- | --- |
| Historical risk | 5 | 0.027157 | 0.171871 | 0.615631 | 0.029862 |
| Historical risk | 10 | 0.045933 | 0.145351 | 0.782191 | 0.04968 |
| Historical risk | 20 | 0.081747 | 0.129342 | 0.910634 | 0.087105 |
| XGBoost neighbor default | 5 | 0.027406 | 0.17345 | 0.618834 | 0.030157 |
| XGBoost neighbor default | 10 | 0.047198 | 0.149357 | 0.790199 | 0.050938 |
| XGBoost neighbor default | 20 | 0.084436 | 0.133596 | 0.921525 | 0.09034 |
| XGBoost neighbor tuned | 5 | 0.026815 | 0.169708 | 0.610826 | 0.029541 |
| XGBoost neighbor tuned | 10 | 0.046893 | 0.148392 | 0.783472 | 0.050868 |
| XGBoost neighbor tuned | 20 | 0.083845 | 0.132661 | 0.91672 | 0.089533 |
