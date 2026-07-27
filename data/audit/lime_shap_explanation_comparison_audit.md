# Final LIME vs SHAP local explanation comparison audit

- Model: default neighbor-feature XGBoost, H3 resolution 8, 3-hour windows, inside-Dubai full-grid setup.
- Training rows used after deterministic cap: `1000000`
- Raw validation-selected threshold: `0.826884`
- Sigmoid-calibrated validation-selected threshold: `0.079937`
- LIME background rows: `2000` maximum, sampled deterministically from final training candidates.
- Local examples compared: `4`
- SHAP explains the XGBoost raw-margin score.
- LIME explains the local XGBoost class-probability surface using a model-agnostic surrogate.
- Sigmoid-calibrated risk is included for dashboard display but is not treated as a separate explanation method.
- Methods explain the trained model, not causal effects.
- Rejected hard-negative sample and tuned XGBoost configuration are not used.

## Example summary

```csv
example_id,candidate_index,h3_cell_res8,window_index,window_start,actual_label,incident_count,raw_xgboost_score,sigmoid_calibrated_risk,raw_threshold,calibrated_threshold,predicted_label,calibrated_predicted_label,top5_feature_overlap,top_shap_features,top_lime_features
high_risk_true_positive,2500356,8843acd9a7fffff,21290,2025-11-25 12:00:00,1,1,0.9714374542236328,0.3504677414894104,0.8268836140632629,0.07993732392787933,1,1,5,hist_cell_hour_risk; prev_7d_incident_count; neighbor_prev_3h_incident_count; year; prev_24h_incident_count,hist_cell_hour_risk > 0.04; prev_7d_incident_count > 2.00; prev_3h_incident_count <= 0.00; prev_24h_incident_count > 0.00; year > 2022.00
high_risk_false_positive,478367,8843a13687fffff,19741,2025-05-15 21:00:00,0,0,0.9739475250244141,0.37074822187423706,0.8268836140632629,0.07993732392787933,1,1,4,hist_cell_hour_risk; prev_7d_incident_count; neighbor_prev_3h_incident_count; prev_3h_incident_count; year,hist_cell_hour_risk > 0.04; prev_7d_incident_count > 2.00; prev_3h_incident_count > 0.00; neighbor_prev_24h_incident_count > 2.00; neighbor_prev_7d_incident_count > 12.00
low_score_false_negative,129134,8843acd887fffff,19473,2025-04-12 09:00:00,1,1,6.944371853023767e-05,2.7448872970126104e-06,0.8268836140632629,0.07993732392787933,0,0,4,hist_cell_hour_risk; hist_cell_risk; neighbor_prev_7d_incident_count; prev_7d_incident_count; year,hist_cell_hour_risk <= 0.00; neighbor_prev_7d_incident_count <= 1.00; prev_7d_incident_count <= 0.00; neighbor_prev_3h_incident_count <= 0.00; hist_cell_risk <= 0.00
low_risk_true_negative,2904581,8843acc329fffff,21600,2026-01-03 06:00:00,0,0,5.968302139081061e-05,2.3840850644774036e-06,0.8268836140632629,0.07993732392787933,0,0,4,hist_cell_hour_risk; hist_cell_risk; neighbor_prev_7d_incident_count; prev_7d_incident_count; is_weekend,hist_cell_hour_risk <= 0.00; prev_3h_incident_count <= 0.00; neighbor_prev_7d_incident_count <= 1.00; prev_7d_incident_count <= 0.00; hist_cell_risk <= 0.00
```

## Top contributor rows

```csv
example_id,method,rank,feature,feature_value,contribution,direction,contribution_scale
high_risk_true_positive,SHAP,1,hist_cell_hour_risk,0.2345864623785019,2.2886502742767334,increases_risk,xgboost_raw_margin
high_risk_true_positive,SHAP,2,prev_7d_incident_count,16.0,0.3927761912345886,increases_risk,xgboost_raw_margin
high_risk_true_positive,SHAP,3,neighbor_prev_3h_incident_count,4.0,0.32398688793182373,increases_risk,xgboost_raw_margin
high_risk_true_positive,SHAP,4,year,2025.0,0.12370584905147552,increases_risk,xgboost_raw_margin
high_risk_true_positive,SHAP,5,prev_24h_incident_count,4.0,0.09561972320079803,increases_risk,xgboost_raw_margin
high_risk_true_positive,SHAP,6,neighbor_prev_24h_incident_count,11.0,0.07248465716838837,increases_risk,xgboost_raw_margin
high_risk_true_positive,SHAP,7,neighbor_prev_7d_severity_weight_sum,36.0,0.04912808537483215,increases_risk,xgboost_raw_margin
high_risk_true_positive,SHAP,8,neighbor_prev_7d_incident_count,41.0,0.03984654322266579,increases_risk,xgboost_raw_margin
high_risk_true_positive,LIME,1,hist_cell_hour_risk > 0.04,,0.41749156841534657,increases_risk,local_probability_surrogate
high_risk_true_positive,LIME,2,prev_7d_incident_count > 2.00,,0.05867439135834116,increases_risk,local_probability_surrogate
high_risk_true_positive,LIME,3,prev_3h_incident_count <= 0.00,,-0.04352843621344578,decreases_risk,local_probability_surrogate
high_risk_true_positive,LIME,4,prev_24h_incident_count > 0.00,,0.03336598569188091,increases_risk,local_probability_surrogate
high_risk_true_positive,LIME,5,year > 2022.00,,0.027473196308826477,increases_risk,local_probability_surrogate
high_risk_true_positive,LIME,6,neighbor_prev_24h_incident_count > 2.00,,0.023131168848244062,increases_risk,local_probability_surrogate
high_risk_true_positive,LIME,7,day_of_week_Friday <= 0.00,,0.02264572332873972,increases_risk,local_probability_surrogate
high_risk_true_positive,LIME,8,neighbor_prev_7d_severity_weight_sum > 12.00,,0.017921719773832796,increases_risk,local_probability_surrogate
high_risk_false_positive,SHAP,1,hist_cell_hour_risk,0.225677028298378,2.2413313388824463,increases_risk,xgboost_raw_margin
high_risk_false_positive,SHAP,2,prev_7d_incident_count,17.0,0.3995806872844696,increases_risk,xgboost_raw_margin
high_risk_false_positive,SHAP,3,neighbor_prev_3h_incident_count,6.0,0.3045238256454468,increases_risk,xgboost_raw_margin
high_risk_false_positive,SHAP,4,prev_3h_incident_count,1.0,0.186416357755661,increases_risk,xgboost_raw_margin
high_risk_false_positive,SHAP,5,year,2025.0,0.15053240954875946,increases_risk,xgboost_raw_margin
high_risk_false_positive,SHAP,6,prev_24h_incident_count,5.0,0.10520335286855698,increases_risk,xgboost_raw_margin
high_risk_false_positive,SHAP,7,neighbor_prev_24h_incident_count,11.0,0.07184723764657974,increases_risk,xgboost_raw_margin
high_risk_false_positive,SHAP,8,neighbor_prev_24h_severity_weight_sum,13.0,0.03214871510863304,increases_risk,xgboost_raw_margin
high_risk_false_positive,LIME,1,hist_cell_hour_risk > 0.04,,0.40221332860521336,increases_risk,local_probability_surrogate
high_risk_false_positive,LIME,2,prev_7d_incident_count > 2.00,,0.057564426420634876,increases_risk,local_probability_surrogate
high_risk_false_positive,LIME,3,prev_3h_incident_count > 0.00,,0.033551239508056885,increases_risk,local_probability_surrogate
high_risk_false_positive,LIME,4,neighbor_prev_24h_incident_count > 2.00,,0.03184941584399774,increases_risk,local_probability_surrogate
high_risk_false_positive,LIME,5,neighbor_prev_7d_incident_count > 12.00,,0.022798654488419598,increases_risk,local_probability_surrogate
high_risk_false_positive,LIME,6,prev_24h_incident_count > 0.00,,0.02238923350870625,increases_risk,local_probability_surrogate
high_risk_false_positive,LIME,7,is_weekend <= 0.00,,0.020792720690029593,increases_risk,local_probability_surrogate
high_risk_false_positive,LIME,8,prev_7d_severity_weight_sum > 2.00,,0.019721399265439154,increases_risk,local_probability_surrogate
low_score_false_negative,SHAP,1,hist_cell_hour_risk,0.0,-8.316967010498047,decreases_risk,xgboost_raw_margin
low_score_false_negative,SHAP,2,hist_cell_risk,0.0,-0.647057056427002,decreases_risk,xgboost_raw_margin
low_score_false_negative,SHAP,3,neighbor_prev_7d_incident_count,0.0,-0.26084840297698975,decreases_risk,xgboost_raw_margin
low_score_false_negative,SHAP,4,prev_7d_incident_count,0.0,-0.12888583540916443,decreases_risk,xgboost_raw_margin
low_score_false_negative,SHAP,5,year,2025.0,0.09871810674667358,increases_risk,xgboost_raw_margin
low_score_false_negative,SHAP,6,is_weekend,1.0,-0.08166393637657166,decreases_risk,xgboost_raw_margin
low_score_false_negative,SHAP,7,neighbor_prev_7d_active_cell_count,0.0,-0.07597900182008743,decreases_risk,xgboost_raw_margin
low_score_false_negative,SHAP,8,neighbor_prev_7d_severity_weight_sum,0.0,-0.050405725836753845,decreases_risk,xgboost_raw_margin
low_score_false_negative,LIME,1,hist_cell_hour_risk <= 0.00,,-0.4285360681457578,decreases_risk,local_probability_surrogate
low_score_false_negative,LIME,2,neighbor_prev_7d_incident_count <= 1.00,,-0.05474953967699239,decreases_risk,local_probability_surrogate
low_score_false_negative,LIME,3,prev_7d_incident_count <= 0.00,,-0.05473167402427553,decreases_risk,local_probability_surrogate
low_score_false_negative,LIME,4,neighbor_prev_3h_incident_count <= 0.00,,-0.03341235702430841,decreases_risk,local_probability_surrogate
low_score_false_negative,LIME,5,hist_cell_risk <= 0.00,,-0.02884496409776843,decreases_risk,local_probability_surrogate
low_score_false_negative,LIME,6,neighbor_prev_7d_active_cell_count <= 1.00,,-0.027494689947978414,decreases_risk,local_probability_surrogate
low_score_false_negative,LIME,7,year > 2022.00,,0.026715514444973443,increases_risk,local_probability_surrogate
low_score_false_negative,LIME,8,day_of_week_Saturday > 0.00,,0.025623291128775812,increases_risk,local_probability_surrogate
low_risk_true_negative,SHAP,1,hist_cell_hour_risk,0.0,-8.356596946716309,decreases_risk,xgboost_raw_margin
low_risk_true_negative,SHAP,2,hist_cell_risk,0.0,-0.5910738706588745,decreases_risk,xgboost_raw_margin
low_risk_true_negative,SHAP,3,neighbor_prev_7d_incident_count,0.0,-0.28385433554649353,decreases_risk,xgboost_raw_margin
low_risk_true_negative,SHAP,4,prev_7d_incident_count,0.0,-0.12885624170303345,decreases_risk,xgboost_raw_margin
low_risk_true_negative,SHAP,5,is_weekend,1.0,-0.08207039535045624,decreases_risk,xgboost_raw_margin
low_risk_true_negative,SHAP,6,neighbor_prev_7d_active_cell_count,0.0,-0.0760248526930809,decreases_risk,xgboost_raw_margin
low_risk_true_negative,SHAP,7,hour_block,2.0,-0.06303787976503372,decreases_risk,xgboost_raw_margin
low_risk_true_negative,SHAP,8,neighbor_prev_24h_incident_count,0.0,-0.060826625674963,decreases_risk,xgboost_raw_margin
low_risk_true_negative,LIME,1,hist_cell_hour_risk <= 0.00,,-0.42239098293578636,decreases_risk,local_probability_surrogate
low_risk_true_negative,LIME,2,prev_3h_incident_count <= 0.00,,-0.054803836277572245,decreases_risk,local_probability_surrogate
low_risk_true_negative,LIME,3,neighbor_prev_7d_incident_count <= 1.00,,-0.05060494638473871,decreases_risk,local_probability_surrogate
low_risk_true_negative,LIME,4,prev_7d_incident_count <= 0.00,,-0.04801120691875821,decreases_risk,local_probability_surrogate
low_risk_true_negative,LIME,5,hist_cell_risk <= 0.00,,-0.035492294027340526,decreases_risk,local_probability_surrogate
low_risk_true_negative,LIME,6,neighbor_prev_24h_incident_count <= 0.00,,-0.029024669654345747,decreases_risk,local_probability_surrogate
low_risk_true_negative,LIME,7,neighbor_prev_3h_incident_count <= 0.00,,-0.027330337454470634,decreases_risk,local_probability_surrogate
low_risk_true_negative,LIME,8,0.00 < is_weekend <= 1.00,,-0.024004103294837677,decreases_risk,local_probability_surrogate
```
