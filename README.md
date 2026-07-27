# Explainable AI for Non-Recurrent Traffic Incident Risk Prediction in Dubai

This repository contains the submission package for a final-year B.E. Computer Science thesis at BITS Pilani, Dubai Campus.

The project builds an explainable machine-learning pipeline for predicting non-recurrent traffic incident risk in Dubai. The final model predicts whether at least one incident is likely in an H3 grid zone during a 3-hour time window, then explains high-risk predictions using SHAP with LIME as a supporting local comparison.

## Repository Contents

- `reports/final_report/main.pdf` - final thesis report.
- `reports/final_report/` - LaTeX source for the final report.
- `reports/research_paper/main.pdf` - IEEE-style research paper draft based on the thesis.
- `reports/research_paper/` - LaTeX source for the research paper draft.
- `reports/submission/` - ready-to-submit PDFs with clear filenames, including the final report, Turnitin similarity report, and research paper draft.
- `dashboard/app.py` - Streamlit dashboard for rolling prediction replay and model explanation.
- `scripts/` - reproducible data preparation, EDA, grid construction, modeling, explainability, calibration, and dashboard-preparation scripts.
- `reports/modeling/` - aggregate model result tables and report-ready figures.
- `reports/eda/` - exploratory data analysis tables, figures, and hotspot map.
- `reports/dashboard/screenshots/` - dashboard screenshots used in the final report.
- `resources/geo/` - Dubai and UAE GeoJSON boundary files used for spatial processing.
- `data/mappings/` - translated incident-category mapping tables.
- `data/audit/` - compact audit summaries for the reproducible pipeline.
- `references.bib` - bibliography used by the LaTeX report.

The raw traffic incident CSV and large generated datasets are not included because of size and submission-safety reasons.

Public repository:

[https://github.com/sparshagg/dubai-traffic-incident-risk-xai](https://github.com/sparshagg/dubai-traffic-incident-risk-xai)

## Data Source

The primary dataset is the Dubai Police traffic incident dataset from Dubai's Data & Statistics portal:

[https://data.dubai/en/l/469979](https://data.dubai/en/l/469979)

The report uses a frozen local snapshot named:

`traffic_incidents_2026-06-01_10-44-37_1.csv`

Expected raw dataset facts for this snapshot:

- 720,155 incident-level records.
- 233 unique Arabic incident-category strings.
- Incident time range from 2018-08-13 to 2026-06-01.
- Raw fields: `acci_id`, `acci_time`, `acci_name`, `acci_x`, `acci_y`, and `load_timestamp`.

To reproduce the complete pipeline, download the dataset from the source portal and place the CSV at the repository root with the filename above.

## Environment Setup

Create a Python environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-data.txt
```

The dashboard can be started with:

```bash
streamlit run dashboard/app.py
```

For the primary English map view, provide a MapTiler key through the environment:

```bash
export MAPTILER_API_KEY="your_key_here"
```

If a MapTiler key is not available, the dashboard falls back to a simpler map view.

## Reproducible Pipeline

Run the scripts in numeric order when reproducing the full analysis:

```bash
python scripts/data/01_audit_raw_incidents.py
python scripts/data/02_extract_acci_name_categories.py
python scripts/data/04_apply_category_mapping.py
python scripts/data/05_build_eda_ready_incidents.py
python scripts/eda/01_generate_initial_eda.py
python scripts/grid/01_build_h3_grid_time_dataset.py
python scripts/modeling/01_train_initial_baselines.py
python scripts/modeling/02_run_full_grid_evaluation.py
python scripts/modeling/03_audit_historical_risk_features.py
python scripts/modeling/04_generate_shap_explanations.py
python scripts/modeling/05_run_h3_time_sensitivity.py
python scripts/modeling/06_compare_lime_shap_explanations.py
python scripts/modeling/07_evaluate_neighbor_lag_features.py
python scripts/modeling/08_evaluate_hard_negative_training.py
python scripts/modeling/09_tune_xgboost_neighbor_features.py
python scripts/modeling/10_calibrate_xgboost_probabilities.py
python scripts/dashboard/01_prepare_dashboard_data.py
```

Large generated outputs are written under `data/processed/` and `data/dashboard/`, which are intentionally ignored.

## Final Model Result

The selected final model is XGBoost with temporal, historical-risk, same-cell lag, and neighboring-cell lag features. It is evaluated on the inside-Dubai full-grid test set using H3 resolution 8 and 3-hour windows.

| Metric | Final value |
| --- | ---: |
| Test candidates | 4,463,100 |
| Positive cell-windows | 108,224 |
| ROC-AUC | 0.812 |
| PR-AUC | 0.099 |
| F1-score | 0.164 |
| Recall | 0.272 |
| Precision | 0.118 |
| Top-20 incidents captured | 10,416 |

The model is best interpreted as a hotspot-ranking and decision-support tool, not as a perfect incident detector. Traffic incident prediction remains difficult because most grid-time combinations contain no incident.

## Dashboard Scope

The dashboard demonstrates a rolling prediction replay on historical test windows before the dataset cutoff. It shows where the model ranked risk highest, when the risk occurs, and why the model gave a high score for selected zones.

The final model uses recent 3-hour, 24-hour, and 7-day lag features. Therefore, predictions after 2026-06-01 require updated consecutive incident records to compute those lag features. The dashboard does not claim live deployment or real-time traffic-control capability.

## Report Compilation

The final report is written in LaTeX. With a working TeX installation, compile it from the final-report folder:

```bash
cd reports/final_report
latexmk -xelatex main.tex
```

The compiled submission PDF is included as `reports/final_report/main.pdf`.

The conference-style research paper draft is included as `reports/research_paper/main.pdf`. It is also attached in Appendix C of the final thesis report.

For direct submission, use the clearly named PDFs in `reports/submission/`:

- `Sparsh_Aggarwal_2022A7TS0279U_Final_Thesis_Report.pdf`
- `Sparsh_Aggarwal_2022A7TS0279U_Turnitin_Similarity_Report.pdf`
- `Sparsh_Aggarwal_2022A7TS0279U_Research_Paper_Draft.pdf`

## Limitations

- The work uses an open historical dataset snapshot, not a live traffic feed.
- Weather, road-speed, road-network, and event-calendar features are not included in the final model.
- SHAP and LIME explain model behavior; they do not prove causal reasons for incidents.
- The dashboard is a thesis prototype for historical replay and explanation.
