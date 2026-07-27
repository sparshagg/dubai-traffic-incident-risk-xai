# Streamlit dashboard data audit

- Dashboard type: rolling replay plus historical thesis demonstration.
- Live deployment claim: none.
- Map used by the app: MapTiler vector style through PyDeck when `MAPTILER_API_KEY` is available; Plotly/Esri label-free fallback only if the key is unavailable.
- Replay model path: default neighbor-feature XGBoost.
- Calibration path: sigmoid calibration fitted on validation predictions only.
- Post-cutoff dates are not scored in the active dashboard.
- Warning shown in app: forecasts after 2026-06-01 require updated incident counts to compute recent lag features.
- H3 resolution: `8`
- Time window hours: `3`
- Inside-Dubai cells: `1305`
- Replay date range: `2026-05-25` to `2026-05-31`
- Default replay date: `2026-05-31`
- Replay windows: `56`
- Replay rows written: `5600`
- Replay SHAP contributor rows: `28000`
- Test windows: `3420`
- Test candidate rows: `4463100`
- Test positive cell/windows: `108224`
- Top-risk rows retained per test window: `100`
- Historical dashboard risk rows written: `342000`
- The dashboard data files are generated and ignored by Git; screenshots and source code are committed.

## Generated files

- cell_geojson: `data/dashboard/inside_dubai_h3_cells.geojson`
- risk_scores: `data/dashboard/risk_scores_top100.csv`
- window_summary: `data/dashboard/window_summary.csv`
- rolling_replay_rows: `data/dashboard/rolling_replay_top_zones.csv`
- rolling_replay_window_summary: `data/dashboard/rolling_replay_window_summary.csv`
- rolling_replay_shap_contributors: `data/dashboard/rolling_replay_shap_contributors.csv`
- rolling_replay_metadata: `data/dashboard/rolling_replay_metadata.json`
- metadata: `data/dashboard/dashboard_metadata.json`

## Data checks

- Unique windows in historical risk table: `3420`
- Maximum historical risk rank: `100`
- Minimum calibrated historical risk: `0.00643977`
- Maximum calibrated historical risk: `0.37074822`
- Replay dates: `2026-05-25, 2026-05-26, 2026-05-27, 2026-05-28, 2026-05-29, 2026-05-30, 2026-05-31`
- Replay windows per date: `8`
- Replay maximum rank: `100`
- Replay minimum predicted risk: `0.02735538`
- Replay maximum predicted risk: `0.35009283`

## Replay model features

The rolling replay uses the final neighbor-lag feature set, including recent same-cell and nearby-cell lag features. It is therefore not used for dates after the data cutoff unless updated incident data is available.
