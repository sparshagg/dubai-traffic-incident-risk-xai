# Thesis Dashboard

The Streamlit dashboard demonstrates the final model through historical rolling replay. It shows:

- ranked H3 risk zones on an English map,
- selected date and 3-hour time-window views,
- predicted incident-risk scores,
- SHAP reasons for selected zones,
- historical-check and model-evidence panels.

Run it from the repository root:

```bash
streamlit run dashboard/app.py
```

For the primary English map, set a MapTiler key as an environment variable:

```bash
export MAPTILER_API_KEY="your_key_here"
```

The dashboard expects compact generated files under `data/dashboard/`. Generate them with:

```bash
python scripts/dashboard/01_prepare_dashboard_data.py
```

The dashboard is a historical thesis prototype. It does not claim live deployment.
