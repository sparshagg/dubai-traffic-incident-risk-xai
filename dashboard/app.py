from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = ROOT / "data" / "dashboard"
EDA_TABLE_DIR = ROOT / "reports" / "eda" / "tables"
MODEL_TABLE_DIR = ROOT / "reports" / "modeling" / "tables"

VIEWS = ["Rolling replay", "Zone explanation", "Historical check", "Model evidence", "Scope notes"]
LABEL_FREE_BASEMAP_LAYERS = [
    {
        "below": "traces",
        "sourcetype": "raster",
        "source": [
            "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"
        ],
        "sourceattribution": "Esri, HERE, Garmin, FAO, NOAA, USGS, OpenStreetMap contributors",
    }
]

ENGLISH_CONTEXT_LABELS = [
    {"name": "Dubai", "lat": 25.2048, "lon": 55.2708},
    {"name": "Sharjah", "lat": 25.3463, "lon": 55.4209},
    {"name": "Jebel Ali", "lat": 25.0118, "lon": 55.0617},
    {"name": "Downtown Dubai", "lat": 25.1972, "lon": 55.2744},
    {"name": "Dubai Marina", "lat": 25.0800, "lon": 55.1400},
]


st.set_page_config(
    page_title="Dubai incident-risk replay",
    page_icon="",
    layout="wide",
)

st.markdown(
    """
    <style>
    [data-testid="stToolbar"], header[data-testid="stHeader"], #MainMenu, footer {visibility: hidden;}
    .stMetric [data-testid="stMetricValue"] {font-size: 1.65rem;}
    .small-note {color: #555; font-size: 0.92rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def fmt_int(value: float | int) -> str:
    return f"{int(value):,}"


def fmt_pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def required_path(path: Path) -> Path:
    if not path.exists():
        st.error("Dashboard data has not been prepared yet. Run the dashboard data-preparation step first.")
        st.stop()
    return path


@st.cache_data(show_spinner=False)
def load_dashboard_data() -> dict:
    metadata = load_json(required_path(DASHBOARD_DIR / "dashboard_metadata.json"))
    replay_metadata = load_json(required_path(DASHBOARD_DIR / "rolling_replay_metadata.json"))
    cells = load_json(required_path(DASHBOARD_DIR / "inside_dubai_h3_cells.geojson"))
    risks = load_csv(required_path(DASHBOARD_DIR / "risk_scores_top100.csv"))
    windows = load_csv(required_path(DASHBOARD_DIR / "window_summary.csv"))
    replay = load_csv(required_path(DASHBOARD_DIR / "rolling_replay_top_zones.csv"))
    replay_windows = load_csv(required_path(DASHBOARD_DIR / "rolling_replay_window_summary.csv"))
    replay_contributors = load_csv(required_path(DASHBOARD_DIR / "rolling_replay_shap_contributors.csv"))
    return {
        "metadata": metadata,
        "replay_metadata": replay_metadata,
        "cells": cells,
        "risks": risks,
        "windows": windows,
        "replay": replay,
        "replay_windows": replay_windows,
        "replay_contributors": replay_contributors,
    }


def chart_layout(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=45, b=10),
        font=dict(family="Arial, sans-serif", size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def read_simple_toml_value(path: Path, key: str) -> str | None:
    if not path.exists():
        return None
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("[") or "=" not in line:
            continue
        candidate, raw_value = line.split("=", 1)
        if candidate.strip() != key:
            continue
        value = raw_value.strip().strip('"').strip("'")
        return value or None
    return None


def local_secret(name: str) -> str | None:
    if os.environ.get(name):
        return os.environ[name].strip()
    try:
        value = st.secrets.get(name)
        if value:
            return str(value).strip()
    except Exception:
        pass
    for path in [
        ROOT / ".streamlit" / "secrets.toml",
        ROOT / ".streamlit" / "credentials.toml",
        Path.home() / ".streamlit" / "secrets.toml",
        Path.home() / ".streamlit" / "credentials.toml",
    ]:
        value = read_simple_toml_value(path, name)
        if value:
            return value
    return None


def risk_color(value: float, max_value: float) -> str:
    rgb = risk_color_rgb(value, max_value)
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def risk_color_rgb(value: float, max_value: float) -> list[int]:
    stops = [
        (255, 255, 178),
        (254, 178, 76),
        (240, 59, 32),
        (189, 0, 38),
    ]
    if max_value <= 0:
        return [255, 255, 178]
    t = max(0.0, min(1.0, float(value) / max_value))
    scaled = t * (len(stops) - 1)
    lower = min(int(scaled), len(stops) - 2)
    frac = scaled - lower
    c0, c1 = stops[lower], stops[lower + 1]
    rgb = tuple(round(c0[i] + (c1[i] - c0[i]) * frac) for i in range(3))
    return [int(rgb[0]), int(rgb[1]), int(rgb[2])]


def safe_json_value(value):
    if isinstance(value, (list, tuple)):
        return [safe_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: safe_json_value(item) for key, item in value.items()}
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def build_map_geojson(
    cells: dict,
    rows: pd.DataFrame,
    risk_col: str,
    rank_col: str,
    time_col: str,
    reason_col: str | None,
    max_risk: float,
) -> dict:
    cell_features = {feature["properties"]["h3_cell_res8"]: feature for feature in cells["features"]}
    features = []
    for _, row in rows.iterrows():
        h3_cell = row["h3_cell_res8"]
        source = cell_features.get(h3_cell)
        if not source:
            continue
        risk = float(row[risk_col])
        reason = str(row[reason_col]) if reason_col and reason_col in row and pd.notna(row[reason_col]) else ""
        properties = {
            "h3_cell_res8": h3_cell,
            "rank": int(row[rank_col]),
            "risk": risk,
            "risk_label": fmt_pct(risk),
            "time_window": str(row[time_col]),
            "actual_label": int(row["actual_label"]) if "actual_label" in row and pd.notna(row["actual_label"]) else None,
            "incident_count": int(row["incident_count"]) if "incident_count" in row and pd.notna(row["incident_count"]) else None,
            "reason": reason,
            "fill_color": risk_color(risk, max_risk),
            "fill_color_rgba": [*risk_color_rgb(risk, max_risk), 176],
        }
        features.append(
            {
                "type": "Feature",
                "geometry": source["geometry"],
                "properties": {key: safe_json_value(value) for key, value in properties.items()},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def render_maptiler_map(
    data: dict,
    rows: pd.DataFrame,
    *,
    title: str,
    risk_col: str,
    rank_col: str,
    time_col: str,
    reason_col: str | None = None,
    max_risk: float | None = None,
    height: int = 650,
) -> bool:
    api_key = local_secret("MAPTILER_API_KEY")
    if not api_key:
        st.info("MapTiler key not found. Falling back to the basic Plotly map.")
        return False
    style_url = local_secret("MAPTILER_STYLE_URL") or f"https://api.maptiler.com/maps/streets-v2/style.json?key={api_key}"
    max_value = max_risk if max_risk is not None else max(0.01, float(rows[risk_col].max()))
    geojson = build_map_geojson(data["cells"], rows, risk_col, rank_col, time_col, reason_col, max_value)
    if not geojson["features"]:
        st.warning("No map zones are available for this selection.")
        return True
    st.markdown(f"**{title}**")
    layer = pdk.Layer(
        "GeoJsonLayer",
        data=geojson,
        pickable=True,
        stroked=True,
        filled=True,
        opacity=0.74,
        get_fill_color="properties.fill_color_rgba",
        get_line_color=[48, 48, 48, 210],
        line_width_min_pixels=1,
    )
    tooltip = {
        "html": (
            "<b>Rank #{rank}</b><br/>"
            "Risk: {risk_label}<br/>"
            "Time: {time_window}<br/>"
            "Zone: {h3_cell_res8}<br/>"
            "Replay incidents: {incident_count}<br/>"
            "{reason}"
        ),
        "style": {"font-family": "Arial, sans-serif", "font-size": "12px"},
    }
    deck = pdk.Deck(
        map_style=style_url,
        initial_view_state=pdk.ViewState(latitude=25.20, longitude=55.30, zoom=9.2, pitch=0),
        layers=[layer],
        tooltip=tooltip,
    )
    st.pydeck_chart(deck, use_container_width=True, height=height)
    st.caption(f"Predicted incident risk color scale: 0% to {fmt_pct(max_value, 1)}.")
    return True


def apply_dashboard_basemap(fig: go.Figure) -> go.Figure:
    """Use label-free raster tiles and add only controlled English dashboard labels."""
    fig.update_layout(mapbox_style="white-bg", mapbox_layers=LABEL_FREE_BASEMAP_LAYERS)
    fig.add_trace(
        go.Scattermapbox(
            lat=[label["lat"] for label in ENGLISH_CONTEXT_LABELS],
            lon=[label["lon"] for label in ENGLISH_CONTEXT_LABELS],
            text=[label["name"] for label in ENGLISH_CONTEXT_LABELS],
            mode="text",
            textfont=dict(size=14, color="#50545a"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    return fig


def summary_value(summary: pd.DataFrame, metric: str) -> str:
    values = summary.loc[summary["metric"] == metric, "value"]
    if values.empty:
        return "not available"
    return str(values.iloc[0])


def render_header(metadata: dict) -> None:
    st.title("Dubai incident-risk replay")
    st.caption(
        "Replay the final XGBoost model on late observed 3-hour windows. "
        "The map shows where the model would rank incident risk, and SHAP explains why a selected zone is ranked high."
    )
    st.caption(
        f"Historical incident data is available up to {metadata['data_cutoff']}. "
        "This is a thesis decision-support prototype, not a live traffic-control system."
    )


def warning_text(metadata: dict) -> str:
    return metadata.get(
        "warning",
        "Forecasts after the data cutoff require updated incident counts to compute recent lag features.",
    )


def replay_filters(data: dict, key_prefix: str) -> tuple[str, str, int]:
    metadata = data["metadata"]
    replay = data["replay"]
    dates = sorted(replay["replay_date"].unique().tolist())
    default_date = metadata.get("default_replay_date", dates[-1])
    default_idx = dates.index(default_date) if default_date in dates else len(dates) - 1
    selected_date = st.sidebar.selectbox("Replay date", dates, index=default_idx, key=f"{key_prefix}_date")
    time_windows = ["Whole day"] + sorted(
        replay.loc[replay["replay_date"] == selected_date, "time_window"].unique().tolist()
    )
    selected_time = st.sidebar.selectbox("Time window", time_windows, index=0, key=f"{key_prefix}_time")
    top_n = int(st.sidebar.slider("Zones shown", min_value=10, max_value=100, value=40, step=10, key=f"{key_prefix}_topn"))
    return selected_date, selected_time, top_n


def selected_replay_rows(data: dict, selected_date: str, selected_time: str, top_n: int) -> pd.DataFrame:
    replay = data["replay"]
    rows = replay[replay["replay_date"] == selected_date].copy()
    if selected_time != "Whole day":
        rows = rows[rows["time_window"] == selected_time].copy()
    rows = rows.sort_values("predicted_incident_risk", ascending=False).head(top_n).copy()
    rows["Rank"] = range(1, len(rows) + 1)
    rows["Predicted incident risk"] = rows["predicted_incident_risk"].map(lambda value: fmt_pct(float(value)))
    rows["Time window"] = rows["time_window"]
    rows["Map zone"] = rows["map_zone"]
    rows["Main reason"] = rows["main_reason"]
    rows["Other reasons"] = rows["other_reasons"].fillna("")
    return rows


def replay_map(data: dict, rows: pd.DataFrame, title: str) -> go.Figure:
    map_rows = rows.sort_values("predicted_incident_risk", ascending=False).drop_duplicates("h3_cell_res8").copy()
    fig = px.choropleth_mapbox(
        map_rows,
        geojson=data["cells"],
        locations="h3_cell_res8",
        featureidkey="properties.h3_cell_res8",
        color="predicted_incident_risk",
        color_continuous_scale="YlOrRd",
        range_color=(0, max(0.01, float(data["replay"]["predicted_incident_risk"].quantile(0.995)))),
        mapbox_style="white-bg",
        center={"lat": 25.20, "lon": 55.30},
        zoom=9.2,
        opacity=0.72,
        hover_data={
            "Rank": True,
            "predicted_incident_risk": ":.4f",
            "time_window": True,
            "main_reason": True,
            "actual_label": True,
            "incident_count": True,
            "h3_cell_res8": False,
        },
        labels={"predicted_incident_risk": "Predicted incident risk"},
    )
    fig.update_layout(title=title, height=620, margin=dict(l=0, r=0, t=35, b=0))
    return apply_dashboard_basemap(fig)


def render_replay_summary(rows: pd.DataFrame) -> None:
    if rows.empty:
        st.warning("No replay rows are available for this selection.")
        return
    top = rows.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Highest-ranked zone", str(top["Map zone"]))
    c2.metric("Highest-ranked time", str(top["Time window"]))
    c3.metric("Predicted incident risk", str(top["Predicted incident risk"]))
    c4.markdown(f"**Main SHAP reason**<br>{top['Main reason']}", unsafe_allow_html=True)


def selected_zone_label(rows: pd.DataFrame) -> str:
    return st.selectbox(
        "Choose a ranked zone",
        rows.apply(lambda row: f"#{row['Rank']} | {row['Time window']} | {row['Predicted incident risk']} | {row['Map zone']}", axis=1),
    )


def row_from_label(rows: pd.DataFrame, label: str) -> pd.Series:
    selected_rank = int(label.split("|", maxsplit=1)[0].replace("#", "").strip())
    return rows.loc[rows["Rank"] == selected_rank].iloc[0]


def contributor_rows(data: dict, selected: pd.Series) -> pd.DataFrame:
    contributors = data["replay_contributors"]
    mask = (
        (contributors["window_index"] == int(selected["window_index"]))
        & (contributors["h3_cell_res8"] == selected["h3_cell_res8"])
        & (contributors["rank"] == int(selected["rank"]))
    )
    return contributors[mask].sort_values("contributor_rank").copy()


def render_rolling_replay(data: dict) -> None:
    st.subheader("Rolling replay")
    st.write(
        "This page replays the final neighbor-lag XGBoost model on late historical windows. "
        "It shows how the model would rank zones when the recent incident counts needed by the model are available."
    )
    st.warning(warning_text(data["metadata"]))
    selected_date, selected_time, top_n = replay_filters(data, "replay")
    rows = selected_replay_rows(data, selected_date, selected_time, top_n)
    render_replay_summary(rows)

    title = f"Top ranked zones for {selected_date}"
    if selected_time != "Whole day":
        title += f", {selected_time}"
    map_rows = rows.sort_values("predicted_incident_risk", ascending=False).drop_duplicates("h3_cell_res8").copy()
    rendered = render_maptiler_map(
        data,
        map_rows,
        title=title,
        risk_col="predicted_incident_risk",
        rank_col="Rank",
        time_col="Time window",
        reason_col="Main reason",
        max_risk=max(0.01, float(data["replay"]["predicted_incident_risk"].quantile(0.995))),
    )
    if not rendered:
        st.plotly_chart(replay_map(data, rows, title), use_container_width=True, config={"scrollZoom": True})

    st.subheader("Top ranked zones")
    display = rows[
        [
            "Rank",
            "Predicted incident risk",
            "Time window",
            "Map zone",
            "actual_label",
            "incident_count",
            "Main reason",
            "Other reasons",
        ]
    ].head(20)
    st.dataframe(display, use_container_width=True, hide_index=True)


def render_zone_explanation(data: dict) -> None:
    st.subheader("Zone explanation")
    st.write(
        "SHAP explains why the final XGBoost model ranked a selected zone and 3-hour window as risky. "
        "These are model reasons, not proof of the real-world cause of an incident."
    )
    st.warning(warning_text(data["metadata"]))
    selected_date, selected_time, top_n = replay_filters(data, "explain")
    rows = selected_replay_rows(data, selected_date, selected_time, top_n)
    if rows.empty:
        st.warning("No replay rows are available for this selection.")
        return

    selected = row_from_label(rows, selected_zone_label(rows))
    detail = contributor_rows(data, selected)

    left, right = st.columns([0.8, 1.2])
    with left:
        st.metric("Rank", f"#{int(selected['Rank'])}")
        st.metric("Predicted incident risk", str(selected["Predicted incident risk"]))
        st.markdown(f"**When:** {selected['replay_date']} {selected['Time window']}")
        st.markdown(f"**Map zone:** {selected['Map zone']}")
        st.markdown(f"**Actual outcome in replay:** {int(selected['incident_count'])} incident(s)")
    with right:
        st.markdown("**Top SHAP reasons**")
        st.write(selected["main_reason"])
        if str(selected.get("reason_2", "")).strip():
            st.write(selected["reason_2"])
        if str(selected.get("reason_3", "")).strip():
            st.write(selected["reason_3"])

    if not detail.empty:
        chart_df = detail.head(8).sort_values("shap_value")
        fig = px.bar(
            chart_df,
            x="shap_value",
            y="reason",
            orientation="h",
            color="direction",
            text="feature_value",
            title="Local SHAP contribution for the selected zone",
            labels={"shap_value": "SHAP contribution", "reason": ""},
            color_discrete_map={"increases risk": "#c1121f", "reduces risk": "#457b9d"},
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(chart_layout(fig, 430), use_container_width=True)
        st.dataframe(
            detail[["contributor_rank", "reason", "feature_value", "shap_value", "direction", "strength"]],
            use_container_width=True,
            hide_index=True,
        )


def selected_historical_window(data: dict) -> int:
    windows = data["windows"].copy()
    windows["label"] = windows["window_start"] + "  | incidents: " + windows["incident_count"].astype(int).astype(str)
    default_idx = int(windows["incident_count"].idxmax())
    selected_label = st.sidebar.selectbox(
        "Historical test window",
        windows["label"].tolist(),
        index=default_idx,
        help="Historical 3-hour test windows only.",
        key="historical_window",
    )
    return int(windows.loc[windows["label"] == selected_label, "window_index"].iloc[0])


def render_historical_check(data: dict) -> None:
    st.subheader("Historical check")
    st.write(
        "This view checks how the final model ranked zones during historical test windows where the actual incident outcomes are already known."
    )

    window_index = selected_historical_window(data)
    top_n = int(st.sidebar.slider("Historical zones shown", min_value=20, max_value=100, value=50, step=10, key="historical_topn"))
    risks = data["risks"]
    windows = data["windows"]
    window_risk = risks[(risks["window_index"] == window_index) & (risks["risk_rank"] <= top_n)].copy()
    window_row = windows.loc[windows["window_index"] == window_index].iloc[0]
    window_risk["time_window"] = str(window_row["window_start"])

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"**Window start**<br><span style='font-size:1.25rem'>{window_row['window_start']}</span>", unsafe_allow_html=True)
    c2.metric("Actual incidents", fmt_int(window_row["incident_count"]))
    c3.metric("Positive cells", fmt_int(window_row["positive_cell_windows"]))
    c4.metric("Top-20 hits", fmt_int(window_row["top20_positive_cells"]))

    rendered = render_maptiler_map(
        data,
        window_risk,
        title=f"Top ranked zones for {window_row['window_start']}",
        risk_col="sigmoid_calibrated_risk",
        rank_col="risk_rank",
        time_col="time_window",
        max_risk=max(0.01, float(data["risks"]["sigmoid_calibrated_risk"].quantile(0.995))),
    )
    if not rendered:
        fig = px.choropleth_mapbox(
            window_risk,
            geojson=data["cells"],
            locations="h3_cell_res8",
            featureidkey="properties.h3_cell_res8",
            color="sigmoid_calibrated_risk",
            color_continuous_scale="YlOrRd",
            range_color=(0, max(0.01, float(data["risks"]["sigmoid_calibrated_risk"].quantile(0.995)))),
            mapbox_style="white-bg",
            center={"lat": 25.20, "lon": 55.30},
            zoom=9.2,
            opacity=0.72,
            hover_data={
                "h3_cell_res8": True,
                "risk_rank": True,
                "sigmoid_calibrated_risk": ":.4f",
                "actual_label": True,
                "incident_count": True,
            },
            labels={"sigmoid_calibrated_risk": "Predicted incident risk"},
        )
        fig.update_layout(height=620, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(apply_dashboard_basemap(fig), use_container_width=True, config={"scrollZoom": True})

    display = window_risk[
        [
            "risk_rank",
            "h3_cell_res8",
            "sigmoid_calibrated_risk",
            "actual_label",
            "incident_count",
            "prev_7d_incident_count",
            "neighbor_prev_7d_incident_count",
            "hist_cell_hour_risk",
        ]
    ].head(20)
    st.dataframe(display, use_container_width=True, hide_index=True)


def render_eda_summary() -> None:
    summary = load_csv(EDA_TABLE_DIR / "summary_metrics.csv")
    top_types = load_csv(EDA_TABLE_DIR / "top_incident_types.csv").head(10)
    severity = load_csv(EDA_TABLE_DIR / "severity_distribution.csv")
    hourly = load_csv(EDA_TABLE_DIR / "hour_of_day_incidents.csv")
    day = load_csv(EDA_TABLE_DIR / "day_of_week_incidents.csv")

    c1, c2, c3 = st.columns(3)
    c1.metric("Raw incident records", fmt_int(float(summary_value(summary, "Total raw records"))))
    c2.metric("Map-usable records", fmt_int(float(summary_value(summary, "Map-EDA usable rows"))))
    c3.metric("Unique Arabic categories", "233")

    left, right = st.columns([1.15, 0.85])
    with left:
        fig = px.bar(
            top_types.sort_values("count"),
            x="count",
            y="incident_type_en",
            orientation="h",
            labels={"count": "Records", "incident_type_en": ""},
            title="Top translated incident categories",
            color_discrete_sequence=["#2b2b2b"],
        )
        st.plotly_chart(chart_layout(fig, 430), use_container_width=True)
    with right:
        fig = px.pie(
            severity,
            values="count",
            names="severity_code",
            title="Severity distribution",
            color_discrete_sequence=px.colors.sequential.Greys_r,
        )
        st.plotly_chart(chart_layout(fig, 430), use_container_width=True)

    left, right = st.columns(2)
    with left:
        fig = px.bar(hourly, x="hour", y="count", title="Incidents by hour of day", labels={"count": "Records"})
        st.plotly_chart(chart_layout(fig, 340), use_container_width=True)
    with right:
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day["day_of_week"] = pd.Categorical(day["day_of_week"], categories=weekday_order, ordered=True)
        fig = px.bar(
            day.sort_values("day_of_week"),
            x="day_of_week",
            y="count",
            title="Incidents by day of week",
            labels={"day_of_week": "", "count": "Records"},
        )
        st.plotly_chart(chart_layout(fig, 340), use_container_width=True)


def render_model_evidence(data: dict) -> None:
    st.subheader("How the model was checked")
    metadata = data["metadata"]
    replay_metadata = data["replay_metadata"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Inside-Dubai H3 cells", fmt_int(metadata["inside_dubai_cells"]))
    c2.metric("Historical test candidates", fmt_int(metadata["test_candidates"]))
    c3.metric("Replay windows", fmt_int(replay_metadata["replay_windows"]))
    c4.metric("Default replay date", replay_metadata["default_replay_date"])

    st.write(
        "The replay uses the same final neighbor-lag XGBoost model used in the report. "
        "It is a replay because the actual outcomes are known, but the model inputs are restricted to information available before each 3-hour window."
    )

    metrics = load_csv(MODEL_TABLE_DIR / "neighbor_lag_feature_metrics.csv")
    calibration = load_csv(MODEL_TABLE_DIR / "xgboost_calibration_metrics.csv")
    selected = load_csv(MODEL_TABLE_DIR / "xgboost_calibration_selected.csv").iloc[0]
    topk = load_csv(MODEL_TABLE_DIR / "xgboost_calibration_topk.csv")
    bins = load_csv(MODEL_TABLE_DIR / "xgboost_calibration_bins.csv")

    st.subheader("Final prediction result")
    selected_model = metrics.loc[metrics["model"] == "XGBoost + neighbor lags"].iloc[0]
    top20 = topk[(topk["model"] == "sigmoid") & (topk["k"] == 20)].iloc[0]
    final_rows = [
        ("Model", "XGBoost + neighbor lags"),
        ("Evaluation", "inside-Dubai full-grid test"),
        ("H3/time setup", "H3 resolution 8, 3-hour windows"),
        ("Test candidates", fmt_int(metadata["test_candidates"])),
        ("Positive cell-windows", fmt_int(int(top20["total_positive_cell_windows"]))),
        ("ROC-AUC", f"{selected_model['test_roc_auc']:.3f}"),
        ("PR-AUC", f"{selected_model['test_pr_auc']:.3f}"),
        ("F1-score", f"{selected_model['test_f1']:.3f}"),
        ("Recall", f"{selected_model['test_recall']:.3f}"),
        ("Precision", f"{selected_model['test_precision']:.3f}"),
        ("Top-20 incidents captured", fmt_int(int(top20["incidents_captured"]))),
        ("Calibration used for display", str(selected["selected_method"])),
    ]
    split_at = len(final_rows) // 2
    final_result = pd.DataFrame(
        [
            {
                "Measure": left[0],
                "Value": left[1],
                "Measure ": right[0],
                "Value ": right[1],
            }
            for left, right in zip(final_rows[:split_at], final_rows[split_at:], strict=True)
        ]
    )
    st.table(final_result)
    st.info(
        "The model is best used as a hotspot-ranking tool, not as a perfect incident detector. "
        "It ranks high-risk zones better than baseline methods, but incident prediction remains difficult because most cell-window combinations have no incident."
    )

    st.subheader("Dataset evidence")
    render_eda_summary()

    st.subheader("Supporting model checks")
    plot_df = metrics[metrics["model"].isin(["Historical risk", "XGBoost current features", "XGBoost + neighbor lags"])]
    plot_df = plot_df.melt(
        id_vars="model",
        value_vars=["test_roc_auc", "test_pr_auc", "test_f1", "test_recall"],
        var_name="metric",
        value_name="score",
    )
    fig = px.bar(
        plot_df,
        x="model",
        y="score",
        color="metric",
        barmode="group",
        title="Full-grid test performance",
        labels={"model": "", "score": "Score"},
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    st.plotly_chart(chart_layout(fig, 390), use_container_width=True)

    left, right = st.columns(2)
    with left:
        topk_view = topk[(topk["model"] == "sigmoid") & (topk["k"].isin([5, 10, 20]))][
            ["k", "precision_at_k", "incident_recall_at_k", "positive_window_hit_rate_at_k"]
        ]
        st.write("Hotspot ranking after sigmoid calibration")
        st.dataframe(topk_view, use_container_width=True, hide_index=True)
    with right:
        sig_bins = bins[bins["method"] == "sigmoid"].copy()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=sig_bins["predicted_mean"],
                y=sig_bins["observed_rate"],
                mode="lines+markers",
                name="Sigmoid calibration",
            )
        )
        max_axis = float(max(sig_bins["predicted_mean"].max(), sig_bins["observed_rate"].max()))
        fig.add_trace(go.Scatter(x=[0, max_axis], y=[0, max_axis], mode="lines", name="Perfect calibration"))
        fig.update_layout(title="Predicted risk vs observed rate", xaxis_title="Mean predicted risk", yaxis_title="Observed incident rate")
        st.plotly_chart(chart_layout(fig, 360), use_container_width=True)

    st.caption("Calibration improves probability reliability for display. It does not change hotspot ranking when the calibration is monotonic.")
    st.dataframe(calibration, use_container_width=True, hide_index=True)


def render_scope_notes(data: dict) -> None:
    st.subheader("Scope and interpretation notes")
    metadata = data["metadata"]
    st.write(
        "This dashboard is a rolling replay and model-inspection prototype for thesis demonstration. "
        "It is not connected to live Dubai Police, RTA, CCTV, signal, or sensor systems."
    )
    st.markdown(
        f"""
- The current incident snapshot ends on **{metadata['data_cutoff']}**.
- The replay uses the final neighbor-lag XGBoost model on complete observed days before the cutoff.
- Forecasts after the cutoff require updated incident data so recent 3-hour, 24-hour, 7-day, and neighboring lag features can be computed.
- The model predicts whether at least one incident is likely in a grid cell and 3-hour time window.
- The map shows ranked risk areas, not a guarantee that an incident will occur.
- SHAP explains why the model predicts high risk. It does not prove the actual cause of an incident.
- LIME remains a report-level local comparison; this dashboard uses SHAP for zone-level explanations.
- Unknown-severity incidents remain in binary incident-risk counts, but they do not add weight to severity-weighted summaries.
"""
    )


def render_view(view: str, data: dict) -> None:
    if view == "Rolling replay":
        render_rolling_replay(data)
    elif view == "Zone explanation":
        render_zone_explanation(data)
    elif view == "Historical check":
        render_historical_check(data)
    elif view == "Model evidence":
        render_model_evidence(data)
    elif view == "Scope notes":
        render_scope_notes(data)


def main() -> None:
    data = load_dashboard_data()
    render_header(data["metadata"])

    query_view = st.query_params.get("view")
    if query_view in VIEWS:
        st.sidebar.title("Dashboard controls")
        render_view(query_view, data)
        return

    st.sidebar.title("Dashboard controls")
    st.sidebar.caption("Rolling replay is the default view.")
    selected_view = st.sidebar.radio("View", VIEWS, index=0)
    render_view(selected_view, data)


if __name__ == "__main__":
    main()
