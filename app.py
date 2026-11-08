"""Stage B CrowdShield AI command-center dashboard."""

from pathlib import Path
import sys
import tempfile

import cv2
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import config
from src.actions import ActionRouter
from src.analysis import AnalysisResult, analyze_video


st.set_page_config(page_title="CrowdShield AI", page_icon="🛡️", layout="wide")
st.markdown(
    """<style>
    .stApp { background: #070d18; color: #e8f5ff; }
    h1, h2, h3 { color: #58e6ff; }
    [data-testid="stMetric"] { background: #101d2d; border: 1px solid #1e8497; border-radius: 12px; padding: 14px; }
    .stage { color: #f7b955; letter-spacing: 1px; }
    .alert { padding: 12px; border-radius: 8px; background: #54220e; border: 1px solid #f78b38; }
    </style>""",
    unsafe_allow_html=True,
)


def _chart(title: str, result: AnalysisResult, column: str) -> go.Figure:
    values = (
        [risk.overall_risk for risk in result.risks]
        if column == "risk"
        else result.mean_flow_speed
        if column == "raw_speed"
        else result.surge_signal
        if column == "raw_surge"
        else [float(max(zones)) if len(zones) else 0.0 for zones in result.zone_counts]
    )
    figure = go.Figure()
    if column == "risk":
        figure.add_hrect(y0=0, y1=config.RISK_LEVEL_THRESHOLDS["green"], fillcolor="#16855b", opacity=0.15, line_width=0)
        figure.add_hrect(y0=config.RISK_LEVEL_THRESHOLDS["green"], y1=config.RISK_LEVEL_THRESHOLDS["yellow"], fillcolor="#e8c547", opacity=0.15, line_width=0)
        figure.add_hrect(y0=config.RISK_LEVEL_THRESHOLDS["yellow"], y1=config.RISK_LEVEL_THRESHOLDS["orange"], fillcolor="#f27c22", opacity=0.15, line_width=0)
        figure.add_hrect(y0=config.RISK_LEVEL_THRESHOLDS["orange"], y1=100, fillcolor="#d63b3b", opacity=0.15, line_width=0)
    if column == "raw":
        figure.add_trace(go.Scatter(x=result.timestamps, y=result.mean_flow_speed, name="Mean flow speed", line={"color": "#58e6ff"}))
        figure.add_trace(go.Scatter(x=result.timestamps, y=result.surge_signal, name="Surge term", line={"color": "#ff9f43"}))
    else:
        figure.add_trace(go.Scatter(x=result.timestamps, y=values, mode="lines+markers", line={"color": "#58e6ff"}))
    for boundary in result.scene_boundaries:
        figure.add_vline(x=boundary, line_dash="dash", line_color="#f7b955")
    for panic_time in config.PANIC_STARTS_S:
        figure.add_vline(x=panic_time, line_dash="dot", line_color="#ff4f67")
    figure.update_layout(
        title=title,
        template="plotly_dark",
        height=280,
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        xaxis_title="Time (s)",
    )
    return figure


def _show_context() -> None:
    st.header("Why CrowdShield")
    st.caption("Figures to be verified against cited sources.")
    left, right = st.columns(2)
    with left:
        st.subheader("Chinnaswamy Stadium — 4 June 2025")
        st.write("**What failed**")
        st.write("About 11 people died after a crowd reported at roughly 2.5–3 lakh gathered around a venue with capacity about 35,000. Inquiry reporting cited entry/gate mismanagement and a free-entry announcement.")
    with right:
        st.subheader("What CrowdShield would do")
        st.write("Fixed-scale zone counts, flow conflict, and risk trend would warn before pressure peaks; simulated voice guidance would identify the exact zone for on-site staff.")
    left, right = st.columns(2)
    with left:
        st.subheader("Maha Kumbh 2025")
        st.write("**What failed**")
        st.write("AI cameras sent timely alerts, but there were not enough police on the ground to act.")
    with right:
        st.subheader("What CrowdShield would do")
        st.write("Simulation alerts repeat with cooldowns, include zone and estimated people, and provide early-warning lead time. Technology cannot create exits.")
    st.info("**Stampedes happen in seconds and help arrives late: what is innovative?** Predict early + auto on-site guidance + entry throttling.")
    st.warning("**No exit, no space, no staff, police far away?** Prevention first: early alerts and entry caps; reduce pressure from the back with voice guidance; alert responders at prediction time with the exact location. Technology cannot create exits.")


def _show_how_it_works() -> None:
    st.header("How It Works")
    st.write("The demo analyzes dense samples, computes Farneback flow between adjacent source frames, smooths measured people and flow features, calibrates normal flow for each scene, then scores a 0–100 risk time series.")
    st.write("Risk = configured density, baseline flow deviation, speed-drop, direction-conflict, sudden speed-surge, and TREND terms. All thresholds, weights, paths, smoothing, baseline, scene, and panic settings live in `src/config.py`.")
    st.caption("TODO: crowd-type profiles, staffing scenarios, dispatch logic, venue assessment, LSTM forecasting, edge deployment, and multi-camera zones.")


def _show_roadmap() -> None:
    st.header("Roadmap — remaining 60%")
    st.write("Crowd-type profiles · staffing scenarios (0 staff / few volunteers / full staff) · volunteer dispatch · police prioritisation · venue risk assessment · LSTM risk forecasting · edge deployment · multi-camera zones")


def _show_limits() -> None:
    st.header("Limits")
    st.write("Detection weakens in very dense crowds and low light. Confidence thresholds affect counts. When staff is available, a human should confirm Red before operational action.")
    st.caption("This dashboard is a Simulation. It does not dispatch people, control gates, or contact emergency services.")


st.title("🛡️ CrowdShield AI")
st.caption("Predict. Prevent. Protect. — Stage B: flow, risk, alerts and explainability.")
st.markdown("<div class='stage'>SIMULATION · CPU-SAFE DEMO MODE · REAL VIDEO DATA</div>", unsafe_allow_html=True)

video_options = []
if config.UMN_VIDEO_PATH.exists():
    video_options.append(config.UMN_VIDEO_PATH)
if config.VIDEOS_DIR.exists():
    video_options.extend(sorted(config.VIDEOS_DIR.glob("*.mp4")))
    video_options.extend(sorted(config.VIDEOS_DIR.glob("*.avi")))
uploaded_video = st.sidebar.file_uploader(
    "Upload video for Simulation",
    type=["avi", "mp4"],
    help="The deployed app cannot access the repository's excluded data/ folder.",
)
selected = None
if uploaded_video is not None:
    upload_key = f"{uploaded_video.name}:{uploaded_video.size}"
    if st.session_state.get("upload_key") != upload_key:
        suffix = Path(uploaded_video.name).suffix.lower()
        temporary_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temporary_file.write(uploaded_video.getvalue())
        temporary_file.close()
        st.session_state.upload_key = upload_key
        st.session_state.upload_path = temporary_file.name
    selected = Path(st.session_state.upload_path)
elif video_options:
    selected = st.sidebar.selectbox("Video", video_options, format_func=lambda path: path.name)
else:
    st.sidebar.info("Simulation: upload an AVI or MP4 video to run the deployed demo.")
st.sidebar.caption(f"Simulation: {config.ANALYSIS_SAMPLES_PER_SECOND:g} analysis samples/sec; flow uses adjacent source frames.")
run = st.sidebar.button("Run demo", type="primary", disabled=selected is None)

if run:
    progress = st.progress(0, text="Simulation: precomputing sampled results...")
    try:
        result = analyze_video(selected, progress.progress)
    except (RuntimeError, ValueError) as exc:
        st.error(str(exc))
        st.stop()
    progress.progress(1.0, text="Simulation: results ready.")
    st.session_state.analysis = result
    st.session_state.actions = []

result = st.session_state.get("analysis")
tabs = st.tabs(["Live Monitor", "Why CrowdShield", "How It Works", "Roadmap", "Limits"])
with tabs[1]:
    _show_context()
with tabs[2]:
    _show_how_it_works()
with tabs[3]:
    _show_roadmap()
with tabs[4]:
    _show_limits()

with tabs[0]:
    if result is None:
        st.info("Select the video and click Run demo. The dashboard will precompute a smooth sampled simulation.")
    else:
        action_router = ActionRouter()
        events = []
        for action_input in result.action_inputs:
            event = action_router.route(*action_input)
            if event:
                events.append((action_input[-1], event))
        st.session_state.actions = events
        default_index = int(max(range(len(result.risks)), key=lambda index: result.risks[index].overall_risk))
        selected_index = st.slider(
            "Simulation sample time",
            min_value=0,
            max_value=len(result.timestamps) - 1,
            value=default_index,
            format="%d",
        )
        selected_time = result.timestamps[selected_index]
        current = result.risks[selected_index]
        if current.level in {"Yellow", "Orange", "Red"}:
            st.markdown(f"<div class='alert'>Simulation alert: {current.level} in {current.top_zone} — {current.why}.</div>", unsafe_allow_html=True)
        st.caption(f"Simulation sample at {selected_time:.2f} seconds · slow zones: {', '.join(result.slow_zones[selected_index]) if result.slow_zones[selected_index] else 'none detected'}")
        top_zone_people = max(result.zone_counts[selected_index]) if len(result.zone_counts[selected_index]) else 0
        active_alerts = sum(timestamp <= selected_time for timestamp, _ in events)
        early_warning = f"{result.early_warning:.1f} s" if result.early_warning is not None else "n/a"
        metrics = st.columns([1.0, 1.0, 1.15, 1.0, 1.0, 1.0])
        metrics[0].metric("People count", int(round(result.counts[selected_index])))
        metrics[1].metric("Max in zone", int(top_zone_people))
        metrics[2].metric("Risk / 100", f"{current.overall_risk:.1f}")
        metrics[3].metric("Active alerts", active_alerts)
        metrics[4].metric("Lead time", early_warning)
        metrics[5].metric("Speed fps", f"{result.processing_fps:.1f}")

        frame_slot, heatmap_slot = st.columns(2)
        with frame_slot:
            st.subheader("Sampled video with boxes and zone grid")
            st.image(cv2.cvtColor(result.frames[selected_index], cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")
        with heatmap_slot:
            st.subheader("Fixed-scale density heatmap")
            st.image(cv2.cvtColor(result.heatmaps[selected_index], cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")
            st.caption("Heatmap scale is fixed from config, not normalised per frame.")

        grid = st.columns(config.GRID_COLS)
        for index, value in enumerate(current.zone_risk):
            grid[index % config.GRID_COLS].metric(f"Z{index + 1} risk", f"{value:.1f} · {('Red' if value >= config.RISK_LEVEL_THRESHOLDS['orange'] else 'Orange' if value >= config.RISK_LEVEL_THRESHOLDS['yellow'] else 'Yellow' if value >= config.RISK_LEVEL_THRESHOLDS['green'] else 'Green')}")
        chart_left, chart_right = st.columns(2)
        chart_left.plotly_chart(_chart("Risk over time", result, "risk"), width="stretch")
        chart_left.plotly_chart(_chart("Raw signals", result, "raw"), width="stretch")
        chart_right.plotly_chart(_chart("Max zone density over time", result, "density"), width="stretch")
        st.subheader("Simulation validation")
        if result.panic_validation:
            st.dataframe(result.panic_validation, width="stretch", hide_index=True)
        else:
            st.caption("Simulation validation unavailable: no panic markers are configured in src/config.py.")
        st.subheader("Action feed")
        if events:
            for timestamp, event in events:
                st.write(f"**Simulation · {event.level}** {event.message}")
                st.caption(f"Simulation time: {timestamp:.2f} seconds")
                if event.payload:
                    st.json(event.payload)
                if event.audio_paths:
                    st.caption(f"Cached voice: {len(event.audio_paths)} language tracks")
        else:
            st.write("Simulation: no threshold alerts in the sampled run.")
