# Streamlit AIR AIM — Enhanced Dashboard
# Features: YOLOv8 detection, ROI, Queue detection, Heatmap, Line charts,
# ARIMA/Prophet forecasting, Multiple simultaneous inputs

import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
import time
from collections import deque
import pandas as pd
import altair as alt
import os

# Optional forecasting libs (try import)
try:
    from statsmodels.tsa.arima.model import ARIMA
    has_arima = True
except Exception:
    has_arima = False

try:
    from prophet import Prophet
    has_prophet = True
except Exception:
    has_prophet = False

# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="AIR AIM Enhanced", layout="wide")
st.title("AIR AIM — Enhanced Monitoring Dashboard")
st.write("Heatmap • Time-series charts • ARIMA/Prophet forecasting • Multiple simultaneous inputs")
st.info(
    "This dashboard is a **demo / prototype** of AIR AIM for testing and presentation purposes. "
    "It is not intended for direct production deployment."
)

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Controls")
model_choice = st.sidebar.selectbox("YOLO Model", ("yolov8n.pt", "yolov8s.pt"), index=0)
source_type = st.sidebar.selectbox("Input Type", ["Webcam", "Upload Video(s)", "RTSP / URLs"])
use_heatmap = st.sidebar.checkbox("Enable Heatmap", value=True)
use_tracking = st.sidebar.checkbox("Enable Tracking (ByteTrack)", value=True)
save_logs = st.sidebar.checkbox("Auto-save CSV logs", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("ROIs (x1,y1,x2,y2)")

# Adjusted ROIs for ~1920x1080 eagle-eye terminal view
# Left: Check-in, Middle: Central Walkway, Right: Security Area
default_rois = {
    "Check-in": (0, 250, 650, 1080),
    "Central Walkway": (650, 200, 1250, 1080),
    "Security Area": (1250, 200, 1920, 1080)
}

rois = {}
for name, coords in default_rois.items():
    txt = st.sidebar.text_input(name, value=",".join(map(str, coords)))
    try:
        rois[name] = tuple(map(int, txt.split(",")))
    except Exception:
        rois[name] = coords

st.sidebar.markdown("---")
st.sidebar.subheader("Queue detection")
queue_distance_px = st.sidebar.slider("Max distance between people for queue (px)", 20, 300, 80)
queue_count_threshold = st.sidebar.slider("Queue pairs threshold", 1, 100, 5)

st.sidebar.markdown("---")
st.sidebar.subheader("Crowd level thresholds (per ROI)")
busy_threshold = st.sidebar.number_input(
    "Busy when people ≥",
    min_value=1,
    max_value=500,
    value=6
)
crowded_threshold = st.sidebar.number_input(
    "Crowded when people ≥",
    min_value=2,
    max_value=1000,
    value=15
)

st.sidebar.markdown("---")
st.sidebar.subheader("Forecasting")
forecast_method = st.sidebar.selectbox(
    "Method",
    ["Moving Average", "ARIMA (statsmodels)", "Prophet"]
)
if forecast_method == "ARIMA (statsmodels)" and not has_arima:
    st.sidebar.warning("statsmodels ARIMA not available — install statsmodels to use this")
if forecast_method == "Prophet" and not has_prophet:
    st.sidebar.warning("prophet not available — install prophet to use this")
history_length = st.sidebar.slider("History length (frames)", 10, 600, 120)

# Input selection
uploaded_files = None
rtsp_text = ""
if source_type == "Webcam":
    webcam_index = st.sidebar.number_input("Webcam index", min_value=0, max_value=4, value=0)
elif source_type == "Upload Video(s)":
    uploaded_files = st.sidebar.file_uploader(
        "Upload one or more videos",
        type=["mp4", "avi", "mov"],
        accept_multiple_files=True
    )
else:
    rtsp_text = st.sidebar.text_area("RTSP / URLs (one per line)")

st.sidebar.markdown("---")
reload_model = st.sidebar.button("Reload Model")

# -----------------------------
# Load model (cached)
# -----------------------------
@st.cache_resource
def load_model(path):
    return YOLO(path)

if reload_model:
    # Clear cache and reload
    load_model.clear()
    st.experimental_rerun()

with st.spinner(f"Loading {model_choice}..."):
    model = load_model(model_choice)

# -----------------------------
# Helpers
# -----------------------------
def save_uploaded_temp(uploaded):
    t = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
    t.write(uploaded.read())
    t.flush()
    return t.name

def inside_roi(cx, cy, roi):
    x1, y1, x2, y2 = roi
    return x1 <= cx <= x2 and y1 <= cy <= y2

def euclidean(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def get_crowd_level(count, busy_threshold, crowded_threshold):
    """
    Returns (label, BGR_color) based on people count:
    - NORMAL  → green
    - BUSY    → yellow
    - CROWDED → red
    """
    if count >= crowded_threshold:
        return "CROWDED", (0, 0, 255)      # red
    elif count >= busy_threshold:
        return "BUSY", (0, 255, 255)       # yellow
    else:
        return "NORMAL", (0, 255, 0)       # green

# Forecast helpers
def moving_average_forecast(history):
    # history is a deque → convert to list for slicing
    hist = list(history)
    if len(hist) < 3:
        return None
    return float(np.mean(hist[-3:]))

def arima_forecast(history):
    if not has_arima:
        return None
    hist = list(history)
    if len(hist) < 10:
        return None
    try:
        model_ar = ARIMA(hist, order=(2, 0, 0))
        res = model_ar.fit()
        pred = res.forecast(steps=1)
        return float(pred[0])
    except Exception:
        return None

def prophet_forecast(history):
    if not has_prophet:
        return None
    hist = list(history)
    if len(hist) < 20:
        return None
    try:
        df = pd.DataFrame({
            "ds": pd.date_range(end=pd.Timestamp.now(), periods=len(hist), freq="S"),
            "y": hist
        })
        m = Prophet()
        m.fit(df)
        future = m.make_future_dataframe(periods=5, freq="S")
        fc = m.predict(future)
        return float(fc["yhat"].iloc[-1])
    except Exception:
        return None

# -----------------------------
# UI layout
# -----------------------------
col_vid, col_side = st.columns([3, 1])
video_container = col_vid.container()  # will hold multiple feeds

chart_placeholder = col_side.empty()
metrics_placeholder = col_side.empty()
log_download_placeholder = col_side.empty()

# Persistent structures
history = deque(maxlen=history_length)
logs = []

# Select/assemble sources
sources = []
source_labels = []

if source_type == "Webcam":
    sources = [int(webcam_index)]
    source_labels = [f"Webcam {webcam_index}"]
elif source_type == "Upload Video(s)":
    if uploaded_files:
        for f in uploaded_files:
            path = save_uploaded_temp(f)
            sources.append(path)
            source_labels.append(f"File: {f.name}")
else:
    if rtsp_text.strip():
        for idx, line in enumerate(rtsp_text.splitlines()):
            url = line.strip()
            if url:
                sources.append(url)
                source_labels.append(f"RTSP/URL {idx+1}")

if len(sources) == 0:
    st.info("Choose an input (webcam, upload videos, or RTSP URLs).")

# Controls
start = st.button("Start")
stop = st.button("Stop")   # NOTE: with this pattern, Stop won't interrupt a running loop; it will be read only at the start of the run.
reset = st.button("Reset")

if reset:
    history.clear()
    logs = []
    st.experimental_rerun()

# -----------------------------
# Main processing (multi-source)
# -----------------------------
if start and sources:
    # Create placeholders for each source
    cols = video_container.columns(len(sources))
    video_placeholders = [c.empty() for c in cols]

    # Create one VideoCapture per source
    caps = []
    for src in sources:
        cap = cv2.VideoCapture(src)
        caps.append(cap)

    # Check at least one source opened
    opened_any = any(cap.isOpened() for cap in caps)
    if not opened_any:
        st.error("Could not open any source. Check path/index/URL.")
        for cap in caps:
            cap.release()
    else:
        st.success("Sources opened — processing all simultaneously...")

        # Initialize heatmaps per source (once we know frame size)
        heatmap_accs = [None] * len(sources)

        frame_id = 0
        start_time = time.time()

        # Time-series containers (for aggregated counts)
        times = []
        people_list = []

        # Track which sources are finished
        finished = [False] * len(sources)

        while True:
            if stop:
                break

            # If all sources finished, stop
            if all(finished):
                break

            total_people_all_sources = 0

            for i, cap in enumerate(caps):
                if finished[i] or not cap.isOpened():
                    continue

                ret, frame = cap.read()
                if not ret or frame is None:
                    finished[i] = True
                    continue

                frame_id += 1

                # Init heatmap for this source if needed
                if heatmap_accs[i] is None:
                    h, w = frame.shape[:2]
                    heatmap_accs[i] = np.zeros((h, w), dtype=np.float32)

                centers = []

                # Inference (tracking or not)
                if use_tracking:
                    results = model.track(frame, persist=True)[0]
                else:
                    results = model(frame)[0]

                # Collect person detections
                if results.boxes is not None:
                    for box in results.boxes:
                        cls = int(box.cls[0])
                        if cls != 0:  # 0 = person
                            continue
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        centers.append((cx, cy))

                        # accumulate heat
                        if use_heatmap and heatmap_accs[i] is not None:
                            cv2.circle(heatmap_accs[i], (cx, cy), 30, 1, -1)

                        # draw on frame
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

                total_people = len(centers)
                total_people_all_sources += total_people

                # ROI counting
                roi_counts = {}
                for name, roi in rois.items():
                    cnt = 0
                    for c in centers:
                        if inside_roi(c[0], c[1], roi):
                            cnt += 1
                    roi_counts[name] = cnt

                # Queue detection via pairwise distances
                queue_pairs = 0
                for a_idx in range(len(centers)):
                    for b_idx in range(a_idx + 1, len(centers)):
                        if euclidean(centers[a_idx], centers[b_idx]) < queue_distance_px:
                            queue_pairs += 1
                queue_detected = queue_pairs >= queue_count_threshold

                # Prepare display frame with ROI crowd levels
                disp = frame.copy()
                for name, roi in rois.items():
                    x1, y1, x2, y2 = roi
                    cnt = roi_counts.get(name, 0)

                    level, color = get_crowd_level(cnt, busy_threshold, crowded_threshold)

                    # Draw ROI
                    cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)

                    # Label: zone + level + count
                    label = f"{name}: {level} ({cnt})"
                    cv2.putText(
                        disp,
                        label,
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )

                # Heatmap rendering per source
                if use_heatmap and heatmap_accs[i] is not None:
                    hm = cv2.normalize(heatmap_accs[i], None, 0, 255, cv2.NORM_MINMAX)
                    hm_uint8 = hm.astype(np.uint8)
                    hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
                    disp = cv2.addWeighted(disp, 0.6, hm_color, 0.4, 0)

                # Overlays
                
                cv2.putText(
                    disp,
                    f"{source_labels[i]}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )
                cv2.putText(
                    disp,
                    f"People: {total_people}",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                cv2.putText(
                    disp,
                    f"Queue: {'YES' if queue_detected else 'NO'}",
                    (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255) if queue_detected else (0, 255, 0),
                    2
                )

                # Update each source's video panel
                video_placeholders[i].image(disp, channels="BGR")

                # Log for this frame+source
                logs.append({
                    "frame_id": frame_id,
                    "source_index": i,
                    "source_label": source_labels[i],
                    "total_people": total_people,
                    "queue_pairs": queue_pairs,
                    "queue_detected": queue_detected,
                    **{f"roi_{k}": v for k, v in roi_counts.items()}
                })

            # After processing all sources for this frame step
            history.append(total_people_all_sources)
            times.append(frame_id)
            people_list.append(total_people_all_sources)

            # Forecast on aggregated people count
            forecast_val = None
            if forecast_method == 'Moving Average':
                forecast_val = moving_average_forecast(history)
            elif forecast_method == 'ARIMA (statsmodels)':
                forecast_val = arima_forecast(history)
            elif forecast_method == 'Prophet':
                forecast_val = prophet_forecast(history)

            # -----------------------------
            # Charts & metrics
            # -----------------------------
            if len(times) > 1:
                df = pd.DataFrame({
                    "frame": times,
                    "people": people_list
                })
                line = alt.Chart(df).mark_line().encode(
                    x="frame:Q",
                    y="people:Q"
                )

                if forecast_val is not None:
                    df_fc = pd.DataFrame({
                        "frame": [times[-1] + 1],
                        "people": [forecast_val]
                    })
                    fc_point = alt.Chart(df_fc).mark_point(size=80).encode(
                        x="frame:Q",
                        y="people:Q"
                    )
                    chart = line + fc_point
                else:
                    chart = line

                chart_placeholder.altair_chart(chart, use_container_width=True)

            metrics_placeholder.metric(
                "People (all sources, current step)",
                value=total_people_all_sources
            )

        # End while loop
        for cap in caps:
            cap.release()

        # -----------------------------
        # Save logs if requested
        # -----------------------------
        if logs and save_logs:
            logs_df = pd.DataFrame(logs)
            csv_name = f"air_aim_logs_{int(time.time())}.csv"
            logs_df.to_csv(csv_name, index=False)
            log_download_placeholder.download_button(
                label="Download CSV Logs",
                data=logs_df.to_csv(index=False).encode("utf-8"),
                file_name=csv_name,
                mime="text/csv"
            )

elif start and not sources:
    st.warning("No valid sources found. Please add a webcam, upload videos, or RTSP URLs.")
