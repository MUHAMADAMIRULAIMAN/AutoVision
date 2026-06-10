import os

os.environ["ULTRALYTICS_NO_AUTOUPDATES"] = "true"

import streamlit as st
import cv2
import time
import pandas as pd
import numpy as np
import hashlib
from ultralytics import YOLO
from supabase import create_client
from datetime import datetime, timezone
import serial
import serial.tools.list_ports
import plotly.express as px
import plotly.graph_objects as go


# =========================================================
# 1. STREAMLIT PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="DRB-HICOM Defect System",
    layout="wide",
    page_icon="🛡️"
)


# =========================================================
# 2. CUSTOM STYLES
# =========================================================

LOGO_PATH = "DRB_HiCOM_Logo.png"

def to_myt(ts_str):
    """Convert a UTC ISO timestamp string to Malaysia Time (UTC+8)."""
    ts = pd.to_datetime(ts_str)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("Asia/Kuala_Lumpur")


def apply_custom_styles():
    st.markdown(
        """
        <style>
            .stApp {
                background-color: #0E1117;
                color: #FFFFFF;
            }

            [data-testid="stSidebar"] {
                background-color: #00153B;
                border-right: 2px solid #00205B;
            }

            div.stButton > button[kind="primary"],
            [data-testid="stFormSubmitButton"] > button {
                background-color: #D6001C !important;
                border-color: #D6001C !important;
                color: #FFFFFF !important;
            }

            div.stButton > button[kind="primary"]:hover,
            [data-testid="stFormSubmitButton"] > button:hover {
                background-color: #FF1A3A !important;
                border-color: #FF1A3A !important;
            }

            div.stButton > button[kind="secondary"] {
                border-color: #00205B !important;
                color: #FFFFFF !important;
                background-color: transparent !important;
            }

            div.stButton > button[kind="secondary"]:hover {
                border-color: #D6001C !important;
                color: #D6001C !important;
            }

            .stTextInput>div>div>input,
            .stSelectbox>div>div>div[aria-expanded="false"],
            .stNumberInput>div>div>input {
                color: #FFFFFF;
                background-color: #1C1E26;
                border: 1px solid #00205B;
            }

            [data-testid="stDataFrame"] {
                border: 1px solid #00205B;
            }

            hr {
                border-color: #D6001C !important;
                opacity: 0.5;
            }

            /* Cap camera feed height so the page fits without scrolling */
            [data-testid="stImage"] img {
                max-height: 340px;
                object-fit: contain;
                width: 100%;
            }

            /* Tighten metric padding on the detection page */
            [data-testid="stMetric"] {
                padding: 4px 8px !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )


apply_custom_styles()


# =========================================================
# 3. SUPABASE CONFIGURATION
# =========================================================

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]


# =========================================================
# 4. SESSION STATE
# =========================================================

if "user" not in st.session_state:
    st.session_state.user = None

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "pending_delete_user_id" not in st.session_state:
    st.session_state.pending_delete_user_id = None

if "trend_data" not in st.session_state:
    st.session_state.trend_data = []

if "camera_active" not in st.session_state:
    st.session_state.camera_active = False


# =========================================================
# 5. CACHED RESOURCES
# =========================================================

def get_available_ports():
    ports = serial.tools.list_ports.comports()
    return [port.device for port in ports]


@st.cache_resource
def connect_arduino(selected_port):
    if selected_port is None:
        print("⚠️ No Arduino COM port selected.")
        return None

    try:
        ser = serial.Serial(selected_port, 9600, timeout=1)
        time.sleep(2)

        ser.reset_input_buffer()
        ser.reset_output_buffer()

        print(f"✅ Arduino connected successfully on {selected_port}.")
        return ser

    except Exception as e:
        print(f"❌ Arduino connection failed on {selected_port}: {e}")
        return None


@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)


@st.cache_resource
def load_model():
    return YOLO("bestimage.pt")


try:
    supabase = init_supabase()
    model = load_model()

except Exception as e:
    st.error(f"System Initialization Error: {e}")
    st.stop()


# =========================================================
# 6. AUTHENTICATION FUNCTIONS
# =========================================================

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def login(username, password):
    if not username or not password:
        st.error("Please enter both username and password.")
        return

    try:
        password_hash = hash_password(password)

        with st.spinner("Logging in..."):
            response = (
                supabase
                .table("users")
                .select("*")
                .eq("username", username)
                .eq("password", password_hash)
                .execute()
            )

        if len(response.data) > 0:
            st.session_state.user = response.data[0]
            st.session_state.logged_in = True
            st.success("Login Successful!")
            st.rerun()
        else:
            st.error("Invalid username or password.")

    except Exception as e:
        st.error(f"Login Error: {e}")


def logout():
    st.session_state.user = None
    st.session_state.logged_in = False
    st.rerun()


# =========================================================
# 7. DATABASE HELPERS
# =========================================================

def log_inspection(status, confidence):
    try:
        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "confidence_score": float(confidence) if confidence is not None else None,
            "operator_id": st.session_state.user["id"]
        }

        response = (
            supabase
            .table("inspections")
            .insert(data)
            .execute()
        )

        if response.data is not None and len(response.data) > 0:
            return True

        return False

    except Exception as e:
        print(f"❌ Database Save Error: {e}")
        return False


def fetch_recent_logs(limit=5):
    try:
        response = (
            supabase
            .table("inspections")
            .select("*")
            .order("timestamp", desc=True)
            .limit(limit)
            .execute()
        )

        return response.data

    except Exception as e:
        st.error(f"Failed to fetch recent logs: {e}")
        return []


def fetch_all_inspection_logs(page_size=1000):
    all_rows = []
    start = 0

    while True:
        end = start + page_size - 1

        response = (
            supabase
            .table("inspections")
            .select("*")
            .order("timestamp", desc=True)
            .range(start, end)
            .execute()
        )

        rows = response.data or []

        if not rows:
            break

        all_rows.extend(rows)

        if len(rows) < page_size:
            break

        start += page_size

    return all_rows


def update_log_display(container):
    logs = fetch_recent_logs(limit=5)

    container.empty()

    with container.container():
        st.subheader("Recent Inspection Logs")

        if logs:
            for row in logs:
                color = "#D6001C" if row["status"] == "Fail" else "#21c354"
                t_stamp = to_myt(row["timestamp"]).strftime("%H:%M:%S")

                st.markdown(
                    f"""
                    <div style="
                        padding:10px;
                        border-left: 5px solid {color};
                        background-color: #1C1E26;
                        margin-bottom: 10px;
                        border-radius: 5px;
                    ">
                        <strong>{row["status"]}</strong> | Confidence: {f'{row["confidence_score"]:.2f}' if row.get("confidence_score") is not None else "N/A"}<br>
                        <small style="color: #bbb;">🕒 {t_stamp}</small>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.info("No logs found yet.")


def display_operator_charts(container):
    try:
        today = datetime.now(timezone.utc).date().isoformat()
        response = (
            supabase
            .table("inspections")
            .select("status")
            .gte("timestamp", f"{today}T00:00:00+00:00")
            .execute()
        )

        df = pd.DataFrame(response.data)

        container.empty()

        with container.container():
            st.subheader("Inspection Summary Charts")

            if df.empty:
                st.info("No inspection data available for charts.")
                return

            status_counts = (
                df["status"]
                .value_counts()
                .reset_index()
            )

            status_counts.columns = ["Status", "Count"]

            chart_col1, chart_col2 = st.columns(2)

            fig_bar = px.bar(
                status_counts,
                x="Status",
                y="Count",
                color="Status",
                text="Count",
                color_discrete_map={
                    "Pass": "#21c354",
                    "Fail": "#D6001C"
                },
                title="Pass vs Fail Count"
            )

            fig_bar.update_layout(
                plot_bgcolor="#0E1117",
                paper_bgcolor="#0E1117",
                font_color="white",
                xaxis_title="Status",
                yaxis_title="Count",
                showlegend=False
            )

            fig_bar.update_traces(textposition="outside")

            chart_col1.plotly_chart(fig_bar, use_container_width=True)

            fig_pie = px.pie(
                status_counts,
                names="Status",
                values="Count",
                color="Status",
                color_discrete_map={
                    "Pass": "#21c354",
                    "Fail": "#D6001C"
                },
                title="Pass vs Fail Percentage",
                hole=0.4
            )

            fig_pie.update_layout(
                plot_bgcolor="#0E1117",
                paper_bgcolor="#0E1117",
                font_color="white"
            )

            chart_col2.plotly_chart(fig_pie, use_container_width=True)

    except Exception as e:
        container.error(f"Chart Error: {e}")


def fetch_today_counts():
    try:
        today = datetime.now(timezone.utc).date().isoformat()
        response = (
            supabase
            .table("inspections")
            .select("status")
            .gte("timestamp", f"{today}T00:00:00+00:00")
            .execute()
        )
        rows = response.data or []
        passed = sum(1 for r in rows if r["status"] == "Pass")
        failed = sum(1 for r in rows if r["status"] == "Fail")
        return passed, failed
    except Exception:
        return 0, 0


def show_result_card(container, status, confidence):
    container.empty()

    if status == "Pass":
        color = "#21c354"
        icon = "✅"
        title = "PRODUCT PASSED"
        message = "No defect detected"
    else:
        color = "#D6001C"
        icon = "❌"
        title = "DEFECT DETECTED"
        message = f"Confidence: {confidence:.2f}" if confidence is not None else "Defect detected"

    container.markdown(
        f"""
        <div style="
            background-color: {color};
            padding: 10px 14px;
            border-radius: 8px;
            color: white;
            text-align: center;
            margin-bottom: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        ">
            <span style="font-size: 22px;">{icon}</span>
            <span style="font-size: 17px; font-weight: 800; margin-left: 8px;">{title}</span>
            <span style="font-size: 13px; margin-left: 10px; opacity: 0.9;">{message}</span>
        </div>
        """,
        unsafe_allow_html=True
    )


def create_confidence_overlay(frame_rgb, confidence, is_defect):
    h, w = frame_rgb.shape[:2]
    Y, X = np.ogrid[:h, :w]
    cx, cy = w // 2, h // 2
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    mask = (1 - dist / max_dist) * confidence
    mask = np.clip(mask * 255, 0, 255).astype(np.uint8)
    color_overlay = np.zeros_like(frame_rgb)
    if is_defect:
        color_overlay[:, :, 0] = mask  # Red channel
    else:
        color_overlay[:, :, 1] = mask  # Green channel
    return cv2.addWeighted(frame_rgb, 0.75, color_overlay, 0.25, 0)


def render_trend(container):
    data = st.session_state.trend_data
    container.empty()
    with container.container():
        if len(data) < 2:
            st.info("Waiting for inspections to build trend...")
            return
        df = pd.DataFrame(data)
        df["defect"] = (df["status"] == "Fail").astype(int)
        df["defect_rate"] = df["defect"].expanding().mean() * 100
        df["#"] = range(1, len(df) + 1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["#"], y=df["defect_rate"],
            mode="lines+markers",
            line=dict(color="#D6001C", width=2),
            marker=dict(size=5),
            name="Defect Rate"
        ))
        fig.add_hline(
            y=10, line_dash="dash",
            line_color="#FFD700",
            annotation_text="10% limit",
            annotation_font_color="#FFD700"
        )
        fig.update_layout(
            title="Live Defect Rate Trend",
            xaxis_title="Inspection #",
            yaxis_title="Defect Rate (%)",
            yaxis=dict(range=[0, 100]),
            plot_bgcolor="#0E1117",
            paper_bgcolor="#0E1117",
            font_color="white",
            height=260,
            margin=dict(l=0, r=0, t=35, b=0),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)


# =========================================================
# 8. LOGIN SCREEN
# =========================================================

if not st.session_state.logged_in:
    try:
        st.image(LOGO_PATH, width=300)
    except Exception:
        pass

    st.title("🔐 AutoVision System Login")

    col1, col2 = st.columns([1, 2])

    with col1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Log In", use_container_width=True, type="primary"):
            login(username, password)

    st.stop()


# =========================================================
# 9. SIDEBAR AND NAVIGATION
# =========================================================

try:
    st.sidebar.image(LOGO_PATH, use_container_width=True)
    st.sidebar.markdown("---")
except Exception:
    pass

st.sidebar.title(f"👤 {st.session_state.user['username']}")
st.sidebar.caption(f"Role: {st.session_state.user.get('role', 'Operator').upper()}")

role = st.session_state.user.get("role", "operator")


# =========================================================
# ARDUINO COM PORT SELECTION
# =========================================================

available_ports = get_available_ports()

if available_ports:
    selected_port = st.sidebar.selectbox(
        "Arduino COM Port",
        available_ports
    )
else:
    selected_port = None
    st.sidebar.warning("No COM ports detected.")

arduino = connect_arduino(selected_port)

st.sidebar.markdown("---")
st.sidebar.subheader("🖥️ System Health")

cam_ok = st.session_state.get("camera_active", False)
arduino_ok = arduino is not None and arduino.is_open
model_ok = model is not None
try:
    supabase.table("inspections").select("id").limit(1).execute()
    db_ok = True
except Exception:
    db_ok = False

st.sidebar.markdown(
    f"{'🟢' if cam_ok else '🔴'} **Camera:** {'Active' if cam_ok else 'Inactive'}\n\n"
    f"{'🟢' if arduino_ok else '🔴'} **Arduino:** {'Connected' if arduino_ok else 'Disconnected'}\n\n"
    f"{'🟢' if db_ok else '🔴'} **Database:** {'Online' if db_ok else 'Offline'}\n\n"
    f"{'🟢' if model_ok else '🔴'} **AI Model:** {'Loaded' if model_ok else 'Not loaded'}"
)


if role == "admin":
    menu_options = [
        "Defect Detection",
        "Inspection Logs",
        "User Management (Admin)",
        "Manage Profile"
    ]
else:
    menu_options = [
        "Defect Detection",
        "Manage Profile"
    ]

selected_page = st.sidebar.radio("Navigation", menu_options)

if st.sidebar.button("Log Out"):
    logout()


# =========================================================
# 10. DEFECT DETECTION PAGE
# =========================================================

if selected_page == "Defect Detection":
    st.title("🛡️ Defect Detection & Operator Dashboard")

    conf_threshold = 0.15

    today_counter = st.empty()

    def render_today_counter(container):
        passed, failed = fetch_today_counts()
        total = passed + failed
        with container.container():
            m1, m2, m3 = st.columns(3)
            m1.metric("Today — Total", total)
            m2.metric("Passed", passed)
            m3.metric("Failed", failed)

    render_today_counter(today_counter)

    col_main, col_side = st.columns([2, 1])

    with col_main:
        st.subheader("Live Defect Detection")
        run_system = st.checkbox("Start Camera System", value=False)
        result_card = st.empty()
        status_box = st.empty()
        frame_window = st.empty()

    with col_side:
        log_placeholder = st.empty()
        trend_placeholder = st.empty()

    update_log_display(log_placeholder)
    render_trend(trend_placeholder)

    if run_system:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        frame_count = 0
        latest_display_frame = None

        if not cap.isOpened():
            st.error("Camera could not be opened.")
            st.session_state.camera_active = False
        else:
            st.session_state.camera_active = True
            status_box.info("System running. Waiting for Arduino SCAN signal...")

        try:
         while cap.isOpened() and run_system:
            ret, frame = cap.read()

            if not ret:
                st.error("Camera error.")
                break

            frame_count += 1

            # =====================================================
            # FRAME SKIPPING FOR PERFORMANCE
            # YOLO only runs on every 3rd frame.
            # Skipped frames show raw camera image.
            # =====================================================

            if frame_count % 3 == 0:
                preview_results = model.predict(
                    frame,
                    verbose=False
                )

                annotated_frame = preview_results[0].plot()
                latest_display_frame = cv2.cvtColor(
                    annotated_frame,
                    cv2.COLOR_BGR2RGB
                )

            else:
                latest_display_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

            frame_window.image(
                latest_display_frame,
                channels="RGB",
                use_container_width=True
            )

            # =====================================================
            # HARDWARE COMMUNICATION
            # =====================================================

            if arduino is not None and arduino.in_waiting > 0:
                try:
                    raw_msg = arduino.read(arduino.in_waiting)
                    arduino_msg = raw_msg.decode("utf-8", errors="ignore").strip()

                    if arduino_msg:
                        print(f"📥 Received from Arduino: '{arduino_msg}'")
                        status_box.caption(f"Arduino: {arduino_msg}")

                    if "SCAN" in arduino_msg.upper():
                        print("🎯 SCAN command detected.")
                        status_box.warning("SCAN received. Capturing inspection image...")

                        time.sleep(0.8)

                        for _ in range(10):
                            cap.read()

                        # =================================================
                        # MULTI-FRAME SCAN
                        # Capture 5 frames, run YOLO on each.
                        # Fail if any frame detects a defect — keeps the
                        # highest-confidence detection as the result.
                        # =================================================

                        SCAN_FRAMES = 5
                        best_conf = None
                        best_result = None

                        cls_names = None

                        for _ in range(SCAN_FRAMES):
                            ret, fresh_frame = cap.read()
                            if not ret:
                                continue

                            result = model.predict(
                                fresh_frame,
                                verbose=False
                            )

                            if cls_names is None:
                                cls_names = result[0].names

                            defect_idx = next(
                                (k for k, v in cls_names.items() if v.lower() == "defect"),
                                None
                            )

                            if defect_idx is not None:
                                frame_conf = float(result[0].probs.data[defect_idx])
                                if best_conf is None or frame_conf > best_conf:
                                    best_conf = frame_conf
                                    best_result = result
                            elif best_result is None:
                                best_result = result

                        if best_result is None:
                            status_box.error("Failed to capture inspection image.")
                        else:
                            # =================================================
                            # RESULT DECISION
                            # =================================================

                            if best_conf is not None and best_conf >= conf_threshold:
                                conf = best_conf
                                status = "Fail"
                                command = b"0"

                                st.toast(f"❌ Defect Detected! ({conf:.2f})")
                                status_box.error(
                                    f"Defect detected. Confidence: {conf:.2f}"
                                )

                            else:
                                conf = best_conf if best_conf is not None else 0.0
                                status = "Pass"
                                command = b"1"

                                st.toast("✅ Product Passed Inspection")
                                status_box.success("Product passed inspection.")

                            # =================================================
                            # HEATMAP OVERLAY
                            # =================================================

                            inspected_rgb = cv2.cvtColor(
                                best_result[0].plot(),
                                cv2.COLOR_BGR2RGB
                            )

                            heatmap_conf = conf if conf is not None else 0.0
                            inspected_rgb = create_confidence_overlay(
                                inspected_rgb,
                                heatmap_conf,
                                is_defect=(status == "Fail")
                            )

                            frame_window.image(
                                inspected_rgb,
                                channels="RGB",
                                use_container_width=True
                            )

                            show_result_card(result_card, status, conf if status == "Fail" else None)

                            # =================================================
                            # SAVE TO SUPABASE FIRST, THEN MOVE MOTOR
                            # =================================================

                            print("💾 Saving inspection result to Supabase...")
                            save_success = log_inspection(status, conf)

                            if save_success:
                                print("✅ Data saved successfully.")

                                st.session_state.trend_data.append({"status": status})
                                if len(st.session_state.trend_data) > 50:
                                    st.session_state.trend_data = st.session_state.trend_data[-50:]

                                render_today_counter(today_counter)
                                update_log_display(log_placeholder)
                                render_trend(trend_placeholder)

                                if arduino is not None and arduino.is_open:
                                    if status == "Pass":
                                        print("📤 Sending '1' to Arduino: Forward/Pass")
                                    else:
                                        print("📤 Sending '0' to Arduino: Backward/Fail")

                                    arduino.write(command)
                                    arduino.flush()

                                    status_box.info(
                                        "Result saved. Motor command sent to Arduino."
                                    )
                                else:
                                    status_box.error(
                                        "Arduino not connected. Motor command not sent."
                                    )

                            else:
                                print("❌ Supabase save failed. Motor command NOT sent.")
                                status_box.error(
                                    "Supabase save failed. Motor command was not sent."
                                )

                        time.sleep(0.5)
                        if arduino is not None:
                            arduino.reset_input_buffer()

                except Exception as e:
                    print(f"❌ Serial communication error: {e}")
                    status_box.error(f"Serial communication error: {e}")

        finally:
            cap.release()
            st.session_state.camera_active = False



# =========================================================
# 12. INSPECTION LOGS PAGE WITH ANALYTICS — ADMIN ONLY
# =========================================================

elif selected_page == "Inspection Logs":
    st.title("📋 Inspection History & Analytics")

    if role != "admin":
        st.warning("You do not have permission to view this page.")
        st.stop()

    try:
        logs_data = fetch_all_inspection_logs(page_size=1000)
        df_logs = pd.DataFrame(logs_data)

        users_res = (
            supabase
            .table("users")
            .select("id, username")
            .execute()
        )

        if not df_logs.empty:
            df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"], utc=True).dt.tz_convert("Asia/Kuala_Lumpur")

            st.subheader("📅 Filter by Date")

            col_date1, col_date2 = st.columns(2)

            min_date = df_logs["timestamp"].min().date()
            max_date = df_logs["timestamp"].max().date()

            with col_date1:
                start_date = st.date_input("Start Date", min_date)

            with col_date2:
                end_date = st.date_input("End Date", max_date)

            mask = (
                (df_logs["timestamp"].dt.date >= start_date)
                & (df_logs["timestamp"].dt.date <= end_date)
            )

            df_logs = df_logs.loc[mask]

            if users_res.data:
                user_map = {
                    user["id"]: user["username"]
                    for user in users_res.data
                }

                if "operator_id" in df_logs.columns:
                    df_logs["Operator"] = (
                        df_logs["operator_id"]
                        .map(user_map)
                        .fillna("Unknown")
                    )

            if df_logs.empty:
                st.warning("No inspection records found for the selected date range.")
                st.stop()

            total_inspections = len(df_logs)
            pass_count = len(df_logs[df_logs["status"] == "Pass"])
            fail_count = len(df_logs[df_logs["status"] == "Fail"])

            fail_rate = 0
            if total_inspections > 0:
                fail_rate = (fail_count / total_inspections) * 100

            avg_confidence = df_logs["confidence_score"].mean()

            st.subheader("📊 Summary Overview")

            metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)

            metric_col1.metric("Total Inspections", total_inspections)
            metric_col2.metric("Pass", pass_count)
            metric_col3.metric("Fail", fail_count)
            metric_col4.metric("Fail Rate", f"{fail_rate:.1f}%")
            metric_col5.metric("Avg Confidence", f"{avg_confidence:.2f}" if pd.notna(avg_confidence) else "N/A")

            st.divider()

            st.subheader("📈 Inspection Analytics")

            status_counts = (
                df_logs["status"]
                .value_counts()
                .reset_index()
            )

            status_counts.columns = ["Status", "Count"]

            chart_col1, chart_col2 = st.columns(2)

            fig_bar = px.bar(
                status_counts,
                x="Status",
                y="Count",
                color="Status",
                text="Count",
                color_discrete_map={
                    "Pass": "#21c354",
                    "Fail": "#D6001C"
                },
                title="Pass vs Fail Count"
            )

            fig_bar.update_layout(
                plot_bgcolor="#0E1117",
                paper_bgcolor="#0E1117",
                font_color="white",
                showlegend=False
            )

            fig_bar.update_traces(textposition="outside")

            chart_col1.plotly_chart(fig_bar, use_container_width=True)

            fig_pie = px.pie(
                status_counts,
                names="Status",
                values="Count",
                color="Status",
                color_discrete_map={
                    "Pass": "#21c354",
                    "Fail": "#D6001C"
                },
                title="Pass vs Fail Percentage",
                hole=0.4
            )

            fig_pie.update_layout(
                plot_bgcolor="#0E1117",
                paper_bgcolor="#0E1117",
                font_color="white"
            )

            chart_col2.plotly_chart(fig_pie, use_container_width=True)

            st.divider()

            st.subheader("🧾 Filtered Inspection Records")

            cols_to_keep = [
                "inspection_id",
                "timestamp",
                "status",
                "confidence_score",
                "Operator"
            ]

            final_cols = [
                col for col in cols_to_keep
                if col in df_logs.columns
            ]

            df_display = df_logs[final_cols].copy()

            df_display["timestamp"] = (
                df_display["timestamp"]
                .dt.strftime("%Y-%m-%d %H:%M:%S MYT")
            )

            col_f1, _ = st.columns(2)

            with col_f1:
                if "status" in df_display.columns:
                    unique_status = df_display["status"].unique().tolist()

                    status_filter = st.multiselect(
                        "Filter by Status",
                        unique_status,
                        default=unique_status
                    )

                    df_display = df_display[
                        df_display["status"].isin(status_filter)
                    ]

            def highlight_status(val):
                if val == "Fail":
                    return "color: #ff4b4b; font-weight: bold"
                if val == "Pass":
                    return "color: #21c354; font-weight: bold"
                return ""

            st.dataframe(
                df_display.style.map(
                    highlight_status,
                    subset=["status"]
                ),
                use_container_width=True,
                height=500
            )

            csv = df_display.to_csv(index=False).encode("utf-8")

            st.download_button(
                label=f"📥 Download {start_date} to {end_date} Logs CSV",
                data=csv,
                file_name=f"inspection_logs_{start_date}_to_{end_date}.csv",
                mime="text/csv",
                type="secondary"
            )

        else:
            st.info("No inspection records found.")

    except Exception as e:
        st.error(f"Error fetching logs: {e}")


# =========================================================
# 12. USER MANAGEMENT PAGE
# =========================================================

elif selected_page == "User Management (Admin)":
    st.title("👥 User Management")

    if role != "admin":
        st.warning("You do not have permission to view this page.")
        st.stop()

    with st.expander("➕ Add New User", expanded=False):
        with st.form("add_user_form"):
            st.subheader("Add New Operator")

            new_user = st.text_input("Username")
            new_email = st.text_input(
                "Email",
                placeholder="e.g. operator@autovision.com"
            )
            new_pass = st.text_input("Password", type="password")
            new_role = st.selectbox("Role", ["operator", "admin"])

            submitted = st.form_submit_button(
                "Create User",
                type="primary"
            )

            if submitted:
                if not new_user or not new_pass:
                    st.warning("⚠️ Username and Password are required.")
                else:
                    try:
                        supabase.table("users").insert(
                            {
                                "username": new_user,
                                "email": new_email,
                                "password": hash_password(new_pass),
                                "role": new_role
                            }
                        ).execute()

                        st.success(f"✅ User '{new_user}' created successfully!")
                        time.sleep(1)
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error creating user: {e}")

    st.divider()
    st.subheader("Existing Users")

    try:
        users_res = (
            supabase
            .table("users")
            .select("*")
            .order("id", desc=False)
            .execute()
        )

        users = users_res.data

        if users:
            h1, h2, h3, h4, h5 = st.columns([1, 2, 2, 1, 1])

            h1.markdown("**ID**")
            h2.markdown("**Username**")
            h3.markdown("**Email**")
            h4.markdown("**Role**")
            h5.markdown("**Action**")

            st.divider()

            for user in users:
                c1, c2, c3, c4, c5 = st.columns([1, 2, 2, 1, 1])

                c1.write(str(user["id"]))
                c2.write(user["username"])
                c3.write(user.get("email", "-"))
                c4.write(user["role"].upper())

                if st.session_state.pending_delete_user_id == user["id"]:
                    c5.warning("Sure?")
                    conf_col, cancel_col = st.columns(2)
                    if conf_col.button("Yes, delete", key=f"confirm_del_{user['id']}", type="primary"):
                        if user["id"] == st.session_state.user["id"]:
                            st.error("You cannot delete your own account!")
                            st.session_state.pending_delete_user_id = None
                        else:
                            try:
                                # Detach inspection records before deleting the
                                # user so the FK constraint is satisfied.
                                # Logs are preserved and show as "Unknown".
                                supabase.table("inspections").update(
                                    {"operator_id": None}
                                ).eq("operator_id", user["id"]).execute()

                                supabase.table("users").delete().eq(
                                    "id",
                                    user["id"]
                                ).execute()

                                st.session_state.pending_delete_user_id = None
                                st.toast(f"🗑️ Deleted user: {user['username']}")
                                time.sleep(0.5)
                                st.rerun()

                            except Exception as e:
                                st.error(f"Failed to delete: {e}")

                    if cancel_col.button("Cancel", key=f"cancel_del_{user['id']}"):
                        st.session_state.pending_delete_user_id = None
                        st.rerun()

                else:
                    if c5.button(
                        "🗑️",
                        key=f"del_user_{user['id']}",
                        help="Delete this user"
                    ):
                        if user["id"] == st.session_state.user["id"]:
                            st.error("You cannot delete your own account!")
                        else:
                            st.session_state.pending_delete_user_id = user["id"]
                            st.rerun()

                st.markdown("<hr>", unsafe_allow_html=True)

        else:
            st.info("No users found.")

    except Exception as e:
        st.error(f"Error loading users: {e}")


# =========================================================
# 13. MANAGE PROFILE PAGE
# =========================================================

elif selected_page == "Manage Profile":
    user = st.session_state.user

    st.title("My Profile Settings")

    with st.container():
        st.header("Edit Profile")

        col1, col2 = st.columns([3, 1])

        with col1:
            st.text_input(
                "Username:",
                value=user["username"],
                disabled=True,
                key="profile_uname"
            )

        with col2:
            st.markdown(
                "<br style='height: 20px;'> (Read Only)",
                unsafe_allow_html=True
            )

        current_email = user.get("email", "")

        if current_email is None:
            current_email = ""

        new_email = st.text_input(
            "Email:",
            value=current_email,
            key="profile_email"
        )

        st.markdown("---")
        st.subheader("Change Password")

        current_pass = st.text_input(
            "Current Password:",
            type="password",
            help="Required to save changes",
            key="p_current"
        )

        new_pass = st.text_input(
            "New Password:",
            type="password",
            key="p_new"
        )

        confirm_pass = st.text_input(
            "Confirm New Password:",
            type="password",
            key="p_confirm"
        )

        st.write("")

        if st.button("Save Changes", type="primary"):
            error_msg = None

            if not current_pass:
                error_msg = "⚠️ Please enter your current password to confirm changes."

            elif hash_password(current_pass) != user["password"]:
                error_msg = "❌ Incorrect current password."

            elif new_pass or confirm_pass:
                if new_pass != confirm_pass:
                    error_msg = "❌ New passwords do not match."

                elif len(new_pass) < 6:
                    error_msg = "⚠️ New password must be at least 6 characters."

            if error_msg:
                st.error(error_msg)

            else:
                try:
                    update_data = {}

                    if new_email != current_email:
                        update_data["email"] = new_email

                    if new_pass:
                        update_data["password"] = hash_password(new_pass)

                    if update_data:
                        (
                            supabase
                            .table("users")
                            .update(update_data)
                            .eq("id", user["id"])
                            .execute()
                        )

                        if "email" in update_data:
                            st.session_state.user["email"] = new_email

                        if "password" in update_data:
                            st.session_state.user["password"] = hash_password(new_pass)

                        st.success("✅ Profile updated successfully!")
                        time.sleep(1)
                        st.rerun()

                    else:
                        st.info("No changes detected.")

                except Exception as e:
                    st.error(f"Failed to update profile: {e}")
