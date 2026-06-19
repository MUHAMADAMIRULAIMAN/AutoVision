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


CHART_PASS_COLOR = "#0D9F4F"
CHART_FAIL_COLOR = "#D6001C"

def _chart_layout(**extra):
    base = dict(
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#1A2744", size=13, family="Segoe UI, sans-serif"),
        xaxis=dict(
            gridcolor="#EEF2F7", linecolor="#DDE4EF", showline=True,
            tickfont=dict(color="#1A2744", size=13),
            title_font=dict(color="#00205B", size=13),
        ),
        yaxis=dict(
            gridcolor="#EEF2F7", linecolor="#DDE4EF", showline=True,
            tickfont=dict(color="#1A2744", size=13),
            title_font=dict(color="#00205B", size=13),
        ),
        title_font=dict(color="#00205B", size=15, family="Segoe UI, sans-serif"),
        showlegend=False,
        margin=dict(l=8, r=8, t=52, b=8),
    )
    base.update(extra)
    return base


def apply_custom_styles():
    st.markdown(
        """
        <style>
            /* ── Design tokens ──────────────────────────────
               white   : #FFFFFF
               navy    : #00205B  (DRB-HICOM primary)
               navy-mid: #003087
               red     : #D6001C  (DRB-HICOM accent)
               red-soft: #FFF0F2  (red tint for backgrounds)
               grey-50 : #F8FAFB  (page background)
               grey-100: #EEF2F7  (subtle fill)
               grey-200: #DDE4EF  (borders)
               text-pri: #1A2744
               text-sec: #5A7299
               green   : #0D9F4F
               green-bg: #EAF7EE
            ──────────────────────────────────────────────── */

            /* ── Base ── */
            .stApp {
                background-color: #FFFFFF;
                color: #1A2744;
                font-family: 'Segoe UI', sans-serif;
            }

            /* ── Sidebar ── */
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #00205B 0%, #001A4A 100%);
                border-right: 3px solid #D6001C;
            }
            [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] span,
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] h1,
            [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3,
            [data-testid="stSidebar"] .stMarkdown,
            [data-testid="stSidebar"] .stCaption,
            [data-testid="stSidebar"] .stRadio label,
            [data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
                color: #FFFFFF !important;
            }
            [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] {
                background-color: #002E7A !important;
                border-color: #4A7CC7 !important;
            }
            [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] * {
                color: #FFFFFF !important;
            }
            [data-testid="stSidebar"] .stAlert {
                background-color: #002E7A;
                border-color: #D6001C;
            }

            /* ── Buttons ── */
            div.stButton > button[kind="primary"],
            [data-testid="stFormSubmitButton"] > button {
                background-color: #D6001C !important;
                border-color: #D6001C !important;
                color: #FFFFFF !important;
                border-radius: 6px !important;
                font-weight: 600 !important;
                letter-spacing: 0.3px;
            }
            div.stButton > button[kind="primary"]:hover,
            [data-testid="stFormSubmitButton"] > button:hover {
                background-color: #B5001A !important;
                border-color: #B5001A !important;
            }
            div.stButton > button[kind="secondary"] {
                border: 1.5px solid #00205B !important;
                color: #00205B !important;
                background-color: #FFFFFF !important;
                border-radius: 6px !important;
                font-weight: 500 !important;
            }
            div.stButton > button[kind="secondary"]:hover {
                border-color: #D6001C !important;
                color: #D6001C !important;
                background-color: #FFF0F2 !important;
            }

            /* ── Download button ── */
            [data-testid="stDownloadButton"] > button {
                background-color: #00205B !important;
                border-color: #00205B !important;
                color: #FFFFFF !important;
                border-radius: 6px !important;
                font-weight: 600 !important;
            }
            [data-testid="stDownloadButton"] > button:hover {
                background-color: #D6001C !important;
                border-color: #D6001C !important;
                color: #FFFFFF !important;
            }

            /* Sidebar log-out button */
            [data-testid="stSidebar"] div.stButton > button {
                border-color: rgba(255,255,255,0.35) !important;
                color: #FFFFFF !important;
                background-color: transparent !important;
            }
            [data-testid="stSidebar"] div.stButton > button:hover {
                background-color: #D6001C !important;
                border-color: #D6001C !important;
            }

            /* ── Inputs ── */
            .stTextInput > div > div > input,
            .stNumberInput > div > div > input,
            .stTextArea textarea {
                background-color: #FFFFFF !important;
                border: 1.5px solid #DDE4EF !important;
                border-radius: 6px !important;
                color: #1A2744 !important;
            }
            .stTextInput > div > div > input:focus,
            .stNumberInput > div > div > input:focus {
                border-color: #00205B !important;
                box-shadow: 0 0 0 2px rgba(0,32,91,0.12) !important;
            }
            .stSelectbox > div > div > div[aria-expanded="false"] {
                background-color: #FFFFFF !important;
                border: 1.5px solid #DDE4EF !important;
                color: #1A2744 !important;
            }

            /* ── Sliders ── */
            [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
                background-color: #D6001C !important;
            }
            [data-testid="stSlider"] [data-baseweb="slider"] div:first-child > div:first-child {
                background-color: #D6001C !important;
            }

            /* ── Metric cards ── */
            [data-testid="stMetric"] {
                background-color: #FFFFFF;
                border: 1px solid #DDE4EF;
                border-top: 3px solid #00205B;
                border-radius: 10px;
                padding: 14px 18px !important;
                box-shadow: 0 2px 8px rgba(0,32,91,0.06);
            }
            [data-testid="stMetric"] label {
                color: #5A7299 !important;
                font-size: 11px !important;
                font-weight: 700 !important;
                text-transform: uppercase;
                letter-spacing: 0.8px;
            }
            [data-testid="stMetric"] [data-testid="stMetricValue"] {
                color: #00205B !important;
                font-size: 28px !important;
                font-weight: 800 !important;
            }

            /* ── Expander ── */
            [data-testid="stExpander"] {
                border: 1px solid #DDE4EF !important;
                border-radius: 10px !important;
                background-color: #FFFFFF !important;
                box-shadow: 0 1px 4px rgba(0,32,91,0.05);
            }
            [data-testid="stExpander"] summary {
                color: #00205B !important;
                font-weight: 600 !important;
                font-size: 14px;
            }
            [data-testid="stExpander"] summary:hover {
                color: #D6001C !important;
            }

            /* ── Data table ── */
            [data-testid="stDataFrame"] {
                border: 1px solid #DDE4EF;
                border-radius: 10px;
                overflow: hidden;
            }

            /* ── Dividers ── */
            hr {
                border: none !important;
                border-top: 1px solid #EEF2F7 !important;
                margin: 16px 0 !important;
            }
            [data-testid="stSidebar"] hr {
                border-top: 1px solid rgba(255,255,255,0.12) !important;
            }

            /* ── Alerts/info boxes ── */
            [data-testid="stAlert"] {
                border-radius: 8px !important;
                border-left-width: 4px !important;
            }

            /* ── Camera image ── */
            [data-testid="stImage"] img {
                max-height: 340px;
                object-fit: contain;
                width: 100%;
                border-radius: 10px;
                border: 1px solid #DDE4EF;
                box-shadow: 0 2px 8px rgba(0,32,91,0.06);
            }

            /* ── Page headings ── */
            h1 { color: #00205B !important; font-weight: 800 !important; }
            h2 { color: #00205B !important; font-weight: 700 !important; }
            h3 { color: #003087 !important; font-weight: 600 !important; }

            /* ── Force all main-content text visible on white ── */

            /* Generic widget label (covers checkbox, slider, number input, selectbox) */
            section[data-testid="stMain"] [data-testid="stWidgetLabel"] p,
            section[data-testid="stMain"] [data-testid="stWidgetLabel"] span,
            section[data-testid="stMain"] label {
                color: #1A2744 !important;
            }

            /* Checkbox text */
            section[data-testid="stMain"] [data-testid="stCheckbox"] span,
            section[data-testid="stMain"] [data-testid="stCheckbox"] p,
            section[data-testid="stMain"] [data-testid="stCheckbox"] label {
                color: #1A2744 !important;
            }

            /* Radio buttons (main page only — sidebar already white) */
            section[data-testid="stMain"] .stRadio label,
            section[data-testid="stMain"] .stRadio p,
            section[data-testid="stMain"] .stRadio span {
                color: #1A2744 !important;
            }

            /* Selectbox options text */
            section[data-testid="stMain"] .stSelectbox label,
            section[data-testid="stMain"] .stSelectbox p {
                color: #1A2744 !important;
            }
            section[data-testid="stMain"] div[data-baseweb="select"] span,
            section[data-testid="stMain"] div[data-baseweb="select"] div {
                color: #1A2744 !important;
            }

            /* Slider label and tick values */
            section[data-testid="stMain"] [data-testid="stSlider"] label,
            section[data-testid="stMain"] [data-testid="stSlider"] p,
            section[data-testid="stMain"] [data-testid="stSlider"] span {
                color: #1A2744 !important;
            }

            /* Number input label */
            section[data-testid="stMain"] [data-testid="stNumberInput"] label,
            section[data-testid="stMain"] [data-testid="stNumberInput"] p {
                color: #1A2744 !important;
            }

            /* Disabled / read-only inputs (e.g. username in Manage Profile) */
            section[data-testid="stMain"] input:disabled,
            section[data-testid="stMain"] textarea:disabled,
            section[data-testid="stMain"] .stTextInput input[disabled],
            section[data-testid="stMain"] .stTextInput input[readonly] {
                color: #1A2744 !important;
                -webkit-text-fill-color: #1A2744 !important;
                opacity: 1 !important;
                background-color: #F4F6FA !important;
                border-color: #DDE4EF !important;
            }

            /* Multiselect */
            section[data-testid="stMain"] [data-testid="stMultiSelect"] label,
            section[data-testid="stMain"] [data-testid="stMultiSelect"] p {
                color: #1A2744 !important;
            }
            section[data-testid="stMain"] [data-baseweb="tag"] span {
                color: #FFFFFF !important;
            }

            /* Date input */
            section[data-testid="stMain"] [data-testid="stDateInput"] label,
            section[data-testid="stMain"] [data-testid="stDateInput"] p {
                color: #1A2744 !important;
            }

            /* General markdown paragraphs in main content */
            section[data-testid="stMain"] .stMarkdown p,
            section[data-testid="stMain"] .stMarkdown span,
            section[data-testid="stMain"] .stMarkdown li {
                color: #1A2744;
            }

            /* Caption / help text */
            section[data-testid="stMain"] [data-testid="stCaptionContainer"] p,
            section[data-testid="stMain"] small {
                color: #5A7299 !important;
            }

            /* Info / warning / error text */
            section[data-testid="stMain"] [data-testid="stAlert"] p {
                color: #1A2744 !important;
            }

            /* Subheader and section headers */
            section[data-testid="stMain"] [data-testid="stHeadingWithActionElements"] h2,
            section[data-testid="stMain"] [data-testid="stHeadingWithActionElements"] h3 {
                color: #00205B !important;
            }

            /* ── Animations ── */
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(6px); }
                to   { opacity: 1; transform: translateY(0); }
            }
            @keyframes slideUp {
                from { opacity: 0; transform: translateY(14px); }
                to   { opacity: 1; transform: translateY(0); }
            }
            @keyframes pulseRed {
                0%   { box-shadow: 0 0 0 0 rgba(214,0,28,0.5); }
                70%  { box-shadow: 0 0 0 14px rgba(214,0,28,0); }
                100% { box-shadow: 0 0 0 0 rgba(214,0,28,0); }
            }
            @keyframes pulseGreen {
                0%   { box-shadow: 0 0 0 0 rgba(13,159,79,0.5); }
                70%  { box-shadow: 0 0 0 14px rgba(13,159,79,0); }
                100% { box-shadow: 0 0 0 0 rgba(13,159,79,0); }
            }

            section[data-testid="stMain"] > div:first-child {
                animation: fadeIn 0.3s ease-out;
            }
            [data-testid="stMetric"] {
                animation: slideUp 0.3s ease-out;
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }
            [data-testid="stMetric"]:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 16px rgba(0,32,91,0.10);
            }
            div.stButton > button {
                transition: all 0.18s ease !important;
            }
            div.stButton > button:hover {
                transform: translateY(-1px) !important;
            }
            [data-testid="stDataFrame"] { animation: fadeIn 0.35s ease-out; }

            .result-fail { animation: pulseRed  1s ease-out 2; }
            .result-pass { animation: pulseGreen 1s ease-out 1; }
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

if "session_passed" not in st.session_state:
    st.session_state.session_passed = 0

if "session_failed" not in st.session_state:
    st.session_state.session_failed = 0

if "show_session_summary" not in st.session_state:
    st.session_state.show_session_summary = False


# =========================================================
# 5. CACHED RESOURCES
# =========================================================

def get_available_ports():
    ports = serial.tools.list_ports.comports()
    return sorted([port.device for port in ports])


def get_default_port(ports):
    """Prefer the port whose description contains 'Arduino' or 'CH340',
    otherwise fall back to the last (highest-numbered) COM port."""
    all_ports = serial.tools.list_ports.comports()
    for p in all_ports:
        desc = (p.description or "").lower()
        if "arduino" in desc or "ch340" in desc or "ch341" in desc or "usb serial" in desc:
            return p.device
    return ports[-1] if ports else None


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
                is_fail  = row["status"] == "Fail"
                accent   = "#D6001C"  if is_fail else "#0D9F4F"
                bg       = "#FFF0F2"  if is_fail else "#EAF7EE"
                label    = "FAIL"     if is_fail else "PASS"
                lbl_bg   = "#D6001C"  if is_fail else "#0D9F4F"
                conf_str = f'{row["confidence_score"]:.2f}' if row.get("confidence_score") is not None else "—"
                t_stamp  = to_myt(row["timestamp"]).strftime("%H:%M:%S")

                st.markdown(
                    f"""
                    <div style="
                        display:flex;align-items:center;gap:12px;
                        padding:10px 14px;
                        border-left:4px solid {accent};
                        background:{bg};
                        margin-bottom:8px;
                        border-radius:0 8px 8px 0;
                    ">
                        <span style="background:{lbl_bg};color:#FFF;font-size:10px;
                            font-weight:800;padding:3px 8px;border-radius:4px;
                            letter-spacing:0.8px;white-space:nowrap;">{label}</span>
                        <span style="color:#1A2744;font-size:13px;flex:1;">
                            Conf: <strong>{conf_str}</strong>
                        </span>
                        <span style="color:#5A7299;font-size:12px;">🕒 {t_stamp}</span>
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
                color_discrete_map={"Pass": CHART_PASS_COLOR, "Fail": CHART_FAIL_COLOR},
                title="Pass vs Fail Count"
            )
            fig_bar.update_layout(**_chart_layout(xaxis_title="Status", yaxis_title="Count"))
            fig_bar.update_traces(textposition="outside",
                                  textfont=dict(color="#1A2744", size=12))
            chart_col1.plotly_chart(fig_bar, use_container_width=True)

            fig_pie = px.pie(
                status_counts,
                names="Status",
                values="Count",
                color="Status",
                color_discrete_map={"Pass": CHART_PASS_COLOR, "Fail": CHART_FAIL_COLOR},
                title="Pass vs Fail Distribution",
                hole=0.45
            )
            fig_pie.update_layout(**_chart_layout(showlegend=True,
                legend=dict(bgcolor="#FFFFFF", bordercolor="#DDE4EF",
                            font=dict(color="#1A2744"))))
            fig_pie.update_traces(textfont=dict(color="#FFFFFF", size=13))
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
        bg        = "#0D9F4F"
        border    = "#0A8040"
        icon      = "✅"
        title     = "PRODUCT PASSED"
        message   = f"Confidence: {confidence:.2f}" if confidence is not None else "No defect detected"
        css_class = "result-pass"
    else:
        bg        = "#D6001C"
        border    = "#B5001A"
        icon      = "❌"
        title     = "DEFECT DETECTED"
        message   = f"Defect confidence: {confidence:.2f}" if confidence is not None else "Defect detected"
        css_class = "result-fail"

    container.markdown(
        f"""
        <div class="{css_class}" style="
            background:{bg};
            border-bottom:4px solid {border};
            padding:14px 20px;
            border-radius:10px;
            color:#FFFFFF;
            display:flex;
            align-items:center;
            gap:14px;
            margin-bottom:10px;
            box-shadow:0 4px 14px rgba(0,0,0,0.18);
        ">
            <span style="font-size:28px;line-height:1;">{icon}</span>
            <div>
                <div style="font-size:16px;font-weight:800;letter-spacing:0.5px;">{title}</div>
                <div style="font-size:12px;opacity:0.88;margin-top:2px;">{message}</div>
            </div>
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
            st.markdown(
                """
                <div style="
                    text-align:center;
                    padding:28px 16px;
                    border:1.5px dashed #DDE4EF;
                    border-radius:10px;
                    background:#FFFFFF;
                    color:#5A7299;
                    animation:fadeIn 0.4s ease-out;
                ">
                    <div style="font-size:32px;">📈</div>
                    <div style="font-size:13px;font-weight:600;color:#00205B;margin-top:8px;">
                        Live Defect Trend
                    </div>
                    <div style="font-size:12px;color:#5A7299;margin-top:4px;">
                        Appears after 2+ inspections
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
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
            **_chart_layout(
                title=dict(text="Live Defect Rate Trend",
                           font=dict(color="#00205B", size=14)),
                xaxis_title="Inspection #",
                yaxis=dict(range=[0, 100], gridcolor="#EEF2F7",
                           tickfont=dict(color="#1A2744", size=12),
                           title_font=dict(color="#00205B", size=12)),
                xaxis=dict(gridcolor="#EEF2F7",
                           tickfont=dict(color="#1A2744", size=12),
                           title_font=dict(color="#00205B", size=12)),
                height=260,
                margin=dict(l=0, r=0, t=40, b=0),
            )
        )
        st.plotly_chart(fig, use_container_width=True)


# =========================================================
# 8. LOGIN SCREEN
# =========================================================

if not st.session_state.logged_in:
    _, center, _ = st.columns([1, 1.4, 1])
    with center:
        st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
        try:
            st.image(LOGO_PATH, use_container_width=True)
        except Exception:
            pass
        st.markdown(
            """
            <div style="
                background:#FFFFFF;
                border:1px solid #DDE4EF;
                border-top:4px solid #D6001C;
                border-radius:12px;
                padding:36px 40px 32px;
                box-shadow:0 4px 24px rgba(0,32,91,0.10);
                margin-top:24px;
            ">
                <h2 style="color:#00205B;font-weight:800;margin:0 0 4px;">Welcome Back</h2>
                <p style="color:#5A7299;font-size:14px;margin:0 0 28px;">
                    Sign in to AutoVision Defect System
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        if st.button("Sign In", use_container_width=True, type="primary"):
            login(username, password)
        st.markdown(
            "<p style='text-align:center;color:#5A7299;font-size:12px;margin-top:20px;'>"
            "DRB-HICOM Automated Defect Detection System</p>",
            unsafe_allow_html=True
        )

    st.stop()


# =========================================================
# 9. SIDEBAR AND NAVIGATION
# =========================================================

role = st.session_state.user.get("role", "operator")

# =========================================================
# SIDEBAR
# =========================================================

try:
    st.sidebar.image(LOGO_PATH, use_container_width=True)
except Exception:
    pass

# ── Navigation at the top ──
st.sidebar.markdown("---")

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

selected_page = st.sidebar.radio("📌 Navigation", menu_options)

# ── User info ──
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"<div style='color:#FFFFFF;font-size:15px;font-weight:700;'>👤 {st.session_state.user['username']}</div>"
    f"<div style='color:#A8C4E8;font-size:12px;margin-top:2px;'>Role: {st.session_state.user.get('role','Operator').upper()}</div>",
    unsafe_allow_html=True
)

# ── Arduino COM Port ──
st.sidebar.markdown("---")
available_ports = get_available_ports()

if available_ports:
    default_port = get_default_port(available_ports)
    default_index = available_ports.index(default_port) if default_port in available_ports else 0
    selected_port = st.sidebar.selectbox(
        "🔌 Arduino COM Port",
        available_ports,
        index=default_index
    )
else:
    selected_port = None
    st.sidebar.warning("No COM ports detected.")

if st.session_state.get("_last_arduino_port") != selected_port:
    connect_arduino.clear()
    st.session_state["_last_arduino_port"] = selected_port

arduino = connect_arduino(selected_port)

# ── System Health ──
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='color:#FFFFFF;font-size:13px;font-weight:700;letter-spacing:0.5px;'>🖥️ SYSTEM HEALTH</div>",
    unsafe_allow_html=True
)

cam_ok = st.session_state.get("camera_active", False)
arduino_ok = arduino is not None and arduino.is_open
model_ok = model is not None

_last_db_check = st.session_state.get("_last_db_check_time", 0)
if time.time() - _last_db_check > 30:
    try:
        supabase.table("inspections").select("id").limit(1).execute()
        st.session_state["_db_ok"] = True
    except Exception:
        st.session_state["_db_ok"] = False
    st.session_state["_last_db_check_time"] = time.time()
db_ok = st.session_state.get("_db_ok", True)

def _health_row(ok, label, on_text, off_text):
    text  = on_text if ok else off_text
    bg    = "rgba(13,159,79,0.18)"  if ok else "rgba(214,0,28,0.18)"
    dot   = "#4ADE80"               if ok else "#FF6B6B"
    badge = "rgba(13,159,79,0.30)"  if ok else "rgba(214,0,28,0.30)"
    return (
        f"<div style='display:flex;align-items:center;gap:10px;"
        f"background:{bg};border-radius:8px;"
        f"padding:8px 12px;margin-bottom:6px;'>"
        f"<span style='width:8px;height:8px;border-radius:50%;"
        f"background:{dot};display:inline-block;flex-shrink:0;'></span>"
        f"<span style='color:#FFFFFF;font-size:13px;flex:1;font-weight:500;'>{label}</span>"
        f"<span style='background:{badge};color:#FFFFFF;font-size:11px;"
        f"font-weight:700;padding:2px 8px;border-radius:20px;'>{text}</span>"
        f"</div>"
    )

st.sidebar.markdown(
    _health_row(cam_ok,    "Camera",   "Active",       "Inactive")    +
    _health_row(arduino_ok,"Arduino",  "Connected",    "Disconnected")+
    _health_row(db_ok,     "Database", "Online",       "Offline")     +
    _health_row(model_ok,  "AI Model", "Loaded",       "Not loaded"),
    unsafe_allow_html=True
)

# ── Log Out ──
st.sidebar.markdown("---")
if st.sidebar.button("🚪 Log Out", use_container_width=True):
    logout()

# page title tag
_page_titles = {
    "Defect Detection": "Defect Detection",
    "Inspection Logs": "Inspection Logs",
    "User Management (Admin)": "User Management",
    "Manage Profile": "My Profile",
}
_title = _page_titles.get(selected_page, selected_page)
st.markdown(
    f"<script>document.title = '{_title} | DRB-HICOM'</script>",
    unsafe_allow_html=True
)

fail_votes_required = 2


# =========================================================
# 10. DEFECT DETECTION PAGE
# =========================================================

if selected_page == "Defect Detection":
    st.title("🛡️ Defect Detection & Operator Dashboard")

    with st.expander("⚙️ Camera & Scan Settings", expanded=False):
        s1, s2, s3 = st.columns(3)
        camera_index = s1.number_input(
            "Camera Index",
            min_value=0, max_value=5, value=0, step=1,
            help="0 = first camera. Change to 1 or 2 if the wrong camera opens."
        )
        scan_delay = s2.slider(
            "Scan Delay (s)",
            min_value=0.5, max_value=4.0, value=1.5, step=0.1,
            help="Wait time after IR trigger before scanning."
        )
        conf_threshold_ui = s3.slider(
            "Defect Confidence Threshold",
            min_value=0.30, max_value=0.90, value=0.50, step=0.05,
            help="Minimum confidence to count a frame as defective. Lower = more sensitive."
        )

    conf_threshold = conf_threshold_ui

    def render_today_counter(container):
        passed, failed = fetch_today_counts()
        total = passed + failed
        container.empty()
        with container.container():
            m1, m2, m3 = st.columns(3)
            m1.metric("Today — Total", total)
            m2.metric("Passed", passed)
            m3.metric("Failed", failed)

    today_counter = st.empty()
    render_today_counter(today_counter)

    col_main, col_side = st.columns([2, 1])

    with col_main:
        st.subheader("Live Defect Detection")
        run_system = st.checkbox("Start Camera System", value=False)
        manual_scan = st.button("🔍 Manual Scan (Demo)", type="secondary", help="Trigger a scan manually without Arduino signal")
        result_card = st.empty()
        status_box = st.empty()
        frame_window = st.empty()

    with col_side:
        log_placeholder = st.empty()
        trend_placeholder = st.empty()

    update_log_display(log_placeholder)
    render_trend(trend_placeholder)

    if run_system:
        cap = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW)

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
            st.session_state.session_passed = 0
            st.session_state.session_failed = 0
            st.session_state.show_session_summary = False
            st.session_state["_manual_scan_consumed"] = False
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

            try:
                arduino_has_data = arduino is not None and arduino.in_waiting > 0
            except Exception:
                arduino_has_data = False

            trigger_scan = manual_scan and not st.session_state.get("_manual_scan_consumed", False)
            if trigger_scan and manual_scan:
                st.session_state["_manual_scan_consumed"] = True

            if arduino_has_data:
                try:
                    raw_msg = arduino.read(arduino.in_waiting)
                    arduino_msg = raw_msg.decode("utf-8", errors="ignore").strip()

                    if arduino_msg:
                        print(f"📥 Received from Arduino: '{arduino_msg}'")
                        status_box.caption(f"Arduino: {arduino_msg}")

                    if "SCAN" in arduino_msg.upper():
                        trigger_scan = True

                except Exception as e:
                    print(f"❌ Serial read error: {e}")

            if trigger_scan:
                try:
                    print("🎯 SCAN triggered.")
                    label = "Manual scan triggered." if manual_scan else "SCAN received. Capturing inspection image..."
                    status_box.warning(label)

                    time.sleep(scan_delay)

                    # Flush stale frames from buffer after the delay
                    flush_count = max(10, int(scan_delay * 10))
                    for _ in range(flush_count):
                        cap.read()

                    # =================================================
                    # MULTI-FRAME SCAN WITH MAJORITY VOTING
                    # Scans 7 frames; the majority decision wins so a
                    # single blurry/partial frame cannot force a Fail.
                    # =================================================

                    SCAN_FRAMES = 7
                    votes = []   # (frame_status, defect_conf, result_obj)
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
                            print(f"🔍 Model classes: {cls_names}")

                        defect_idx = next(
                            (k for k, v in cls_names.items() if v.lower() == "defect"),
                            None
                        )

                        all_confs = {cls_names[k]: round(float(result[0].probs.data[k]), 3) for k in cls_names}
                        print(f"  Frame {len(votes)+1}: classes={all_confs}, defect_idx={defect_idx}")

                        if defect_idx is not None:
                            frame_conf = float(result[0].probs.data[defect_idx])
                            frame_status = "Fail" if frame_conf >= conf_threshold else "Pass"
                        else:
                            frame_conf = 0.0
                            frame_status = "Pass"

                        votes.append((frame_status, frame_conf, result))

                    if not votes:
                        status_box.error("Failed to capture inspection image.")
                    else:
                        # =================================================
                        # RESULT DECISION — majority vote
                        # =================================================

                        fail_votes = [v for v in votes if v[0] == "Fail"]
                        pass_votes = [v for v in votes if v[0] == "Pass"]

                        if len(fail_votes) >= fail_votes_required:
                            status = "Fail"
                            command = b"0"
                            # Display the frame with the highest defect confidence
                            best_vote = max(fail_votes, key=lambda v: v[1])
                        else:
                            status = "Pass"
                            command = b"1"
                            # Display the frame with the lowest defect confidence (clearest pass)
                            best_vote = min(pass_votes, key=lambda v: v[1]) if pass_votes else votes[0]

                        conf = best_vote[1]
                        best_result = best_vote[2]
                        vote_summary = f"{len(fail_votes)}F / {len(pass_votes)}P of {len(votes)} frames"

                        if status == "Fail":
                            st.toast(f"❌ Defect Detected! ({conf:.2f}) [{vote_summary}]")
                            status_box.error(
                                f"Defect detected. Confidence: {conf:.2f} | Votes: {vote_summary}"
                            )
                        else:
                            st.toast(f"✅ Product Passed ({vote_summary})")
                            status_box.success(f"Product passed inspection. | Votes: {vote_summary}")

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
                        # MOTOR COMMAND THEN SUPABASE SAVE
                        # =================================================

                        print("💾 Saving inspection result to Supabase...")
                        if arduino is not None and arduino.is_open:
                            print(f"📤 Sending {'1' if status == 'Pass' else '0'} to Arduino")
                            arduino.write(command)
                            arduino.flush()
                            status_box.info("Motor command sent. Saving to database...")
                        else:
                            status_box.warning("Arduino not connected. Motor command skipped.")

                        save_success = log_inspection(status, conf)

                        if save_success:
                            print("✅ Data saved successfully.")

                            if status == "Pass":
                                st.session_state.session_passed += 1
                            else:
                                st.session_state.session_failed += 1

                            st.session_state.trend_data.append({"status": status})
                            if len(st.session_state.trend_data) > 50:
                                st.session_state.trend_data = st.session_state.trend_data[-50:]

                            render_today_counter(today_counter)
                            update_log_display(log_placeholder)
                            render_trend(trend_placeholder)
                            status_box.info("Result saved. Inspection complete.")

                        else:
                            print("❌ Supabase save failed.")
                            status_box.error(
                                "Motor command sent but database save failed. Check connection."
                            )

                    time.sleep(0.5)
                    st.session_state["_manual_scan_consumed"] = False
                    if arduino is not None:
                        arduino.reset_input_buffer()

                except Exception as e:
                    print(f"❌ Scan error: {e}")
                    status_box.error(f"Scan error: {e}")

        finally:
            cap.release()
            st.session_state.camera_active = False
            total_session = st.session_state.session_passed + st.session_state.session_failed
            if total_session > 0:
                st.session_state.show_session_summary = True

    if st.session_state.get("show_session_summary", False) and not run_system and (st.session_state.session_passed + st.session_state.session_failed) > 0:
        sp = st.session_state.session_passed
        sf = st.session_state.session_failed
        st_total = sp + sf
        rate = (sf / st_total * 100) if st_total > 0 else 0.0
        st.markdown(
            f"""
            <div style="
                background:#FFFFFF;
                border:1px solid #DDE4EF;
                border-top:4px solid #00205B;
                border-radius:10px;
                padding:20px 24px;
                margin-top:16px;
                animation:fadeIn 0.5s ease-out;
                box-shadow:0 2px 10px rgba(0,32,91,0.07);
            ">
                <div style="font-size:13px;font-weight:700;color:#5A7299;
                    text-transform:uppercase;letter-spacing:0.8px;margin-bottom:16px;">
                    📊 Session Summary
                </div>
                <div style="display:flex;gap:0;flex-wrap:wrap;">
                    <div style="flex:1;min-width:80px;border-right:1px solid #EEF2F7;padding:0 20px 0 0;margin-right:20px;">
                        <div style="font-size:24px;font-weight:800;color:#00205B;">{st_total}</div>
                        <div style="font-size:11px;color:#5A7299;text-transform:uppercase;letter-spacing:0.5px;">Total</div>
                    </div>
                    <div style="flex:1;min-width:80px;border-right:1px solid #EEF2F7;padding:0 20px 0 0;margin-right:20px;">
                        <div style="font-size:24px;font-weight:800;color:#0D9F4F;">{sp}</div>
                        <div style="font-size:11px;color:#5A7299;text-transform:uppercase;letter-spacing:0.5px;">Passed</div>
                    </div>
                    <div style="flex:1;min-width:80px;border-right:1px solid #EEF2F7;padding:0 20px 0 0;margin-right:20px;">
                        <div style="font-size:24px;font-weight:800;color:#D6001C;">{sf}</div>
                        <div style="font-size:11px;color:#5A7299;text-transform:uppercase;letter-spacing:0.5px;">Failed</div>
                    </div>
                    <div style="flex:1;min-width:80px;">
                        <div style="font-size:24px;font-weight:800;color:#1A2744;">{rate:.1f}%</div>
                        <div style="font-size:11px;color:#5A7299;text-transform:uppercase;letter-spacing:0.5px;">Fail Rate</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# 12. INSPECTION LOGS PAGE WITH ANALYTICS — ADMIN ONLY
# =========================================================

elif selected_page == "Inspection Logs":
    title_col, btn_col = st.columns([5, 1])
    title_col.title("📋 Inspection History & Analytics")
    if btn_col.button("🔄 Refresh", use_container_width=True):
        st.rerun()

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
                x="Status", y="Count", color="Status", text="Count",
                color_discrete_map={"Pass": CHART_PASS_COLOR, "Fail": CHART_FAIL_COLOR},
                title="Pass vs Fail Count"
            )
            fig_bar.update_layout(**_chart_layout(xaxis_title="Status", yaxis_title="Count"))
            fig_bar.update_traces(textposition="outside",
                                  textfont=dict(color="#1A2744", size=12))
            chart_col1.plotly_chart(fig_bar, use_container_width=True)

            fig_pie = px.pie(
                status_counts,
                names="Status", values="Count", color="Status",
                color_discrete_map={"Pass": CHART_PASS_COLOR, "Fail": CHART_FAIL_COLOR},
                title="Pass vs Fail Distribution",
                hole=0.45
            )
            fig_pie.update_layout(**_chart_layout(showlegend=True,
                legend=dict(bgcolor="#FFFFFF", bordercolor="#DDE4EF",
                            font=dict(color="#1A2744"))))
            fig_pie.update_traces(textfont=dict(color="#FFFFFF", size=13))
            chart_col2.plotly_chart(fig_pie, use_container_width=True)

            st.divider()

            st.subheader("🧾 Filtered Inspection Records")

            cols_to_keep = [
                "id",
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
                    return "color: #D6001C; font-weight: 700"
                if val == "Pass":
                    return "color: #0D9F4F; font-weight: 700"
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
