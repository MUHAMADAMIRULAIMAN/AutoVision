import os

# --- FIX FOR GITHUB TIMEOUT ERROR ---
os.environ["ULTRALYTICS_NO_AUTOUPDATES"] = "true"

import streamlit as st
import cv2
import time
import pandas as pd
import numpy as np
from ultralytics import YOLO
from supabase import create_client
from datetime import datetime
from PIL import Image
import serial
import plotly.express as px


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

LOGO_PATH = "DRB_HICOM_Logo.png"


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
        </style>
        """,
        unsafe_allow_html=True
    )


apply_custom_styles()


# =========================================================
# 3. SUPABASE CONFIGURATION
# =========================================================
# IMPORTANT:
# For FYP demo this can work, but for safety you should later move
# these credentials into Streamlit secrets or environment variables.

SUPABASE_URL = "https://easbotszklartfkakkxm.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVhc2JvdHN6a2xhcnRma2Fra3htIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Njc1MzI5NzgsImV4cCI6MjA4MzEwODk3OH0.O9tQ6ZR56UTsRqmXfOrYtCBPmAyQKtjh3_quBKXHYfE"


# =========================================================
# 4. SESSION STATE
# =========================================================

if "user" not in st.session_state:
    st.session_state.user = None

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


# =========================================================
# 5. CACHED RESOURCES
# =========================================================

@st.cache_resource
def connect_arduino():
    try:
        ser = serial.Serial("COM7", 9600, timeout=1)
        time.sleep(2)

        ser.reset_input_buffer()
        ser.reset_output_buffer()

        print("✅ Arduino connected successfully.")
        return ser

    except Exception as e:
        print(f"❌ Arduino connection failed: {e}")
        return None


@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)


@st.cache_resource
def load_model():
    return YOLO("bestv4.pt")


arduino = connect_arduino()

try:
    supabase = init_supabase()
    model = load_model()

except Exception as e:
    st.error(f"System Initialization Error: {e}")
    st.stop()


# =========================================================
# 6. AUTHENTICATION FUNCTIONS
# =========================================================

def login(username, password):
    try:
        response = (
            supabase
            .table("users")
            .select("*")
            .eq("username", username)
            .eq("password", password)
            .execute()
        )

        if len(response.data) > 0:
            st.session_state.user = response.data[0]
            st.session_state.logged_in = True
            st.success("Login Successful!")
            st.rerun()
        else:
            st.error("Invalid Username or Password")

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
    """
    Saves inspection result to Supabase.
    Returns True only when save is successful.
    """
    try:
        data = {
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "confidence_score": float(confidence),
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


def update_log_display(container):
    try:
        response = (
            supabase
            .table("inspections")
            .select("*")
            .order("timestamp", desc=True)
            .limit(5)
            .execute()
        )

        container.empty()

        with container.container():
            st.subheader("Recent Logs")

            if response.data:
                for row in response.data:
                    color = "#D6001C" if row["status"] == "Fail" else "#21c354"
                    t_stamp = pd.to_datetime(row["timestamp"]).strftime("%H:%M:%S")

                    st.markdown(
                        f"""
                        <div style="
                            padding:10px;
                            border-left: 5px solid {color};
                            background-color: #1C1E26;
                            margin-bottom: 10px;
                            border-radius: 5px;
                        ">
                            <strong>{row["status"]}</strong> | Conf: {row.get("confidence_score", 0):.2f}<br>
                            <small style="color: #bbb;">🕒 {t_stamp}</small>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.info("No logs found yet.")

    except Exception as e:
        container.error(f"Sync Error: {e}")


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

menu_options = [
    "Defect Detection",
    "Inspection Logs",
    "Manage Profile"
]

if st.session_state.user.get("role") == "admin":
    menu_options.insert(2, "User Management (Admin)")

selected_page = st.sidebar.radio("Navigation", menu_options)

if st.sidebar.button("Log Out"):
    logout()


# =========================================================
# 10. DEFECT DETECTION PAGE
# =========================================================

if selected_page == "Defect Detection":
    st.title("🛡️ Defect Detection Interface")

    conf_threshold = 0.15

    input_source = st.radio(
        "Select Input Source:",
        ["Webcam", "Upload Image"],
        horizontal=True
    )

    st.divider()

    col_main, col_logs = st.columns([2, 1])

    # -----------------------------------------------------
    # WEBCAM MODE
    # -----------------------------------------------------
    if input_source == "Webcam":
        with col_main:
            st.subheader("Live Feed")
            run_system = st.checkbox("Start Camera System", value=False)
            frame_window = st.empty()
            status_box = st.empty()

        log_placeholder = col_logs.empty()
        update_log_display(log_placeholder)

        if run_system:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

            # Camera settings
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)

            if not cap.isOpened():
                st.error("Camera could not be opened.")
            else:
                status_box.info("System running. Waiting for Arduino SCAN signal...")

            while cap.isOpened() and run_system:
                ret, frame = cap.read()

                if not ret:
                    st.error("Camera error.")
                    break

                # Prevent UI lag
                time.sleep(0.1)

                # Continuous preview detection
                results = model.predict(
                    frame,
                    conf=conf_threshold,
                    verbose=False
                )

                annotated_frame = results[0].plot()
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                frame_window.image(
                    frame_rgb,
                    channels="RGB",
                    width=900
                )

                # -----------------------------------------------------
                # HARDWARE COMMUNICATION
                # -----------------------------------------------------
                if arduino is not None and arduino.in_waiting > 0:
                    try:
                        while arduino.in_waiting > 0:
                            raw_msg = arduino.readline()
                            arduino_msg = raw_msg.decode(
                                "utf-8",
                                errors="ignore"
                            ).strip()

                            if arduino_msg != "":
                                print(f"📥 Received from Arduino: '{arduino_msg}'")

                            if "SCAN" in arduino_msg:
                                print("🎯 SCAN command detected.")
                                status_box.warning("SCAN received. Capturing inspection image...")

                                # Wait for product and belt vibration to settle
                                time.sleep(0.8)

                                # Flush camera buffer to remove blurry frames
                                for _ in range(10):
                                    cap.read()

                                # Capture fresh inspection image
                                ret, fresh_frame = cap.read()

                                if not ret:
                                    st.error("Failed to capture inspection image.")
                                    break

                                # Strict AI detection
                                strict_conf = min(0.20, conf_threshold)

                                final_results = model.predict(
                                    fresh_frame,
                                    conf=strict_conf,
                                    verbose=False
                                )

                                inspected_frame = final_results[0].plot()
                                inspected_rgb = cv2.cvtColor(
                                    inspected_frame,
                                    cv2.COLOR_BGR2RGB
                                )

                                frame_window.image(
                                    inspected_rgb,
                                    channels="RGB",
                                    width=900
                                )

                                # -----------------------------------------------------
                                # RESULT DECISION
                                # -----------------------------------------------------
                                if len(final_results[0].boxes) > 0:
                                    box = final_results[0].boxes[0]
                                    conf = float(box.conf[0])
                                    status = "Fail"
                                    command = b"0"

                                    st.toast(f"❌ Defect Detected! ({conf:.2f})")
                                    status_box.error(
                                        f"Defect detected. Confidence: {conf:.2f}"
                                    )

                                else:
                                    conf = 1.0
                                    status = "Pass"
                                    command = b"1"

                                    st.toast("✅ Product Passed Inspection")
                                    status_box.success("Product passed inspection.")

                                # -----------------------------------------------------
                                # IMPORTANT FIX:
                                # SAVE TO SUPABASE FIRST, THEN MOVE MOTOR
                                # -----------------------------------------------------
                                print("💾 Saving inspection result to Supabase...")
                                save_success = log_inspection(status, conf)

                                if save_success:
                                    print("✅ Data saved successfully.")
                                    update_log_display(log_placeholder)

                                    # Send command to Arduino only after successful save
                                    if arduino is not None:
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

                                break

                    except Exception as e:
                        print(f"❌ Serial communication error: {e}")
                        status_box.error(f"Serial communication error: {e}")

            cap.release()

    # -----------------------------------------------------
    # UPLOAD IMAGE MODE
    # -----------------------------------------------------
    elif input_source == "Upload Image":
        with col_main:
            st.subheader("Static Image Analysis")

            uploaded_file = st.file_uploader(
                "Upload image...",
                type=["jpg", "png", "jpeg"]
            )

            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                img_array = np.array(image)

                results = model.predict(
                    img_array,
                    conf=conf_threshold,
                    verbose=False
                )

                st.image(
                    results[0].plot(),
                    caption="Analyzed Image",
                    use_container_width=True
                )

                if len(results[0].boxes) > 0:
                    box = results[0].boxes[0]
                    conf = float(box.conf[0])
                    status = "Fail"
                    st.error(f"Defect detected. Confidence: {conf:.2f}")
                else:
                    conf = 1.0
                    status = "Pass"
                    st.success("No defect detected. Product passed.")

                if st.button("Save Static Image Result", type="primary"):
                    save_success = log_inspection(status, conf)

                    if save_success:
                        st.success("Static image result saved successfully.")
                    else:
                        st.error("Failed to save static image result.")

        log_placeholder = col_logs.empty()
        update_log_display(log_placeholder)


# =========================================================
# 11. INSPECTION LOGS PAGE WITH ANALYTICS
# =========================================================

elif selected_page == "Inspection Logs":
    st.title("📋 Inspection History & Analytics")

    try:
        logs_res = (
            supabase
            .table("inspections")
            .select("*")
            .order("timestamp", desc=True)
            .execute()
        )

        df_logs = pd.DataFrame(logs_res.data)

        users_res = (
            supabase
            .table("users")
            .select("id, username")
            .execute()
        )

        if not df_logs.empty:
            df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"])

            # =========================================================
            # DATE RANGE FILTER
            # =========================================================

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

            # =========================================================
            # USERNAME MAPPING
            # =========================================================

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

            # =========================================================
            # SUMMARY METRICS
            # =========================================================

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
            metric_col5.metric("Avg Confidence", f"{avg_confidence:.2f}")

            st.divider()

            # =========================================================
            # GRAPHS
            # =========================================================

            st.subheader("📈 Inspection Analytics")

            chart_col1, chart_col2 = st.columns(2)

            # -----------------------------
            # PASS / FAIL BAR CHART
            # -----------------------------
            status_counts = (
                df_logs["status"]
                .value_counts()
                .reset_index()
            )

            status_counts.columns = ["Status", "Count"]

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
                font_color="white"
            )

            fig_bar.update_traces(
                textposition="outside"
            )

            chart_col1.plotly_chart(fig_bar, use_container_width=True)

            # -----------------------------
            # PASS / FAIL PIE CHART
            # -----------------------------
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

            # -----------------------------
            # DAILY PASS / FAIL TREND
            # -----------------------------
            df_logs["date"] = df_logs["timestamp"].dt.date

            daily_status = (
                df_logs
                .groupby(["date", "status"])
                .size()
                .reset_index(name="Count")
            )

            fig_daily = px.line(
                daily_status,
                x="date",
                y="Count",
                color="status",
                markers=True,
                color_discrete_map={
                    "Pass": "#21c354",
                    "Fail": "#D6001C"
                },
                title="Daily Pass/Fail Trend"
            )

            fig_daily.update_layout(
                plot_bgcolor="#0E1117",
                paper_bgcolor="#0E1117",
                font_color="white",
                xaxis_title="Date",
                yaxis_title="Number of Inspections"
            )

            st.plotly_chart(fig_daily, use_container_width=True)

            # -----------------------------
            # CONFIDENCE SCORE DISTRIBUTION
            # -----------------------------
            fig_conf = px.histogram(
                df_logs,
                x="confidence_score",
                color="status",
                nbins=10,
                color_discrete_map={
                    "Pass": "#21c354",
                    "Fail": "#D6001C"
                },
                title="Confidence Score Distribution"
            )

            fig_conf.update_layout(
                plot_bgcolor="#0E1117",
                paper_bgcolor="#0E1117",
                font_color="white",
                xaxis_title="Confidence Score",
                yaxis_title="Frequency"
            )

            st.plotly_chart(fig_conf, use_container_width=True)

            st.divider()

            # =========================================================
            # FILTERED TABLE
            # =========================================================

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
                .dt.strftime("%Y-%m-%d %H:%M:%S")
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

            # =========================================================
            # DOWNLOAD BUTTON
            # =========================================================

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
                                "password": new_pass,
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

                if c5.button(
                    "🗑️",
                    key=f"del_user_{user['id']}",
                    help="Delete this user"
                ):
                    if user["id"] == st.session_state.user["id"]:
                        st.error("You cannot delete your own account!")
                    else:
                        try:
                            supabase.table("inspections").delete().eq(
                                "operator_id",
                                user["id"]
                            ).execute()

                            supabase.table("users").delete().eq(
                                "id",
                                user["id"]
                            ).execute()

                            st.toast(
                                f"🗑️ Deleted user and their logs: {user['username']}"
                            )

                            time.sleep(0.5)
                            st.rerun()

                        except Exception as e:
                            st.error(f"Failed to delete: {e}")

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

            elif current_pass != user["password"]:
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
                        update_data["password"] = new_pass

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
                            st.session_state.user["password"] = new_pass

                        st.success("✅ Profile updated successfully!")
                        time.sleep(1)
                        st.rerun()

                    else:
                        st.info("No changes detected.")

                except Exception as e:
                    st.error(f"Failed to update profile: {e}")
