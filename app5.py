import streamlit as st
import datetime
from plyer import notification

# ML libraries
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import plotly.express as px

# -----------------------------
# Load pretrained models
# -----------------------------
model_bolus = None
model_gap = None
try:
    model_bolus = xgb.XGBRegressor()
    model_bolus.load_model("model_bolus.json")

    model_gap = xgb.XGBRegressor()
    model_gap.load_model("model_gap.json")
    MODELS_AVAILABLE = True
except Exception:
    MODELS_AVAILABLE = False

# Try to load recursive model (if available)
try:
    recursive_model = pickle.load(open("recursive_model.pkl", "rb"))
    RECURSIVE_MODEL_AVAILABLE = True
except Exception:
    RECURSIVE_MODEL_AVAILABLE = False

# ==============================
# ⚠️ Disclaimer
# ==============================
st.set_page_config(page_title="Insulin Reminder & Predictor", page_icon="💉", layout="wide")
st.warning("⚠️ This app is a helper tool. It does not replace medical advice.")

# -----------------------------
# Mock Database
# -----------------------------
if "db" not in st.session_state:
    st.session_state.db = {"users": {}, "injections": {}}
db = st.session_state.db

if "settings" not in st.session_state:
    st.session_state.settings = {
        "low": 6, "normal": 5, "slightly_high": 4,
        "high": 3, "very_high": 2, "night_buffer": 1,
        "time_display": "both"
    }

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.page = "Home"

# -----------------------------
# Utility Functions
# -----------------------------
def recommend_dosage(sugar_level: int) -> int:
    if sugar_level < 70: return 0
    elif 70 <= sugar_level <= 130: return 2
    elif 131 <= sugar_level <= 180: return 4
    elif 181 <= sugar_level <= 250: return 6
    elif 251 <= sugar_level <= 300: return 8
    else: return 10

def calculate_next_due(sugar_level: int, time_of_day: str) -> str:
    s = st.session_state.settings
    if sugar_level < 70: hours = s["low"]
    elif sugar_level <= 130: hours = s["normal"]
    elif sugar_level <= 180: hours = s["slightly_high"]
    elif sugar_level <= 250: hours = s["high"]
    else: hours = s["very_high"]
    if time_of_day == "Night": hours += s["night_buffer"]
    next_due_dt = datetime.datetime.now() + datetime.timedelta(hours=hours)
    return next_due_dt.strftime("%Y-%m-%d %I:%M %p")

def calculate_next_due_from_gap(predicted_gap_hours: float) -> str:
    next_due_dt = datetime.datetime.now() + datetime.timedelta(hours=float(predicted_gap_hours))
    return next_due_dt.strftime("%Y-%m-%d %I:%M %p")

def parse_next_due(next_due_str: str):
    return datetime.datetime.strptime(next_due_str, "%Y-%m-%d %I:%M %p")

def time_remaining_str(next_due_str: str) -> str:
    s = st.session_state.settings
    try:
        next_due = parse_next_due(next_due_str)
        now = datetime.datetime.now()
        diff = next_due - now
        mins = int(diff.total_seconds() // 60)
        if mins <= 0: return "(Overdue)"
        hrs, m = divmod(mins, 60)
        verbose = f"{hrs}h {m}m" if hrs else f"{m}m"
        decimal = f"{round(mins/60,1)}h"
        if s["time_display"] == "verbose": return f"({verbose})"
        if s["time_display"] == "decimal": return f"({decimal})"
        return f"({decimal} | {verbose})"
    except: return ""

def current_time_12h(): return datetime.datetime.now().strftime("%I:%M %p")

def get_time_of_day_from_now():
    hr = datetime.datetime.now().hour
    if 5 <= hr < 12: return "Morning"
    elif 12 <= hr < 17: return "Afternoon"
    elif 17 <= hr < 21: return "Evening"
    else: return "Night"

# -----------------------------
# LOGIN PAGE
# -----------------------------
if not st.session_state.logged_in:
    st.title("💉 Insulin Reminder & Predictor")

    username = st.text_input("👤 Username")
    password = st.text_input("🔑 Password", type="password")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login"):
            if username in db["users"] and db["users"][username] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("✅ Login successful")
            else:
                st.error("❌ Invalid username or password")
    with col2:
        if st.button("Register"):
            if username in db["users"]:
                st.error("⚠️ Username already exists")
            else:
                db["users"][username] = password
                db["injections"][username] = []
                st.success("✅ Registered successfully")
else:
    # -----------------------------
    # SIDEBAR NAV
    # -----------------------------
    username = st.session_state.username
    st.sidebar.title(f"👋 {username}")
    nav = ["Home","Log Injection","History","Predictor","Settings","Logout"]
    choice = st.sidebar.radio("Navigation", nav)
    st.session_state.page = choice

    # -----------------------------
    # HOME
    # -----------------------------
    if choice == "Home":
        st.title("🏠 Dashboard")
        st.metric("Current Time", current_time_12h())
        total_logs = len(db["injections"].get(username, []))
        st.metric("Total Logs", total_logs)

        # -----------------------------
        # Insulin Dosage Trend Graph (New)
        # -----------------------------
        user_logs = db["injections"].get(username, [])
        if user_logs:
            logs_df = pd.DataFrame(user_logs)
            logs_df["datetime"] = pd.to_datetime(logs_df["date"] + " " + logs_df["time"])

            dosage_fig = px.line(
                logs_df,
                x="datetime",
                y="recommended_dosage",
                title="💉 Recommended Insulin Dosage Trend",
                labels={"recommended_dosage":"Units","datetime":"Date & Time"},
                markers=True
            )
            st.plotly_chart(dosage_fig, use_container_width=True)

    # -----------------------------
    # LOG INJECTION
    # -----------------------------
    elif choice == "Log Injection":
        st.title("📝 Log New Injection")
        sugar = st.number_input("Blood sugar (mg/dL)", 40, 600, 120)
        tod = st.selectbox(
            "Time of Day",
            ["Morning", "Afternoon", "Evening", "Night"],
            index=["Morning", "Afternoon", "Evening", "Night"].index(get_time_of_day_from_now())
        )
        
        if st.button("Log Injection"):
            dosage = recommend_dosage(sugar)
            next_due = calculate_next_due(sugar, tod)
            
            entry = {
                "date": datetime.datetime.now().strftime("%Y-%m-%d"),
                "time": current_time_12h(),
                "sugar_level": sugar,
                "recommended_dosage": dosage,
                "next_due": next_due
            }
            db["injections"][username].append(entry)
            st.success(f"✅ Logged! Dose {dosage} units. Next due: {next_due} {time_remaining_str(next_due)}")
            
            # Desktop Notification
            try:
                notification.notify(
                    title="💉 Insulin Logged",
                    message=f"Hi {username}, {dosage} units of insulin have been recorded at {current_time_12h()}",
                    timeout=5
                )
            except Exception as e:
                st.warning(f"Notification failed: {e}")

    # -----------------------------
    # HISTORY
    # -----------------------------
    elif choice == "History":
        st.title("📜 Injection History")
        history = db["injections"].get(username, [])
        if not history:
            st.info("No logs yet.")
        else:
            for entry in reversed(history):
                line = f"📅 {entry['date']} ⏰ {entry['time']} | 🩸 {entry['sugar_level']} mg/dL | 💉 {entry['recommended_dosage']} u"
                overdue = ""
                if entry.get("next_due"):
                    try:
                        next_due = parse_next_due(entry["next_due"])
                        if datetime.datetime.now() > next_due:
                            overdue = f" | ⏳ **Overdue since {entry['next_due']}**"
                        else:
                            overdue = f" | ⏳ Next: {entry['next_due']} {time_remaining_str(entry['next_due'])}"
                    except: pass
                st.write(line + overdue)

    # -----------------------------
    # PREDICTOR
    # -----------------------------
    elif choice == "Predictor":
        st.title("🔮 Insulin Predictor")
        if not MODELS_AVAILABLE:
            st.error("❌ Models not found. Please run train_model.py first.")
        else:
            st.subheader("Baseline Inputs")
            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.number_input("Age",1,100,35)
                bmi = st.number_input("BMI",10.0,50.0,22.5)
            with col2:
                glucose = st.number_input("Glucose (mg/dL)",50,400,120)
                basal = st.number_input("Basal Insulin (u)",0,50,20)
            with col3:
                meal = st.number_input("Meal Bolus (u)",0,20,5)
                corr = st.number_input("Correction Bolus (u)",0,20,2)

            user_df = pd.DataFrame([{
                "Age": age, "BMI": bmi, "Glucose": glucose,
                "Basal Insulin (Units)": basal,
                "Meal Bolus (Units)": meal,
                "Correction Bolus (Units)": corr
            }])

            required_cols = [
                'Age', 'BMI', 'Glucose', 'pred_type_enc',
                'Basal Insulin (Units)', 'Meal Bolus (Units)',
                'Correction Bolus (Units)', 'Total Bolus (Units)',
                'hour', 'dayofweek'
            ]

            if 'pred_type_enc' not in user_df.columns:
                user_df['pred_type_enc'] = 0

            if 'Total Bolus (Units)' not in user_df.columns:
                user_df['Total Bolus (Units)'] = (
                    user_df.get('Meal Bolus (Units)', 0) + user_df.get('Correction Bolus (Units)', 0)
                )

            now = datetime.datetime.now()
            if 'hour' not in user_df.columns:
                user_df['hour'] = now.hour
            if 'dayofweek' not in user_df.columns:
                user_df['dayofweek'] = now.weekday()

            user_df = user_df[required_cols]

            # Initialize actual injection time in session state only once
            if 'actual_injection_time' not in st.session_state:
                st.session_state.actual_injection_time = datetime.datetime.now().time()

            st.time_input(
                "Enter Actual Injection Time",
                value=st.session_state.actual_injection_time,
                key='actual_injection_time'
            )
            actual_time = st.session_state.actual_injection_time

            if st.button("Predict"):
                bolus_pred = model_bolus.predict(user_df)[0]
                gap_pred = model_gap.predict(user_df)[0]
                bolus_pred = max(0, bolus_pred)
                gap_pred = max(0.1, gap_pred)
                next_due = calculate_next_due_from_gap(gap_pred)

                tab1, tab2 = st.tabs(["📊 Predictions", "🚨 Safety Alerts"])

                with tab1:
                    st.success("✅ Predictions Complete")
                    st.write(f"**Predicted Bolus:** {bolus_pred:.2f} u")
                    st.write(f"**Predicted Gap:** {gap_pred:.2f} hrs")
                    st.write(f"**Next Shot At:** {next_due} {time_remaining_str(next_due)}")

                with tab2:
                    st.subheader("Injection Timing Safety Check")
                    predicted_time = parse_next_due(next_due)
                    actual_datetime = datetime.datetime.combine(datetime.datetime.now().date(), actual_time)
                    diff = (actual_datetime - predicted_time).total_seconds() / 60

                    if diff < 0:
                        st.error(f"🔴 Overdose Risk! Taken {-int(diff)} minutes earlier than recommended.")
                    elif diff > 0:
                        st.warning(f"🟠 Underdose Risk! Taken {int(diff)} minutes later than recommended.")
                    else:
                        st.success("🟢 Safe: Taken exactly at recommended time.")

                    # Safety Timing Bar Graph
                    safety_df = pd.DataFrame({
                        "Type": ["Early", "Late", "On-time"],
                        "Minutes": [
                            max(0, -diff) if diff < 0 else 0,
                            max(0, diff) if diff > 0 else 0,
                            0 if diff != 0 else 1
                        ]
                    })

                    fig_safety = px.bar(
                        safety_df,
                        x="Type",
                        y="Minutes",
                        title="⏱ Injection Timing Safety",
                        text_auto=True,
                        color="Type",
                        color_discrete_map={"Early":"red", "Late":"orange", "On-time":"green"}
                    )
                    st.plotly_chart(fig_safety, use_container_width=True)

                    # Desktop Notification for Predictor
                    try:
                        notification.notify(
                            title="💉 Injection Safety Check",
                            message=f"Actual injection time logged. Safety status updated.",
                            timeout=5
                        )
                    except:
                        pass

    # -----------------------------
    # SETTINGS
    # -----------------------------
    elif choice == "Settings":
        st.title("⚙️ Settings")
        s = st.session_state.settings
        s["low"] = st.number_input("Low sugar (<70)",1,12,s["low"])
        s["normal"] = st.number_input("Normal (70–130)",1,12,s["normal"])
        s["slightly_high"] = st.number_input("Slightly high (131–180)",1,12,s["slightly_high"])
        s["high"] = st.number_input("High (181–250)",1,12,s["high"])
        s["very_high"] = st.number_input("Very high (>250)",1,12,s["very_high"])
        s["night_buffer"] = st.number_input("Night buffer (+hrs)",0,4,s["night_buffer"])
        s["time_display"] = st.selectbox("Display Style",["verbose","decimal","both"],
                                         index=["verbose","decimal","both"].index(s["time_display"]))
        st.success("✅ Settings saved")

    # -----------------------------
    # LOGOUT
    # -----------------------------
    elif choice == "Logout":
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.page = "Home"
        st.success("✅ Logged out")
