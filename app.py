import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pytz
from datetime import datetime, timedelta

# UI Setup & Custom Styling
st.set_page_config(page_title="Iron Dome C2 System", page_icon="🛡️", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
    .main { background-color: #030305; color: #00f2ff; font-family: 'Orbitron', sans-serif; }
    .stMetric { background: rgba(0, 242, 255, 0.03); border: 1px solid rgba(0, 242, 255, 0.3); border-radius: 8px; padding: 20px; text-align: center; }
    .prediction-card {
        background: rgba(0, 242, 255, 0.02);
        padding: 30px; border-radius: 12px; border: 1px solid #00f2ff;
        box-shadow: 0 0 25px rgba(0, 242, 255, 0.1); margin-top: 25px;
    }
    .danger-card { border-color: #ff4b4b; box-shadow: 0 0 25px rgba(255, 75, 75, 0.1); }
    .safe-card { border-color: #00ff00; box-shadow: 0 0 25px rgba(0, 255, 0, 0.1); }
    </style>
    """, unsafe_allow_html=True)

# Load Machine Learning Assets Securely
@st.cache_resource
def load_c2_assets():
    return (
        joblib.load("alert_classifier.joblib"),
        joblib.load("alert_regressor.joblib"),
        joblib.load("location_encoder.joblib"),
        joblib.load("system_memory.joblib"),
        joblib.load("feature_names.joblib")
    )

try:
    clf, reg, encoder, sys_mem, FEATURE_ORDER = load_c2_assets()
except Exception as e:
    st.error(f"System initialization failed. Ensure all .joblib files are present. {e}")
    st.stop()

# --- LIVE GITHUB INGESTION ENGINE ---
# Cache data for 60 seconds to prevent rate-limiting from GitHub
@st.cache_data(ttl=60)
def fetch_live_country_load():
    csv_url = "https://raw.githubusercontent.com/yuval-harpaz/alarms/master/data/alarms.csv"
    try:
        # Load only the Time column to save memory and processing power
        df = pd.read_csv(csv_url, usecols=[0], names=["Time"], skiprows=1)
        df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
        
        # Calculate the exact time 60 minutes ago in Israel time
        israel_tz = pytz.timezone('Asia/Jerusalem')
        one_hour_ago = datetime.now(israel_tz).replace(tzinfo=None) - timedelta(hours=1)
        
        # Filter for alerts that happened in the last hour
        recent_alerts = df[df["Time"] >= one_hour_ago]
        return len(recent_alerts)
        
    except Exception as e:
        print(f"Sync error with GitHub Data Stream: {e}")
        return 0

# --- C2 COMMAND PANEL (SIDEBAR) ---
with st.sidebar:
    st.header("🕹️ COMMAND PANEL")
    selected_city = st.selectbox("TARGET LOCATION", sorted(encoder.classes_))
    
    # Force Israel Timezone to prevent server discrepancies
    israel_tz = pytz.timezone('Asia/Jerusalem')
    current_time = datetime.now(israel_tz).strftime("%Y-%m-%d %H:%M:%S")
    time_str = st.text_input("REFERENCE TIME", current_time)
    
    st.divider()
    
    # War-Game Simulation Toggle
    sim_mode = st.toggle("ACTIVATE SIMULATION", False)
    if sim_mode:
        live_load = st.slider("SIMULATED NATIONAL LOAD", 0, 300, 80)
        local_load_24h_sim = st.slider("SIMULATED LOCAL 24H LOAD", 0, 50, 5)
        st.warning("SYSTEM IN SIMULATION MODE")
    else:
        live_load = fetch_live_country_load()
        local_load_24h_sim = 0 
        if live_load > 0:
            st.error(f"🔴 LIVE ALERT: {live_load} ACTIVE ZONES (Past Hour)")
        else:
            st.success("🟢 LIVE STATUS: CLEAR NATIONWIDE")

    run_btn = st.button("EXECUTE ANALYSIS", use_container_width=True)

# --- MAIN DASHBOARD ENGINE ---
st.title("🛡️ IRON DOME C2 SYSTEM")
st.subheader("PURE TACTICAL ENGINE V9.1 (LIVE GITHUB SYNC)")

if run_btn:
    try:
        ref_time = pd.to_datetime(time_str)
        loc_idx = encoder.transform([selected_city])[0]
        
        # Pull Historical Details
        loc_total = sys_mem['counts'].get(loc_idx, 0)
        risk_ratio = sys_mem['ratios'].get(loc_idx, 0.0001)
        is_rare = 1 if loc_total < 15 else 0
        
        last_t = sys_mem['time'].get(loc_idx)
        prev_gap = sys_mem['gap'].get(loc_idx, 0)
        
        # Dynamic Context Math
        min_since = (ref_time - last_t).total_seconds() / 60.0 if last_t else -1
        accel = min_since - prev_gap if min_since > 0 else 0
        local_impact = live_load * risk_ratio
        
        if not sim_mode:
            local_alerts_24h = 1 if 0 < min_since <= 1440 else 0 
        else:
            local_alerts_24h = local_load_24h_sim

        # Construct exact feature map for the model
        input_dict = {
            "Location_Encoded": loc_idx,
            "Hour_sin": np.sin(2 * np.pi * ref_time.hour / 24.0),
            "Hour_cos": np.cos(2 * np.pi * ref_time.hour / 24.0),
            "min_since_last": float(min_since),
            "acceleration": float(accel),
            "country_load_1h": float(live_load),
            "localized_impact": float(local_impact),
            "local_alerts_24h": float(local_alerts_24h),
            "is_rare": float(is_rare),
            "total_counts": float(loc_total)
        }
        X_in = pd.DataFrame([input_dict])[FEATURE_ORDER]

        # Execute Predictive Models
        prob = clf.predict_proba(X_in)[0][1]
        tactical_time = reg.predict(X_in)[0] # Raw predicted minutes

        # --- SMART UI LOGIC & MANUAL TWEAKS ---
        if prob < 0.40:
            is_safe = True
            display_time = "CLEAR"
        else:
            is_safe = False
            
            # The Tweak: Compress time proportionally to the danger probability
            tweak_multiplier = (1.01 - prob)
            adjusted_time = tactical_time * tweak_multiplier
            
            # Hard floor to prevent zero/negative times
            if adjusted_time < 2:
                adjusted_time = 2
                
            # Clean formatting
            if adjusted_time >= 60:
                display_time = f"{int(adjusted_time/60)}h {int(adjusted_time%60)}m"
            else:
                display_time = f"{int(adjusted_time)} min"

        # Display Live Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("6H THREAT PROBABILITY", f"{prob*100:.1f}%")
        c2.metric("ESTIMATED WINDOW", display_time)
        c3.metric("GEOGRAPHIC PROFILE", f"{loc_total} Historical Hits")

        st.markdown("---")
        
        # Threat Intelligence Card
        card_class = "safe-card" if is_safe else "danger-card"
        threat_status = "STABLE / CLEAR" if is_safe else "ELEVATED COMBAT READINESS"
        
        st.markdown(f"""
            <div class="prediction-card {card_class}">
                <h3>🔍 Tactical Intelligence Briefing: {selected_city}</h3>
                <p>System analysis finalized for reference time: <b>{ref_time}</b>.</p>
                <ul>
                    <li><b>Operational Status:</b> {threat_status}</li>
                    <li><b>Live Sync:</b> Fetching live payload securely via decentralized datastream.</li>
                    <li><b>System Calibration:</b> Adjusted using {live_load} active national threats in the past 60 minutes.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Tactical Engine Error: {e}")

st.divider()
st.caption("C2 PROTOCOL V9.1 | LIVE GITHUB DATALINK | TWEAKED SYNCHRONIZATION")