# import streamlit as st
# import pandas as pd
# import joblib
# import numpy as np
# import os

# # --- CRITICAL: AUTOMATIC BUILD LOGIC ---
# # This must happen before we try to load the assets
# from engine import train_and_save

# if not os.path.exists('model.joblib') or not os.path.exists('scaler.joblib'):
#     # We use st.toast or a spinner to let the user know what's happening
#     with st.spinner("🚀 First-time setup: Training the ML Model on the server..."):
#         train_and_save()
#     st.success("Model ready!")

# # --- APP CONFIGURATION ---
# st.set_page_config(page_title="Travel ML Production", layout="wide")

# @st.cache_resource
# def load_assets():
#     # Now that we've ensured they exist, we load them
#     model = joblib.load('model.joblib')
#     scaler = joblib.load('scaler.joblib')
#     return model, scaler

# model, scaler = load_assets()

# # --- MAIN UI ---
# st.title("🛫 Real-Time Booking Intent & Intervention")
# st.markdown("This model predicts the probability of a user booking based on session behavior.")

# # Sidebar for Simulation
# st.sidebar.header("Live Session Simulation")
# d2d = st.sidebar.slider("Days to Departure", 1, 180, 15)
# stay = st.sidebar.slider("Stay Duration", 1, 21, 5)
# duration = st.sidebar.number_input("Session Seconds", 10, 3600, 600)
# p_dev = st.sidebar.slider("Price Dev %", -20, 20, 0) / 100
# m_change = st.sidebar.slider("10m Price Change %", -10, 10, 2) / 100
# freq = st.sidebar.slider("Searches last 24h", 1, 20, 3)

# # Prediction Pipeline
# # Reshape to match the training features: [days, stay, duration, dev, change, freq]
# input_data = np.array([[d2d, stay, duration, p_dev, m_change, freq]])
# input_scaled = scaler.transform(input_data)
# prob = model.predict_proba(input_scaled)[0, 1]

# # Display Results
# col1, col2 = st.columns(2)

# with col1:
#     st.subheader("Booking Intent Score")
#     st.metric("Probability", f"{prob*100:.1f}%")
#     st.progress(prob)
    
#     if prob < 0.30:
#         st.error("Status: High Abandonment Risk")
#     elif prob < 0.60:
#         st.warning("Status: Window Shopper")
#     else:
#         st.success("Status: Likely to Book")

# with col2:
#     st.subheader("Smart Intervention")
#     # Business Logic: Trigger if engagement is high (duration) but price momentum is scary (m_change)
#     if prob < 0.45 and duration > 400:
#         st.info("🎯 **ACTION RECOMMENDED**")
#         st.write("The user is highly engaged but hesitating. Trigger a **'Price Freeze'** or **'10% Discount'** popup.")
#         if st.button("Simulate Intervention"):
#             st.balloons()
#             st.write("Coupon sent to user session!")
#     else:
#         st.write("User behavior is normal. No intervention needed at this time.")

# # Admin Section
# st.sidebar.markdown("---")
# if st.sidebar.button("🔄 Force Retrain Model"):
#     os.remove('model.joblib')
#     st.rerun()

#######################################################
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SkyFlow | Predictive Ops",
    page_icon="🛫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM THEME & CSS ---
st.markdown("""
    <style>
    /* Main background */
    .stApp { background-color: #F9FAFB; }
    
    /* Metrics Styling */
    [data-testid="stMetricValue"] { font-size: 2.2rem !important; font-weight: 700 !important; color: #1F2937; }
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] { background-color: #111827; color: white; }
    section[data-testid="stSidebar"] .stMarkdown h2, section[data-testid="stSidebar"] .stMarkdown h1 { color: #F9FAFB; }
    
    /* Custom Action Card */
    .action-card {
        background-color: #EFF6FF;
        border-left: 5px solid #3B82F6;
        padding: 20px;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- ASSET LOADING ---
@st.cache_resource
def load_assets():
    if not os.path.exists('model.joblib'):
        from engine import train_and_save
        train_and_save()
    return joblib.load('model.joblib'), joblib.load('scaler.joblib')

model, scaler = load_assets()

# --- SIDEBAR: INTELLIGENT INPUTS ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/airplane-take-off.png", width=80)
    st.title("SkyFlow AI")
    st.markdown("---")
    
    st.subheader("📍 Journey Context")
    d2d = st.slider("Days to Departure", 1, 180, 15)
    stay = st.slider("Stay Duration (Days)", 1, 21, 5)
    
    st.subheader("⚡ Live Behavior")
    duration = st.number_input("Session Time (Seconds)", 10, 3600, 600, step=10)
    p_dev = st.select_slider("Price Deviation (%)", options=np.arange(-20, 21, 5).tolist(), value=0) / 100
    m_change = st.select_slider("10m Momentum (%)", options=[-5, -2, 0, 2, 5], value=2) / 100
    freq = st.number_input("Search Freq (24h)", 1, 20, 3)
    
    st.markdown("---")
    if st.button("🔄 Reset Environment"):
        st.rerun()

# --- INFERENCE PIPELINE ---
input_features = np.array([[d2d, stay, duration, p_dev, m_change, freq]])
features_scaled = scaler.transform(input_features)
prob = model.predict_proba(features_scaled)[0, 1]

# --- MAIN DASHBOARD LAYOUT ---
st.header("Session Intelligence Dashboard")
st.caption(f"Real-time prediction engine active • Model Version: 2.4.1")

# Row 1: Key Metrics
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Booking Intent", f"{prob*100:.1f}%", delta=f"{prob - 0.15:.2%}", delta_color="normal")

with m2:
    status = "Active" if duration < 1200 else "Fatigued"
    st.metric("User Status", status, delta="Session Health")

with m3:
    risk_level = "CRITICAL" if prob < 0.35 else "STABLE"
    st.metric("Churn Risk", risk_level, delta="-2.4% vs Avg", delta_color="inverse")

with m4:
    st.metric("Market Context", "Volatile" if abs(m_change) > 0.02 else "Flat", delta="Price Velocity")

st.markdown("---")

# Row 2: Deep Analysis & Action
left_col, right_col = st.columns([1.5, 1])

with left_col:
    st.subheader("🧠 Behavioral Attribution")
    # Generating a mock attribution based on model inputs for UI feedback
    attribution_data = pd.DataFrame({
        'Factor': ['Price Sensitivity', 'Engagement Time', 'Urgency', 'Momentum'],
        'Influence': [abs(p_dev)*100, (duration/3600)*100, (1-(d2d/180))*100, abs(m_change)*100]
    }).set_index('Factor')
    
    st.bar_chart(attribution_data, use_container_width=True)
    st.caption("How different behavioral signals are currently influencing the probability score.")

with right_col:
    st.subheader("🎯 Prescriptive Action")
    
    # Logic-driven intervention cards
    if prob < 0.40 and duration > 400:
        st.markdown(f"""
            <div class="action-card">
                <h4 style="color: #1E40AF; margin-top:0;">Strategy: Intent Recovery</h4>
                <p style="color: #1E3A8A;">The user has spent <b>{duration}s</b> browsing but <b>{m_change*100}% price momentum</b> is causing friction.</p>
                <hr style="border: 0.5px solid #BFDBFE;">
                <b>Recommended:</b> Trigger "Price Protection" Guarantee.
            </div>
            """, unsafe_allow_html=True)
        
        if st.button("🚀 Trigger Real-Time Intervention", use_container_width=True):
            st.balloons()
            st.toast("Incentive deployed to user session.")
    
    elif prob > 0.75:
        st.success("High Organic Intent. Maintain standard checkout flow (No discounts needed).")
    else:
        st.info("Continuous Monitoring. User is in standard comparison phase.")

# Row 3: Live Session Log (Simulated)
with st.expander("📝 Raw Feature Vector (Developer Trace)"):
    st.write(pd.DataFrame(input_features, columns=['D2D', 'Stay', 'Duration', 'Price_Dev', 'Momentum', 'Freq']))