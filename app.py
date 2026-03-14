import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Travel ML Production", layout="wide")

@st.cache_resource
def load_assets():
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

try:
    model, scaler = load_assets()
except:
    st.error("Model files not found. Run 'python engine.py' first!")
    st.stop()

st.title("🛫 Real-Time Booking Intent & Intervention")

# Simulation Sidebar
st.sidebar.header("Session Parameters")
d2d = st.sidebar.slider("Days to Departure", 1, 180, 15)
stay = st.sidebar.slider("Stay Duration", 1, 21, 5)
duration = st.sidebar.number_input("Session Seconds", 10, 3600, 600)
p_dev = st.sidebar.slider("Price Dev %", -20, 20, 0) / 100
m_change = st.sidebar.slider("10m Price Change %", -10, 10, 2) / 100
freq = st.sidebar.slider("Searches last 24h", 1, 20, 3)

# Prediction Logic
features = np.array([[d2d, stay, duration, p_dev, m_change, freq]])
features_scaled = scaler.transform(features)
prob = model.predict_proba(features_scaled)[0, 1]

# Display Results
col1, col2 = st.columns(2)
with col1:
    st.metric("Booking Probability", f"{prob*100:.1f}%")
    st.progress(prob)

with col2:
    if prob < 0.35 and duration > 500:
        st.error("⚠️ INTERVENTION TRIGGERED")
        st.markdown("### Offer: 'Price Freeze for $5'")
    else:
        st.success("✅ MONITORING: No intervention needed")

st.info("The model suggests that long sessions with low booking probability indicate high comparison-shopping intent. These users are prime for incentives.")