import streamlit as st
import time
import pandas as pd
import numpy as np
import joblib
import os
import sys
from io import BytesIO

# --- Path Fix ---
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from agent import FeatureEngineeringAgent
except ImportError:
    from src.agent import FeatureEngineeringAgent

# --- Page Config ---
st.set_page_config(
    page_title="METOOLOK | Space Agent 2026",
    page_icon="👨‍🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Space & Cyberpunk 2026 Theme (Custom CSS) ---
st.markdown("""
    <style>
    /* Ana Arka Plan */
    .stApp {
        background: radial-gradient(circle at top, #0f0f1a 0%, #050505 100%);
        color: #E0E0E0;
    }

    /* Sidebar Tasarımı */
    section[data-testid="stSidebar"] {
        background-color: #05050a;
        border-right: 2px solid #58A6FF;
    }

    /* 2026 Metoolok Logo Tasarımı */
    .logo-container {
        padding: 20px;
        text-align: center;
        border: 1px solid #58A6FF;
        border-radius: 15px;
        background: rgba(88, 166, 255, 0.05);
        box-shadow: 0 0 20px rgba(88, 166, 255, 0.2);
        margin-bottom: 25px;
    }
    .logo-text {
        font-family: 'Orbitron', sans-serif;
        font-weight: 900;
        font-size: 32px;
        color: #58A6FF;
        letter-spacing: 4px;
        text-shadow: 0 0 15px #58A6FF;
        margin: 0;
    }
    .logo-subtext {
        font-size: 9px;
        color: #8B949E;
        letter-spacing: 1.5px;
        margin-top: 8px;
        text-transform: uppercase;
    }

    /* Terminal Ekranı */
    .terminal {
        background-color: #000000;
        border: 2px solid #58A6FF;
        border-radius: 10px;
        padding: 20px;
        font-family: 'Courier New', monospace;
        color: #00d4ff;
        height: 320px;
        overflow-y: auto;
        box-shadow: inset 0 0 25px #00d4ff22;
        margin-bottom: 20px;
    }

    /* Metric Kartları: Glassmorphism */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(88, 166, 255, 0.2);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        transition: 0.3s ease;
    }
    .metric-card:hover {
        border-color: #58A6FF;
        box-shadow: 0 0 30px rgba(88, 166, 255, 0.25);
        transform: translateY(-3px);
    }

    /* Tabs ve Butonlar */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #161B22;
        border-radius: 5px 5px 0 0;
        color: #8B949E;
    }
    .stTabs [aria-selected="true"] { color: #58A6FF !important; border-bottom-color: #58A6FF !important; }

    .stButton>button {
        background: linear-gradient(135deg, #1f6feb 0%, #58a6ff 100%);
        border: none;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        box-shadow: 0 0 20px #58A6FF;
        transform: scale(1.02);
    }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;900&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)


# --- Helper Functions ---
def type_log(placeholder, current_logs, new_message):
    current_logs.append(f"> {new_message}")
    terminal_html = "<br>".join(current_logs)
    placeholder.markdown(f'<div class="terminal">{terminal_html}</div>', unsafe_allow_html=True)


# --- Session State ---
if "agent" not in st.session_state: st.session_state.agent = None
if "processed_df" not in st.session_state: st.session_state.processed_df = None

# --- Sidebar ---
with st.sidebar:
    st.markdown("""
        <div class="logo-container">
            <p class="logo-text">METOOLOK</p>
            <p class="logo-subtext">DESIGNED BY METIN MERT | © 2026</p>
        </div>
    """, unsafe_allow_html=True)

    st.subheader("⚙️ Agent Command")
    problem_type = st.selectbox("Problem Type", ["classification", "regression"])
    uploaded_file = st.file_uploader("Infiltrate Data (CSV)", type=["csv"])

    target_col = None
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"📡 Data uplink successful: {len(df)} rows")
        target_col = st.selectbox("Select Target Objective", df.columns)

# --- Main Layout ---
st.markdown(
    "<h1 style='text-align: center; color: #58A6FF; text-shadow: 0 0 15px #58A6FF;'>🌌 NEURAL SPACE COMMAND</h1>",
    unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #8B949E; margin-bottom: 30px;'>Autonomous Feature Engineering Agent v6.0</p>",
    unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
feature_count = str(len(st.session_state.agent.state.get("final_columns", []))) if st.session_state.agent else "0"

with col1:
    st.markdown(
        f'<div class="metric-card"><h2 style="margin:0;color:#58A6FF;">{feature_count}</h2><p style="margin:0;font-size:12px;">ACTIVE FEATURES</p></div>',
        unsafe_allow_html=True)
with col2:
    st.markdown(
        f'<div class="metric-card"><h2 style="margin:0;color:#58A6FF;">v{FeatureEngineeringAgent.__version__}</h2><p style="margin:0;font-size:12px;">SYSTEM VERSION</p></div>',
        unsafe_allow_html=True)
with col3:
    status = "READY" if st.session_state.agent else "IDLE"
    st.markdown(
        f'<div class="metric-card"><h2 style="margin:0;color:#58A6FF;">{status}</h2><p style="margin:0;font-size:12px;">AGENT STATUS</p></div>',
        unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["⚡ Mission Control", "🔭 Data Exploration", "🛸 Payload Export"])

with tab1:
    log_placeholder = st.empty()
    logs = []
    if uploaded_file and target_col:
        if st.button("🚀 INITIATE NEURAL OPTIMIZATION"):
            type_log(log_placeholder, logs, "Establishing Neural Link...")
            time.sleep(0.5)

            agent = FeatureEngineeringAgent(target_column=target_col, problem_type=problem_type)

            with st.spinner("Processing Deep Space Data..."):
                agent.fit(df, y=df[target_col])
                full_processed_df = agent.transform(df, detect_drift=True)

            st.session_state.agent = agent
            st.session_state.processed_df = full_processed_df

            type_log(log_placeholder, logs, f"✅ Optimization Complete. {len(full_processed_df)} entities processed.")
            st.rerun()
    elif not uploaded_file:
        log_placeholder.markdown('<div class="terminal">> System Offline. Waiting for Data Uplink...</div>',
                                 unsafe_allow_html=True)

with tab2:
    if st.session_state.processed_df is not None:
        st.subheader("🔭 Processed Entity Preview")
        st.dataframe(st.session_state.processed_df.head(100))
        st.subheader("📊 Statistical Signature")
        st.write(st.session_state.processed_df.describe())
    else:
        st.info("Initiate mission control to view analysis.")

with tab3:
    if st.session_state.processed_df is not None:
        st.subheader("📦 Payload Ready for Dispatch")

        full_df = st.session_state.processed_df
        csv_buffer = BytesIO()
        full_df.to_csv(csv_buffer, index=False)

        st.download_button(
            label="⬇️ DOWNLOAD PROCESSED CSV",
            data=csv_buffer.getvalue(),
            file_name=f"metoolok_space_data_{int(time.time())}.csv",
            mime="text/csv"
        )

        st.divider()
        st.subheader("🧠 Core Memory Export")
        agent_buffer = BytesIO()
        joblib.dump(st.session_state.agent, agent_buffer)
        st.download_button(
            label="⬇️ DOWNLOAD AGENT (.JOBLIB)",
            data=agent_buffer.getvalue(),
            file_name="metoolok_space_agent.joblib"
        )