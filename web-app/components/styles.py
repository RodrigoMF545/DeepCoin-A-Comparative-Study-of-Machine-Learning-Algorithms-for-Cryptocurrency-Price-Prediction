# components/styles.py
import streamlit as st

def inject_styles():
    st.markdown("""
    <style>
    .main {background-color: #0f1724; color: #d1d9e6;}
    .card {
        background: linear-gradient(180deg, rgba(20,28,40,0.6), rgba(14,20,30,0.6));
        border: 1px solid rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 4px 18px rgba(2,6,23,0.6);
        margin-bottom: 12px;
    }
    .muted { color: #9aa4b2; font-size: 0.9rem; }
    .title { font-weight:700; color: #ffffff; font-size: 1.25rem; }
    div[data-testid="stSidebar"] {background-color: #071022; color: #cfd8e6;}
    .metric-box {
        background: rgba(255,255,255,0.02);
        border-radius: 8px;
        padding: 10px 14px;
        text-align: left;
        margin-bottom: 8px;
    }
    .stAlert {background-color: rgba(20,28,40,0.8);}
    </style>
    """, unsafe_allow_html=True)
