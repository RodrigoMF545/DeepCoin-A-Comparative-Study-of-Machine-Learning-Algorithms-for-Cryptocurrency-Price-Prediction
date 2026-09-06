# components/header.py
import streamlit as st

def render_header(df, sidebar_state):
    st.markdown(f"""
    <div style='display:flex;align-items:center;justify-content:space-between'>
        <div>
            <h1 style='margin:0;color:#fff'>📈 DeepCoin Prediction Dashboard</h1>
            <div class='muted'>Análise avançada com Deep Learning para predição de criptomoedas</div>
        </div>
        <div style='text-align:right'>
            <div style='color:{sidebar_state["color"]};font-weight:700;font-size:1.5rem'>
                {sidebar_state["name"]}
            </div>
            <div class='muted' style='font-size:0.75rem'>{len(df):,} registros</div>
            <div class='muted' style='font-size:0.75rem'>
                {df['date'].min().strftime('%d/%m/%Y')} - {df['date'].max().strftime('%d/%m/%Y')}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
