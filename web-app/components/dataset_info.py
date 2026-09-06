# components/dataset_info.py
import streamlit as st
import numpy as np

def render_dataset_info(df, sidebar_state):
    st.markdown(f"""
    <div class='card'>
        <div style='display:flex;gap:12px;align-items:center;margin-bottom:12px'>
            <div style='width:46px;height:46px;border-radius:10px;background:linear-gradient(90deg,#16a34a,#34d399);display:flex;align-items:center;justify-content:center'>
                <svg height='20' width='20' viewBox='0 0 24 24' fill='white'><path d='M3 13h8V3H3v10zm0 8h8v-6H3v6zM13 21h8V11h-8v10zM13 3v6h8V3h-8z'/></svg>
            </div>
            <div>
                <div class='title'>📚 Dataset Selecionado - {sidebar_state['name']}</div>
                <div class='muted'>Arquivo: {sidebar_state['file']}</div>
            </div>
        </div>
        <hr style='opacity:0.06;margin-bottom:12px'>
    """, unsafe_allow_html=True)

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.markdown(f"<div class='metric-box'><div class='muted'>Registros</div><h3 style='margin:4px'>{len(df):,}</h3></div>", unsafe_allow_html=True)
    with d2:
        st.markdown(f"<div class='metric-box'><div class='muted'>Período</div><h4 style='margin:4px'>{df['date'].min().strftime('%d/%m/%Y')} - {df['date'].max().strftime('%d/%m/%Y')}</h4></div>", unsafe_allow_html=True)
    with d3:
        st.markdown(f"<div class='metric-box'><div class='muted'>Mínimo / Máximo</div><h4 style='margin:4px'>${df['close'].min():,.2f} / ${df['close'].max():,.2f}</h4></div>", unsafe_allow_html=True)
    with d4:
        volatility_avg = df['returns'].std() * np.sqrt(252) * 100
        st.markdown(f"<div class='metric-box'><div class='muted'>Volatilidade Anual</div><h4 style='margin:4px'>{volatility_avg:.2f}%</h4></div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:12px'><div class='muted'><b>Features Disponíveis:</b></div></div>", unsafe_allow_html=True)
    feature_cols = [col for col in df.columns if col != 'date']
    feature_info = ", ".join([f"<code style='background:rgba(255,255,255,0.05);padding:2px 6px;border-radius:4px'>{col}</code>" for col in feature_cols])
    st.markdown(f"<div class='muted'>{feature_info}</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:12px'><div class='muted'>Últimas 5 linhas do dataset</div></div>", unsafe_allow_html=True)
    preview = df.tail(5).copy()
    preview['date'] = preview['date'].dt.strftime('%Y-%m-%d')
    st.dataframe(preview, use_container_width=True, hide_index=True)

    st.markdown("</div>", unsafe_allow_html=True)
