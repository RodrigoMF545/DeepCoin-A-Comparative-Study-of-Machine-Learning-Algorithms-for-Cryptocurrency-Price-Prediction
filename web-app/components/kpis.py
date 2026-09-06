# components/kpis.py
import streamlit as st
import numpy as np

def render_kpis(df, df_pred, sidebar_state):
    k1, k2, k3, k4 = st.columns([1,1,1,1])

    with k1:
        current_price = df['close'].iloc[-1]
        st.markdown(f"""
        <div class='card metric-box'>
            <div class='muted'>💵 Preço Atual</div>
            <h3 style='margin:4px'>${current_price:,.2f}</h3>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        if df_pred is not None and not df_pred.empty:
            pred_after = df_pred['predicted_close'].iloc[-1]
            change_pct = ((pred_after - current_price) / current_price) * 100
            color = "#4ade80" if change_pct > 0 else "#f87171"
            arrow = "↗" if change_pct > 0 else "↘"
            st.markdown(f"""
            <div class='card metric-box'>
                <div class='muted'>🎯 Preço Previsto ({sidebar_state['predict_days']}d)</div>
                <h3 style='margin:4px'>${pred_after:,.2f}</h3>
                <div style='color:{color};font-size:0.85rem;font-weight:600'>
                    {arrow} {'+' if change_pct > 0 else ''}{change_pct:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='card metric-box'>
                <div class='muted'>🎯 Preço Previsto ({sidebar_state['predict_days']}d)</div>
                <h3 style='margin:4px'>Execute a predição</h3>
            </div>
            """, unsafe_allow_html=True)

    with k3:
        volume_current = int(df['volume'].iloc[-1])
        st.markdown(f"""
        <div class='card metric-box'>
            <div class='muted'>📊 Volume Atual</div>
            <h3 style='margin:4px'>{volume_current:,}</h3>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        if len(df) >= 2:
            change_24h = ((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100
        else:
            change_24h = 0
        color_24h = "#4ade80" if change_24h > 0 else "#f87171"
        arrow_24h = "↗" if change_24h > 0 else "↘"
        st.markdown(f"""
        <div class='card metric-box'>
            <div class='muted'>📅 Variação 24h</div>
            <h3 style='margin:4px;color:{color_24h}'>
                {arrow_24h} {'+' if change_24h > 0 else ''}{change_24h:.2f}%
            </h3>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
