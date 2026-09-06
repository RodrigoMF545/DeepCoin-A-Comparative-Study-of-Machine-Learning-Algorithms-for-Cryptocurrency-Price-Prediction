# components/prediction_chart.py
import streamlit as st
import plotly.graph_objects as go

def render_prediction_chart(df, df_pred, sidebar_state):
    st.markdown(f"""
    <div class='card'>
        <div style='display:flex;gap:12px;align-items:center;margin-bottom:12px'>
            <div style='width:46px;height:46px;border-radius:10px;background:linear-gradient(90deg,#2b6ef6,#6c8ef7);display:flex;align-items:center;justify-content:center'>
                <svg height='20' width='20' viewBox='0 0 24 24' fill='white'><path d='M13 2L3 14h8l-2 8 10-12h-8l2-8z'/></svg>
            </div>
            <div>
                <div class='title'>Predição de Preço - {sidebar_state['symbol']}</div>
                <div class='muted'>Modelo: {sidebar_state['model_choice'] if sidebar_state['model_choice'] else 'Selecione um modelo'}</div>
            </div>
        </div>
        <hr style='opacity:0.06;margin-bottom:12px'>
    """, unsafe_allow_html=True)

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(
        x=df['date'],
        y=df['close'],
        name='Histórico',
        line=dict(width=2, color='#6c8ef7'),
        hovertemplate='<b>Data:</b> %{x}<br><b>Preço:</b> $%{y:,.2f}<extra></extra>'
    ))

    if df_pred is not None and not df_pred.empty:
        fig_pred.add_trace(go.Scatter(
            x=df_pred['date'],
            y=df_pred['predicted_close'],
            name='Predição',
            line=dict(width=3, dash='dash', color='#4ade80'),
            hovertemplate='<b>Data:</b> %{x}<br><b>Previsto:</b> $%{y:,.2f}<extra></extra>'
        ))
        fig_pred.add_vline(x=df['date'].iloc[-1], line=dict(color='rgba(255,255,255,0.2)', dash='dot', width=2))
        fig_pred.add_annotation(x=df['date'].iloc[-1], y=df['close'].max() * 1.05, text="← Histórico | Predição →", showarrow=False, font=dict(size=11, color='#9aa4b2'))

    fig_pred.update_layout(
        template='plotly_dark',
        height=450,
        margin=dict(t=10,b=10,l=10,r=10),
        legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        xaxis_title="Data",
        yaxis_title="Preço (USD)",
        hovermode='x unified'
    )

    st.plotly_chart(fig_pred, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
