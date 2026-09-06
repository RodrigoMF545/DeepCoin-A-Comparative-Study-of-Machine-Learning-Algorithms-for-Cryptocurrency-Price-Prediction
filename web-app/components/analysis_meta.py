# components/analysis_meta.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from datetime import datetime

def _get_file_last_update(filename):
    """
    Tenta obter a data de modificação do ficheiro em data/<filename>.
    Retorna None se não encontrar.
    """
    try:
        filepath = os.path.join('data', filename)
        if os.path.exists(filepath):
            ts = os.path.getmtime(filepath)
            return datetime.fromtimestamp(ts)
    except Exception:
        pass
    return None

def _detect_scaling_columns(df):
    """Procura colunas que indiquem que os dados já estão escalados (heurística)."""
    scaled_cols = [c for c in df.columns if ('scaled' in c.lower()) or ('scaler' in c.lower()) or ('norm' in c.lower())]
    return scaled_cols

def _outliers_iqr_count(series):
    """Conta outliers usando o método IQR (1.5 * IQR)."""
    if series.dropna().empty:
        return 0
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return int(((series < lower) | (series > upper)).sum())

def render_analysis_meta(df: pd.DataFrame, sidebar_state: dict):
    """
    Renderiza a Row 4: Heatmap de Correlação + Dataset Info (source, last update, features, preview)
    + Notas simples sobre processamento (missing values, outliers, scaling heurística).
    """
    st.markdown("<div style='margin-top:12px'></div>", unsafe_allow_html=True)

    # Grid: left (6 cols) heatmap, right (6 cols) dataset info + notas
    left_col, right_col = st.columns([1.2, 1])

    # -----------------------
    # LEFT: Heatmap de Correlação
    # -----------------------
    with left_col:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("""
            <div style='display:flex;gap:12px;align-items:center;margin-bottom:12px'>
                <div style='width:46px;height:46px;border-radius:10px;background:linear-gradient(90deg,#d946ef,#8b5cf6);
                            display:flex;align-items:center;justify-content:center'>
                    <svg height='20' width='20' viewBox='0 0 24 24' fill='white'>
                        <path d='M3 13h8V3H3v10zm0 8h8v-6H3v6zM13 21h8V11h-8v10zM13 3v6h8V3h-8z'/>
                    </svg>
                </div>
                <div>
                    <div class='title'>Heatmap de Correlação</div>
                    <div class='muted'>Relação entre features presentes no dataset</div>
                </div>
            </div>
            <hr style='opacity:0.06;margin-bottom:12px'>
        """, unsafe_allow_html=True)

        # Escolhe features numéricas (exceto date)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.markdown("<div class='muted'>Sem colunas numéricas suficientes para calcular correlação.</div>", unsafe_allow_html=True)
        else:
            corr = df[numeric_cols].corr()

            # Cria heatmap com Plotly
            fig = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                zmin=-1, zmax=1,
                colorscale='RdBu',
                colorbar=dict(title="r")
            ))
            fig.update_layout(
                template='plotly_dark',
                margin=dict(t=10, b=10, l=10, r=10),
                height=380
            )
            # Anota valores (opcional: só mostra hover)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------
    # RIGHT: Dataset Info + Notas
    # -----------------------
    with right_col:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("""
            <div style='display:flex;gap:12px;align-items:center;margin-bottom:12px'>
                <div style='width:46px;height:46px;border-radius:10px;background:linear-gradient(90deg,#16a34a,#34d399);
                            display:flex;align-items:center;justify-content:center'>
                    <svg height='20' width='20' viewBox='0 0 24 24' fill='white'>
                        <path d='M12 2L2 7v7c0 5 4 9 10 9s10-4 10-9V7l-10-5z'/>
                    </svg>
                </div>
                <div>
                    <div class='title'>Dataset Info & Metadados</div>
                    <div class='muted'>Fonte, última atualização e preview</div>
                </div>
            </div>
            <hr style='opacity:0.06;margin-bottom:12px'>
        """, unsafe_allow_html=True)

        # Fonte (heurística: sidebar_state contém o nome do ficheiro)
        src = sidebar_state.get('file', 'data/??.csv')
        st.markdown(f"<div class='muted'><b>Fonte:</b> <code>{src}</code></div>", unsafe_allow_html=True)

        # última atualização do ficheiro (se possível)
        last_update = _get_file_last_update(src)
        if last_update:
            st.markdown(f"<div class='muted'><b>Última atualização do ficheiro:</b> {last_update.strftime('%d/%m/%Y %H:%M:%S')}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='muted'><b>Última atualização do ficheiro:</b> Não disponível</div>", unsafe_allow_html=True)

        # features
        feature_cols = [c for c in df.columns if c != 'date']
        st.markdown("<div style='margin-top:8px'/>", unsafe_allow_html=True)
        st.markdown("<div class='muted'><b>Features:</b></div>", unsafe_allow_html=True)
        feature_info = ", ".join([f"<code style='background:rgba(255,255,255,0.03);padding:2px 6px;border-radius:4px'>{c}</code>" for c in feature_cols])
        st.markdown(f"<div class='muted' style='margin-bottom:8px'>{feature_info}</div>", unsafe_allow_html=True)

        # preview
        st.markdown("<div class='muted'><b>Preview (últimas 5 linhas):</b></div>", unsafe_allow_html=True)
        preview = df.tail(5).copy()
        if 'date' in preview.columns:
            preview['date'] = preview['date'].dt.strftime('%Y-%m-%d')
        st.dataframe(preview, use_container_width=True, hide_index=True)

        st.markdown("</div>", unsafe_allow_html=True)

        
