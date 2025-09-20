"""
Página de Análise de Séries Temporais
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

# Adicionar o diretório modules ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))

try:
    from modules.data_loader import DataLoader
    from modules.time_series import TimeSeriesAnalyzer
    from modules.visualizations import DashboardVisualizer
except ImportError as e:
    st.error(f"Erro ao importar módulos: {e}")
    st.stop()

st.set_page_config(
    page_title="Análise de Séries Temporais - Sistema de Monitoramento",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Análise de Séries Temporais")

# Inicializar componentes
data_loader = DataLoader()
time_series = TimeSeriesAnalyzer()
visualizer = DashboardVisualizer()

# Configurações das estações
stations_config = {
    'Two Mouths': {
        'parameters': ['pH', 'turbidity', 'chlorine', 'flow_rate', 'temperature']
    },
    'New Rose of Rocky': {
        'parameters': ['pH', 'DO', 'BOD', 'COD', 'TSS']
    },
    'Botanic Garden': {
        'parameters': ['pH', 'nitrogen', 'phosphorus', 'coliform', 'turbidity']
    }
}

# Sidebar para configurações
with st.sidebar:
    st.header("⚙️ Configurações")
    
    # Seleção de estação
    selected_station = st.selectbox(
        "📍 Selecionar Estação",
        list(stations_config.keys())
    )
    
    # Seleção de parâmetro
    station_params = stations_config[selected_station]['parameters']
    selected_param = st.selectbox(
        "📊 Parâmetro",
        station_params
    )
    
    # Período de análise
    time_period = st.selectbox(
        "📅 Período",
        ["Última semana", "Último mês", "Últimos 3 meses", "Último ano"]
    )
    
    # Tipo de análise
    analysis_type = st.selectbox(
        "🔍 Tipo de Análise",
        ["Decomposição Sazonal", "ARIMA", "Prophet", "Análise de Tendências", "Análise Espectral"]
    )

# Carregar dados
@st.cache_data
def load_station_data(station_name, days=90):
    """Carregar dados da estação"""
    return data_loader.generate_synthetic_data(station_name, days)

# Carregar dados baseado no período
days_map = {
    "Última semana": 7,
    "Último mês": 30,
    "Últimos 3 meses": 90,
    "Último ano": 365
}

df = load_station_data(selected_station, days_map[time_period])

# Filtrar dados do parâmetro selecionado
if selected_param in df.columns:
    ts_data = df.set_index('timestamp')[selected_param].dropna()
    
    st.header(f"📊 Análise de {selected_param}")
    
    # Estatísticas básicas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Média", f"{ts_data.mean():.3f}")
        
    with col2:
        st.metric("Desvio Padrão", f"{ts_data.std():.3f}")
        
    with col3:
        st.metric("Mínimo", f"{ts_data.min():.3f}")
        
    with col4:
        st.metric("Máximo", f"{ts_data.max():.3f}")
    
    # Gráfico da série temporal
    st.subheader("📈 Série Temporal")
    
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(
        x=ts_data.index,
        y=ts_data.values,
        mode='lines',
        name=selected_param,
        line=dict(width=2, color='blue')
    ))
    
    fig_ts.update_layout(
        title=f"Série Temporal - {selected_param}",
        xaxis_title="Tempo",
        yaxis_title=selected_param,
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_ts, use_container_width=True)
    
    # Análises específicas
    if analysis_type == "Decomposição Sazonal":
        st.subheader("🔍 Decomposição Sazonal")
        
        # Configurações
        col1, col2, col3 = st.columns(3)
        with col1:
            model_type = st.radio("Modelo:", ["additive", "multiplicative"])
        with col2:
            period = st.number_input("Período:", min_value=2, max_value=len(ts_data)//2, value=96)
        with col3:
            extrapolate_trend = st.checkbox("Extrapolar tendência", value=True)
        
        if st.button("Realizar Decomposição"):
            with st.spinner("Realizando decomposição sazonal..."):
                try:
                    decomposition = time_series.seasonal_decomposition(
                        ts_data, model=model_type, period=period
                    )
                    
                    if 'error' not in decomposition:
                        # Visualização da decomposição
                        fig_decomp = visualizer.create_seasonal_decomposition_chart(decomposition)
                        st.plotly_chart(fig_decomp, use_container_width=True)
                        
                        # Métricas da decomposição
                        st.subheader("📊 Métricas da Decomposição")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            trend_strength = 1 - np.var(decomposition['residual'].dropna()) / np.var(ts_data - decomposition['seasonal'])
                            st.metric("Força da Tendência", f"{trend_strength:.3f}")
                            
                        with col2:
                            seasonal_strength = 1 - np.var(decomposition['residual'].dropna()) / np.var(ts_data - decomposition['trend'])
                            st.metric("Força da Sazonalidade", f"{seasonal_strength:.3f}")
                            
                        with col3:
                            st.metric("Variância dos Resíduos", f"{np.var(decomposition['residual'].dropna()):.3f}")
                            
                        with col4:
                            st.metric("Período Dominante", f"{period} obs")
                            
                    else:
                        st.error(f"Erro na decomposição: {decomposition['error']}")
                        
                except Exception as e:
                    st.error(f"Erro ao realizar decomposição: {str(e)}")
    
    elif analysis_type == "ARIMA":
        st.subheader("🔍 Análise ARIMA")
        
        # Configurações do modelo
        col1, col2, col3 = st.columns(3)
        with col1:
            p = st.number_input("p (AR order):", min_value=0, max_value=10, value=1)
        with col2:
            d = st.number_input("d (Diferenciação):", min_value=0, max_value=3, value=1)
        with col3:
            q = st.number_input("q (MA order):", min_value=0, max_value=10, value=1)
        
        if st.button("Ajustar Modelo ARIMA"):
            with st.spinner("Ajustando modelo ARIMA..."):
                try:
                    result = time_series.fit_arima(ts_data, order=(p, d, q))
                    
                    if 'error' not in result:
                        # Resumo do modelo
                        st.subheader("📋 Resumo do Modelo")
                        st.text(str(result['summary']))
                        
                        # Métricas do modelo
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("AIC", f"{result['aic']:.2f}")
                        with col2:
                            st.metric("BIC", f"{result['bic']:.2f}")
                        with col3:
                            st.metric("Log-Likelihood", f"{result['model'].llf:.2f}")
                        
                        # Previsões
                        st.subheader("🔮 Previsões")
                        n_periods = st.slider("Períodos para previsão:", 1, 100, 24)
                        
                        forecast = time_series.forecast_arima(result['model'], n_periods)
                        
                        # Visualização
                        fig_arima = go.Figure()
                        
                        # Dados históricos
                        fig_arima.add_trace(go.Scatter(
                            x=ts_data.index,
                            y=ts_data.values,
                            mode='lines',
                            name='Dados Históricos',
                            line=dict(color='blue', width=2)
                        ))
                        
                        # Valores ajustados
                        fig_arima.add_trace(go.Scatter(
                            x=ts_data.index,
                            y=result['fitted_values'],
                            mode='lines',
                            name='Valores Ajustados',
                            line=dict(color='red', width=2, dash='dash')
                        ))
                        
                        # Previsões
                        future_dates = pd.date_range(
                            start=ts_data.index[-1] + pd.Timedelta(minutes=15),
                            periods=n_periods,
                            freq='15T'
                        )
                        
                        fig_arima.add_trace(go.Scatter(
                            x=future_dates,
                            y=forecast,
                            mode='lines',
                            name='Previsão',
                            line=dict(color='green', width=2)
                        ))
                        
                        fig_arima.update_layout(
                            title=f"Modelo ARIMA({p},{d},{q}) - {selected_param}",
                            xaxis_title="Tempo",
                            yaxis_title=selected_param,
                            height=500,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_arima, use_container_width=True)
                        
                        # Diagnóstico dos resíduos
                        st.subheader("🔍 Diagnóstico dos Resíduos")
                        
                        residuals = result['residuals']
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Histograma dos resíduos
                            fig_hist = go.Figure(data=[go.Histogram(x=residuals, nbinsx=30)])
                            fig_hist.update_layout(
                                title="Distribuição dos Resíduos",
                                xaxis_title="Resíduos",
                                yaxis_title="Frequência",
                                height=300
                            )
                            st.plotly_chart(fig_hist, use_container_width=True)
                            
                        with col2:
                            # Q-Q Plot
                            from scipy import stats
                            qq = stats.probplot(residuals, dist="norm")
                            
                            fig_qq = go.Figure()
                            fig_qq.add_trace(go.Scatter(
                                x=qq[0][0],
                                y=qq[0][1],
                                mode='markers',
                                name='Resíduos'
                            ))
                            fig_qq.add_trace(go.Scatter(
                                x=qq[0][0],
                                y=qq[0][0],
                                mode='lines',
                                name='Normal Teórica',
                                line=dict(color='red', dash='dash')
                            ))
                            fig_qq.update_layout(
                                title="Q-Q Plot",
                                xaxis_title="Quantis Teóricos",
                                yaxis_title="Quantis Amostrais",
                                height=300
                            )
                            st.plotly_chart(fig_qq, use_container_width=True)
                            
                    else:
                        st.error(f"Erro no modelo ARIMA: {result['error']}")
                        
                except Exception as e:
                    st.error(f"Erro na análise ARIMA: {str(e)}")
    
    elif analysis_type == "Prophet":
        st.subheader("🔍 Análise Prophet")
        
        if st.button("Ajustar Modelo Prophet"):
            with st.spinner("Ajustando modelo Prophet..."):
                try:
                    result = time_series.fit_prophet(df, selected_param)
                    
                    if 'error' not in result:
                        # Visualizar previsões
                        fig_prophet = visualizer.create_forecast_chart(df, selected_param, result['forecast'])
                        st.plotly_chart(fig_prophet, use_container_width=True)
                        
                        # Mostrar componentes
                        if 'components' in result:
                            st.subheader("🧩 Componentes do Modelo")
                            components = result['components']
                            
                            fig_comp = make_subplots(
                                rows=3, cols=1,
                                subplot_titles=('Tendência', 'Sazonalidade Diária', 'Sazonalidade Semanal'),
                                vertical_spacing=0.1
                            )
                            
                            fig_comp.add_trace(
                                go.Scatter(x=components['ds'], y=components['trend'], 
                                         mode='lines', name='Tendência'),
                                row=1, col=1
                            )
                            
                            fig_comp.add_trace(
                                go.Scatter(x=components['ds'], y=components['daily'], 
                                         mode='lines', name='Diária'),
                                row=2, col=1
                            )
                            
                            fig_comp.add_trace(
                                go.Scatter(x=components['ds'], y=components['weekly'], 
                                         mode='lines', name='Semanal'),
                                row=3, col=1
                            )
                            
                            fig_comp.update_layout(height=600, showlegend=False)
                            st.plotly_chart(fig_comp, use_container_width=True)
                            
                    else:
                        st.error(f"Erro no modelo Prophet: {result['error']}")
                        
                except Exception as e:
                    st.error(f"Erro na análise Prophet: {str(e)}")
    
    elif analysis_type == "Análise de Tendências":
        st.subheader("🔍 Análise de Tendências")
        
        # Médias móveis
        col1, col2 = st.columns(2)
        with col1:
            ma_short = st.number_input("Média móvel curta (períodos):", min_value=2, max_value=100, value=24)
        with col2:
            ma_long = st.number_input("Média móvel longa (períodos):", min_value=ma_short+1, max_value=500, value=96)
        
        # Calcular médias móveis
        ts_data_ma = ts_data.copy()
        ts_data_ma[f'MA_{ma_short}'] = ts_data_ma.rolling(window=ma_short).mean()
        ts_data_ma[f'MA_{ma_long}'] = ts_data_ma.rolling(window=ma_long).mean()
        
        # Regressão linear
        from sklearn.linear_model import LinearRegression
        X = np.arange(len(ts_data)).reshape(-1, 1)
        y = ts_data.values
        model = LinearRegression()
        model.fit(X, y)
        trend_line = model.predict(X)
        
        # Visualização
        fig_trend = go.Figure()
        
        # Dados originais
        fig_trend.add_trace(go.Scatter(
            x=ts_data.index,
            y=ts_data.values,
            mode='lines',
            name='Dados Originais',
            line=dict(color='lightgray', width=1),
            opacity=0.5
        ))
        
        # Médias móveis
        fig_trend.add_trace(go.Scatter(
            x=ts_data.index,
            y=ts_data_ma[f'MA_{ma_short}'],
            mode='lines',
            name=f'Média Móvel {ma_short} períodos',
            line=dict(color='blue', width=2)
        ))
        
        fig_trend.add_trace(go.Scatter(
            x=ts_data.index,
            y=ts_data_ma[f'MA_{ma_long}'],
            mode='lines',
            name=f'Média Móvel {ma_long} períodos',
            line=dict(color='red', width=2)
        ))
        
        # Tendência linear
        fig_trend.add_trace(go.Scatter(
            x=ts_data.index,
            y=trend_line,
            mode='lines',
            name='Tendência Linear',
            line=dict(color='green', width=2, dash='dash')
        ))
        
        fig_trend.update_layout(
            title=f"Análise de Tendências - {selected_param}",
            xaxis_title="Tempo",
            yaxis_title=selected_param,
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Métricas de tendência
        col1, col2, col3 = st.columns(3)
        
        with col1:
            slope = model.coef_[0]
            st.metric("Taxa de Mudança", f"{slope:.4f}", "unidades/obs")
            
        with col2:
            r2 = model.score(X, y)
            st.metric("R² da Tendência", f"{r2:.3f}")
            
        with col3:
            direction = "Crescente" if slope > 0 else "Decrescente"
            st.metric("Direção", direction)
    
    elif analysis_type == "Análise Espectral":
        st.subheader("🔍 Análise Espectral")
        
        # Análise de frequências usando FFT
        from scipy.fft import fft, fftfreq
        
        # Remover tendência
        ts_detrended = ts_data - ts_data.rolling(window=96).mean()
        ts_detrended = ts_detrended.dropna()
        
        if len(ts_detrended) > 0:
            # FFT
            N = len(ts_detrended)
            yf = fft(ts_detrended.values)
            xf = fftfreq(N, 1/96)  # 96 observações por dia
            
            # Plotar espectro de potência
            fig_spectrum = go.Figure()
            fig_spectrum.add_trace(go.Scatter(
                x=xf[:N//2],
                y=np.abs(yf[:N//2]),
                mode='lines',
                name='Espectro de Potência'
            ))
            
            fig_spectrum.update_layout(
                title="Espectro de Potência",
                xaxis_title="Frequência (ciclos/dia)",
                yaxis_title="Magnitude",
                height=400
            )
            
            st.plotly_chart(fig_spectrum, use_container_width=True)
            
            # Identificar frequências dominantes
            power_spectrum = np.abs(yf[:N//2])
            dominant_freq_idx = np.argmax(power_spectrum[1:]) + 1  # Excluir DC component
            dominant_freq = xf[dominant_freq_idx]
            dominant_period = 1 / dominant_freq if dominant_freq > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Frequência Dominante", f"{dominant_freq:.3f}", "ciclos/dia")
                
            with col2:
                st.metric("Período Dominante", f"{dominant_period:.1f}", "dias")
                
            with col3:
                st.metric("Potência Máxima", f"{power_spectrum[dominant_freq_idx]:.1f}")

else:
    st.error(f"Parâmetro {selected_param} não encontrado nos dados da estação {selected_station}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
    Página de Análise de Séries Temporais | Sistema de Monitoramento v2.0 | 
    Última atualização: {:%Y-%m-%d %H:%M:%S}
    </div>
    """.format(datetime.now()),
    unsafe_allow_html=True
)
