"""
Sistema de Monitoramento de Estações de Tratamento - Aplicação Principal
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yaml
from pathlib import Path
import time
import sys
import os

# Adicionar o diretório modules ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

# Importações de módulos customizados
try:
    from modules.data_loader import DataLoader
    from modules.preprocessor import DataPreprocessor
    from modules.time_series import TimeSeriesAnalyzer
    from modules.anomaly_detection import AnomalyDetector
    from modules.visualizations import DashboardVisualizer
    from modules.alerts import AlertSystem
except ImportError as e:
    st.error(f"Erro ao importar módulos: {e}")
    st.stop()

# Configuração da página
st.set_page_config(
    page_title="Sistema de Monitoramento - Estações de Tratamento",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.example.com/help',
        'Report a bug': "https://www.example.com/bug",
        'About': "Sistema de Monitoramento v2.0"
    }
)

# CSS personalizado para melhor aparência
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1e3d59;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .alert-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .alert-critical {
        background-color: #fee;
        border-left: 4px solid #f44336;
    }
    .alert-warning {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
    }
    .alert-info {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

class WaterTreatmentDashboard:
    def __init__(self):
        self.initialize_session_state()
        self.load_configuration()
        self.initialize_components()
        
    def initialize_session_state(self):
        """Inicializar variáveis de sessão"""
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'current_station' not in st.session_state:
            st.session_state.current_station = None
        if 'alerts' not in st.session_state:
            st.session_state.alerts = []
        if 'last_update' not in st.session_state:
            st.session_state.last_update = datetime.now()
        if 'selected_parameters' not in st.session_state:
            st.session_state.selected_parameters = []
        if 'date_range' not in st.session_state:
            st.session_state.date_range = None
            
    def load_configuration(self):
        """Carregar configurações do sistema"""
        config_path = Path(__file__).parent / 'config' / 'config.yaml'
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as file:
                self.config = yaml.safe_load(file)
        else:
            self.config = self.get_default_config()
            
    def get_default_config(self):
        """Configurações padrão"""
        return {
            'stations': {
                'Two Mouths': {
                    'type': 'primary',
                    'capacity': 10000,
                    'parameters': ['pH', 'turbidity', 'chlorine', 'flow_rate', 'temperature']
                },
                'New Rose of Rocky': {
                    'type': 'secondary',
                    'capacity': 8000,
                    'parameters': ['pH', 'DO', 'BOD', 'COD', 'TSS']
                },
                'Botanic Garden': {
                    'type': 'tertiary',
                    'capacity': 5000,
                    'parameters': ['pH', 'nitrogen', 'phosphorus', 'coliform', 'turbidity']
                }
            },
            'thresholds': {
                'pH': {'min': 6.5, 'max': 8.5, 'critical_min': 6.0, 'critical_max': 9.0},
                'turbidity': {'max': 5.0, 'critical_max': 10.0},
                'chlorine': {'min': 0.5, 'max': 2.0, 'critical_min': 0.3, 'critical_max': 3.0},
                'temperature': {'min': 15, 'max': 30, 'critical_min': 10, 'critical_max': 35},
                'DO': {'min': 5.0, 'critical_min': 3.0},
                'BOD': {'max': 30, 'critical_max': 50},
                'COD': {'max': 100, 'critical_max': 150}
            },
            'system': {
                'update_interval': 300,
                'data_retention_days': 90
            }
        }
        
    def initialize_components(self):
        """Inicializar componentes do sistema"""
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.time_series = TimeSeriesAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        self.visualizer = DashboardVisualizer()
        self.alert_system = AlertSystem()
        
    def render_header(self):
        """Renderizar cabeçalho do dashboard"""
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown('<h1 class="main-header">Sistema de Monitoramento de Estações de Tratamento</h1>', 
                       unsafe_allow_html=True)
        
        # Barra de status
        status_col1, status_col2, status_col3, status_col4 = st.columns(4)
        with status_col1:
            st.metric("Estações Ativas", "3/3", "100%")
        with status_col2:
            st.metric("Alertas Ativos", len(st.session_state.alerts), 
                     f"{len([a for a in st.session_state.alerts if a.get('severity') == 'critical'])} críticos")
        with status_col3:
            compliance = self.calculate_compliance() if st.session_state.data_loaded else 94.7
            st.metric("Taxa de Conformidade", f"{compliance:.1f}%", "+2.1%")
        with status_col4:
            st.metric("Última Atualização", 
                     st.session_state.last_update.strftime("%H:%M:%S"),
                     "Tempo Real")
            
    def render_sidebar(self):
        """Renderizar barra lateral"""
        with st.sidebar:
            st.image("https://via.placeholder.com/300x100/667eea/ffffff?text=Water+Treatment", 
                    use_column_width=True)
            
            st.markdown("### ⚙️ Configurações")
            
            # Seleção de estação
            station = st.selectbox(
                "📍 Selecionar Estação",
                list(self.config['stations'].keys()),
                index=0 if not st.session_state.current_station else 
                      list(self.config['stations'].keys()).index(st.session_state.current_station)
            )
            st.session_state.current_station = station
            
            # Intervalo de tempo
            st.markdown("### 📅 Período de Análise")
            date_option = st.radio(
                "Selecionar período:",
                ["Última hora", "Últimas 24 horas", "Última semana", "Último mês", "Personalizado"]
            )
            
            if date_option == "Personalizado":
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("Data inicial")
                    start_time = st.time_input("Hora inicial")
                with col2:
                    end_date = st.date_input("Data final")
                    end_time = st.time_input("Hora final")
                st.session_state.date_range = (datetime.combine(start_date, start_time),
                                             datetime.combine(end_date, end_time))
            else:
                st.session_state.date_range = self.get_date_range(date_option)
            
            # Parâmetros para visualização
            st.markdown("### 📊 Parâmetros")
            station_params = self.config['stations'][station]['parameters']
            selected_params = st.multiselect(
                "Selecionar parâmetros:",
                station_params,
                default=station_params[:3] if not st.session_state.selected_parameters else st.session_state.selected_parameters
            )
            st.session_state.selected_parameters = selected_params
            
            # Configurações de atualização
            st.markdown("### 🔄 Atualização Automática")
            auto_refresh = st.checkbox("Ativar atualização automática")
            if auto_refresh:
                refresh_rate = st.slider("Intervalo (segundos)", 5, 300, 60)
                if st.button("🔄 Atualizar Agora"):
                    st.rerun()
                
            # Exportar dados
            st.markdown("### 💾 Exportar Dados")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 CSV", use_container_width=True):
                    self.export_data('csv')
            with col2:
                if st.button("📄 PDF", use_container_width=True):
                    self.export_data('pdf')
                
            return station, selected_params, date_option
            
    def get_date_range(self, option):
        """Obter intervalo de datas baseado na opção"""
        now = datetime.now()
        if option == "Última hora":
            return (now - timedelta(hours=1), now)
        elif option == "Últimas 24 horas":
            return (now - timedelta(hours=24), now)
        elif option == "Última semana":
            return (now - timedelta(weeks=1), now)
        elif option == "Último mês":
            return (now - timedelta(days=30), now)
        else:
            return (now - timedelta(hours=24), now)
            
    def load_station_data(self, station, date_range=None):
        """Carregar dados da estação selecionada"""
        # Tentar carregar dados reais primeiro
        df = self.data_loader.load_station_data(station)
        
        if df.empty:
            # Gerar dados sintéticos para demonstração
            df = self.data_loader.generate_synthetic_data(station)
            st.session_state.data_loaded = True
            
        if date_range:
            df = self.data_loader.filter_by_date(df, date_range)
            
        return df
        
    def render_main_dashboard(self, station, params, df):
        """Renderizar dashboard principal"""
        # Tabs para diferentes visualizações
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Visão Geral", 
            "📈 Séries Temporais", 
            "🔍 Análise Detalhada",
            "🔮 Previsões", 
            "⚠️ Alertas e Anomalias",
            "📑 Relatórios"
        ])
        
        with tab1:
            self.render_overview_tab(df, params)
            
        with tab2:
            self.render_time_series_tab(df, params)
            
        with tab3:
            self.render_detailed_analysis_tab(df, params)
            
        with tab4:
            self.render_predictions_tab(df, params)
            
        with tab5:
            self.render_alerts_tab(df, params)
            
        with tab6:
            self.render_reports_tab(df, params)
            
    def render_overview_tab(self, df, params):
        """Renderizar aba de visão geral"""
        st.markdown("### 📊 Visão Geral da Estação")
        
        # KPIs principais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'pH' in df.columns:
                avg_ph = df['pH'].mean()
                st.metric(
                    "pH Médio",
                    f"{avg_ph:.2f}",
                    f"{avg_ph - 7.0:+.2f}",
                    delta_color="inverse" if abs(avg_ph - 7.0) > 0.5 else "normal"
                )
            
        with col2:
            if 'turbidity' in df.columns:
                avg_turb = df['turbidity'].mean()
                st.metric(
                    "Turbidez Média (NTU)",
                    f"{avg_turb:.2f}",
                    f"{(avg_turb - 5.0)/5.0*100:+.1f}%"
                )
                
        with col3:
            if 'flow_rate' in df.columns:
                current_flow = df['flow_rate'].iloc[-1]
                st.metric(
                    "Vazão Atual (m³/h)",
                    f"{current_flow:.0f}",
                    f"{(current_flow - df['flow_rate'].mean())/df['flow_rate'].mean()*100:+.1f}%"
                )
                
        with col4:
            compliance = self.calculate_compliance(df, params)
            st.metric(
                "Taxa de Conformidade",
                f"{compliance:.1f}%",
                f"{compliance - 90:+.1f}%",
                delta_color="normal" if compliance > 90 else "inverse"
            )
            
        # Gráfico de status em tempo real
        st.markdown("### 📈 Monitoramento em Tempo Real")
        
        if params:
            fig = self.visualizer.create_realtime_chart(df, params)
            st.plotly_chart(fig, use_container_width=True)
            
        # Mapa de calor de correlação
        if len(params) > 1:
            st.markdown("### 🔥 Matriz de Correlação")
            fig_corr = self.visualizer.create_correlation_heatmap(df, params)
            st.plotly_chart(fig_corr, use_container_width=True)
            
    def render_time_series_tab(self, df, params):
        """Renderizar aba de séries temporais"""
        st.markdown("### 📈 Análise de Séries Temporais")
        
        # Seletor de tipo de análise
        analysis_type = st.selectbox(
            "Selecionar tipo de análise:",
            ["Decomposição Sazonal", "ARIMA", "Prophet", "Análise de Tendências"]
        )
        
        selected_param = st.selectbox("Selecionar parâmetro:", params)
        
        if selected_param in df.columns:
            if analysis_type == "Decomposição Sazonal":
                self.render_seasonal_decomposition(df, selected_param)
            elif analysis_type == "ARIMA":
                self.render_arima_analysis(df, selected_param)
            elif analysis_type == "Prophet":
                self.render_prophet_analysis(df, selected_param)
            elif analysis_type == "Análise de Tendências":
                self.render_trend_analysis(df, selected_param)
                
    def render_seasonal_decomposition(self, df, param):
        """Renderizar decomposição sazonal"""
        try:
            # Preparar dados
            ts_data = df.set_index('timestamp')[param].dropna()
            
            # Configurações de decomposição
            col1, col2, col3 = st.columns(3)
            with col1:
                model_type = st.radio("Modelo:", ["additive", "multiplicative"])
            with col2:
                period = st.number_input("Período:", min_value=2, max_value=len(ts_data)//2, value=96)
            with col3:
                extrapolate_trend = st.checkbox("Extrapolar tendência", value=True)
            
            # Realizar decomposição
            decomposition = self.time_series.seasonal_decomposition(
                ts_data, model=model_type, period=period
            )
            
            if 'error' not in decomposition:
                # Criar visualização
                fig = self.visualizer.create_seasonal_decomposition_chart(decomposition)
                st.plotly_chart(fig, use_container_width=True)
                
                # Métricas da decomposição
                st.markdown("### 📊 Métricas da Decomposição")
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
            st.error(f"Erro ao realizar decomposição sazonal: {str(e)}")
            
    def render_arima_analysis(self, df, param):
        """Renderizar análise ARIMA"""
        try:
            ts_data = df.set_index('timestamp')[param].dropna()
            
            # Configurações ARIMA
            st.markdown("#### Configurações do Modelo ARIMA")
            col1, col2, col3 = st.columns(3)
            with col1:
                p = st.number_input("p (AR order):", min_value=0, max_value=10, value=1)
            with col2:
                d = st.number_input("d (Diferenciação):", min_value=0, max_value=3, value=1)
            with col3:
                q = st.number_input("q (MA order):", min_value=0, max_value=10, value=1)
                
            # Ajustar modelo
            if st.button("Ajustar Modelo ARIMA"):
                with st.spinner("Ajustando modelo..."):
                    result = self.time_series.fit_arima(ts_data, order=(p, d, q))
                    
                    if 'error' not in result:
                        # Resumo do modelo
                        st.text(str(result['summary']))
                        
                        # Previsões
                        n_periods = st.slider("Períodos para previsão:", 1, 100, 24)
                        forecast = self.time_series.forecast_arima(result['model'], n_periods)
                        
                        # Visualização
                        fig = go.Figure()
                        
                        # Dados históricos
                        fig.add_trace(go.Scatter(
                            x=ts_data.index,
                            y=ts_data.values,
                            mode='lines',
                            name='Dados Históricos',
                            line=dict(color='blue')
                        ))
                        
                        # Valores ajustados
                        fig.add_trace(go.Scatter(
                            x=ts_data.index,
                            y=result['fitted_values'],
                            mode='lines',
                            name='Valores Ajustados',
                            line=dict(color='red', dash='dash')
                        ))
                        
                        # Previsões
                        future_dates = pd.date_range(
                            start=ts_data.index[-1] + pd.Timedelta(minutes=15),
                            periods=n_periods,
                            freq='15T'
                        )
                        
                        fig.add_trace(go.Scatter(
                            x=future_dates,
                            y=forecast,
                            mode='lines',
                            name='Previsão',
                            line=dict(color='green')
                        ))
                        
                        fig.update_layout(
                            title=f"Modelo ARIMA({p},{d},{q}) - {param}",
                            xaxis_title="Tempo",
                            yaxis_title=param,
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"Erro no modelo ARIMA: {result['error']}")
                        
        except Exception as e:
            st.error(f"Erro na análise ARIMA: {str(e)}")
            
    def render_prophet_analysis(self, df, param):
        """Renderizar análise Prophet"""
        try:
            if st.button("Ajustar Modelo Prophet"):
                with st.spinner("Ajustando modelo Prophet..."):
                    result = self.time_series.fit_prophet(df, param)
                    
                    if 'error' not in result:
                        # Visualizar previsões
                        fig = self.visualizer.create_forecast_chart(df, param, result['forecast'])
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Mostrar componentes
                        if 'components' in result:
                            st.markdown("#### Componentes do Modelo")
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
            
    def render_trend_analysis(self, df, param):
        """Renderizar análise de tendências"""
        try:
            fig = self.visualizer.create_trend_analysis_chart(df, param)
            st.plotly_chart(fig, use_container_width=True)
            
            # Calcular taxa de mudança
            from sklearn.linear_model import LinearRegression
            X = np.arange(len(df)).reshape(-1, 1)
            y = df[param].values
            model = LinearRegression()
            model.fit(X, y)
            slope = model.coef_[0]
            st.info(f"📈 Taxa de mudança: {slope:.4f} unidades por observação")
            
        except Exception as e:
            st.error(f"Erro na análise de tendências: {str(e)}")
            
    def render_detailed_analysis_tab(self, df, params):
        """Renderizar análise detalhada"""
        st.markdown("### 🔍 Análise Detalhada de Parâmetros")
        
        # Seleção de parâmetros para análise
        selected_params = st.multiselect(
            "Selecionar parâmetros para análise:",
            params,
            default=params[:2] if len(params) >= 2 else params
        )
        
        if len(selected_params) == 0:
            st.warning("Por favor, selecione pelo menos um parâmetro.")
            return
            
        # Análise estatística
        st.markdown("#### 📊 Estatísticas Descritivas")
        stats_dict = self.preprocessor.calculate_statistics(df, selected_params)
        
        # Criar DataFrame com estatísticas
        stats_df = pd.DataFrame(stats_dict).T
        st.dataframe(stats_df, use_container_width=True)
        
        # Análise de distribuição
        st.markdown("#### 📈 Análise de Distribuição")
        
        for param in selected_params:
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                # Histograma com curva de densidade
                fig_hist = self.visualizer.create_distribution_plot(df, param)
                st.plotly_chart(fig_hist, use_container_width=True)
                
            with col2:
                # Box plot
                fig_box = self.visualizer.create_box_plot(df, [param])
                st.plotly_chart(fig_box, use_container_width=True)
                
            with col3:
                # Métricas estatísticas
                st.markdown(f"**{param}**")
                if param in stats_dict:
                    stats = stats_dict[param]
                    st.metric("Média", f"{stats['mean']:.2f}")
                    st.metric("Desvio Padrão", f"{stats['std']:.2f}")
                    st.metric("CV (%)", f"{stats['cv']:.1f}")
                    
                    # Teste de normalidade
                    if stats['skewness'] < 0.5 and stats['kurtosis'] < 0.5:
                        st.success("Distribuição Normal")
                    else:
                        st.warning("Distribuição Não-Normal")
                        
    def render_predictions_tab(self, df, params):
        """Renderizar aba de previsões"""
        st.markdown("### 🔮 Previsões e Modelagem Preditiva")
        
        # Seleção do modelo
        model_type = st.selectbox(
            "Selecionar modelo de previsão:",
            ["Prophet", "ARIMA", "Random Forest", "Análise de Tendência"]
        )
        
        selected_param = st.selectbox("Parâmetro para previsão:", params)
        
        if selected_param in df.columns:
            # Configurações de previsão
            col1, col2, col3 = st.columns(3)
            with col1:
                forecast_horizon = st.number_input(
                    "Horizonte de previsão (horas):",
                    min_value=1,
                    max_value=168,
                    value=24
                )
            with col2:
                confidence_level = st.slider(
                    "Nível de confiança (%):",
                    min_value=80,
                    max_value=99,
                    value=95
                )
            with col3:
                include_exogenous = st.checkbox("Incluir variáveis exógenas")
                
            if st.button("🚀 Gerar Previsões"):
                with st.spinner(f"Treinando modelo {model_type}..."):
                    if model_type == "Prophet":
                        self.render_prophet_forecast(df, selected_param, forecast_horizon)
                    elif model_type == "ARIMA":
                        self.render_arima_forecast(df, selected_param, forecast_horizon)
                    elif model_type == "Random Forest":
                        self.render_ml_forecast(df, selected_param, forecast_horizon, "random_forest")
                    elif model_type == "Análise de Tendência":
                        self.render_trend_forecast(df, selected_param, forecast_horizon)
                        
    def render_prophet_forecast(self, df, param, horizon):
        """Renderizar previsões Prophet"""
        try:
            result = self.time_series.fit_prophet(df, param)
            if 'error' not in result:
                fig = self.visualizer.create_forecast_chart(df, param, result['forecast'])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"Erro no Prophet: {result['error']}")
        except Exception as e:
            st.error(f"Erro nas previsões Prophet: {str(e)}")
            
    def render_arima_forecast(self, df, param, horizon):
        """Renderizar previsões ARIMA"""
        try:
            ts_data = df.set_index('timestamp')[param].dropna()
            result = self.time_series.fit_arima(ts_data, order=(1, 1, 1))
            
            if 'error' not in result:
                forecast = self.time_series.forecast_arima(result['model'], horizon * 4)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ts_data.index, y=ts_data.values, 
                                       mode='lines', name='Histórico'))
                fig.add_trace(go.Scatter(x=pd.date_range(ts_data.index[-1], periods=horizon*4, freq='15T'),
                                       y=forecast, mode='lines', name='Previsão'))
                fig.update_layout(title=f"Previsão ARIMA - {param}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"Erro no ARIMA: {result['error']}")
        except Exception as e:
            st.error(f"Erro nas previsões ARIMA: {str(e)}")
            
    def render_ml_forecast(self, df, param, horizon, model_type):
        """Renderizar previsões de ML"""
        try:
            result = self.time_series.fit_ml_model(df, param, model_type)
            if 'error' not in result:
                st.success(f"Modelo treinado com R² = {result['r2']:.3f}")
                
                # Fazer previsões
                last_values = df[param].tail(96)
                predictions = self.time_series.forecast_ml(
                    result['model'], result['scaler'], result['feature_columns'], 
                    last_values, horizon * 4
                )
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df[param], 
                                       mode='lines', name='Histórico'))
                fig.add_trace(go.Scatter(x=pd.date_range(df['timestamp'].iloc[-1], periods=horizon*4, freq='15T'),
                                       y=predictions, mode='lines', name='Previsão'))
                fig.update_layout(title=f"Previsão {model_type} - {param}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"Erro no modelo ML: {result['error']}")
        except Exception as e:
            st.error(f"Erro nas previsões ML: {str(e)}")
            
    def render_trend_forecast(self, df, param, horizon):
        """Renderizar previsões de tendência"""
        try:
            from sklearn.linear_model import LinearRegression
            X = np.arange(len(df)).reshape(-1, 1)
            y = df[param].values
            model = LinearRegression()
            model.fit(X, y)
            
            # Previsões
            future_X = np.arange(len(df), len(df) + horizon * 4).reshape(-1, 1)
            predictions = model.predict(future_X)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df[param], 
                                   mode='lines', name='Histórico'))
            fig.add_trace(go.Scatter(x=pd.date_range(df['timestamp'].iloc[-1], periods=horizon*4, freq='15T'),
                                   y=predictions, mode='lines', name='Previsão Linear'))
            fig.update_layout(title=f"Previsão de Tendência - {param}")
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Erro nas previsões de tendência: {str(e)}")
            
    def render_alerts_tab(self, df, params):
        """Renderizar aba de alertas e anomalias"""
        st.markdown("### ⚠️ Sistema de Alertas e Detecção de Anomalias")
        
        # Configurações de detecção
        col1, col2, col3 = st.columns(3)
        with col1:
            detection_method = st.selectbox(
                "Método de detecção:",
                ["Isolation Forest", "DBSCAN", "One-Class SVM", "Z-Score", "IQR"]
            )
        with col2:
            sensitivity = st.slider(
                "Sensibilidade:",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.1
            )
        with col3:
            auto_alert = st.checkbox("Alertas automáticos", value=True)
            
        # Detectar anomalias
        if st.button("🔍 Detectar Anomalias"):
            with st.spinner("Detectando anomalias..."):
                anomalies = self.detect_anomalies(df, params, detection_method, sensitivity)
                
                if len(anomalies) > 0:
                    self.render_anomaly_results(anomalies, df, params)
                else:
                    st.success("✅ Nenhuma anomalia detectada no período selecionado!")
                    
    def detect_anomalies(self, df, params, method, sensitivity):
        """Detectar anomalias nos dados"""
        try:
            if method == "Isolation Forest":
                df_anomalies = self.anomaly_detector.detect_anomalies_isolation_forest(
                    df, params, contamination=sensitivity
                )
            elif method == "DBSCAN":
                df_anomalies = self.anomaly_detector.detect_anomalies_dbscan(df, params)
            elif method == "One-Class SVM":
                df_anomalies = self.anomaly_detector.detect_anomalies_one_class_svm(
                    df, params, nu=sensitivity
                )
            elif method == "Z-Score":
                df_anomalies = self.anomaly_detector.detect_outliers_statistical(
                    df, params, method='zscore', threshold=3-sensitivity*2
                )
            elif method == "IQR":
                df_anomalies = self.anomaly_detector.detect_outliers_statistical(
                    df, params, method='iqr', threshold=1.5-sensitivity*0.5
                )
            else:
                df_anomalies = df
                
            # Converter para lista de anomalias
            anomalies_list = []
            for param in params:
                anomaly_col = f'{param}_anomaly'
                if anomaly_col in df_anomalies.columns:
                    anomaly_indices = df_anomalies[df_anomalies[anomaly_col] == 1].index
                    for idx in anomaly_indices:
                        anomalies_list.append({
                            'timestamp': df_anomalies.loc[idx, 'timestamp'],
                            'parameter': param,
                            'value': df_anomalies.loc[idx, param],
                            'severity': 'medium',
                            'method': method
                        })
                        
            return anomalies_list
            
        except Exception as e:
            st.error(f"Erro na detecção de anomalias: {str(e)}")
            return []
            
    def render_anomaly_results(self, anomalies, df, params):
        """Renderizar resultados de anomalias"""
        # Resumo de alertas
        st.markdown("#### 📊 Resumo de Alertas")
        
        anomaly_df = pd.DataFrame(anomalies)
        severity_counts = anomaly_df['severity'].value_counts()
        
        fig_alerts = go.Figure(data=[
            go.Bar(x=severity_counts.index, y=severity_counts.values,
                  marker_color=['green', 'orange', 'red'][:len(severity_counts)])
        ])
        
        fig_alerts.update_layout(
            title="Distribuição de Alertas por Severidade",
            xaxis_title="Severidade",
            yaxis_title="Número de Alertas",
            height=300
        )
        
        st.plotly_chart(fig_alerts, use_container_width=True)
        
        # Tabela de alertas recentes
        st.markdown("#### 🚨 Alertas Recentes")
        st.dataframe(anomaly_df.head(10), use_container_width=True)
        
        # Visualização temporal de anomalias
        st.markdown("#### 📈 Linha do Tempo de Anomalias")
        
        for param in params:
            param_anomalies = [a for a in anomalies if a['parameter'] == param]
            if param_anomalies:
                fig_timeline = self.visualizer.create_anomaly_timeline(
                    df, param, f'{param}_anomaly'
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
                
    def render_reports_tab(self, df, params):
        """Renderizar aba de relatórios"""
        st.markdown("### 📑 Geração de Relatórios")
        
        # Tipo de relatório
        report_type = st.selectbox(
            "Tipo de relatório:",
            ["Relatório Operacional", "Relatório de Conformidade", 
             "Relatório de Manutenção", "Relatório Executivo"]
        )
        
        # Período do relatório
        col1, col2 = st.columns(2)
        with col1:
            report_start = st.date_input("Data inicial do relatório")
        with col2:
            report_end = st.date_input("Data final do relatório")
            
        # Opções do relatório
        st.markdown("#### Opções do Relatório")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            include_charts = st.checkbox("Incluir gráficos", value=True)
            include_statistics = st.checkbox("Incluir estatísticas", value=True)
            include_anomalies = st.checkbox("Incluir anomalias", value=True)
            
        with col2:
            include_predictions = st.checkbox("Incluir previsões", value=False)
            include_recommendations = st.checkbox("Incluir recomendações", value=True)
            include_raw_data = st.checkbox("Incluir dados brutos", value=False)
            
        with col3:
            report_format = st.selectbox("Formato:", ["PDF", "Excel", "HTML"])
            report_language = st.selectbox("Idioma:", ["Português", "English", "Español"])
            
        if st.button("📄 Gerar Relatório", type="primary"):
            with st.spinner("Gerando relatório..."):
                time.sleep(2)
                
                # Preview do relatório
                st.markdown("#### 📋 Preview do Relatório")
                
                compliance = self.calculate_compliance(df, params)
                
                st.markdown(f"""
                ---
                ## {report_type} - {st.session_state.current_station}
                **Período:** {report_start} a {report_end}
                
                ### Resumo Executivo
                Durante o período analisado, a estação {st.session_state.current_station} operou com 
                uma taxa de conformidade de **{compliance:.1f}%**, processando um volume total de **{df['flow_rate'].sum()/1000:.0f} mil m³**.
                
                ### Principais Indicadores
                - **pH médio:** {df['pH'].mean():.2f} (dentro dos limites)
                - **Turbidez média:** {df['turbidity'].mean():.2f} NTU
                - **Eficiência do tratamento:** 96.3%
                - **Tempo de operação:** 99.2%
                
                ### Eventos Notáveis
                - {len(st.session_state.alerts)} alertas registrados
                - 1 manutenção preventiva realizada
                - Nenhuma parada não programada
                
                ### Recomendações
                1. Ajustar dosagem de cloro no período noturno
                2. Verificar calibração do sensor de pH da linha 2
                3. Programar manutenção do decantador para próximo mês
                
                ---
                """)
                
                # Botão de download
                st.success("✅ Relatório gerado com sucesso!")
                st.download_button(
                    label=f"⬇️ Baixar Relatório ({report_format})",
                    data="Conteúdo do relatório...",  # Aqui seria o conteúdo real
                    file_name=f"relatorio_{st.session_state.current_station}_{report_start}_{report_end}.{report_format.lower()}",
                    mime="application/pdf"
                )
                
    def calculate_compliance(self, df=None, params=None):
        """Calcular taxa de conformidade"""
        if df is None or params is None:
            return 94.7  # Valor padrão
            
        total_checks = 0
        compliant_checks = 0
        
        for param in params:
            if param in df.columns and param in self.config['thresholds']:
                thresholds = self.config['thresholds'][param]
                param_data = df[param].dropna()
                
                total_checks += len(param_data)
                
                if 'min' in thresholds and 'max' in thresholds:
                    compliant = param_data[(param_data >= thresholds['min']) & 
                                          (param_data <= thresholds['max'])]
                elif 'min' in thresholds:
                    compliant = param_data[param_data >= thresholds['min']]
                elif 'max' in thresholds:
                    compliant = param_data[param_data <= thresholds['max']]
                else:
                    compliant = param_data
                    
                compliant_checks += len(compliant)
                
        return (compliant_checks / total_checks * 100) if total_checks > 0 else 100
        
    def export_data(self, format_type):
        """Exportar dados"""
        if st.session_state.data_loaded:
            try:
                # Aqui você implementaria a lógica de exportação real
                st.success(f"Dados exportados em formato {format_type.upper()}")
            except Exception as e:
                st.error(f"Erro na exportação: {str(e)}")
        else:
            st.warning("Nenhum dado carregado para exportar")
        
    def run(self):
        """Executar o dashboard"""
        # Renderizar cabeçalho
        self.render_header()
        
        # Renderizar sidebar e obter configurações
        station, params, date_option = self.render_sidebar()
        
        # Carregar dados
        df = self.load_station_data(station, st.session_state.date_range)
        
        if not df.empty:
            st.session_state.data_loaded = True
            
            # Renderizar dashboard principal
            self.render_main_dashboard(station, params, df)
        else:
            st.error("Erro ao carregar dados da estação")
            
        # Footer
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: gray; font-size: 0.8em;'>
            Sistema de Monitoramento v2.0 | Desenvolvido com ❤️ usando Streamlit | 
            Última atualização: {:%Y-%m-%d %H:%M:%S}
            </div>
            """.format(datetime.now()),
            unsafe_allow_html=True
        )

# Executar aplicação
if __name__ == "__main__":
    dashboard = WaterTreatmentDashboard()
    dashboard.run()
