"""
Sistema de Monitoramento de Estações de Tratamento - Aplicação Unificada
Versão consolidada que combina funcionalidades simples e avançadas
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
from typing import Dict, List, Optional, Tuple
import scipy.stats as stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

# Adicionar o diretório modules ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

# Configuração da página
st.set_page_config(
    page_title="Sistema de Monitoramento - Estações de Tratamento",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.example.com/help',
        'Report a bug': "https://www.example.com/bug",
        'About': "Sistema de Monitoramento v3.0 - Versão Unificada"
    }
)

# Constantes de configuração
ALLOWED_STATIONS = {'Two Mouths', 'New Rose of Rocky', 'Botanic Garden'}
DEFAULT_DATA_PATH = Path(__file__).parent / "data"
CONFIG_PATH = Path(__file__).parent / "config"

# CSS personalizado
CUSTOM_CSS = """
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
    .mode-selector {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
"""

class UnifiedWaterTreatmentApp:
    """Aplicação unificada para monitoramento de estações de tratamento"""

    def __init__(self):
        self.initialize_session_state()
        self.load_configuration()
        self.setup_modules()

    def initialize_session_state(self):
        """Inicializar variáveis de sessão"""
        defaults = {
            'app_mode': 'Simples',
            'data_loaded': False,
            'current_station': None,
            'selected_parameters': [],
            'date_range': None,
            'last_update': datetime.now()
        }

        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    def load_configuration(self):
        """Carregar configurações do sistema"""
        config_file = CONFIG_PATH / 'config.yaml'
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as file:
                self.config = yaml.safe_load(file)
        else:
            self.config = self.get_default_config()

    def get_default_config(self) -> Dict:
        """Configurações padrão do sistema"""
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
            }
        }

    def setup_modules(self):
        """Configurar módulos do sistema"""
        try:
            from modules.data_loader import DataLoader
            from modules.preprocessor import DataPreprocessor
            from modules.time_series import TimeSeriesAnalyzer
            from modules.anomaly_detection import AnomalyDetector
            from modules.visualizations import DashboardVisualizer
            from modules.alerts import AlertSystem

            self.data_loader = DataLoader()
            self.preprocessor = DataPreprocessor()
            self.time_series = TimeSeriesAnalyzer()
            self.anomaly_detector = AnomalyDetector()
            self.visualizer = DashboardVisualizer()
            self.alert_system = AlertSystem()
            self.modules_loaded = True
        except ImportError as e:
            st.error(f"Erro ao carregar módulos avançados: {e}")
            self.modules_loaded = False

    @st.cache_data
    def load_station_data(_self, station_name: str) -> pd.DataFrame:
        """Carregar dados da estação com cache otimizado"""
        if station_name not in ALLOWED_STATIONS:
            st.error(f"Estação '{station_name}' não permitida")
            return pd.DataFrame()

        file_path = DEFAULT_DATA_PATH / f"{station_name}.csv"

        if not file_path.exists():
            st.warning(f"Arquivo não encontrado: {file_path}")
            return pd.DataFrame()

        try:
            df = pd.read_csv(file_path)

            # Converter timestamp
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            else:
                # Encontrar coluna de data automaticamente
                date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
                if date_columns:
                    df['timestamp'] = pd.to_datetime(df[date_columns[0]])
                else:
                    # Gerar timestamps
                    df['timestamp'] = pd.date_range(
                        start=datetime.now() - timedelta(days=30),
                        periods=len(df),
                        freq='15T'
                    )

            return df

        except Exception as e:
            st.error(f"Erro ao carregar dados: {str(e)}")
            return pd.DataFrame()

    def render_header(self):
        """Renderizar cabeçalho da aplicação"""
        st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

        # Título principal
        st.markdown('<h1 class="main-header">Sistema de Monitoramento de Estações de Tratamento</h1>',
                   unsafe_allow_html=True)

        # Seletor de modo
        with st.container():
            st.markdown('<div class="mode-selector">', unsafe_allow_html=True)

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                mode = st.radio(
                    "Selecione o modo de operação:",
                    ["Simples", "Avançado"],
                    horizontal=True,
                    key="app_mode"
                )

            st.markdown('</div>', unsafe_allow_html=True)

        return mode

    def render_simple_mode(self):
        """Renderizar interface simples"""
        st.markdown("## 📊 Modo Simples - Visualização Básica")

        # Seleção da estação
        col1, col2 = st.columns([1, 2])

        with col1:
            station = st.selectbox(
                "Selecionar Estação:",
                list(ALLOWED_STATIONS),
                key="simple_station"
            )

        with col2:
            # Carregar dados
            df = self.load_station_data(station)
            if df.empty:
                st.error("Não foi possível carregar os dados")
                return

        # Seleção de parâmetros
        available_params = [col for col in df.columns if col != 'timestamp']
        selected_params = st.multiselect(
            "Selecionar parâmetros para visualização:",
            available_params,
            default=available_params[:3] if len(available_params) >= 3 else available_params
        )

        if not selected_params:
            st.warning("Selecione pelo menos um parâmetro")
            return

        # Filtro de tempo
        st.markdown("### 📅 Período de Análise")
        min_date = df['timestamp'].min().to_pydatetime()
        max_date = df['timestamp'].max().to_pydatetime()

        date_range = st.slider(
            'Selecione o intervalo de tempo:',
            min_value=min_date,
            max_value=max_date,
            value=[min_date, max_date],
            format="DD/MM/YYYY"
        )

        # Filtrar dados
        filtered_df = df[(df['timestamp'] >= date_range[0]) & (df['timestamp'] <= date_range[1])]

        # Tipo de visualização
        chart_type = st.selectbox(
            "Tipo de gráfico:",
            ["Linha", "Matriz de Correlação", "Box Plot", "Histograma"]
        )

        # Renderizar gráficos
        self.render_simple_charts(filtered_df, selected_params, chart_type)

    def render_simple_charts(self, df: pd.DataFrame, params: List[str], chart_type: str):
        """Renderizar gráficos do modo simples"""
        if chart_type == "Linha":
            fig = go.Figure()
            for param in params:
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df[param],
                    mode='lines',
                    name=param
                ))
            fig.update_layout(
                title="Séries Temporais dos Parâmetros",
                xaxis_title="Tempo",
                yaxis_title="Valores"
            )
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Matriz de Correlação":
            if len(params) > 1:
                corr_matrix = df[params].corr()
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='RdBu_r',
                    title="Matriz de Correlação"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Selecione mais de um parâmetro para correlação")

        elif chart_type == "Box Plot":
            melted_df = df.melt(
                id_vars=['timestamp'],
                value_vars=params,
                var_name='Parâmetro',
                value_name='Valor'
            )
            fig = px.box(
                melted_df,
                x='Parâmetro',
                y='Valor',
                title="Distribuição dos Parâmetros"
            )
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Histograma":
            for param in params:
                fig = px.histogram(
                    df,
                    x=param,
                    nbins=30,
                    title=f'Histograma - {param}'
                )
                st.plotly_chart(fig, use_container_width=True)

    def render_advanced_mode(self):
        """Renderizar interface avançada"""
        if not self.modules_loaded:
            st.error("Módulos avançados não disponíveis. Execute: pip install -r requirements.txt")
            return

        st.markdown("## 🚀 Modo Avançado - Dashboard Completo")

        # Importar funcionalidades do app.py original
        try:
            from app import WaterTreatmentDashboard
            dashboard = WaterTreatmentDashboard()

            # Renderizar sidebar
            station, params, date_option = dashboard.render_sidebar()

            # Carregar dados
            df = dashboard.load_station_data(station, st.session_state.date_range)

            if not df.empty:
                # Renderizar dashboard principal
                dashboard.render_main_dashboard(station, params, df)
            else:
                st.error("Erro ao carregar dados da estação")

        except Exception as e:
            st.error(f"Erro no modo avançado: {str(e)}")
            st.info("Usando funcionalidades básicas do modo avançado")
            self.render_basic_advanced_mode()

    def render_basic_advanced_mode(self):
        """Versão básica do modo avançado quando módulos não estão disponíveis"""
        st.markdown("### Funcionalidades Avançadas Básicas")

        # Seleção da estação
        station = st.selectbox(
            "Selecionar Estação:",
            list(ALLOWED_STATIONS),
            key="advanced_station"
        )

        df = self.load_station_data(station)
        if df.empty:
            return

        # Tabs para diferentes análises
        tab1, tab2, tab3 = st.tabs(["📊 Visão Geral", "📈 Análise Temporal", "📋 Estatísticas"])

        with tab1:
            self.render_overview_basic(df)

        with tab2:
            self.render_time_analysis_basic(df)

        with tab3:
            self.render_statistics_basic(df)

    def render_overview_basic(self, df: pd.DataFrame):
        """Visão geral básica"""
        st.markdown("#### Resumo dos Dados")

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        # Métricas principais
        cols = st.columns(min(4, len(numeric_cols)))
        for i, col in enumerate(numeric_cols[:4]):
            with cols[i]:
                mean_val = df[col].mean()
                std_val = df[col].std()
                st.metric(
                    label=col.title(),
                    value=f"{mean_val:.2f}",
                    delta=f"±{std_val:.2f}"
                )

        # Gráfico de linha de todos os parâmetros
        if len(numeric_cols) > 0:
            fig = go.Figure()
            for col in numeric_cols:
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df[col],
                    mode='lines',
                    name=col
                ))
            fig.update_layout(title="Todos os Parâmetros - Série Temporal")
            st.plotly_chart(fig, use_container_width=True)

    def render_time_analysis_basic(self, df: pd.DataFrame):
        """Análise temporal básica"""
        st.markdown("#### Análise de Séries Temporais")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        selected_param = st.selectbox("Selecionar parâmetro:", numeric_cols)

        if selected_param:
            # Decomposição sazonal básica
            try:
                ts_data = df.set_index('timestamp')[selected_param].dropna()
                if len(ts_data) > 24:  # Mínimo para decomposição
                    period = min(96, len(ts_data) // 2)  # 96 = 24h com dados de 15min
                    decomposition = seasonal_decompose(ts_data, model='additive', period=period)

                    # Plot da decomposição
                    fig = make_subplots(
                        rows=4, cols=1,
                        shared_xaxes=True,
                        subplot_titles=("Original", "Tendência", "Sazonalidade", "Resíduos")
                    )

                    fig.add_trace(go.Scatter(x=ts_data.index, y=ts_data, name='Original'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, name='Tendência'), row=2, col=1)
                    fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, name='Sazonalidade'), row=3, col=1)
                    fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, name='Resíduos'), row=4, col=1)

                    fig.update_layout(height=800, title=f"Decomposição Sazonal - {selected_param}")
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.warning(f"Não foi possível realizar decomposição: {str(e)}")

    def render_statistics_basic(self, df: pd.DataFrame):
        """Estatísticas básicas"""
        st.markdown("#### Estatísticas Descritivas")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats_df = df[numeric_cols].describe()
            st.dataframe(stats_df, use_container_width=True)

            # Teste de normalidade
            st.markdown("#### Testes de Normalidade (Shapiro-Wilk)")
            for col in numeric_cols:
                try:
                    stat, p_value = stats.shapiro(df[col].dropna()[:5000])  # Máximo 5000 amostras
                    is_normal = p_value > 0.05
                    st.write(f"**{col}**: p-value = {p_value:.6f} {'✅ Normal' if is_normal else '❌ Não Normal'}")
                except:
                    st.write(f"**{col}**: Não foi possível calcular")

    def run(self):
        """Executar a aplicação"""
        # Renderizar cabeçalho e obter modo
        mode = self.render_header()

        # Barra de status
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Modo Atual", mode)
        with col2:
            st.metric("Estações Disponíveis", len(ALLOWED_STATIONS))
        with col3:
            st.metric("Última Atualização", datetime.now().strftime("%H:%M:%S"))

        st.markdown("---")

        # Renderizar interface baseada no modo
        if mode == "Simples":
            self.render_simple_mode()
        else:
            self.render_advanced_mode()

        # Footer
        st.markdown("---")
        st.markdown(
            f"""
            <div style='text-align: center; color: gray; font-size: 0.8em;'>
            Sistema de Monitoramento v3.0 - Versão Unificada |
            Desenvolvido com ❤️ usando Streamlit |
            Última atualização: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            </div>
            """,
            unsafe_allow_html=True
        )

# Executar aplicação
if __name__ == "__main__":
    app = UnifiedWaterTreatmentApp()
    app.run()