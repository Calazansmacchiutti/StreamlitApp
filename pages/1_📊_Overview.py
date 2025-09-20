"""
Página de Visão Geral do Dashboard
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
    from modules.visualizations import DashboardVisualizer
except ImportError as e:
    st.error(f"Erro ao importar módulos: {e}")
    st.stop()

st.set_page_config(
    page_title="Visão Geral - Sistema de Monitoramento",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Visão Geral do Sistema")

# Inicializar componentes
data_loader = DataLoader()
visualizer = DashboardVisualizer()

# Configurações das estações
stations_config = {
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
}

# Sidebar para seleção
with st.sidebar:
    st.header("⚙️ Configurações")
    
    # Seleção de estação
    selected_station = st.selectbox(
        "📍 Selecionar Estação",
        list(stations_config.keys())
    )
    
    # Período de análise
    time_period = st.selectbox(
        "📅 Período",
        ["Última hora", "Últimas 24 horas", "Última semana", "Último mês"]
    )
    
    # Parâmetros
    station_params = stations_config[selected_station]['parameters']
    selected_params = st.multiselect(
        "📊 Parâmetros",
        station_params,
        default=station_params[:3]
    )

# Carregar dados
@st.cache_data
def load_station_data(station_name, days=30):
    """Carregar dados da estação"""
    return data_loader.generate_synthetic_data(station_name, days)

# Carregar dados
df = load_station_data(selected_station)

# Filtrar por período
if time_period == "Última hora":
    df = df.tail(4)  # 4 pontos = 1 hora
elif time_period == "Últimas 24 horas":
    df = df.tail(96)  # 96 pontos = 24 horas
elif time_period == "Última semana":
    df = df.tail(672)  # 672 pontos = 1 semana
elif time_period == "Último mês":
    df = df  # Todos os dados

# KPIs principais
st.header("📈 Indicadores Principais")

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
    # Taxa de conformidade simulada
    compliance = 94.7 + np.random.normal(0, 2)
    st.metric(
        "Taxa de Conformidade",
        f"{compliance:.1f}%",
        f"{compliance - 90:+.1f}%",
        delta_color="normal" if compliance > 90 else "inverse"
    )

# Status das estações
st.header("🏭 Status das Estações")

status_col1, status_col2, status_col3 = st.columns(3)

with status_col1:
    st.metric("Two Mouths", "🟢 Operacional", "100%")
    
with status_col2:
    st.metric("New Rose of Rocky", "🟢 Operacional", "100%")
    
with status_col3:
    st.metric("Botanic Garden", "🟢 Operacional", "100%")

# Gráficos de monitoramento
st.header("📊 Monitoramento em Tempo Real")

if selected_params:
    # Gráfico principal
    fig = visualizer.create_realtime_chart(df, selected_params)
    st.plotly_chart(fig, use_container_width=True)
    
    # Gráficos individuais com limites
    st.subheader("📈 Parâmetros com Limites de Controle")
    
    thresholds = {
        'pH': {'min': 6.5, 'max': 8.5, 'critical_min': 6.0, 'critical_max': 9.0},
        'turbidity': {'max': 5.0, 'critical_max': 10.0},
        'chlorine': {'min': 0.5, 'max': 2.0, 'critical_min': 0.3, 'critical_max': 3.0},
        'temperature': {'min': 15, 'max': 30, 'critical_min': 10, 'critical_max': 35},
        'flow_rate': {'min': 500, 'max': 1500, 'critical_min': 300, 'critical_max': 2000},
        'DO': {'min': 5.0, 'critical_min': 3.0},
        'BOD': {'max': 30, 'critical_max': 50},
        'COD': {'max': 100, 'critical_max': 150},
        'TSS': {'max': 50, 'critical_max': 100},
        'nitrogen': {'max': 10, 'critical_max': 20},
        'phosphorus': {'max': 2, 'critical_max': 5},
        'coliform': {'max': 100, 'critical_max': 1000}
    }
    
    for param in selected_params:
        if param in df.columns and param in thresholds:
            fig_threshold = visualizer.create_threshold_chart(
                df, param, thresholds[param]
            )
            st.plotly_chart(fig_threshold, use_container_width=True)

# Matriz de correlação
if len(selected_params) > 1:
    st.header("🔥 Matriz de Correlação")
    fig_corr = visualizer.create_correlation_heatmap(df, selected_params)
    st.plotly_chart(fig_corr, use_container_width=True)

# Alertas recentes
st.header("⚠️ Alertas Recentes")

# Simular alertas
alerts_data = {
    'timestamp': [datetime.now() - timedelta(hours=i) for i in range(5)],
    'station': [selected_station] * 5,
    'parameter': ['pH', 'turbidity', 'chlorine', 'flow_rate', 'temperature'][:5],
    'severity': ['medium', 'low', 'high', 'low', 'medium'],
    'message': [
        'pH ligeiramente acima do normal',
        'Turbidez dentro dos limites',
        'Cloro residual baixo',
        'Vazão normal',
        'Temperatura estável'
    ]
}

alerts_df = pd.DataFrame(alerts_data)

# Cores por severidade
severity_colors = {
    'low': '🟢',
    'medium': '🟡', 
    'high': '🟠',
    'critical': '🔴'
}

for _, alert in alerts_df.iterrows():
    with st.container():
        col1, col2, col3, col4 = st.columns([1, 2, 2, 3])
        
        with col1:
            st.write(severity_colors.get(alert['severity'], '⚪'))
            
        with col2:
            st.write(f"**{alert['parameter']}**")
            
        with col3:
            st.write(alert['timestamp'].strftime("%H:%M"))
            
        with col4:
            st.write(alert['message'])

# Resumo de performance
st.header("📊 Resumo de Performance")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Eficiência por Parâmetro")
    
    # Dados simulados de eficiência
    efficiency_data = {
        'Parameter': selected_params,
        'Efficiency': [95 + np.random.normal(0, 3) for _ in selected_params]
    }
    
    fig_eff = px.bar(
        efficiency_data, 
        x='Parameter', 
        y='Efficiency',
        title="Eficiência de Tratamento por Parâmetro (%)",
        color='Efficiency',
        color_continuous_scale='RdYlGn'
    )
    st.plotly_chart(fig_eff, use_container_width=True)

with col2:
    st.subheader("Tendências dos Últimos 7 Dias")
    
    # Simular tendências
    days = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='D')
    trend_data = {
        'Date': days,
        'Compliance': [94 + np.random.normal(0, 2) for _ in days],
        'Efficiency': [96 + np.random.normal(0, 1.5) for _ in days]
    }
    
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=trend_data['Date'], 
        y=trend_data['Compliance'],
        mode='lines+markers',
        name='Conformidade (%)',
        line=dict(color='blue')
    ))
    fig_trend.add_trace(go.Scatter(
        x=trend_data['Date'], 
        y=trend_data['Efficiency'],
        mode='lines+markers',
        name='Eficiência (%)',
        line=dict(color='green'),
        yaxis='y2'
    ))
    
    fig_trend.update_layout(
        title="Tendências de Performance",
        xaxis_title="Data",
        yaxis=dict(title="Conformidade (%)", side="left"),
        yaxis2=dict(title="Eficiência (%)", side="right", overlaying="y"),
        height=400
    )
    
    st.plotly_chart(fig_trend, use_container_width=True)

# Informações da estação
st.header("ℹ️ Informações da Estação")

station_info = stations_config[selected_station]

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Tipo", station_info['type'].title())
    
with col2:
    st.metric("Capacidade", f"{station_info['capacity']:,} m³/dia")
    
with col3:
    st.metric("Parâmetros Monitorados", len(station_info['parameters']))

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
    Página de Visão Geral | Sistema de Monitoramento v2.0 | 
    Última atualização: {:%Y-%m-%d %H:%M:%S}
    </div>
    """.format(datetime.now()),
    unsafe_allow_html=True
)
