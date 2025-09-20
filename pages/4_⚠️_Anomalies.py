"""
Página de Detecção de Anomalias e Alertas
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
    from modules.anomaly_detection import AnomalyDetector
    from modules.alerts import AlertSystem
    from modules.visualizations import DashboardVisualizer
except ImportError as e:
    st.error(f"Erro ao importar módulos: {e}")
    st.stop()

st.set_page_config(
    page_title="Anomalias e Alertas - Sistema de Monitoramento",
    page_icon="⚠️",
    layout="wide"
)

st.title("⚠️ Detecção de Anomalias e Alertas")

# Inicializar componentes
data_loader = DataLoader()
anomaly_detector = AnomalyDetector()
alert_system = AlertSystem()
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

# Limites de controle
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

# Sidebar para configurações
with st.sidebar:
    st.header("⚙️ Configurações de Detecção")
    
    # Seleção de estação
    selected_station = st.selectbox(
        "📍 Selecionar Estação",
        list(stations_config.keys())
    )
    
    # Seleção de parâmetros
    station_params = stations_config[selected_station]['parameters']
    selected_params = st.multiselect(
        "📊 Parâmetros para Análise",
        station_params,
        default=station_params
    )
    
    # Método de detecção
    detection_method = st.selectbox(
        "🔍 Método de Detecção",
        ["Isolation Forest", "DBSCAN", "One-Class SVM", "Z-Score", "IQR", "LSTM Autoencoder"]
    )
    
    # Sensibilidade
    sensitivity = st.slider(
        "🎚️ Sensibilidade",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Valores mais altos detectam mais anomalias"
    )
    
    # Período de análise
    time_period = st.selectbox(
        "📅 Período de Análise",
        ["Última hora", "Últimas 24 horas", "Última semana", "Último mês"]
    )
    
    # Configurações de alertas
    st.subheader("🚨 Configurações de Alertas")
    
    auto_alerts = st.checkbox("Ativar Alertas Automáticos", value=True)
    
    if auto_alerts:
        min_severity = st.selectbox(
            "Severidade Mínima",
            ["low", "medium", "high", "critical"]
        )
        
        notification_methods = st.multiselect(
            "Métodos de Notificação",
            ["Email", "Slack", "Teams", "Webhook"],
            default=["Email"]
        )

# Carregar dados
@st.cache_data
def load_station_data(station_name, days=30):
    """Carregar dados da estação"""
    return data_loader.generate_synthetic_data(station_name, days)

# Carregar dados baseado no período
days_map = {
    "Última hora": 1,
    "Últimas 24 horas": 1,
    "Última semana": 7,
    "Último mês": 30
}

df = load_station_data(selected_station, days_map[time_period])

# Filtrar dados por período
if time_period == "Última hora":
    df = df.tail(4)  # 4 pontos = 1 hora
elif time_period == "Últimas 24 horas":
    df = df.tail(96)  # 96 pontos = 24 horas
elif time_period == "Última semana":
    df = df.tail(672)  # 672 pontos = 1 semana

# Dashboard de alertas
st.header("🚨 Dashboard de Alertas")

# Verificar alertas de limite
if selected_params:
    threshold_alerts = alert_system.check_threshold_alerts(df, thresholds)
    
    # Resumo de alertas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total de Alertas", len(threshold_alerts))
        
    with col2:
        critical_alerts = len([a for a in threshold_alerts if a['severity'] == 'critical'])
        st.metric("Alertas Críticos", critical_alerts, delta_color="inverse" if critical_alerts > 0 else "normal")
        
    with col3:
        high_alerts = len([a for a in threshold_alerts if a['severity'] == 'high'])
        st.metric("Alertas Altos", high_alerts)
        
    with col4:
        medium_alerts = len([a for a in threshold_alerts if a['severity'] == 'medium'])
        st.metric("Alertas Médios", medium_alerts)
    
    # Lista de alertas ativos
    if threshold_alerts:
        st.subheader("📋 Alertas Ativos")
        
        alerts_df = pd.DataFrame(threshold_alerts)
        
        # Cores por severidade
        def color_severity(val):
            colors = {
                'critical': 'background-color: #ffcccc',
                'high': 'background-color: #ffe6cc',
                'medium': 'background-color: #fff9cc',
                'low': 'background-color: #ccffcc'
            }
            return colors.get(val, '')
        
        styled_alerts = alerts_df.style.applymap(
            color_severity, 
            subset=['severity']
        )
        
        st.dataframe(styled_alerts, use_container_width=True)
        
        # Gráfico de distribuição de alertas
        st.subheader("📊 Distribuição de Alertas por Severidade")
        
        severity_counts = alerts_df['severity'].value_counts()
        
        fig_alerts = go.Figure(data=[
            go.Bar(
                x=severity_counts.index,
                y=severity_counts.values,
                marker_color=['red', 'orange', 'yellow', 'green'][:len(severity_counts)],
                text=severity_counts.values,
                textposition='auto'
            )
        ])
        
        fig_alerts.update_layout(
            title="Alertas por Nível de Severidade",
            xaxis_title="Severidade",
            yaxis_title="Número de Alertas",
            height=400
        )
        
        st.plotly_chart(fig_alerts, use_container_width=True)
    else:
        st.success("✅ Nenhum alerta de limite detectado!")

# Detecção de anomalias
st.header("🔍 Detecção de Anomalias")

if st.button("🔍 Detectar Anomalias", type="primary"):
    with st.spinner("Detectando anomalias..."):
        
        anomalies_list = []
        
        try:
            if detection_method == "Isolation Forest":
                df_anomalies = anomaly_detector.detect_anomalies_isolation_forest(
                    df, selected_params, contamination=sensitivity
                )
            elif detection_method == "DBSCAN":
                df_anomalies = anomaly_detector.detect_anomalies_dbscan(df, selected_params)
            elif detection_method == "One-Class SVM":
                df_anomalies = anomaly_detector.detect_anomalies_one_class_svm(
                    df, selected_params, nu=sensitivity
                )
            elif detection_method == "Z-Score":
                df_anomalies = anomaly_detector.detect_outliers_statistical(
                    df, selected_params, method='zscore', threshold=3-sensitivity*2
                )
            elif detection_method == "IQR":
                df_anomalies = anomaly_detector.detect_outliers_statistical(
                    df, selected_params, method='iqr', threshold=1.5-sensitivity*0.5
                )
            elif detection_method == "LSTM Autoencoder":
                df_anomalies = anomaly_detector.detect_anomalies_lstm_autoencoder(
                    df, selected_params, threshold=sensitivity
                )
            else:
                df_anomalies = df
            
            # Converter para lista de anomalias
            for param in selected_params:
                anomaly_col = f'{param}_anomaly'
                if anomaly_col in df_anomalies.columns:
                    anomaly_indices = df_anomalies[df_anomalies[anomaly_col] == 1].index
                    for idx in anomaly_indices:
                        # Determinar severidade baseada nos limites
                        value = df_anomalies.loc[idx, param]
                        severity = 'low'
                        
                        if param in thresholds:
                            param_thresholds = thresholds[param]
                            
                            if 'critical_min' in param_thresholds and value < param_thresholds['critical_min']:
                                severity = 'critical'
                            elif 'critical_max' in param_thresholds and value > param_thresholds['critical_max']:
                                severity = 'critical'
                            elif 'min' in param_thresholds and value < param_thresholds['min']:
                                severity = 'high'
                            elif 'max' in param_thresholds and value > param_thresholds['max']:
                                severity = 'high'
                            else:
                                severity = 'medium'
                        
                        anomalies_list.append({
                            'timestamp': df_anomalies.loc[idx, 'timestamp'],
                            'parameter': param,
                            'value': value,
                            'severity': severity,
                            'method': detection_method
                        })
            
            if anomalies_list:
                st.success(f"✅ {len(anomalies_list)} anomalias detectadas!")
                
                # Resumo de anomalias
                st.subheader("📊 Resumo de Anomalias Detectadas")
                
                anomalies_df = pd.DataFrame(anomalies_list)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total de Anomalias", len(anomalies_list))
                    
                with col2:
                    param_counts = anomalies_df['parameter'].value_counts()
                    st.metric("Parâmetros Afetados", len(param_counts))
                    
                with col3:
                    severity_counts = anomalies_df['severity'].value_counts()
                    st.metric("Severidade Máxima", severity_counts.index[0] if len(severity_counts) > 0 else "N/A")
                
                # Distribuição por parâmetro
                st.subheader("📈 Anomalias por Parâmetro")
                
                fig_param = px.bar(
                    x=param_counts.index,
                    y=param_counts.values,
                    title="Número de Anomalias por Parâmetro",
                    labels={'x': 'Parâmetro', 'y': 'Número de Anomalias'}
                )
                st.plotly_chart(fig_param, use_container_width=True)
                
                # Distribuição por severidade
                st.subheader("🎯 Anomalias por Severidade")
                
                fig_severity = px.pie(
                    values=severity_counts.values,
                    names=severity_counts.index,
                    title="Distribuição de Anomalias por Severidade",
                    color_discrete_map={
                        'critical': 'red',
                        'high': 'orange',
                        'medium': 'yellow',
                        'low': 'green'
                    }
                )
                st.plotly_chart(fig_severity, use_container_width=True)
                
                # Tabela de anomalias
                st.subheader("📋 Detalhes das Anomalias")
                
                # Formatar tabela
                display_df = anomalies_df.copy()
                display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                display_df['value'] = display_df['value'].round(3)
                
                st.dataframe(display_df, use_container_width=True)
                
                # Visualização temporal
                st.subheader("📈 Linha do Tempo de Anomalias")
                
                for param in selected_params:
                    param_anomalies = [a for a in anomalies_list if a['parameter'] == param]
                    if param_anomalies:
                        fig_timeline = visualizer.create_anomaly_timeline(
                            df, param, f'{param}_anomaly'
                        )
                        st.plotly_chart(fig_timeline, use_container_width=True)
                
                # Exportar resultados
                st.subheader("💾 Exportar Resultados")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    csv_data = anomalies_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Baixar CSV",
                        data=csv_data,
                        file_name=f"anomalias_{selected_station}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    json_data = anomalies_df.to_json(orient='records', date_format='iso')
                    st.download_button(
                        label="📄 Baixar JSON",
                        data=json_data,
                        file_name=f"anomalias_{selected_station}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                
            else:
                st.success("✅ Nenhuma anomalia detectada no período selecionado!")
                
        except Exception as e:
            st.error(f"❌ Erro na detecção de anomalias: {str(e)}")

# Análise de padrões
st.header("🔍 Análise de Padrões")

if st.button("📊 Analisar Padrões"):
    with st.spinner("Analisando padrões..."):
        
        # Análise de correlação entre parâmetros
        if len(selected_params) > 1:
            st.subheader("🔗 Correlação entre Parâmetros")
            
            correlation_matrix = df[selected_params].corr()
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=correlation_matrix.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 10}
            ))
            
            fig_corr.update_layout(
                title="Matriz de Correlação",
                height=500
            )
            
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # Análise de distribuição
        st.subheader("📊 Distribuição dos Parâmetros")
        
        for param in selected_params:
            if param in df.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histograma
                    fig_hist = visualizer.create_distribution_plot(df, param)
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    # Box plot
                    fig_box = visualizer.create_box_plot(df, [param])
                    st.plotly_chart(fig_box, use_container_width=True)

# Configurações de notificação
st.header("📧 Configurações de Notificação")

with st.expander("⚙️ Configurar Notificações"):
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📧 Email")
        
        email_enabled = st.checkbox("Ativar notificações por email")
        
        if email_enabled:
            smtp_host = st.text_input("SMTP Host", value="smtp.gmail.com")
            smtp_port = st.number_input("SMTP Port", value=587)
            smtp_username = st.text_input("Username")
            smtp_password = st.text_input("Password", type="password")
            email_recipients = st.text_area(
                "Destinatários (um por linha)",
                value="operator@watertreatment.com\nsupervisor@watertreatment.com"
            )
    
    with col2:
        st.subheader("💬 Slack")
        
        slack_enabled = st.checkbox("Ativar notificações no Slack")
        
        if slack_enabled:
            slack_webhook = st.text_input("Webhook URL")
            slack_channel = st.text_input("Canal", value="#water-treatment-alerts")
    
    # Testar notificações
    if st.button("🧪 Testar Notificações"):
        st.info("Funcionalidade de teste em desenvolvimento")

# Histórico de alertas
st.header("📚 Histórico de Alertas")

# Simular histórico
history_data = {
    'timestamp': [datetime.now() - timedelta(hours=i) for i in range(24)],
    'station': [selected_station] * 24,
    'parameter': np.random.choice(selected_params, 24),
    'severity': np.random.choice(['low', 'medium', 'high', 'critical'], 24, p=[0.4, 0.3, 0.2, 0.1]),
    'message': [f"Alerta {i+1}" for i in range(24)]
}

history_df = pd.DataFrame(history_data)

# Filtros
col1, col2, col3 = st.columns(3)

with col1:
    severity_filter = st.multiselect(
        "Filtrar por Severidade",
        ['low', 'medium', 'high', 'critical'],
        default=['high', 'critical']
    )

with col2:
    parameter_filter = st.multiselect(
        "Filtrar por Parâmetro",
        selected_params,
        default=selected_params
    )

with col3:
    hours_filter = st.slider(
        "Últimas N horas",
        min_value=1,
        max_value=168,
        value=24
    )

# Aplicar filtros
filtered_history = history_df[
    (history_df['severity'].isin(severity_filter)) &
    (history_df['parameter'].isin(parameter_filter)) &
    (history_df['timestamp'] >= datetime.now() - timedelta(hours=hours_filter))
]

st.dataframe(filtered_history, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
    Página de Anomalias e Alertas | Sistema de Monitoramento v2.0 | 
    Última atualização: {:%Y-%m-%d %H:%M:%S}
    </div>
    """.format(datetime.now()),
    unsafe_allow_html=True
)
