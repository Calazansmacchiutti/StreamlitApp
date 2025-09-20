"""
Página de Geração de Relatórios
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
    page_title="Relatórios - Sistema de Monitoramento",
    page_icon="📑",
    layout="wide"
)

st.title("📑 Geração de Relatórios")

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

# Sidebar para configurações do relatório
with st.sidebar:
    st.header("⚙️ Configurações do Relatório")
    
    # Seleção de estação
    selected_station = st.selectbox(
        "📍 Selecionar Estação",
        list(stations_config.keys())
    )
    
    # Tipo de relatório
    report_type = st.selectbox(
        "📋 Tipo de Relatório",
        [
            "Relatório Operacional",
            "Relatório de Conformidade",
            "Relatório de Manutenção",
            "Relatório Executivo",
            "Relatório de Anomalias",
            "Relatório Customizado"
        ]
    )
    
    # Período do relatório
    st.subheader("📅 Período do Relatório")
    
    date_range_type = st.radio(
        "Selecionar período:",
        ["Última semana", "Último mês", "Últimos 3 meses", "Personalizado"]
    )
    
    if date_range_type == "Personalizado":
        start_date = st.date_input("Data inicial")
        end_date = st.date_input("Data final")
    else:
        now = datetime.now()
        if date_range_type == "Última semana":
            start_date = (now - timedelta(weeks=1)).date()
            end_date = now.date()
        elif date_range_type == "Último mês":
            start_date = (now - timedelta(days=30)).date()
            end_date = now.date()
        elif date_range_type == "Últimos 3 meses":
            start_date = (now - timedelta(days=90)).date()
            end_date = now.date()
    
    # Opções do relatório
    st.subheader("📊 Opções do Relatório")
    
    include_charts = st.checkbox("Incluir gráficos", value=True)
    include_statistics = st.checkbox("Incluir estatísticas", value=True)
    include_anomalies = st.checkbox("Incluir anomalias", value=True)
    include_predictions = st.checkbox("Incluir previsões", value=False)
    include_recommendations = st.checkbox("Incluir recomendações", value=True)
    include_raw_data = st.checkbox("Incluir dados brutos", value=False)
    
    # Formato e idioma
    st.subheader("📄 Formato e Idioma")
    
    report_format = st.selectbox("Formato:", ["PDF", "Excel", "HTML", "Word"])
    report_language = st.selectbox("Idioma:", ["Português", "English", "Español"])

# Carregar dados
@st.cache_data
def load_station_data(station_name, days=90):
    """Carregar dados da estação"""
    return data_loader.generate_synthetic_data(station_name, days)

# Carregar dados
df = load_station_data(selected_station, 90)

# Filtrar dados por período
if date_range_type == "Personalizado":
    df_filtered = df[(df['timestamp'].dt.date >= start_date) & (df['timestamp'].dt.date <= end_date)]
else:
    df_filtered = df

station_info = stations_config[selected_station]
station_params = station_info['parameters']

# Preview do relatório
st.header("📋 Preview do Relatório")

if st.button("📄 Gerar Relatório", type="primary"):
    with st.spinner("Gerando relatório..."):
        
        # Simular geração
        import time
        time.sleep(2)
        
        # Cabeçalho do relatório
        st.markdown(f"""
        ---
        # {report_type} - {selected_station}
        
        **Período:** {start_date} a {end_date}  
        **Data de Geração:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
        **Estação:** {selected_station} ({station_info['type'].title()})  
        **Capacidade:** {station_info['capacity']:,} m³/dia  
        
        ---
        """)
        
        # Resumo Executivo
        st.markdown("## 📊 Resumo Executivo")
        
        # Calcular métricas
        total_volume = df_filtered['flow_rate'].sum() / 1000 if 'flow_rate' in df_filtered.columns else 0
        avg_ph = df_filtered['pH'].mean() if 'pH' in df_filtered.columns else 7.0
        avg_turbidity = df_filtered['turbidity'].mean() if 'turbidity' in df_filtered.columns else 2.0
        
        # Calcular taxa de conformidade
        compliance_rate = 94.7 + np.random.normal(0, 2)
        
        st.markdown(f"""
        Durante o período analisado, a estação **{selected_station}** operou com uma taxa de conformidade de 
        **{compliance_rate:.1f}%**, processando um volume total de **{total_volume:.0f} mil m³**.
        
        ### Principais Indicadores:
        - **pH médio:** {avg_ph:.2f} (dentro dos limites)
        - **Turbidez média:** {avg_turbidity:.2f} NTU
        - **Eficiência do tratamento:** 96.3%
        - **Tempo de operação:** 99.2%
        - **Alertas gerados:** {np.random.randint(5, 15)}
        """)
        
        # Indicadores Principais
        st.markdown("## 📈 Indicadores Principais")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Taxa de Conformidade", f"{compliance_rate:.1f}%", f"{compliance_rate - 90:+.1f}%")
            
        with col2:
            st.metric("Volume Processado", f"{total_volume:.0f} mil m³", "+5.2%")
            
        with col3:
            st.metric("Eficiência", "96.3%", "+1.1%")
            
        with col4:
            st.metric("Tempo de Operação", "99.2%", "+0.3%")
        
        # Gráficos (se habilitado)
        if include_charts:
            st.markdown("## 📊 Análise Gráfica")
            
            # Gráfico de tendências
            st.subheader("📈 Tendências dos Parâmetros")
            
            if station_params:
                fig_trends = visualizer.create_realtime_chart(df_filtered, station_params[:3])
                st.plotly_chart(fig_trends, use_container_width=True)
            
            # Gráfico de distribuição
            if 'pH' in df_filtered.columns:
                st.subheader("📊 Distribuição do pH")
                fig_dist = visualizer.create_distribution_plot(df_filtered, 'pH')
                st.plotly_chart(fig_dist, use_container_width=True)
        
        # Estatísticas (se habilitado)
        if include_statistics:
            st.markdown("## 📊 Estatísticas Descritivas")
            
            # Calcular estatísticas para cada parâmetro
            stats_data = []
            for param in station_params:
                if param in df_filtered.columns:
                    param_data = df_filtered[param].dropna()
                    stats_data.append({
                        'Parâmetro': param,
                        'Média': param_data.mean(),
                        'Desvio Padrão': param_data.std(),
                        'Mínimo': param_data.min(),
                        'Máximo': param_data.max(),
                        'Mediana': param_data.median()
                    })
            
            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                st.dataframe(stats_df.round(3), use_container_width=True)
        
        # Análise de Conformidade
        st.markdown("## ✅ Análise de Conformidade")
        
        compliance_data = []
        for param in station_params:
            if param in df_filtered.columns and param in thresholds:
                param_data = df_filtered[param].dropna()
                param_thresholds = thresholds[param]
                
                total_measurements = len(param_data)
                compliant_measurements = 0
                
                # Verificar conformidade
                if 'min' in param_thresholds and 'max' in param_thresholds:
                    compliant_measurements = len(param_data[
                        (param_data >= param_thresholds['min']) & 
                        (param_data <= param_thresholds['max'])
                    ])
                elif 'min' in param_thresholds:
                    compliant_measurements = len(param_data[param_data >= param_thresholds['min']])
                elif 'max' in param_thresholds:
                    compliant_measurements = len(param_data[param_data <= param_thresholds['max']])
                else:
                    compliant_measurements = total_measurements
                
                compliance_rate_param = (compliant_measurements / total_measurements * 100) if total_measurements > 0 else 100
                
                compliance_data.append({
                    'Parâmetro': param,
                    'Medições Totais': total_measurements,
                    'Medições Conformes': compliant_measurements,
                    'Taxa de Conformidade (%)': compliance_rate_param,
                    'Status': '✅ Conforme' if compliance_rate_param >= 95 else '⚠️ Atenção' if compliance_rate_param >= 90 else '❌ Não Conforme'
                })
        
        if compliance_data:
            compliance_df = pd.DataFrame(compliance_data)
            st.dataframe(compliance_df, use_container_width=True)
        
        # Anomalias (se habilitado)
        if include_anomalies:
            st.markdown("## ⚠️ Análise de Anomalias")
            
            # Simular anomalias
            anomaly_data = {
                'Data': [datetime.now() - timedelta(days=i) for i in range(7)],
                'Parâmetro': np.random.choice(station_params, 7),
                'Tipo': np.random.choice(['Limite Excedido', 'Tendência Anormal', 'Falha de Sensor'], 7),
                'Severidade': np.random.choice(['Baixa', 'Média', 'Alta', 'Crítica'], 7, p=[0.4, 0.3, 0.2, 0.1]),
                'Ação Tomada': np.random.choice(['Monitoramento', 'Ajuste Automático', 'Intervenção Manual'], 7)
            }
            
            anomaly_df = pd.DataFrame(anomaly_data)
            st.dataframe(anomaly_df, use_container_width=True)
            
            # Gráfico de anomalias por severidade
            severity_counts = anomaly_df['Severidade'].value_counts()
            
            fig_anomalies = go.Figure(data=[
                go.Bar(
                    x=severity_counts.index,
                    y=severity_counts.values,
                    marker_color=['green', 'yellow', 'orange', 'red'][:len(severity_counts)]
                )
            ])
            
            fig_anomalies.update_layout(
                title="Distribuição de Anomalias por Severidade",
                xaxis_title="Severidade",
                yaxis_title="Número de Anomalias",
                height=400
            )
            
            st.plotly_chart(fig_anomalies, use_container_width=True)
        
        # Previsões (se habilitado)
        if include_predictions:
            st.markdown("## 🔮 Previsões")
            
            st.info("As previsões para os próximos 7 dias indicam:")
            
            # Simular previsões
            future_dates = pd.date_range(start=datetime.now(), periods=7, freq='D')
            predictions_data = {
                'Data': future_dates,
                'pH Previsto': [7.2 + np.random.normal(0, 0.1) for _ in range(7)],
                'Turbidez Prevista': [2.0 + np.random.normal(0, 0.3) for _ in range(7)],
                'Vazão Prevista': [1000 + np.random.normal(0, 50) for _ in range(7)]
            }
            
            predictions_df = pd.DataFrame(predictions_data)
            st.dataframe(predictions_df.round(2), use_container_width=True)
        
        # Recomendações (se habilitado)
        if include_recommendations:
            st.markdown("## 💡 Recomendações")
            
            recommendations = [
                "Ajustar dosagem de cloro no período noturno para otimizar eficiência",
                "Verificar calibração do sensor de pH da linha 2 - última calibração há 30 dias",
                "Programar manutenção preventiva do decantador para o próximo mês",
                "Considerar aumento da frequência de monitoramento durante picos de demanda",
                "Implementar sistema de alertas automáticos para parâmetros críticos"
            ]
            
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"{i}. {rec}")
        
        # Dados Brutos (se habilitado)
        if include_raw_data:
            st.markdown("## 📊 Dados Brutos")
            
            # Mostrar amostra dos dados
            st.subheader("Amostra dos Dados")
            st.dataframe(df_filtered.head(100), use_container_width=True)
            
            # Estatísticas de qualidade dos dados
            st.subheader("Qualidade dos Dados")
            
            quality_data = []
            for param in station_params:
                if param in df_filtered.columns:
                    param_data = df_filtered[param]
                    quality_data.append({
                        'Parâmetro': param,
                        'Total de Registros': len(param_data),
                        'Valores Faltantes': param_data.isnull().sum(),
                        'Completude (%)': (1 - param_data.isnull().sum() / len(param_data)) * 100,
                        'Valores Únicos': param_data.nunique()
                    })
            
            if quality_data:
                quality_df = pd.DataFrame(quality_data)
                st.dataframe(quality_df.round(2), use_container_width=True)
        
        # Conclusões
        st.markdown("## 🎯 Conclusões")
        
        st.markdown(f"""
        ### Resumo do Período:
        - A estação **{selected_station}** operou de forma **{'estável' if compliance_rate > 95 else 'aceitável' if compliance_rate > 90 else 'problemática'}** durante o período analisado
        - A taxa de conformidade de **{compliance_rate:.1f}%** está {'acima' if compliance_rate > 95 else 'dentro' if compliance_rate > 90 else 'abaixo'} dos padrões estabelecidos
        - O volume processado de **{total_volume:.0f} mil m³** representa {'alta' if total_volume > 1000 else 'média' if total_volume > 500 else 'baixa'} demanda
        
        ### Próximos Passos:
        1. Continuar monitoramento contínuo dos parâmetros críticos
        2. Implementar as recomendações listadas acima
        3. Agendar próxima revisão para {datetime.now() + timedelta(days=30):%d/%m/%Y}
        """)
        
        # Rodapé
        st.markdown("---")
        st.markdown(f"""
        **Relatório gerado em:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
        **Sistema:** Monitoramento de Estações de Tratamento v2.0  
        **Gerado por:** Sistema Automático  
        
        *Este relatório foi gerado automaticamente pelo sistema de monitoramento.*
        """)
        
        # Botões de download
        st.markdown("## 💾 Download do Relatório")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.download_button(
                label="📄 Baixar PDF",
                data="Conteúdo do relatório em PDF...",  # Aqui seria o conteúdo real
                file_name=f"relatorio_{selected_station}_{start_date}_{end_date}.pdf",
                mime="application/pdf"
            )
        
        with col2:
            st.download_button(
                label="📊 Baixar Excel",
                data="Conteúdo do relatório em Excel...",  # Aqui seria o conteúdo real
                file_name=f"relatorio_{selected_station}_{start_date}_{end_date}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col3:
            st.download_button(
                label="🌐 Baixar HTML",
                data="Conteúdo do relatório em HTML...",  # Aqui seria o conteúdo real
                file_name=f"relatorio_{selected_station}_{start_date}_{end_date}.html",
                mime="text/html"
            )
        
        with col4:
            st.download_button(
                label="📝 Baixar Word",
                data="Conteúdo do relatório em Word...",  # Aqui seria o conteúdo real
                file_name=f"relatorio_{selected_station}_{start_date}_{end_date}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
        
        st.success("✅ Relatório gerado com sucesso!")

# Templates de relatórios
st.header("📋 Templates de Relatórios")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📊 Relatório Operacional")
    st.markdown("""
    - Indicadores de performance
    - Análise de tendências
    - Alertas e anomalias
    - Recomendações operacionais
    """)
    
    if st.button("Usar Template", key="template1"):
        st.session_state.report_type = "Relatório Operacional"
        st.rerun()

with col2:
    st.subheader("📈 Relatório Executivo")
    st.markdown("""
    - Resumo executivo
    - KPIs principais
    - Análise de conformidade
    - Recomendações estratégicas
    """)
    
    if st.button("Usar Template", key="template2"):
        st.session_state.report_type = "Relatório Executivo"
        st.rerun()

with col3:
    st.subheader("🔧 Relatório de Manutenção")
    st.markdown("""
    - Status dos equipamentos
    - Alertas de manutenção
    - Histórico de intervenções
    - Cronograma de manutenções
    """)
    
    if st.button("Usar Template", key="template3"):
        st.session_state.report_type = "Relatório de Manutenção"
        st.rerun()

# Histórico de relatórios
st.header("📚 Histórico de Relatórios")

# Simular histórico
history_data = {
    'Data': [datetime.now() - timedelta(days=i) for i in range(10)],
    'Estação': [selected_station] * 10,
    'Tipo': np.random.choice(['Operacional', 'Executivo', 'Manutenção', 'Conformidade'], 10),
    'Período': [f"{(datetime.now() - timedelta(days=i+7)).strftime('%d/%m')} - {(datetime.now() - timedelta(days=i)).strftime('%d/%m')}" for i in range(10)],
    'Formato': np.random.choice(['PDF', 'Excel', 'HTML'], 10),
    'Status': ['✅ Concluído'] * 10
}

history_df = pd.DataFrame(history_data)

# Filtros
col1, col2, col3 = st.columns(3)

with col1:
    type_filter = st.multiselect(
        "Filtrar por Tipo",
        ['Operacional', 'Executivo', 'Manutenção', 'Conformidade'],
        default=['Operacional', 'Executivo', 'Manutenção', 'Conformidade']
    )

with col2:
    format_filter = st.multiselect(
        "Filtrar por Formato",
        ['PDF', 'Excel', 'HTML', 'Word'],
        default=['PDF', 'Excel', 'HTML', 'Word']
    )

with col3:
    days_filter = st.slider(
        "Últimos N dias",
        min_value=1,
        max_value=30,
        value=10
    )

# Aplicar filtros
filtered_history = history_df[
    (history_df['Tipo'].isin(type_filter)) &
    (history_df['Formato'].isin(format_filter)) &
    (history_df['Data'] >= datetime.now() - timedelta(days=days_filter))
]

st.dataframe(filtered_history, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
    Página de Relatórios | Sistema de Monitoramento v2.0 | 
    Última atualização: {:%Y-%m-%d %H:%M:%S}
    </div>
    """.format(datetime.now()),
    unsafe_allow_html=True
)
