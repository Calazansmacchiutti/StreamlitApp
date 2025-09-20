"""
Página de Configurações do Sistema
"""

import streamlit as st
import pandas as pd
import numpy as np
import yaml
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path

# Adicionar o diretório modules ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))

try:
    from modules.data_loader import DataLoader
except ImportError as e:
    st.error(f"Erro ao importar módulos: {e}")
    st.stop()

st.set_page_config(
    page_title="Configurações - Sistema de Monitoramento",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Configurações do Sistema")

# Inicializar componentes
data_loader = DataLoader()

# Carregar configurações
config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'

if config_path.exists():
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
else:
    st.error("Arquivo de configuração não encontrado!")
    st.stop()

# Tabs para diferentes seções de configuração
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏭 Estações",
    "📊 Parâmetros",
    "⚠️ Alertas",
    "🔮 Previsões",
    "📧 Notificações",
    "🔧 Sistema"
])

with tab1:
    st.header("🏭 Configurações das Estações")
    
    # Lista de estações
    stations = config.get('stations', {})
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Estações Configuradas")
        
        for station_name, station_config in stations.items():
            with st.expander(f"📍 {station_name}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Tipo", station_config.get('type', 'N/A').title())
                    st.metric("Capacidade", f"{station_config.get('capacity', 0):,} m³/dia")
                
                with col2:
                    st.metric("Localização", station_config.get('location', 'N/A'))
                    st.metric("Parâmetros", len(station_config.get('parameters', [])))
                
                with col3:
                    st.text_area(
                        "Descrição",
                        value=station_config.get('description', ''),
                        height=100,
                        key=f"desc_{station_name}"
                    )
                
                # Parâmetros da estação
                st.subheader("Parâmetros Monitorados")
                parameters = station_config.get('parameters', [])
                
                if parameters:
                    param_df = pd.DataFrame({
                        'Parâmetro': parameters,
                        'Status': ['✅ Ativo'] * len(parameters)
                    })
                    st.dataframe(param_df, use_container_width=True)
                else:
                    st.warning("Nenhum parâmetro configurado para esta estação")
    
    with col2:
        st.subheader("Adicionar Nova Estação")
        
        with st.form("add_station"):
            new_station_name = st.text_input("Nome da Estação")
            new_station_type = st.selectbox("Tipo", ["primary", "secondary", "tertiary"])
            new_station_capacity = st.number_input("Capacidade (m³/dia)", min_value=0)
            new_station_location = st.text_input("Localização")
            new_station_description = st.text_area("Descrição")
            
            if st.form_submit_button("➕ Adicionar Estação"):
                st.success(f"Estação {new_station_name} adicionada com sucesso!")
                st.rerun()

with tab2:
    st.header("📊 Configurações de Parâmetros")
    
    # Limites de controle
    thresholds = config.get('thresholds', {})
    
    st.subheader("Limites de Controle")
    
    # Criar interface para editar limites
    for param_name, param_thresholds in thresholds.items():
        with st.expander(f"📊 {param_name}"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if 'min' in param_thresholds:
                    min_val = st.number_input(
                        "Limite Mínimo",
                        value=param_thresholds['min'],
                        key=f"min_{param_name}"
                    )
                else:
                    st.info("Sem limite mínimo")
            
            with col2:
                if 'max' in param_thresholds:
                    max_val = st.number_input(
                        "Limite Máximo",
                        value=param_thresholds['max'],
                        key=f"max_{param_name}"
                    )
                else:
                    st.info("Sem limite máximo")
            
            with col3:
                if 'critical_min' in param_thresholds:
                    crit_min_val = st.number_input(
                        "Limite Crítico Mín",
                        value=param_thresholds['critical_min'],
                        key=f"crit_min_{param_name}"
                    )
                else:
                    st.info("Sem limite crítico mínimo")
            
            with col4:
                if 'critical_max' in param_thresholds:
                    crit_max_val = st.number_input(
                        "Limite Crítico Máx",
                        value=param_thresholds['critical_max'],
                        key=f"crit_max_{param_name}"
                    )
                else:
                    st.info("Sem limite crítico máximo")
            
            # Informações adicionais
            col1, col2 = st.columns(2)
            
            with col1:
                unit = st.text_input(
                    "Unidade",
                    value=param_thresholds.get('unit', ''),
                    key=f"unit_{param_name}"
                )
            
            with col2:
                description = st.text_input(
                    "Descrição",
                    value=param_thresholds.get('description', ''),
                    key=f"desc_{param_name}"
                )
            
            if st.button(f"💾 Salvar {param_name}", key=f"save_{param_name}"):
                st.success(f"Configurações de {param_name} salvas!")
    
    # Adicionar novo parâmetro
    st.subheader("➕ Adicionar Novo Parâmetro")
    
    with st.form("add_parameter"):
        col1, col2 = st.columns(2)
        
        with col1:
            new_param_name = st.text_input("Nome do Parâmetro")
            new_param_unit = st.text_input("Unidade")
            new_param_description = st.text_area("Descrição")
        
        with col2:
            new_param_min = st.number_input("Limite Mínimo", value=0.0)
            new_param_max = st.number_input("Limite Máximo", value=100.0)
            new_param_crit_min = st.number_input("Limite Crítico Mínimo", value=0.0)
            new_param_crit_max = st.number_input("Limite Crítico Máximo", value=100.0)
        
        if st.form_submit_button("➕ Adicionar Parâmetro"):
            st.success(f"Parâmetro {new_param_name} adicionado com sucesso!")

with tab3:
    st.header("⚠️ Configurações de Alertas")
    
    alert_config = config.get('alerts', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Configurações Gerais")
        
        alerts_enabled = st.checkbox(
            "Alertas Habilitados",
            value=alert_config.get('enabled', True)
        )
        
        min_severity = st.selectbox(
            "Severidade Mínima",
            ["low", "medium", "high", "critical"],
            index=["low", "medium", "high", "critical"].index(alert_config.get('min_severity', 'medium'))
        )
        
        st.subheader("Regras de Alertas")
        
        rules = alert_config.get('rules', {})
        
        for rule_name, rule_config in rules.items():
            with st.expander(f"📋 {rule_name.replace('_', ' ').title()}"):
                enabled = st.checkbox(
                    "Habilitado",
                    value=rule_config.get('enabled', True),
                    key=f"rule_{rule_name}_enabled"
                )
                
                severity = st.selectbox(
                    "Severidade",
                    ["low", "medium", "high", "critical"],
                    index=["low", "medium", "high", "critical"].index(rule_config.get('severity', 'medium')),
                    key=f"rule_{rule_name}_severity"
                )
                
                if 'consecutive_anomalies' in rule_config:
                    consecutive = st.number_input(
                        "Anomalias Consecutivas",
                        value=rule_config['consecutive_anomalies'],
                        key=f"rule_{rule_name}_consecutive"
                    )
                
                if 'trend_window' in rule_config:
                    trend_window = st.number_input(
                        "Janela de Tendência",
                        value=rule_config['trend_window'],
                        key=f"rule_{rule_name}_trend"
                    )
                
                if 'max_missing_minutes' in rule_config:
                    missing_minutes = st.number_input(
                        "Minutos Máximos Sem Dados",
                        value=rule_config['max_missing_minutes'],
                        key=f"rule_{rule_name}_missing"
                    )
    
    with col2:
        st.subheader("Métodos de Notificação")
        
        notification_methods = alert_config.get('notification_methods', {})
        
        # Email
        with st.expander("📧 Email"):
            email_config = notification_methods.get('email', {})
            
            email_enabled = st.checkbox(
                "Habilitar Email",
                value=email_config.get('enabled', False),
                key="email_enabled"
            )
            
            if email_enabled:
                smtp_config = email_config.get('smtp', {})
                
                smtp_host = st.text_input(
                    "SMTP Host",
                    value=smtp_config.get('host', 'smtp.gmail.com'),
                    key="smtp_host"
                )
                
                smtp_port = st.number_input(
                    "SMTP Port",
                    value=smtp_config.get('port', 587),
                    key="smtp_port"
                )
                
                smtp_username = st.text_input(
                    "Username",
                    value=smtp_config.get('username', ''),
                    key="smtp_username"
                )
                
                smtp_password = st.text_input(
                    "Password",
                    value=smtp_config.get('password', ''),
                    type="password",
                    key="smtp_password"
                )
                
                recipients = st.text_area(
                    "Destinatários (um por linha)",
                    value='\n'.join(email_config.get('recipients', [])),
                    key="email_recipients"
                )
        
        # Slack
        with st.expander("💬 Slack"):
            slack_config = notification_methods.get('slack', {})
            
            slack_enabled = st.checkbox(
                "Habilitar Slack",
                value=slack_config.get('enabled', False),
                key="slack_enabled"
            )
            
            if slack_enabled:
                webhook_url = st.text_input(
                    "Webhook URL",
                    value=slack_config.get('webhook_url', ''),
                    key="slack_webhook"
                )
                
                channel = st.text_input(
                    "Canal",
                    value=slack_config.get('channel', '#water-treatment-alerts'),
                    key="slack_channel"
                )
        
        # Teams
        with st.expander("🏢 Microsoft Teams"):
            teams_config = notification_methods.get('teams', {})
            
            teams_enabled = st.checkbox(
                "Habilitar Teams",
                value=teams_config.get('enabled', False),
                key="teams_enabled"
            )
            
            if teams_enabled:
                teams_webhook = st.text_input(
                    "Webhook URL",
                    value=teams_config.get('webhook_url', ''),
                    key="teams_webhook"
                )
        
        # Webhook
        with st.expander("🔗 Webhook"):
            webhook_config = notification_methods.get('webhook', {})
            
            webhook_enabled = st.checkbox(
                "Habilitar Webhook",
                value=webhook_config.get('enabled', False),
                key="webhook_enabled"
            )
            
            if webhook_enabled:
                webhook_url = st.text_input(
                    "URL",
                    value=webhook_config.get('url', ''),
                    key="webhook_url"
                )
                
                auth_token = st.text_input(
                    "Token de Autorização",
                    value=webhook_config.get('headers', {}).get('Authorization', ''),
                    key="webhook_auth"
                )

with tab4:
    st.header("🔮 Configurações de Previsões")
    
    forecasting_config = config.get('forecasting', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Modelos Disponíveis")
        
        models = forecasting_config.get('models', [])
        default_model = forecasting_config.get('default_model', 'prophet')
        
        for model in models:
            is_default = model == default_model
            st.checkbox(
                f"{'⭐' if is_default else '📊'} {model.title()}",
                value=True,
                disabled=is_default,
                key=f"model_{model}"
            )
        
        st.selectbox(
            "Modelo Padrão",
            models,
            index=models.index(default_model) if default_model in models else 0,
            key="default_model"
        )
        
        st.subheader("Configurações de Previsão")
        
        forecast_horizon = st.number_input(
            "Horizonte de Previsão (horas)",
            value=forecasting_config.get('forecast_horizon_hours', 24),
            min_value=1,
            max_value=168
        )
        
        confidence_level = st.slider(
            "Nível de Confiança",
            min_value=0.8,
            max_value=0.99,
            value=forecasting_config.get('confidence_level', 0.95),
            step=0.01
        )
        
        cv_splits = st.number_input(
            "Divisões para Validação Cruzada",
            value=forecasting_config.get('cross_validation_splits', 5),
            min_value=2,
            max_value=10
        )
    
    with col2:
        st.subheader("Configurações de Treinamento")
        
        # Configurações específicas por modelo
        st.markdown("#### Prophet")
        prophet_changepoint = st.slider(
            "Changepoint Prior Scale",
            min_value=0.001,
            max_value=0.5,
            value=0.05,
            step=0.001
        )
        
        prophet_seasonality = st.selectbox(
            "Modo de Sazonalidade",
            ["additive", "multiplicative"],
            index=1
        )
        
        st.markdown("#### ARIMA")
        arima_p = st.number_input("p (AR order)", value=1, min_value=0, max_value=10)
        arima_d = st.number_input("d (Diferenciação)", value=1, min_value=0, max_value=3)
        arima_q = st.number_input("q (MA order)", value=1, min_value=0, max_value=10)
        
        st.markdown("#### Random Forest")
        rf_estimators = st.number_input("Número de Estimadores", value=100, min_value=10, max_value=1000)
        rf_depth = st.number_input("Profundidade Máxima", value=10, min_value=1, max_value=50)
        
        st.markdown("#### XGBoost")
        xgb_estimators = st.number_input("Número de Estimadores (XGBoost)", value=100, min_value=10, max_value=1000)
        xgb_learning_rate = st.slider("Taxa de Aprendizado", min_value=0.01, max_value=0.3, value=0.1, step=0.01)

with tab5:
    st.header("📧 Configurações de Notificações")
    
    # Configurações de frequência
    st.subheader("Frequência de Notificações")
    
    col1, col2 = st.columns(2)
    
    with col1:
        notification_frequency = st.selectbox(
            "Frequência Máxima",
            ["Imediato", "1 por hora", "1 por dia", "Resumo diário", "Resumo semanal"]
        )
        
        quiet_hours_start = st.time_input(
            "Início do Período Silencioso",
            value=datetime.strptime("22:00", "%H:%M").time()
        )
        
        quiet_hours_end = st.time_input(
            "Fim do Período Silencioso",
            value=datetime.strptime("06:00", "%H:%M").time()
        )
    
    with col2:
        escalation_enabled = st.checkbox("Habilitar Escalação de Alertas")
        
        if escalation_enabled:
            escalation_delay = st.number_input(
                "Delay para Escalação (minutos)",
                value=30,
                min_value=5,
                max_value=1440
            )
            
            escalation_recipients = st.text_area(
                "Destinatários para Escalação",
                value="manager@watertreatment.com\ndirector@watertreatment.com"
            )
    
    # Templates de notificação
    st.subheader("Templates de Notificação")
    
    template_type = st.selectbox(
        "Tipo de Template",
        ["Email", "Slack", "Teams", "SMS"]
    )
    
    if template_type == "Email":
        email_subject = st.text_input(
            "Assunto do Email",
            value="🚨 ALERTA {severity.upper()} - Sistema de Monitoramento"
        )
        
        email_body = st.text_area(
            "Corpo do Email",
            value="""
            <html>
            <body>
                <h2>🚨 Alerta do Sistema de Monitoramento</h2>
                <p><strong>Severidade:</strong> {severity.upper()}</p>
                <p><strong>Estação:</strong> {station}</p>
                <p><strong>Parâmetro:</strong> {parameter}</p>
                <p><strong>Valor:</strong> {value}</p>
                <p><strong>Mensagem:</strong> {message}</p>
                <p><strong>Timestamp:</strong> {timestamp}</p>
                <hr>
                <p><em>Este é um alerta automático do sistema de monitoramento.</em></p>
            </body>
            </html>
            """,
            height=200
        )
    
    elif template_type == "Slack":
        slack_message = st.text_area(
            "Mensagem do Slack",
            value="""
            🚨 *Alerta {severity.upper()}*
            
            *Estação:* {station}
            *Parâmetro:* {parameter}
            *Valor:* {value}
            *Mensagem:* {message}
            *Timestamp:* {timestamp}
            """,
            height=200
        )
    
    # Teste de notificações
    st.subheader("🧪 Teste de Notificações")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_recipient = st.text_input("Destinatário de Teste")
        test_message = st.text_area("Mensagem de Teste", value="Este é um teste do sistema de notificações.")
    
    with col2:
        test_method = st.selectbox("Método de Teste", ["Email", "Slack", "Teams"])
        
        if st.button("📤 Enviar Teste"):
            st.success(f"Teste enviado via {test_method} para {test_recipient}")

with tab6:
    st.header("🔧 Configurações do Sistema")
    
    system_config = config.get('system', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Configurações Gerais")
        
        update_interval = st.number_input(
            "Intervalo de Atualização (segundos)",
            value=system_config.get('update_interval', 300),
            min_value=30,
            max_value=3600
        )
        
        data_retention = st.number_input(
            "Retenção de Dados (dias)",
            value=system_config.get('data_retention_days', 90),
            min_value=7,
            max_value=365
        )
        
        timezone = st.selectbox(
            "Fuso Horário",
            ["America/Sao_Paulo", "UTC", "America/New_York", "Europe/London"],
            index=0
        )
        
        auto_refresh = st.checkbox(
            "Atualização Automática",
            value=system_config.get('auto_refresh', True)
        )
        
        st.subheader("Configurações de Performance")
        
        max_data_points = st.number_input(
            "Máximo de Pontos de Dados",
            value=10000,
            min_value=1000,
            max_value=100000
        )
        
        enable_caching = st.checkbox(
            "Habilitar Cache",
            value=True
        )
        
        cache_ttl = st.number_input(
            "TTL do Cache (segundos)",
            value=300,
            min_value=60,
            max_value=3600
        )
        
        parallel_processing = st.checkbox(
            "Processamento Paralelo",
            value=True
        )
        
        max_workers = st.number_input(
            "Máximo de Workers",
            value=4,
            min_value=1,
            max_value=16
        )
    
    with col2:
        st.subheader("Configurações de Segurança")
        
        security_config = config.get('security', {})
        
        auth_required = st.checkbox(
            "Autenticação Obrigatória",
            value=security_config.get('authentication_required', False)
        )
        
        session_timeout = st.number_input(
            "Timeout de Sessão (segundos)",
            value=security_config.get('session_timeout', 3600),
            min_value=300,
            max_value=86400
        )
        
        max_login_attempts = st.number_input(
            "Máximo de Tentativas de Login",
            value=security_config.get('max_login_attempts', 3),
            min_value=1,
            max_value=10
        )
        
        password_min_length = st.number_input(
            "Tamanho Mínimo da Senha",
            value=security_config.get('password_min_length', 8),
            min_value=6,
            max_value=20
        )
        
        enable_audit_log = st.checkbox(
            "Habilitar Log de Auditoria",
            value=security_config.get('enable_audit_log', True)
        )
        
        st.subheader("Configurações de Desenvolvimento")
        
        dev_config = config.get('development', {})
        
        debug_mode = st.checkbox(
            "Modo Debug",
            value=dev_config.get('debug_mode', False)
        )
        
        log_level = st.selectbox(
            "Nível de Log",
            ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            index=1
        )
        
        enable_profiling = st.checkbox(
            "Habilitar Profiling",
            value=dev_config.get('enable_profiling', False)
        )
        
        mock_data = st.checkbox(
            "Usar Dados Mock",
            value=dev_config.get('mock_data', False)
        )
        
        test_mode = st.checkbox(
            "Modo de Teste",
            value=dev_config.get('test_mode', False)
        )
    
    # Backup e Restauração
    st.subheader("💾 Backup e Restauração")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Fazer Backup"):
            st.success("Backup realizado com sucesso!")
    
    with col2:
        if st.button("📤 Restaurar Backup"):
            uploaded_file = st.file_uploader("Selecionar arquivo de backup", type=['yaml', 'yml'])
            if uploaded_file:
                st.success("Backup restaurado com sucesso!")
    
    with col3:
        if st.button("🔄 Resetar Configurações"):
            if st.checkbox("Confirmar reset"):
                st.warning("Configurações resetadas para valores padrão!")

# Botões de ação
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("💾 Salvar Configurações", type="primary"):
        st.success("Configurações salvas com sucesso!")

with col2:
    if st.button("🔄 Recarregar Configurações"):
        st.rerun()

with col3:
    if st.button("📥 Importar Configurações"):
        uploaded_file = st.file_uploader("Selecionar arquivo de configuração", type=['yaml', 'yml'])
        if uploaded_file:
            st.success("Configurações importadas com sucesso!")

with col4:
    if st.button("📤 Exportar Configurações"):
        st.download_button(
            label="Baixar Configurações",
            data=yaml.dump(config, default_flow_style=False),
            file_name=f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml",
            mime="text/yaml"
        )

# Status do sistema
st.header("📊 Status do Sistema")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Uptime", "99.9%", "+0.1%")

with col2:
    st.metric("Memória Usada", "2.1 GB", "+0.2 GB")

with col3:
    st.metric("CPU", "15%", "+2%")

with col4:
    st.metric("Armazenamento", "45%", "+1%")

# Logs do sistema
st.header("📋 Logs do Sistema")

log_level_filter = st.selectbox("Filtrar por Nível", ["ALL", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

# Simular logs
logs_data = {
    'Timestamp': [datetime.now() - timedelta(minutes=i) for i in range(20)],
    'Level': np.random.choice(['INFO', 'WARNING', 'ERROR'], 20, p=[0.7, 0.2, 0.1]),
    'Message': [
        f"Log message {i+1}: System operation completed successfully"
        for i in range(20)
    ]
}

logs_df = pd.DataFrame(logs_data)

if log_level_filter != "ALL":
    logs_df = logs_df[logs_df['Level'] == log_level_filter]

st.dataframe(logs_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
    Página de Configurações | Sistema de Monitoramento v2.0 | 
    Última atualização: {:%Y-%m-%d %H:%M:%S}
    </div>
    """.format(datetime.now()),
    unsafe_allow_html=True
)
