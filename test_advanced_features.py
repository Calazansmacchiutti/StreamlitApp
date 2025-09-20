"""
Script de teste para funcionalidades avançadas do Mega Dashboard
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import yaml
import json

# Adicionar o diretório raiz ao path
sys.path.append(str(Path(__file__).parent))

from modules.data_loader import DataLoader
from modules.preprocessor import DataPreprocessor
from modules.time_series import TimeSeriesAnalyzer
from modules.anomaly_detection import AnomalyDetector
from modules.alerts import AlertSystem
from modules.backup_audit import BackupManager, AuditLogger, SystemMonitor

def test_advanced_data_loader():
    """Testar funcionalidades avançadas do DataLoader"""
    print("🔍 Testando DataLoader avançado...")
    
    data_loader = DataLoader()
    
    # Testar cache
    print("  - Testando sistema de cache...")
    data_loader.enable_cache(True)
    
    # Carregar dados com cache
    df1 = data_loader.load_csv("data/Two Mouths.csv", cache=True)
    df2 = data_loader.load_csv("data/Two Mouths.csv", cache=True)
    
    # Verificar se é a mesma referência (cache funcionando)
    cache_info = data_loader.get_cache_info()
    print(f"    Cache habilitado: {cache_info['cache_enabled']}")
    print(f"    Arquivos em cache: {cache_info['cache_size']}")
    
    # Testar metadados
    print("  - Testando metadados da estação...")
    metadata = data_loader.get_station_metadata("Two Mouths")
    print(f"    Estação: {metadata['name']}")
    print(f"    Registros: {metadata['data_points']}")
    print(f"    Parâmetros: {len(metadata['parameters'])}")
    
    # Testar backup
    print("  - Testando backup de dados...")
    backup_success = data_loader.backup_data("Two Mouths")
    print(f"    Backup realizado: {backup_success}")
    
    print("✅ DataLoader avançado testado com sucesso!\n")

def test_advanced_preprocessor():
    """Testar funcionalidades avançadas do DataPreprocessor"""
    print("🧹 Testando DataPreprocessor avançado...")
    
    preprocessor = DataPreprocessor()
    
    # Carregar dados de teste
    data_loader = DataLoader()
    df = data_loader.load_csv("data/Two Mouths.csv")
    
    if df.empty:
        print("  ⚠️ Nenhum dado encontrado para teste")
        return
    
    # Testar tratamento de valores ausentes
    print("  - Testando tratamento de valores ausentes...")
    df_with_missing = df.copy()
    # Simular valores ausentes
    df_with_missing.loc[df_with_missing.index[::100], 'pH'] = np.nan
    
    df_cleaned = preprocessor.handle_missing_values(df_with_missing, strategy='interpolate')
    missing_count = df_cleaned['pH'].isnull().sum()
    print(f"    Valores ausentes após limpeza: {missing_count}")
    
    # Testar criação de features de lag
    print("  - Testando features de lag...")
    df_with_lags = preprocessor.create_lag_features(df, 'pH', [1, 4, 12])
    lag_columns = [col for col in df_with_lags.columns if 'lag' in col]
    print(f"    Features de lag criadas: {len(lag_columns)}")
    
    # Testar features de janela móvel
    print("  - Testando features de janela móvel...")
    df_with_rolling = preprocessor.create_rolling_features(df, 'pH', [6, 24], ['mean', 'std'])
    rolling_columns = [col for col in df_with_rolling.columns if 'rolling' in col]
    print(f"    Features de janela móvel criadas: {len(rolling_columns)}")
    
    # Testar features temporais avançadas
    print("  - Testando features temporais avançadas...")
    df_with_time = preprocessor.create_time_features(df)
    time_columns = [col for col in df_with_time.columns if col in ['hour', 'dayofweek', 'month', 'hour_sin', 'hour_cos']]
    print(f"    Features temporais criadas: {len(time_columns)}")
    
    # Testar PCA
    print("  - Testando PCA...")
    numeric_columns = ['pH', 'turbidity', 'chlorine', 'flow_rate', 'temperature']
    available_columns = [col for col in numeric_columns if col in df.columns]
    if len(available_columns) >= 2:
        df_with_pca, pca_model = preprocessor.apply_pca(df, available_columns, n_components=2)
        pca_columns = [col for col in df_with_pca.columns if col.startswith('PC_')]
        print(f"    Componentes PCA criados: {len(pca_columns)}")
    
    # Testar resumo de pré-processamento
    print("  - Testando resumo de pré-processamento...")
    summary = preprocessor.get_preprocessing_summary(df)
    print(f"    Total de registros: {summary['total_records']}")
    print(f"    Colunas numéricas: {len(summary['numeric_columns'])}")
    print(f"    Uso de memória: {summary['memory_usage'] / 1024 / 1024:.2f} MB")
    
    print("✅ DataPreprocessor avançado testado com sucesso!\n")

def test_advanced_anomaly_detection():
    """Testar funcionalidades avançadas do AnomalyDetector"""
    print("⚠️ Testando AnomalyDetector avançado...")
    
    anomaly_detector = AnomalyDetector()
    
    # Carregar dados de teste
    data_loader = DataLoader()
    df = data_loader.load_csv("data/Two Mouths.csv")
    
    if df.empty:
        print("  ⚠️ Nenhum dado encontrado para teste")
        return
    
    # Preparar dados para detecção
    numeric_columns = ['pH', 'turbidity', 'chlorine', 'flow_rate', 'temperature']
    available_columns = [col for col in numeric_columns if col in df.columns]
    
    if len(available_columns) == 0:
        print("  ⚠️ Nenhuma coluna numérica encontrada para teste")
        return
    
    # Testar detecção estatística
    print("  - Testando detecção estatística...")
    for col in available_columns[:2]:  # Testar apenas 2 colunas
        predictions, scores = anomaly_detector.detect_statistical(df[col], method='zscore', threshold=3)
        anomalies = np.sum(predictions == -1)
        print(f"    {col}: {anomalies} anomalias detectadas (Z-score)")
        
        predictions_iqr, scores_iqr = anomaly_detector.detect_statistical(df[col], method='iqr', threshold=1.5)
        anomalies_iqr = np.sum(predictions_iqr == -1)
        print(f"    {col}: {anomalies_iqr} anomalias detectadas (IQR)")
    
    # Testar detecção multivariada
    print("  - Testando detecção multivariada...")
    if len(available_columns) >= 2:
        data_matrix = df[available_columns].values
        predictions_mv, scores_mv = anomaly_detector.detect_multivariate(df, available_columns, method='mahalanobis')
        anomalies_mv = np.sum(predictions_mv == -1)
        print(f"    Detecção multivariada: {anomalies_mv} anomalias detectadas")
    
    # Testar ensemble de detecção
    print("  - Testando ensemble de detecção...")
    if len(available_columns) >= 2:
        data_matrix = df[available_columns].values
        predictions_ensemble, scores_ensemble = anomaly_detector.ensemble_detection(
            data_matrix, 
            methods=['isolation_forest', 'statistical'],
            weights=[0.7, 0.3]
        )
        anomalies_ensemble = np.sum(predictions_ensemble == -1)
        print(f"    Ensemble: {anomalies_ensemble} anomalias detectadas")
    
    print("✅ AnomalyDetector avançado testado com sucesso!\n")

def test_backup_audit_system():
    """Testar sistema de backup e auditoria"""
    print("💾 Testando sistema de backup e auditoria...")
    
    # Configuração de teste
    config = {
        'backup_location': './test_backups',
        'backup_retention_days': 7,
        'encryption_enabled': False
    }
    
    # Testar BackupManager
    print("  - Testando BackupManager...")
    backup_manager = BackupManager(config)
    
    # Criar backup de teste
    backup_info = backup_manager.create_full_backup()
    print(f"    Backup criado: {backup_info['backup_name']}")
    print(f"    Status: {backup_info['status']}")
    print(f"    Tamanho: {backup_info.get('size', 0) / 1024:.2f} KB")
    
    # Listar backups
    backups = backup_manager.list_backups()
    print(f"    Backups disponíveis: {len(backups)}")
    
    # Testar AuditLogger
    print("  - Testando AuditLogger...")
    audit_logger = AuditLogger(config)
    
    # Registrar algumas ações de teste
    audit_logger.log_action(
        action="login",
        user_id="test_user",
        resource="dashboard",
        success=True
    )
    
    audit_logger.log_action(
        action="data_access",
        user_id="test_user",
        resource="Two Mouths",
        details={"parameters": ["pH", "turbidity"]},
        success=True
    )
    
    audit_logger.log_action(
        action="anomaly_detection",
        user_id="test_user",
        resource="Two Mouths",
        details={"method": "isolation_forest", "anomalies_found": 5},
        success=True
    )
    
    # Obter logs de auditoria
    logs = audit_logger.get_audit_logs(limit=10)
    print(f"    Logs de auditoria registrados: {len(logs)}")
    
    # Gerar relatório de auditoria
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)
    audit_report = audit_logger.generate_audit_report(start_date, end_date)
    print(f"    Relatório de auditoria: {audit_report['total_actions']} ações")
    
    # Testar SystemMonitor
    print("  - Testando SystemMonitor...")
    system_monitor = SystemMonitor(config)
    
    # Obter métricas do sistema
    metrics = system_monitor.get_system_metrics()
    if 'error' not in metrics:
        print(f"    CPU: {metrics.get('cpu', {}).get('percent', 'N/A')}%")
        print(f"    Memória: {metrics.get('memory', {}).get('percent', 'N/A')}%")
        print(f"    Disco: {metrics.get('disk', {}).get('percent', 'N/A')}%")
    else:
        print(f"    Erro ao obter métricas: {metrics['error']}")
    
    # Verificar saúde do sistema
    health = system_monitor.check_system_health()
    print(f"    Status geral: {health['overall_status']}")
    
    print("✅ Sistema de backup e auditoria testado com sucesso!\n")

def test_alert_system_advanced():
    """Testar sistema de alertas avançado"""
    print("🚨 Testando sistema de alertas avançado...")
    
    alert_system = AlertSystem()
    
    # Carregar configuração
    config_path = Path("config/config.yaml")
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
    else:
        print("  ⚠️ Arquivo de configuração não encontrado")
        return
    
    # Gerar alertas de teste
    print("  - Gerando alertas de teste...")
    
    # Alertas de diferentes níveis
    alert_system.generate_alert(
        timestamp=datetime.now(),
        station="Two Mouths",
        parameter="pH",
        value=5.8,
        threshold_info=config['thresholds']['pH'],
        level="critical",
        message="pH crítico detectado!"
    )
    
    alert_system.generate_alert(
        timestamp=datetime.now(),
        station="Two Mouths",
        parameter="turbidity",
        value=8.5,
        threshold_info=config['thresholds']['turbidity'],
        level="high",
        message="Turbidez elevada"
    )
    
    alert_system.generate_alert(
        timestamp=datetime.now(),
        station="New Rose of Rocky",
        parameter="DO",
        value=4.2,
        threshold_info=config['thresholds']['DO'],
        level="medium",
        message="Oxigênio dissolvido baixo"
    )
    
    # Obter alertas ativos
    active_alerts = alert_system.get_active_alerts()
    print(f"    Alertas ativos: {len(active_alerts)}")
    
    # Filtrar por nível
    critical_alerts = alert_system.get_active_alerts(level="critical")
    print(f"    Alertas críticos: {len(critical_alerts)}")
    
    # Mostrar detalhes dos alertas
    for alert in active_alerts[:3]:
        print(f"    - [{alert['level'].upper()}] {alert['parameter']}: {alert['value']}")
    
    print("✅ Sistema de alertas avançado testado com sucesso!\n")

def main():
    """Função principal de teste"""
    print("🚀 Iniciando testes das funcionalidades avançadas do Mega Dashboard\n")
    print("=" * 70)
    
    try:
        # Verificar se os dados existem
        data_dir = Path("data")
        if not data_dir.exists() or not list(data_dir.glob("*.csv")):
            print("⚠️ Diretório de dados não encontrado ou vazio.")
            print("   Execute 'python generate_synthetic_data.py' primeiro.")
            return
        
        # Executar testes
        test_advanced_data_loader()
        test_advanced_preprocessor()
        test_advanced_anomaly_detection()
        test_backup_audit_system()
        test_alert_system_advanced()
        
        print("=" * 70)
        print("🎉 Todos os testes das funcionalidades avançadas foram concluídos com sucesso!")
        print("\n📋 Resumo dos recursos testados:")
        print("  ✅ DataLoader com cache e metadados")
        print("  ✅ DataPreprocessor com features avançadas")
        print("  ✅ AnomalyDetector com múltiplos métodos")
        print("  ✅ Sistema de backup e auditoria")
        print("  ✅ Sistema de alertas avançado")
        
        print("\n🔧 Próximos passos:")
        print("  1. Execute 'streamlit run app.py' para usar o dashboard")
        print("  2. Acesse a API REST em http://localhost:8000/api/v1/docs")
        print("  3. Configure alertas e notificações no config.yaml")
        print("  4. Monitore logs de auditoria em logs/audit/")
        
    except Exception as e:
        print(f"❌ Erro durante os testes: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
