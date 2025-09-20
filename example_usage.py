"""
Exemplo de uso dos módulos do Sistema de Monitoramento
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Adicionar o diretório modules ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from modules.data_loader import DataLoader
from modules.preprocessor import DataPreprocessor
from modules.time_series import TimeSeriesAnalyzer
from modules.anomaly_detection import AnomalyDetector
from modules.visualizations import DashboardVisualizer
from modules.alerts import AlertSystem

def main():
    """Exemplo de uso dos módulos"""
    
    print("🌊 Sistema de Monitoramento de Estações de Tratamento")
    print("=" * 60)
    
    # 1. Carregar dados
    print("\n1. 📊 Carregando dados...")
    data_loader = DataLoader()
    
    # Carregar dados de uma estação
    station_name = "Two Mouths"
    df = data_loader.load_station_data(station_name)
    
    if df.empty:
        print(f"   Gerando dados sintéticos para {station_name}...")
        df = data_loader.generate_synthetic_data(station_name, days=30)
    
    print(f"   ✅ Dados carregados: {len(df)} registros")
    print(f"   📅 Período: {df['timestamp'].min()} a {df['timestamp'].max()}")
    print(f"   📊 Parâmetros: {list(df.columns[1:])}")
    
    # 2. Pré-processamento
    print("\n2. 🔧 Pré-processamento dos dados...")
    preprocessor = DataPreprocessor()
    
    # Limpar dados
    df_clean = preprocessor.clean_data(df, method='interpolate')
    print(f"   ✅ Dados limpos: {len(df_clean)} registros")
    
    # Calcular estatísticas
    parameters = ['pH', 'turbidity', 'chlorine']
    stats = preprocessor.calculate_statistics(df_clean, parameters)
    
    print("   📊 Estatísticas principais:")
    for param, stat in stats.items():
        print(f"      {param}: Média={stat['mean']:.2f}, Std={stat['std']:.2f}")
    
    # 3. Análise de séries temporais
    print("\n3. 📈 Análise de séries temporais...")
    time_series = TimeSeriesAnalyzer()
    
    # Verificar estacionariedade
    ts_data = df_clean.set_index('timestamp')['pH'].dropna()
    stationarity = time_series.check_stationarity(ts_data, method='adf')
    
    if 'error' not in stationarity:
        print(f"   📊 Teste de estacionariedade (ADF):")
        print(f"      Estatística: {stationarity['test_statistic']:.4f}")
        print(f"      P-valor: {stationarity['p_value']:.4f}")
        print(f"      Estacionário: {'Sim' if stationarity['is_stationary'] else 'Não'}")
    
    # Decomposição sazonal
    decomposition = time_series.seasonal_decomposition(ts_data, period=96)
    
    if 'error' not in decomposition:
        print("   ✅ Decomposição sazonal realizada com sucesso")
    
    # 4. Detecção de anomalias
    print("\n4. ⚠️ Detecção de anomalias...")
    anomaly_detector = AnomalyDetector()
    
    # Detectar anomalias usando Isolation Forest
    df_anomalies = anomaly_detector.detect_anomalies_isolation_forest(
        df_clean, parameters, contamination=0.1
    )
    
    anomaly_count = df_anomalies['isolation_forest_anomaly'].sum()
    print(f"   🔍 Anomalias detectadas: {anomaly_count}")
    
    # 5. Sistema de alertas
    print("\n5. 🚨 Sistema de alertas...")
    alert_system = AlertSystem()
    
    # Configurar limites
    thresholds = {
        'pH': {'min': 6.5, 'max': 8.5, 'critical_min': 6.0, 'critical_max': 9.0},
        'turbidity': {'max': 5.0, 'critical_max': 10.0},
        'chlorine': {'min': 0.5, 'max': 2.0, 'critical_min': 0.3, 'critical_max': 3.0}
    }
    
    # Verificar alertas de limite
    alerts = alert_system.check_threshold_alerts(df_clean, thresholds)
    print(f"   🚨 Alertas de limite: {len(alerts)}")
    
    for alert in alerts[:3]:  # Mostrar apenas os 3 primeiros
        print(f"      - {alert['parameter']}: {alert['value']:.2f} ({alert['severity']})")
    
    # 6. Visualizações
    print("\n6. 📊 Criando visualizações...")
    visualizer = DashboardVisualizer()
    
    # Criar gráfico de tempo real
    fig = visualizer.create_realtime_chart(df_clean, parameters[:2])
    print("   ✅ Gráfico de tempo real criado")
    
    # Criar matriz de correlação
    if len(parameters) > 1:
        fig_corr = visualizer.create_correlation_heatmap(df_clean, parameters)
        print("   ✅ Matriz de correlação criada")
    
    # 7. Previsões
    print("\n7. 🔮 Gerando previsões...")
    
    # Tentar modelo Prophet
    try:
        prophet_result = time_series.fit_prophet(df_clean, 'pH')
        if 'error' not in prophet_result:
            print("   ✅ Modelo Prophet treinado com sucesso")
        else:
            print(f"   ❌ Erro no Prophet: {prophet_result['error']}")
    except Exception as e:
        print(f"   ⚠️ Prophet não disponível: {str(e)}")
    
    # Tentar modelo ARIMA
    try:
        arima_result = time_series.fit_arima(ts_data, order=(1, 1, 1))
        if 'error' not in arima_result:
            print("   ✅ Modelo ARIMA treinado com sucesso")
            print(f"      AIC: {arima_result['aic']:.2f}")
        else:
            print(f"   ❌ Erro no ARIMA: {arima_result['error']}")
    except Exception as e:
        print(f"   ⚠️ ARIMA não disponível: {str(e)}")
    
    # 8. Resumo final
    print("\n" + "=" * 60)
    print("📋 RESUMO DA ANÁLISE")
    print("=" * 60)
    
    print(f"📊 Dados processados: {len(df_clean)} registros")
    print(f"📅 Período analisado: {df_clean['timestamp'].min().strftime('%d/%m/%Y')} a {df_clean['timestamp'].max().strftime('%d/%m/%Y')}")
    print(f"🔍 Anomalias detectadas: {anomaly_count}")
    print(f"🚨 Alertas gerados: {len(alerts)}")
    print(f"📈 Parâmetros analisados: {len(parameters)}")
    
    # Qualidade dos dados
    data_quality = data_loader.validate_data_quality(df_clean)
    print(f"📊 Qualidade dos dados:")
    for param, completeness in data_quality['data_completeness'].items():
        print(f"      {param}: {completeness}% completo")
    
    print("\n🎉 Análise concluída com sucesso!")
    print("\n💡 Para usar a interface web, execute:")
    print("   streamlit run app.py")

if __name__ == "__main__":
    main()
