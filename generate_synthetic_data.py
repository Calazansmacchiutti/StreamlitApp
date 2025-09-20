"""
Script para gerar dados sintéticos realistas para demonstração
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_station_data(station_name, days=90):
    """Gerar dados sintéticos realistas para uma estação"""
    
    # Configurações específicas por estação
    station_configs = {
        'Two Mouths': {
            'parameters': ['pH', 'turbidity', 'chlorine', 'flow_rate', 'temperature'],
            'base_values': {
                'pH': 7.2, 'turbidity': 2.0, 'chlorine': 1.2, 
                'flow_rate': 1000, 'temperature': 20
            },
            'variations': {
                'pH': 0.3, 'turbidity': 1.0, 'chlorine': 0.2,
                'flow_rate': 200, 'temperature': 3
            }
        },
        'New Rose of Rocky': {
            'parameters': ['pH', 'DO', 'BOD', 'COD', 'TSS'],
            'base_values': {
                'pH': 7.0, 'DO': 6.5, 'BOD': 15, 'COD': 50, 'TSS': 25
            },
            'variations': {
                'pH': 0.2, 'DO': 1.0, 'BOD': 5, 'COD': 15, 'TSS': 8
            }
        },
        'Botanic Garden': {
            'parameters': ['pH', 'nitrogen', 'phosphorus', 'coliform', 'turbidity'],
            'base_values': {
                'pH': 7.1, 'nitrogen': 5.0, 'phosphorus': 1.5, 
                'coliform': 100, 'turbidity': 1.8
            },
            'variations': {
                'pH': 0.25, 'nitrogen': 2.0, 'phosphorus': 0.5,
                'coliform': 50, 'turbidity': 0.8
            }
        }
    }
    
    config = station_configs[station_name]
    
    # Gerar timestamps (dados a cada 15 minutos)
    start_date = datetime.now() - timedelta(days=days)
    dates = pd.date_range(
        start=start_date,
        end=datetime.now(),
        freq='15T'
    )
    
    data = {'timestamp': dates}
    
    # Gerar dados para cada parâmetro
    for param in config['parameters']:
        base_value = config['base_values'][param]
        variation = config['variations'][param]
        
        # Padrões sazonais e temporais
        n_points = len(dates)
        
        if param == 'pH':
            # pH com variação sazonal suave
            seasonal = 0.2 * np.sin(np.arange(n_points) * 2 * np.pi / (96*7))  # Semanal
            daily = 0.1 * np.sin(np.arange(n_points) * 2 * np.pi / 96)  # Diário
            noise = np.random.normal(0, 0.05, n_points)
            data[param] = base_value + seasonal + daily + noise
            
        elif param == 'turbidity':
            # Turbidez com distribuição log-normal
            log_mean = np.log(base_value)
            log_std = 0.3
            data[param] = np.random.lognormal(log_mean, log_std, n_points)
            
        elif param == 'chlorine':
            # Cloro com padrão diário
            daily_pattern = 0.3 * np.sin(np.arange(n_points) * 2 * np.pi / 96)  # Diário
            weekly_pattern = 0.1 * np.sin(np.arange(n_points) * 2 * np.pi / (96*7))  # Semanal
            noise = np.random.normal(0, 0.05, n_points)
            data[param] = base_value + daily_pattern + weekly_pattern + noise
            
        elif param == 'flow_rate':
            # Vazão com padrão diário e sazonal
            daily_pattern = 300 * np.sin(np.arange(n_points) * 2 * np.pi / 96)  # Diário
            weekly_pattern = 100 * np.sin(np.arange(n_points) * 2 * np.pi / (96*7))  # Semanal
            noise = np.random.normal(0, 50, n_points)
            data[param] = base_value + daily_pattern + weekly_pattern + noise
            
        elif param == 'temperature':
            # Temperatura com variação sazonal e diária
            seasonal = 4 * np.sin(np.arange(n_points) * 2 * np.pi / (96*365))  # Anual
            daily = 2 * np.sin(np.arange(n_points) * 2 * np.pi / 96)  # Diário
            noise = np.random.normal(0, 0.5, n_points)
            data[param] = base_value + seasonal + daily + noise
            
        elif param == 'DO':  # Dissolved Oxygen
            # DO com padrão diário (mais baixo à noite)
            daily_pattern = 1.5 * np.sin(np.arange(n_points) * 2 * np.pi / 96 + np.pi/2)  # Diário
            noise = np.random.normal(0, 0.2, n_points)
            data[param] = base_value + daily_pattern + noise
            
        elif param in ['BOD', 'COD', 'TSS']:
            # Parâmetros orgânicos com variação sazonal
            seasonal = 0.3 * base_value * np.sin(np.arange(n_points) * 2 * np.pi / (96*365))  # Anual
            noise = np.random.normal(0, 0.1 * base_value, n_points)
            data[param] = base_value + seasonal + noise
            
        elif param in ['nitrogen', 'phosphorus']:
            # Nutrientes com padrão sazonal
            seasonal = 0.4 * base_value * np.sin(np.arange(n_points) * 2 * np.pi / (96*365))  # Anual
            noise = np.random.normal(0, 0.15 * base_value, n_points)
            data[param] = base_value + seasonal + noise
            
        elif param == 'coliform':
            # Coliformes com distribuição log-normal
            log_mean = np.log(base_value)
            log_std = 0.5
            data[param] = np.random.lognormal(log_mean, log_std, n_points)
            
        else:
            # Outros parâmetros com variação normal
            noise = np.random.normal(0, 0.1 * base_value, n_points)
            data[param] = base_value + noise
    
    # Adicionar alguns outliers e anomalias para tornar mais realista
    for param in config['parameters']:
        # 1% de outliers
        n_outliers = int(0.01 * len(data[param]))
        outlier_indices = np.random.choice(len(data[param]), n_outliers, replace=False)
        
        for idx in outlier_indices:
            # Outliers podem ser 2-3x maiores ou menores
            multiplier = np.random.choice([0.3, 0.5, 2.0, 3.0])
            data[param][idx] *= multiplier
    
    # Adicionar alguns valores faltantes (0.5% dos dados)
    for param in config['parameters']:
        n_missing = int(0.005 * len(data[param]))
        missing_indices = np.random.choice(len(data[param]), n_missing, replace=False)
        for idx in missing_indices:
            data[param][idx] = np.nan
    
    return pd.DataFrame(data)

def main():
    """Gerar dados para todas as estações"""
    
    stations = ['Two Mouths', 'New Rose of Rocky', 'Botanic Garden']
    data_dir = 'data'
    
    # Criar diretório se não existir
    os.makedirs(data_dir, exist_ok=True)
    
    print("Gerando dados sintéticos para demonstração...")
    
    for station in stations:
        print(f"Gerando dados para {station}...")
        
        # Gerar dados para 90 dias
        df = generate_station_data(station, days=90)
        
        # Salvar arquivo CSV
        filename = os.path.join(data_dir, f"{station}.csv")
        df.to_csv(filename, index=False)
        
        print(f"✅ Dados salvos em {filename}")
        print(f"   - {len(df)} registros")
        print(f"   - {len(df.columns)-1} parâmetros")
        print(f"   - Período: {df['timestamp'].min()} a {df['timestamp'].max()}")
        print()
    
    print("🎉 Geração de dados concluída!")
    print("\nArquivos gerados:")
    for station in stations:
        filename = os.path.join(data_dir, f"{station}.csv")
        if os.path.exists(filename):
            file_size = os.path.getsize(filename) / 1024  # KB
            print(f"  - {filename} ({file_size:.1f} KB)")

if __name__ == "__main__":
    main()
