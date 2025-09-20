"""
Módulo para carregamento e manipulação de dados das estações de tratamento
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import streamlit as st
from typing import Optional, Dict, List, Tuple
import sqlite3
import json
import pickle
from functools import lru_cache


class DataLoader:
    """Classe para carregamento e manipulação de dados"""
    
    def __init__(self):
        self.data_path = Path(__file__).parent.parent / "data"
        self.data_cache = {}
        self.cache_enabled = True
        
    def load_csv(self, file_path: str, cache: bool = True) -> pd.DataFrame:
        """
        Carregar dados de arquivo CSV com cache opcional
        
        Args:
            file_path: Caminho para o arquivo CSV
            cache: Se deve usar cache
            
        Returns:
            DataFrame com os dados carregados
        """
        if cache and self.cache_enabled and file_path in self.data_cache:
            return self.data_cache[file_path].copy()
            
        try:
            df = pd.read_csv(file_path)
            
            # Converter timestamp se existir
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            elif 'datetime' in df.columns:
                df['timestamp'] = pd.to_datetime(df['datetime'])
                df = df.drop('datetime', axis=1)
            else:
                # Criar timestamp se não existir
                df['timestamp'] = pd.date_range(
                    start=datetime.now() - timedelta(days=30),
                    periods=len(df),
                    freq='15T'
                )
            
            # Cache dos dados
            if cache and self.cache_enabled:
                self.data_cache[file_path] = df.copy()
                
            return df
            
        except Exception as e:
            st.error(f"Erro ao carregar arquivo {file_path}: {str(e)}")
            return pd.DataFrame()
            
    def load_from_database(self, connection_string: str, query: str) -> pd.DataFrame:
        """
        Carregar dados de banco de dados
        
        Args:
            connection_string: String de conexão com o banco
            query: Query SQL para executar
            
        Returns:
            DataFrame com os dados
        """
        try:
            # Para SQLite (exemplo)
            if connection_string.startswith('sqlite:///'):
                db_path = connection_string.replace('sqlite:///', '')
                conn = sqlite3.connect(db_path)
                df = pd.read_sql_query(query, conn)
                conn.close()
                
                # Converter timestamp se existir
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    
                return df
            else:
                st.error("Tipo de banco de dados não suportado")
                return pd.DataFrame()
                
        except Exception as e:
            st.error(f"Erro ao carregar dados do banco: {str(e)}")
            return pd.DataFrame()
            
    def save_to_database(self, df: pd.DataFrame, connection_string: str, 
                        table_name: str) -> bool:
        """
        Salvar DataFrame no banco de dados
        
        Args:
            df: DataFrame para salvar
            connection_string: String de conexão
            table_name: Nome da tabela
            
        Returns:
            True se salvou com sucesso
        """
        try:
            if connection_string.startswith('sqlite:///'):
                db_path = connection_string.replace('sqlite:///', '')
                conn = sqlite3.connect(db_path)
                df.to_sql(table_name, conn, if_exists='replace', index=False)
                conn.close()
                return True
            else:
                st.error("Tipo de banco de dados não suportado")
                return False
                
        except Exception as e:
            st.error(f"Erro ao salvar no banco: {str(e)}")
            return False
            
    def load_station_data(self, station_name: str) -> pd.DataFrame:
        """
        Carregar dados de uma estação específica
        
        Args:
            station_name: Nome da estação
            
        Returns:
            DataFrame com os dados da estação
        """
        file_path = self.data_path / f"{station_name}.csv"
        
        if file_path.exists():
            return self.load_csv(str(file_path))
        else:
            st.warning(f"Arquivo não encontrado para a estação {station_name}")
            return pd.DataFrame()
            
    def filter_by_date(self, df: pd.DataFrame, date_range: Tuple[datetime, datetime]) -> pd.DataFrame:
        """
        Filtrar dados por intervalo de datas
        
        Args:
            df: DataFrame com os dados
            date_range: Tupla com (data_inicial, data_final)
            
        Returns:
            DataFrame filtrado
        """
        if 'timestamp' not in df.columns:
            return df
            
        start_date, end_date = date_range
        mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
        return df.loc[mask].copy()
        
    def filter_by_parameter(self, df: pd.DataFrame, parameters: List[str]) -> pd.DataFrame:
        """
        Filtrar dados por parâmetros específicos
        
        Args:
            df: DataFrame com os dados
            parameters: Lista de parâmetros para incluir
            
        Returns:
            DataFrame filtrado
        """
        if not parameters:
            return df
            
        # Sempre incluir timestamp
        columns_to_keep = ['timestamp'] + [p for p in parameters if p in df.columns]
        return df[columns_to_keep].copy()
        
    def get_latest_data(self, df: pd.DataFrame, hours: int = 24) -> pd.DataFrame:
        """
        Obter dados mais recentes
        
        Args:
            df: DataFrame com os dados
            hours: Número de horas para retornar
            
        Returns:
            DataFrame com os dados mais recentes
        """
        if 'timestamp' not in df.columns:
            return df.tail(96)  # Últimos 96 pontos (24h com dados a cada 15min)
            
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return df[df['timestamp'] >= cutoff_time].copy()
        
    def resample_data(self, df: pd.DataFrame, freq: str = '1H') -> pd.DataFrame:
        """
        Reamostrar dados para uma frequência específica
        
        Args:
            df: DataFrame com os dados
            freq: Frequência de reamostragem
            
        Returns:
            DataFrame reamostrado
        """
        if 'timestamp' not in df.columns:
            return df
            
        df_resampled = df.set_index('timestamp').resample(freq).agg({
            col: 'mean' if df[col].dtype in ['float64', 'int64'] else 'first'
            for col in df.columns if col != 'timestamp'
        }).reset_index()
        
        return df_resampled
        
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validar qualidade dos dados
        
        Args:
            df: DataFrame com os dados
            
        Returns:
            Dicionário com métricas de qualidade
        """
        quality_metrics = {
            'total_records': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_records': df.duplicated().sum(),
            'date_range': {
                'start': df['timestamp'].min() if 'timestamp' in df.columns else None,
                'end': df['timestamp'].max() if 'timestamp' in df.columns else None
            },
            'data_completeness': {}
        }
        
        # Calcular completude dos dados
        for col in df.columns:
            if col != 'timestamp':
                completeness = (1 - df[col].isnull().sum() / len(df)) * 100
                quality_metrics['data_completeness'][col] = round(completeness, 2)
                
        return quality_metrics
        
    def generate_synthetic_data(self, station_name: str, days: int = 30) -> pd.DataFrame:
        """
        Gerar dados sintéticos para demonstração
        
        Args:
            station_name: Nome da estação
            days: Número de dias de dados para gerar
            
        Returns:
            DataFrame com dados sintéticos
        """
        np.random.seed(42)
        
        # Configurações por estação
        station_configs = {
            'Two Mouths': {
                'parameters': ['pH', 'turbidity', 'chlorine', 'flow_rate', 'temperature'],
                'base_values': {'pH': 7.2, 'turbidity': 2.0, 'chlorine': 1.2, 'flow_rate': 1000, 'temperature': 20}
            },
            'New Rose of Rocky': {
                'parameters': ['pH', 'DO', 'BOD', 'COD', 'TSS'],
                'base_values': {'pH': 7.0, 'DO': 6.5, 'BOD': 15, 'COD': 50, 'TSS': 25}
            },
            'Botanic Garden': {
                'parameters': ['pH', 'nitrogen', 'phosphorus', 'coliform', 'turbidity'],
                'base_values': {'pH': 7.1, 'nitrogen': 5.0, 'phosphorus': 1.5, 'coliform': 100, 'turbidity': 1.8}
            }
        }
        
        config = station_configs.get(station_name, station_configs['Two Mouths'])
        
        # Gerar timestamps
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=days),
            end=datetime.now(),
            freq='15T'
        )
        
        data = {'timestamp': dates}
        
        # Gerar dados para cada parâmetro
        for param in config['parameters']:
            base_value = config['base_values'][param]
            
            if param == 'pH':
                # pH com variação sazonal
                seasonal = 0.3 * np.sin(np.arange(len(dates)) * 2 * np.pi / (96*7))  # Semanal
                noise = np.random.normal(0, 0.1, len(dates))
                data[param] = base_value + seasonal + noise
                
            elif param == 'turbidity':
                # Turbidez com distribuição exponencial
                data[param] = np.random.exponential(base_value, len(dates))
                
            elif param == 'chlorine':
                # Cloro com variação diária
                daily = 0.2 * np.sin(np.arange(len(dates)) * 2 * np.pi / 96)  # Diário
                noise = np.random.normal(0, 0.05, len(dates))
                data[param] = base_value + daily + noise
                
            elif param == 'flow_rate':
                # Vazão com padrão diário
                daily_pattern = 200 * np.sin(np.arange(len(dates)) * 2 * np.pi / 96)  # Diário
                noise = np.random.normal(0, 50, len(dates))
                data[param] = base_value + daily_pattern + noise
                
            elif param == 'temperature':
                # Temperatura com variação sazonal e diária
                seasonal = 5 * np.sin(np.arange(len(dates)) * 2 * np.pi / (96*365))  # Anual
                daily = 2 * np.sin(np.arange(len(dates)) * 2 * np.pi / 96)  # Diário
                noise = np.random.normal(0, 0.5, len(dates))
                data[param] = base_value + seasonal + daily + noise
                
            else:
                # Outros parâmetros com variação normal
                noise = np.random.normal(0, base_value * 0.1, len(dates))
                data[param] = base_value + noise
                
        return pd.DataFrame(data)
        
    def export_data(self, df: pd.DataFrame, format: str = 'csv') -> bytes:
        """
        Exportar dados em diferentes formatos
        
        Args:
            df: DataFrame com os dados
            format: Formato de exportação ('csv', 'excel', 'json')
            
        Returns:
            Dados em bytes
        """
        if format.lower() == 'csv':
            return df.to_csv(index=False).encode('utf-8')
        elif format.lower() == 'excel':
            import io
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Data')
            return output.getvalue()
        elif format.lower() == 'json':
            return df.to_json(orient='records', date_format='iso').encode('utf-8')
        else:
            raise ValueError(f"Formato {format} não suportado")
            
    def clear_cache(self):
        """Limpar cache de dados"""
        self.data_cache.clear()
        
    def get_cache_info(self) -> Dict:
        """Obter informações do cache"""
        return {
            'cache_enabled': self.cache_enabled,
            'cached_files': list(self.data_cache.keys()),
            'cache_size': len(self.data_cache),
            'memory_usage': sum(df.memory_usage(deep=True).sum() for df in self.data_cache.values())
        }
        
    def enable_cache(self, enabled: bool = True):
        """Habilitar/desabilitar cache"""
        self.cache_enabled = enabled
        if not enabled:
            self.clear_cache()
            
    @lru_cache(maxsize=128)
    def get_station_metadata(self, station_name: str) -> Dict:
        """Obter metadados da estação (com cache LRU)"""
        metadata = {
            'name': station_name,
            'last_update': datetime.now(),
            'data_points': 0,
            'parameters': [],
            'date_range': None
        }
        
        df = self.load_station_data(station_name)
        if not df.empty:
            metadata.update({
                'data_points': len(df),
                'parameters': [col for col in df.columns if col != 'timestamp'],
                'date_range': (df['timestamp'].min(), df['timestamp'].max())
            })
            
        return metadata
        
    def backup_data(self, station_name: str, backup_path: str = None) -> bool:
        """Fazer backup dos dados da estação"""
        try:
            df = self.load_station_data(station_name)
            if df.empty:
                return False
                
            if backup_path is None:
                backup_path = self.data_path / f"{station_name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                
            df.to_csv(backup_path, index=False)
            return True
            
        except Exception as e:
            st.error(f"Erro no backup: {str(e)}")
            return False
            
    def restore_data(self, station_name: str, backup_path: str) -> bool:
        """Restaurar dados de backup"""
        try:
            df = pd.read_csv(backup_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Salvar dados restaurados
            original_path = self.data_path / f"{station_name}.csv"
            df.to_csv(original_path, index=False)
            
            # Limpar cache
            if original_path in self.data_cache:
                del self.data_cache[original_path]
                
            return True
            
        except Exception as e:
            st.error(f"Erro na restauração: {str(e)}")
            return False
