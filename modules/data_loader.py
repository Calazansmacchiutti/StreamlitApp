"""
Module for loading and manipulating water treatment station data
Optimized version with Streamlit cache and security validation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import streamlit as st
from typing import Optional, Dict, List, Tuple, Union, Any
import sqlite3
import json
import pickle
from functools import lru_cache
import re
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Class for loading and manipulating data with security and optimized performance

    Attributes:
        data_path (Path): Path to data directory
        allowed_stations (set): Set of allowed stations
    """

    def __init__(self) -> None:
        """Initialize DataLoader with secure configurations"""
        self.data_path: Path = Path(__file__).parent.parent / "data"

        # List of allowed stations to prevent path traversal
        self.allowed_stations: set[str] = {
            'Two Mouths', 'New Rose of Rocky', 'Botanic Garden'
        }

        logger.info(f"DataLoader initialized with data_path: {self.data_path}")

    @staticmethod
    @st.cache_data(ttl=300, show_spinner=True)
    def load_csv_cached(file_path: str) -> pd.DataFrame:
        """
        Carregar arquivo CSV com cache do Streamlit

        Args:
            file_path: Caminho validado para o arquivo CSV

        Returns:
            DataFrame com os dados carregados
        """
        try:
            logger.info(f"Carregando arquivo: {file_path}")
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

            logger.info(f"Arquivo carregado: {len(df)} registros")
            return df

        except Exception as e:
            logger.error(f"Erro ao carregar arquivo {file_path}: {str(e)}")
            st.error(f"Erro ao carregar arquivo {file_path}: {str(e)}")
            return pd.DataFrame()

    def _validate_station_name(self, station_name: str) -> str:
        """
        Validar e sanitizar nome da estação para prevenir path traversal

        Args:
            station_name: Nome da estação a ser validado

        Returns:
            Nome da estação validado

        Raises:
            ValueError: Se o nome da estação for inválido
        """
        if not station_name or not isinstance(station_name, str):
            raise ValueError("Nome da estação deve ser uma string não vazia")

        # Remover caracteres perigosos
        sanitized_name = re.sub(r'[^\w\s-]', '', station_name.strip())

        # Verificar se está na lista de estações permitidas
        if sanitized_name not in self.allowed_stations:
            raise ValueError(f"Estação '{station_name}' não permitida. Estações válidas: {list(self.allowed_stations)}")

        return sanitized_name

    def _validate_file_path(self, file_path: str) -> Path:
        """
        Validar caminho do arquivo para prevenir path traversal

        Args:
            file_path: Caminho do arquivo

        Returns:
            Path validado

        Raises:
            ValueError: Se o caminho for inválido ou inseguro
        """
        try:
            path = Path(file_path).resolve()
            data_path = self.data_path.resolve()

            # Verificar se o path está dentro do diretório de dados permitido
            if not str(path).startswith(str(data_path)):
                raise ValueError("Caminho do arquivo fora do diretório permitido")

            return path
        except Exception as e:
            raise ValueError(f"Caminho de arquivo inválido: {str(e)}")
        
    def load_csv(self, file_path: str, cache: bool = True) -> pd.DataFrame:
        """
        Carregar dados de arquivo CSV com validação e cache

        Args:
            file_path: Caminho para o arquivo CSV
            cache: Se deve usar cache (agora usa @st.cache_data)

        Returns:
            DataFrame com os dados carregados
        """
        # Validar caminho do arquivo
        validated_path = self._validate_file_path(file_path)
        file_path_str = str(validated_path)

        # Usar cache do Streamlit em vez de cache manual
        if cache:
            return self.load_csv_cached(file_path_str)
        else:
            # Carregamento direto sem cache
            try:
                df = pd.read_csv(validated_path)

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

                return df

            except Exception as e:
                logger.error(f"Erro ao carregar arquivo {file_path}: {str(e)}")
                st.error(f"Erro ao carregar arquivo {file_path}: {str(e)}")
                return pd.DataFrame()
            
    def _validate_database_path(self, connection_string: str) -> str:
        """
        Validar string de conexão do banco de dados

        Args:
            connection_string: String de conexão

        Returns:
            String de conexão validada

        Raises:
            ValueError: Se a conexão for inválida
        """
        if not connection_string or not isinstance(connection_string, str):
            raise ValueError("String de conexão deve ser não vazia")

        if connection_string.startswith('sqlite:///'):
            db_path = connection_string.replace('sqlite:///', '')
            # Validar que o path do banco está em local seguro
            validated_path = self._validate_file_path(db_path)
            return f"sqlite:///{validated_path}"
        else:
            raise ValueError("Apenas conexões SQLite são suportadas")

    def load_from_database(self, connection_string: str, query: str, params: Optional[Tuple] = None) -> pd.DataFrame:
        """
        Carregar dados de banco de dados com proteção contra SQL injection

        Args:
            connection_string: String de conexão com o banco
            query: Query SQL para executar (use ? para parâmetros)
            params: Parâmetros para a query (para evitar SQL injection)

        Returns:
            DataFrame com os dados
        """
        try:
            # Validar string de conexão
            validated_connection = self._validate_database_path(connection_string)

            # Validar query (básico - apenas permitir SELECT)
            query_clean = query.strip().upper()
            if not query_clean.startswith('SELECT'):
                raise ValueError("Apenas queries SELECT são permitidas")

            # Para SQLite
            if validated_connection.startswith('sqlite:///'):
                db_path = validated_connection.replace('sqlite:///', '')
                conn = sqlite3.connect(db_path)

                # Usar parâmetros para evitar SQL injection
                if params:
                    df = pd.read_sql_query(query, conn, params=params)
                else:
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
            
    def _validate_table_name(self, table_name: str) -> str:
        """
        Validar nome da tabela para prevenir SQL injection

        Args:
            table_name: Nome da tabela

        Returns:
            Nome da tabela validado

        Raises:
            ValueError: Se o nome for inválido
        """
        if not table_name or not isinstance(table_name, str):
            raise ValueError("Nome da tabela deve ser uma string não vazia")

        # Apenas permitir caracteres alfanuméricos e underscore
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', table_name):
            raise ValueError("Nome da tabela deve conter apenas letras, números e underscore")

        # Limitar tamanho
        if len(table_name) > 64:
            raise ValueError("Nome da tabela muito longo (máximo 64 caracteres)")

        return table_name

    def save_to_database(self, df: pd.DataFrame, connection_string: str,
                        table_name: str) -> bool:
        """
        Salvar DataFrame no banco de dados com validação de segurança

        Args:
            df: DataFrame para salvar
            connection_string: String de conexão
            table_name: Nome da tabela

        Returns:
            True se salvou com sucesso
        """
        try:
            # Validar inputs
            validated_connection = self._validate_database_path(connection_string)
            validated_table = self._validate_table_name(table_name)

            if validated_connection.startswith('sqlite:///'):
                db_path = validated_connection.replace('sqlite:///', '')
                conn = sqlite3.connect(db_path)
                df.to_sql(validated_table, conn, if_exists='replace', index=False)
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
        Carregar dados de uma estação específica com validação de segurança

        Args:
            station_name: Nome da estação

        Returns:
            DataFrame com os dados da estação
        """
        try:
            # Validar nome da estação
            validated_station = self._validate_station_name(station_name)
            file_path = self.data_path / f"{validated_station}.csv"

            if file_path.exists():
                return self.load_csv(str(file_path))
            else:
                st.warning(f"Arquivo não encontrado para a estação {validated_station}")
                return pd.DataFrame()

        except ValueError as e:
            st.error(f"Erro de validação: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            st.error(f"Erro ao carregar dados da estação: {str(e)}")
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
            
    @staticmethod
    def clear_cache() -> None:
        """Limpar cache do Streamlit"""
        st.cache_data.clear()
        logger.info("Cache do Streamlit limpo")

    @staticmethod
    def get_cache_info() -> Dict[str, Any]:
        """Obter informações do cache do Streamlit"""
        return {
            'cache_type': 'Streamlit @st.cache_data',
            'cache_enabled': True,
            'note': 'Cache gerenciado automaticamente pelo Streamlit'
        }
            
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
        """Fazer backup dos dados da estação com validação de segurança"""
        try:
            # Validar nome da estação
            validated_station = self._validate_station_name(station_name)
            df = self.load_station_data(validated_station)
            if df.empty:
                return False

            if backup_path is None:
                # Gerar path seguro para backup
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_filename = f"{validated_station}_backup_{timestamp}.csv"
                backup_path = self.data_path / backup_filename
            else:
                # Validar path fornecido
                backup_path = self._validate_file_path(backup_path)

            df.to_csv(backup_path, index=False)
            return True

        except Exception as e:
            st.error(f"Erro no backup: {str(e)}")
            return False
            
    def restore_data(self, station_name: str, backup_path: str) -> bool:
        """Restaurar dados de backup com validação de segurança"""
        try:
            # Validar inputs
            validated_station = self._validate_station_name(station_name)
            validated_backup_path = self._validate_file_path(backup_path)

            df = pd.read_csv(validated_backup_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Salvar dados restaurados
            original_path = self.data_path / f"{validated_station}.csv"
            df.to_csv(original_path, index=False)

            # Limpar cache
            original_path_str = str(original_path)
            if original_path_str in self.data_cache:
                del self.data_cache[original_path_str]

            return True

        except Exception as e:
            st.error(f"Erro na restauração: {str(e)}")
            return False
