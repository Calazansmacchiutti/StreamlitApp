"""
Constantes e configurações centralizadas do sistema
"""

from pathlib import Path
from typing import Dict, Set, List
from dataclasses import dataclass
from enum import Enum

# =============================================================================
# CAMINHOS E DIRETÓRIOS
# =============================================================================

# Diretório raiz do projeto
PROJECT_ROOT = Path(__file__).parent.parent

# Diretórios principais
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "config"
MODULES_DIR = PROJECT_ROOT / "modules"
PAGES_DIR = PROJECT_ROOT / "pages"

# Arquivos de configuração
CONFIG_FILE = CONFIG_DIR / "config.yaml"
ADVANCED_CONFIG_FILE = CONFIG_DIR / "config_advanced.yaml"

# =============================================================================
# SEGURANÇA
# =============================================================================

# Estações permitidas (whitelist para segurança)
ALLOWED_STATIONS: Set[str] = {
    'Two Mouths',
    'New Rose of Rocky',
    'Botanic Garden'
}

# Extensões de arquivo permitidas
ALLOWED_FILE_EXTENSIONS: Set[str] = {'.csv', '.xlsx', '.json'}

# Tamanho máximo de arquivo (em bytes) - 50MB
MAX_FILE_SIZE: int = 50 * 1024 * 1024

# Regex para validação de nomes seguros
SAFE_NAME_PATTERN = r'^[a-zA-Z_][a-zA-Z0-9_\s-]*$'

# =============================================================================
# INTERFACE DO USUÁRIO
# =============================================================================

# Configuração da página Streamlit
PAGE_CONFIG = {
    "page_title": "Sistema de Monitoramento - Estações de Tratamento",
    "page_icon": "🌊",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "menu_items": {
        'Get Help': 'https://www.example.com/help',
        'Report a bug': "https://www.example.com/bug",
        'About': "Sistema de Monitoramento v3.0"
    }
}

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
    .alert-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .alert-critical {
        background-color: #fee;
        border-left: 4px solid #f44336;
    }
    .alert-warning {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
    }
    .alert-info {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
    }
    </style>
"""

# =============================================================================
# PARÂMETROS DAS ESTAÇÕES
# =============================================================================

class StationType(Enum):
    """Tipos de estações de tratamento"""
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"

@dataclass
class StationConfig:
    """Configuração de uma estação"""
    name: str
    type: StationType
    capacity: int
    parameters: List[str]
    data_file: str

# Configurações das estações
STATION_CONFIGS: Dict[str, StationConfig] = {
    'Two Mouths': StationConfig(
        name='Two Mouths',
        type=StationType.PRIMARY,
        capacity=10000,
        parameters=['pH', 'turbidity', 'chlorine', 'flow_rate', 'temperature'],
        data_file='Two Mouths.csv'
    ),
    'New Rose of Rocky': StationConfig(
        name='New Rose of Rocky',
        type=StationType.SECONDARY,
        capacity=8000,
        parameters=['pH', 'DO', 'BOD', 'COD', 'TSS'],
        data_file='New Rose of Rocky.csv'
    ),
    'Botanic Garden': StationConfig(
        name='Botanic Garden',
        type=StationType.TERTIARY,
        capacity=5000,
        parameters=['pH', 'nitrogen', 'phosphorus', 'coliform', 'turbidity'],
        data_file='Botanic Garden.csv'
    )
}

# =============================================================================
# LIMITES E THRESHOLDS
# =============================================================================

@dataclass
class ParameterThreshold:
    """Limites para um parâmetro"""
    min_value: float = None
    max_value: float = None
    critical_min: float = None
    critical_max: float = None
    unit: str = ""

# Thresholds dos parâmetros
PARAMETER_THRESHOLDS: Dict[str, ParameterThreshold] = {
    'pH': ParameterThreshold(
        min_value=6.5, max_value=8.5,
        critical_min=6.0, critical_max=9.0,
        unit=""
    ),
    'turbidity': ParameterThreshold(
        max_value=5.0, critical_max=10.0,
        unit="NTU"
    ),
    'chlorine': ParameterThreshold(
        min_value=0.5, max_value=2.0,
        critical_min=0.3, critical_max=3.0,
        unit="mg/L"
    ),
    'temperature': ParameterThreshold(
        min_value=15, max_value=30,
        critical_min=10, critical_max=35,
        unit="°C"
    ),
    'DO': ParameterThreshold(
        min_value=5.0, critical_min=3.0,
        unit="mg/L"
    ),
    'BOD': ParameterThreshold(
        max_value=30, critical_max=50,
        unit="mg/L"
    ),
    'COD': ParameterThreshold(
        max_value=100, critical_max=150,
        unit="mg/L"
    ),
    'nitrogen': ParameterThreshold(
        max_value=10, critical_max=15,
        unit="mg/L"
    ),
    'phosphorus': ParameterThreshold(
        max_value=2, critical_max=5,
        unit="mg/L"
    ),
    'coliform': ParameterThreshold(
        max_value=1000, critical_max=5000,
        unit="CFU/100mL"
    ),
    'TSS': ParameterThreshold(
        max_value=30, critical_max=50,
        unit="mg/L"
    ),
    'flow_rate': ParameterThreshold(
        min_value=500, max_value=1500,
        unit="m³/h"
    )
}

# =============================================================================
# CONFIGURAÇÕES DE TEMPO
# =============================================================================

# Frequências de amostragem
SAMPLING_FREQUENCIES = {
    '1min': '1T',
    '5min': '5T',
    '15min': '15T',
    '30min': '30T',
    '1hour': '1H',
    '1day': '1D'
}

# Intervalos de tempo predefinidos
TIME_RANGES = {
    "Última hora": {"hours": 1},
    "Últimas 24 horas": {"hours": 24},
    "Última semana": {"weeks": 1},
    "Último mês": {"days": 30},
    "Últimos 3 meses": {"days": 90}
}

# Configurações de cache
CACHE_CONFIG = {
    'ttl': 300,  # 5 minutos
    'max_entries': 100,
    'show_spinner': True
}

# =============================================================================
# CONFIGURAÇÕES DE ANÁLISE
# =============================================================================

# Algoritmos de machine learning disponíveis
ML_ALGORITHMS = {
    'Random Forest': 'random_forest',
    'XGBoost': 'xgboost',
    'SVR': 'svr',
    'Linear Regression': 'linear_regression'
}

# Métodos de detecção de anomalias
ANOMALY_DETECTION_METHODS = {
    'Isolation Forest': 'isolation_forest',
    'DBSCAN': 'dbscan',
    'One-Class SVM': 'one_class_svm',
    'Z-Score': 'zscore',
    'IQR': 'iqr'
}

# Modelos de séries temporais
TIME_SERIES_MODELS = {
    'ARIMA': 'arima',
    'Prophet': 'prophet',
    'Seasonal Decompose': 'seasonal_decompose',
    'LSTM': 'lstm'
}

# =============================================================================
# CONFIGURAÇÕES DE EXPORTAÇÃO
# =============================================================================

# Formatos de exportação suportados
EXPORT_FORMATS = {
    'CSV': 'csv',
    'Excel': 'xlsx',
    'JSON': 'json',
    'PDF': 'pdf'
}

# Tipos de relatório
REPORT_TYPES = {
    'Operacional': 'operational',
    'Conformidade': 'compliance',
    'Manutenção': 'maintenance',
    'Executivo': 'executive'
}

# =============================================================================
# CONFIGURAÇÕES DE LOG
# =============================================================================

# Níveis de log
LOG_LEVELS = {
    'DEBUG': 10,
    'INFO': 20,
    'WARNING': 30,
    'ERROR': 40,
    'CRITICAL': 50
}

# Formato padrão de log
LOG_FORMAT = {
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S'
}

# Arquivo de log
LOG_FILE = PROJECT_ROOT / "logs" / "app.log"

# =============================================================================
# MENSAGENS PADRÃO
# =============================================================================

# Mensagens de erro
ERROR_MESSAGES = {
    'station_not_found': "Estação '{station}' não encontrada",
    'file_not_found': "Arquivo '{file}' não encontrado",
    'invalid_data': "Dados inválidos ou corrompidos",
    'module_import_error': "Erro ao importar módulo: {module}",
    'database_error': "Erro de conexão com o banco de dados",
    'validation_error': "Erro de validação: {error}",
    'permission_denied': "Permissão negada para acessar {resource}"
}

# Mensagens de sucesso
SUCCESS_MESSAGES = {
    'data_loaded': "Dados carregados com sucesso",
    'export_complete': "Exportação concluída: {file}",
    'backup_created': "Backup criado: {file}",
    'configuration_saved': "Configuração salva com sucesso"
}

# Mensagens de aviso
WARNING_MESSAGES = {
    'large_dataset': "Dataset grande detectado, operação pode ser lenta",
    'missing_data': "Dados faltantes detectados em {columns}",
    'threshold_exceeded': "Limite excedido para {parameter}: {value}",
    'cache_miss': "Cache miss para {key}, recarregando dados"
}

# =============================================================================
# CONFIGURAÇÕES DE PERFORMANCE
# =============================================================================

# Limites para otimização
PERFORMANCE_LIMITS = {
    'max_plot_points': 10000,  # Máximo de pontos em gráficos
    'max_correlation_vars': 20,  # Máximo de variáveis para correlação
    'sample_size_large_dataset': 100000,  # Amostra para datasets grandes
    'max_forecast_periods': 1000,  # Máximo de períodos para previsão
    'max_memory_usage_mb': 500  # Máximo uso de memória em MB
}

# Configurações de otimização
OPTIMIZATION_CONFIG = {
    'use_sampling': True,
    'enable_multiprocessing': True,
    'chunk_size': 10000,
    'lazy_loading': True
}

# =============================================================================
# VERSÃO E METADADOS
# =============================================================================

APP_VERSION = "3.0.0"
APP_NAME = "Sistema de Monitoramento de Estações de Tratamento"
LAST_UPDATE = "2024-12-21"
AUTHORS = ["Equipe de Desenvolvimento"]
LICENSE = "MIT"