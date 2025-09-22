"""
Configuração de logging estruturado para o sistema
"""

import logging
import logging.handlers
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import structlog


class JSONFormatter(logging.Formatter):
    """Formatter customizado para logs em JSON"""

    def format(self, record: logging.LogRecord) -> str:
        """Formatar log record como JSON"""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        # Adicionar informações extras se disponíveis
        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id
        if hasattr(record, 'session_id'):
            log_entry['session_id'] = record.session_id
        if hasattr(record, 'station'):
            log_entry['station'] = record.station
        if hasattr(record, 'operation'):
            log_entry['operation'] = record.operation

        # Adicionar informações de exceção se presente
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


class StructuredLogger:
    """Classe para logging estruturado"""

    def __init__(self, name: str = "water_treatment_system"):
        self.name = name
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Configurar logger estruturado"""
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.INFO)

        # Evitar duplicação de handlers
        if logger.handlers:
            return logger

        # Handler para console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)

        # Handler para arquivo
        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"{self.name}.log"

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JSONFormatter())

        # Handler para erros críticos
        error_file = log_dir / f"{self.name}_errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_file,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(JSONFormatter())

        # Adicionar handlers
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        logger.addHandler(error_handler)

        return logger

    def info(self, message: str, **kwargs) -> None:
        """Log de informação com contexto estruturado"""
        extra = self._build_extra(**kwargs)
        self.logger.info(message, extra=extra)

    def warning(self, message: str, **kwargs) -> None:
        """Log de aviso com contexto estruturado"""
        extra = self._build_extra(**kwargs)
        self.logger.warning(message, extra=extra)

    def error(self, message: str, exception: Optional[Exception] = None, **kwargs) -> None:
        """Log de erro com contexto estruturado"""
        extra = self._build_extra(**kwargs)
        if exception:
            self.logger.error(message, exc_info=exception, extra=extra)
        else:
            self.logger.error(message, extra=extra)

    def debug(self, message: str, **kwargs) -> None:
        """Log de debug com contexto estruturado"""
        extra = self._build_extra(**kwargs)
        self.logger.debug(message, extra=extra)

    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs) -> None:
        """Log crítico com contexto estruturado"""
        extra = self._build_extra(**kwargs)
        if exception:
            self.logger.critical(message, exc_info=exception, extra=extra)
        else:
            self.logger.critical(message, extra=extra)

    def _build_extra(self, **kwargs) -> Dict[str, Any]:
        """Construir informações extras para o log"""
        extra = {}
        for key, value in kwargs.items():
            if key in ['user_id', 'session_id', 'station', 'operation', 'duration', 'records_count']:
                extra[key] = value
        return extra

    def log_operation(self, operation: str, **kwargs) -> None:
        """Log específico para operações do sistema"""
        self.info(f"Operação executada: {operation}", operation=operation, **kwargs)

    def log_data_load(self, station: str, records_count: int, duration: float) -> None:
        """Log específico para carregamento de dados"""
        self.info(
            f"Dados carregados para estação {station}",
            station=station,
            records_count=records_count,
            duration=duration,
            operation="data_load"
        )

    def log_security_event(self, event: str, details: Dict[str, Any]) -> None:
        """Log específico para eventos de segurança"""
        self.warning(
            f"Evento de segurança: {event}",
            operation="security_event",
            event_type=event,
            **details
        )

    def log_performance(self, operation: str, duration: float, **kwargs) -> None:
        """Log específico para métricas de performance"""
        self.info(
            f"Performance - {operation}",
            operation="performance_metric",
            duration=duration,
            **kwargs
        )


# Instância global do logger
app_logger = StructuredLogger("water_treatment_app")

# Loggers específicos para diferentes módulos
data_logger = StructuredLogger("data_module")
analysis_logger = StructuredLogger("analysis_module")
security_logger = StructuredLogger("security_module")


def get_logger(module_name: str) -> StructuredLogger:
    """Obter logger para um módulo específico"""
    return StructuredLogger(f"water_treatment_{module_name}")


def log_function_call(func_name: str, args: Dict[str, Any] = None, **kwargs):
    """Decorator para logar chamadas de função"""
    def decorator(func):
        def wrapper(*args_inner, **kwargs_inner):
            start_time = datetime.now()

            try:
                result = func(*args_inner, **kwargs_inner)
                duration = (datetime.now() - start_time).total_seconds()

                app_logger.debug(
                    f"Função {func_name} executada com sucesso",
                    function=func_name,
                    duration=duration,
                    **kwargs
                )

                return result

            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()

                app_logger.error(
                    f"Erro na função {func_name}",
                    exception=e,
                    function=func_name,
                    duration=duration,
                    **kwargs
                )
                raise

        return wrapper
    return decorator


# Configurações de log por nível
LOG_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
        'json': {
            '()': JSONFormatter
        }
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'level': 'DEBUG',
            'class': 'logging.handlers.RotatingFileHandler',
            'formatter': 'json',
            'filename': 'logs/app.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5
        }
    },
    'loggers': {
        'water_treatment_system': {
            'level': 'DEBUG',
            'handlers': ['console', 'file'],
            'propagate': False
        }
    }
}


def setup_logging():
    """Configurar logging para toda a aplicação"""
    import logging.config

    # Criar diretório de logs se não existir
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)

    # Aplicar configuração
    logging.config.dictConfig(LOG_CONFIG)

    # Log inicial
    app_logger.info("Sistema de logging inicializado")


# Configurar logging automaticamente quando o módulo for importado
setup_logging()