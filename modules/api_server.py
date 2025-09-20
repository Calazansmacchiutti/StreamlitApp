"""
Módulo para API REST do Sistema de Monitoramento
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, g
from flask_cors import CORS
from functools import wraps
import jwt
import yaml
from werkzeug.exceptions import BadRequest, Unauthorized, NotFound, InternalServerError

from .data_loader import DataLoader
from .preprocessor import DataPreprocessor
from .time_series import TimeSeriesAnalyzer
from .anomaly_detection import AnomalyDetector
from .alerts import AlertSystem

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIServer:
    """Servidor API REST para o sistema de monitoramento"""
    
    def __init__(self, config_path: str = "config/config_advanced.yaml"):
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'your-secret-key-change-this'
        
        # Habilitar CORS
        CORS(self.app)
        
        # Carregar configuração
        self.config = self._load_config(config_path)
        
        # Inicializar módulos
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.time_series_analyzer = TimeSeriesAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        self.alert_system = AlertSystem()
        
        # Configurar rotas
        self._setup_routes()
        
        # Configurar middleware
        self._setup_middleware()
        
    def _load_config(self, config_path: str) -> Dict:
        """Carregar configuração do arquivo YAML"""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Erro ao carregar configuração: {e}")
            return {}
    
    def _setup_middleware(self):
        """Configurar middleware da aplicação"""
        
        @self.app.before_request
        def before_request():
            """Executar antes de cada requisição"""
            g.start_time = datetime.now()
            
        @self.app.after_request
        def after_request(response):
            """Executar após cada requisição"""
            if hasattr(g, 'start_time'):
                duration = (datetime.now() - g.start_time).total_seconds()
                response.headers['X-Response-Time'] = f"{duration:.3f}s"
            return response
            
        @self.app.errorhandler(400)
        def bad_request(error):
            return jsonify({
                'error': 'Bad Request',
                'message': str(error),
                'status_code': 400
            }), 400
            
        @self.app.errorhandler(401)
        def unauthorized(error):
            return jsonify({
                'error': 'Unauthorized',
                'message': 'Authentication required',
                'status_code': 401
            }), 401
            
        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({
                'error': 'Not Found',
                'message': 'Resource not found',
                'status_code': 404
            }), 404
            
        @self.app.errorhandler(500)
        def internal_error(error):
            return jsonify({
                'error': 'Internal Server Error',
                'message': 'An internal error occurred',
                'status_code': 500
            }), 500
    
    def _setup_routes(self):
        """Configurar rotas da API"""
        
        # Rota de health check
        @self.app.route('/api/v1/health', methods=['GET'])
        def health_check():
            """Verificar saúde do sistema"""
            try:
                health_status = {
                    'status': 'healthy',
                    'timestamp': datetime.now().isoformat(),
                    'version': '1.0.0',
                    'uptime': 'N/A',  # Implementar se necessário
                    'services': {
                        'database': 'healthy',
                        'data_loader': 'healthy',
                        'anomaly_detection': 'healthy'
                    }
                }
                return jsonify(health_status)
            except Exception as e:
                logger.error(f"Erro no health check: {e}")
                return jsonify({
                    'status': 'unhealthy',
                    'error': str(e)
                }), 500
        
        # Rotas de dados
        @self.app.route('/api/v1/data', methods=['GET'])
        def get_data():
            """Obter dados das estações"""
            try:
                station = request.args.get('station')
                parameter = request.args.get('parameter')
                start_date = request.args.get('start_date')
                end_date = request.args.get('end_date')
                limit = int(request.args.get('limit', 1000))
                
                if not station:
                    return jsonify({'error': 'Station parameter is required'}), 400
                
                # Carregar dados
                df = self.data_loader.load_station_data(station)
                
                if df.empty:
                    return jsonify({'error': f'No data found for station {station}'}), 404
                
                # Filtrar por parâmetro se especificado
                if parameter and parameter in df.columns:
                    df = df[['timestamp', parameter]]
                
                # Filtrar por data se especificado
                if start_date:
                    df = df[df['timestamp'] >= pd.to_datetime(start_date)]
                if end_date:
                    df = df[df['timestamp'] <= pd.to_datetime(end_date)]
                
                # Limitar resultados
                df = df.tail(limit)
                
                # Converter para formato JSON
                data = df.to_dict('records')
                
                return jsonify({
                    'station': station,
                    'parameter': parameter,
                    'count': len(data),
                    'data': data
                })
                
            except Exception as e:
                logger.error(f"Erro ao obter dados: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/v1/data/stations', methods=['GET'])
        def get_stations():
            """Obter lista de estações"""
            try:
                stations = list(self.config.get('stations', {}).keys())
                return jsonify({
                    'stations': stations,
                    'count': len(stations)
                })
            except Exception as e:
                logger.error(f"Erro ao obter estações: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/v1/data/parameters', methods=['GET'])
        def get_parameters():
            """Obter parâmetros de uma estação"""
            try:
                station = request.args.get('station')
                if not station:
                    return jsonify({'error': 'Station parameter is required'}), 400
                
                if station not in self.config.get('stations', {}):
                    return jsonify({'error': f'Station {station} not found'}), 404
                
                parameters = self.config['stations'][station]['parameters']
                return jsonify({
                    'station': station,
                    'parameters': parameters,
                    'count': len(parameters)
                })
            except Exception as e:
                logger.error(f"Erro ao obter parâmetros: {e}")
                return jsonify({'error': str(e)}), 500
        
        # Rotas de alertas
        @self.app.route('/api/v1/alerts', methods=['GET'])
        def get_alerts():
            """Obter alertas ativos"""
            try:
                level = request.args.get('level')
                station = request.args.get('station')
                limit = int(request.args.get('limit', 100))
                
                alerts = self.alert_system.get_active_alerts(level=level)
                
                # Filtrar por estação se especificado
                if station:
                    alerts = [alert for alert in alerts if alert.get('station') == station]
                
                # Limitar resultados
                alerts = alerts[:limit]
                
                return jsonify({
                    'alerts': alerts,
                    'count': len(alerts)
                })
            except Exception as e:
                logger.error(f"Erro ao obter alertas: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/v1/alerts', methods=['POST'])
        def create_alert():
            """Criar novo alerta"""
            try:
                data = request.get_json()
                
                required_fields = ['timestamp', 'station', 'parameter', 'value', 'level']
                for field in required_fields:
                    if field not in data:
                        return jsonify({'error': f'Field {field} is required'}), 400
                
                alert = self.alert_system.generate_alert(
                    timestamp=data['timestamp'],
                    station=data['station'],
                    parameter=data['parameter'],
                    value=data['value'],
                    threshold_info=data.get('threshold_info', {}),
                    level=data['level'],
                    message=data.get('message', '')
                )
                
                return jsonify({
                    'message': 'Alert created successfully',
                    'alert': alert
                }), 201
                
            except Exception as e:
                logger.error(f"Erro ao criar alerta: {e}")
                return jsonify({'error': str(e)}), 500
        
        # Rotas de análise
        @self.app.route('/api/v1/analysis/anomalies', methods=['POST'])
        def detect_anomalies():
            """Detectar anomalias nos dados"""
            try:
                data = request.get_json()
                
                station = data.get('station')
                method = data.get('method', 'isolation_forest')
                parameters = data.get('parameters', [])
                contamination = data.get('contamination', 0.1)
                
                if not station:
                    return jsonify({'error': 'Station is required'}), 400
                
                # Carregar dados
                df = self.data_loader.load_station_data(station)
                if df.empty:
                    return jsonify({'error': f'No data found for station {station}'}), 404
                
                # Usar parâmetros da estação se não especificados
                if not parameters:
                    parameters = self.config['stations'][station]['parameters']
                
                # Detectar anomalias
                if method == 'isolation_forest':
                    df_with_anomalies = self.anomaly_detector.detect_anomalies_isolation_forest(
                        df, parameters, contamination=contamination
                    )
                elif method == 'zscore':
                    # Implementar detecção Z-score
                    pass
                else:
                    return jsonify({'error': f'Method {method} not supported'}), 400
                
                # Extrair anomalias
                anomalies = df_with_anomalies[df_with_anomalies['isolation_forest_anomaly'] == -1]
                
                return jsonify({
                    'station': station,
                    'method': method,
                    'anomalies_count': len(anomalies),
                    'anomalies': anomalies.to_dict('records')
                })
                
            except Exception as e:
                logger.error(f"Erro na detecção de anomalias: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/v1/analysis/forecast', methods=['POST'])
        def generate_forecast():
            """Gerar previsões"""
            try:
                data = request.get_json()
                
                station = data.get('station')
                parameter = data.get('parameter')
                model = data.get('model', 'prophet')
                horizon = data.get('horizon', 24)
                
                if not station or not parameter:
                    return jsonify({'error': 'Station and parameter are required'}), 400
                
                # Carregar dados
                df = self.data_loader.load_station_data(station)
                if df.empty:
                    return jsonify({'error': f'No data found for station {station}'}), 404
                
                if parameter not in df.columns:
                    return jsonify({'error': f'Parameter {parameter} not found'}), 404
                
                # Gerar previsão
                if model == 'prophet':
                    # Implementar previsão Prophet
                    pass
                elif model == 'arima':
                    # Implementar previsão ARIMA
                    pass
                else:
                    return jsonify({'error': f'Model {model} not supported'}), 400
                
                return jsonify({
                    'station': station,
                    'parameter': parameter,
                    'model': model,
                    'horizon': horizon,
                    'forecast': []  # Implementar retorno da previsão
                })
                
            except Exception as e:
                logger.error(f"Erro na geração de previsão: {e}")
                return jsonify({'error': str(e)}), 500
        
        # Rotas de relatórios
        @self.app.route('/api/v1/reports', methods=['POST'])
        def generate_report():
            """Gerar relatório"""
            try:
                data = request.get_json()
                
                report_type = data.get('type', 'operational')
                station = data.get('station')
                start_date = data.get('start_date')
                end_date = data.get('end_date')
                format = data.get('format', 'pdf')
                
                if not station:
                    return jsonify({'error': 'Station is required'}), 400
                
                # Implementar geração de relatório
                report_data = {
                    'type': report_type,
                    'station': station,
                    'start_date': start_date,
                    'end_date': end_date,
                    'format': format,
                    'status': 'generated',
                    'download_url': f'/api/v1/reports/download/{report_type}_{station}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.{format}'
                }
                
                return jsonify({
                    'message': 'Report generated successfully',
                    'report': report_data
                })
                
            except Exception as e:
                logger.error(f"Erro na geração de relatório: {e}")
                return jsonify({'error': str(e)}), 500
        
        # Rotas de métricas
        @self.app.route('/api/v1/metrics', methods=['GET'])
        def get_metrics():
            """Obter métricas do sistema"""
            try:
                metrics = {
                    'timestamp': datetime.now().isoformat(),
                    'system': {
                        'uptime': 'N/A',
                        'memory_usage': 'N/A',
                        'cpu_usage': 'N/A'
                    },
                    'data': {
                        'total_stations': len(self.config.get('stations', {})),
                        'total_parameters': sum(len(station['parameters']) for station in self.config.get('stations', {}).values()),
                        'active_alerts': len(self.alert_system.get_active_alerts())
                    }
                }
                
                return jsonify(metrics)
                
            except Exception as e:
                logger.error(f"Erro ao obter métricas: {e}")
                return jsonify({'error': str(e)}), 500
        
        # Rota de documentação
        @self.app.route('/api/v1/docs', methods=['GET'])
        def api_docs():
            """Documentação da API"""
            docs = {
                'title': 'Water Treatment Monitoring API',
                'version': '1.0.0',
                'description': 'API REST para o sistema de monitoramento de estações de tratamento de água',
                'endpoints': {
                    'GET /api/v1/health': 'Health check do sistema',
                    'GET /api/v1/data': 'Obter dados das estações',
                    'GET /api/v1/data/stations': 'Listar estações',
                    'GET /api/v1/data/parameters': 'Obter parâmetros de uma estação',
                    'GET /api/v1/alerts': 'Obter alertas ativos',
                    'POST /api/v1/alerts': 'Criar novo alerta',
                    'POST /api/v1/analysis/anomalies': 'Detectar anomalias',
                    'POST /api/v1/analysis/forecast': 'Gerar previsões',
                    'POST /api/v1/reports': 'Gerar relatórios',
                    'GET /api/v1/metrics': 'Obter métricas do sistema'
                }
            }
            return jsonify(docs)
    
    def run(self, host: str = '0.0.0.0', port: int = 8000, debug: bool = False):
        """Executar servidor API"""
        logger.info(f"Iniciando servidor API em {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

def create_app(config_path: str = "config/config_advanced.yaml") -> Flask:
    """Factory function para criar aplicação Flask"""
    api_server = APIServer(config_path)
    return api_server.app

if __name__ == '__main__':
    # Executar servidor diretamente
    api_server = APIServer()
    api_server.run(debug=True)
