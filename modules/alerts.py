"""
Módulo para sistema de alertas e notificações
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import streamlit as st
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json
import requests


class AlertSystem:
    """Sistema de alertas e notificações"""
    
    def __init__(self):
        self.alert_rules = {}
        self.notification_config = {}
        self.alert_history = []
        
    def add_alert_rule(self, rule_name: str, condition: str, severity: str,
                      parameters: List[str], threshold: float = None) -> None:
        """
        Adicionar regra de alerta
        
        Args:
            rule_name: Nome da regra
            condition: Condição do alerta ('threshold_exceeded', 'anomaly_detected', 'trend_anomaly')
            severity: Severidade ('low', 'medium', 'high', 'critical')
            parameters: Lista de parâmetros monitorados
            threshold: Limiar para alertas de threshold
        """
        self.alert_rules[rule_name] = {
            'condition': condition,
            'severity': severity,
            'parameters': parameters,
            'threshold': threshold,
            'created_at': datetime.now(),
            'active': True
        }
        
    def check_threshold_alerts(self, df: pd.DataFrame, thresholds: Dict) -> List[Dict]:
        """
        Verificar alertas de limite de controle
        
        Args:
            df: DataFrame com os dados
            thresholds: Dicionário com limites por parâmetro
            
        Returns:
            Lista de alertas gerados
        """
        alerts = []
        
        for param, param_thresholds in thresholds.items():
            if param not in df.columns:
                continue
                
            latest_value = df[param].iloc[-1] if len(df) > 0 else None
            if pd.isna(latest_value):
                continue
                
            # Verificar limites críticos
            if 'critical_min' in param_thresholds and latest_value < param_thresholds['critical_min']:
                alerts.append({
                    'timestamp': datetime.now(),
                    'parameter': param,
                    'value': latest_value,
                    'severity': 'critical',
                    'type': 'threshold_exceeded',
                    'message': f'{param} está abaixo do limite crítico mínimo ({param_thresholds["critical_min"]})',
                    'threshold': param_thresholds['critical_min'],
                    'deviation': latest_value - param_thresholds['critical_min']
                })
                
            elif 'critical_max' in param_thresholds and latest_value > param_thresholds['critical_max']:
                alerts.append({
                    'timestamp': datetime.now(),
                    'parameter': param,
                    'value': latest_value,
                    'severity': 'critical',
                    'type': 'threshold_exceeded',
                    'message': f'{param} está acima do limite crítico máximo ({param_thresholds["critical_max"]})',
                    'threshold': param_thresholds['critical_max'],
                    'deviation': latest_value - param_thresholds['critical_max']
                })
                
            # Verificar limites normais
            elif 'min' in param_thresholds and latest_value < param_thresholds['min']:
                alerts.append({
                    'timestamp': datetime.now(),
                    'parameter': param,
                    'value': latest_value,
                    'severity': 'high',
                    'type': 'threshold_exceeded',
                    'message': f'{param} está abaixo do limite mínimo ({param_thresholds["min"]})',
                    'threshold': param_thresholds['min'],
                    'deviation': latest_value - param_thresholds['min']
                })
                
            elif 'max' in param_thresholds and latest_value > param_thresholds['max']:
                alerts.append({
                    'timestamp': datetime.now(),
                    'parameter': param,
                    'value': latest_value,
                    'severity': 'high',
                    'type': 'threshold_exceeded',
                    'message': f'{param} está acima do limite máximo ({param_thresholds["max"]})',
                    'threshold': param_thresholds['max'],
                    'deviation': latest_value - param_thresholds['max']
                })
                
        return alerts
        
    def check_anomaly_alerts(self, df: pd.DataFrame, anomaly_columns: List[str]) -> List[Dict]:
        """
        Verificar alertas de anomalias
        
        Args:
            df: DataFrame com os dados
            anomaly_columns: Lista de colunas com indicadores de anomalia
            
        Returns:
            Lista de alertas gerados
        """
        alerts = []
        
        for col in anomaly_columns:
            if col not in df.columns:
                continue
                
            # Verificar anomalias recentes (últimas 4 observações)
            recent_anomalies = df[col].tail(4).sum()
            
            if recent_anomalies > 0:
                # Determinar severidade baseada na frequência
                if recent_anomalies >= 3:
                    severity = 'critical'
                elif recent_anomalies >= 2:
                    severity = 'high'
                else:
                    severity = 'medium'
                    
                alerts.append({
                    'timestamp': datetime.now(),
                    'parameter': col.replace('_anomaly', ''),
                    'value': recent_anomalies,
                    'severity': severity,
                    'type': 'anomaly_detected',
                    'message': f'Anomalias detectadas em {col.replace("_anomaly", "")} ({recent_anomalies} nas últimas 4 observações)',
                    'anomaly_count': int(recent_anomalies)
                })
                
        return alerts
        
    def check_trend_alerts(self, df: pd.DataFrame, parameters: List[str],
                          window: int = 24) -> List[Dict]:
        """
        Verificar alertas de tendência anormal
        
        Args:
            df: DataFrame com os dados
            parameters: Lista de parâmetros para analisar
            window: Janela de tempo para análise de tendência
            
        Returns:
            Lista de alertas gerados
        """
        alerts = []
        
        for param in parameters:
            if param not in df.columns or len(df) < window:
                continue
                
            # Calcular tendência usando regressão linear simples
            recent_data = df[param].tail(window)
            if len(recent_data.dropna()) < window * 0.8:  # Pelo menos 80% dos dados
                continue
                
            x = np.arange(len(recent_data))
            y = recent_data.values
            
            # Regressão linear
            slope = np.polyfit(x, y, 1)[0]
            
            # Calcular desvio padrão da tendência
            trend_std = np.std(y)
            
            # Alertar se tendência é muito acentuada
            if abs(slope) > trend_std * 0.5:  # Tendência > 50% do desvio padrão
                severity = 'high' if abs(slope) > trend_std else 'medium'
                direction = 'crescente' if slope > 0 else 'decrescente'
                
                alerts.append({
                    'timestamp': datetime.now(),
                    'parameter': param,
                    'value': slope,
                    'severity': severity,
                    'type': 'trend_anomaly',
                    'message': f'Tendência {direction} anormal detectada em {param} (slope: {slope:.4f})',
                    'slope': slope,
                    'direction': direction
                })
                
        return alerts
        
    def check_sensor_failure_alerts(self, df: pd.DataFrame, parameters: List[str],
                                   max_missing_minutes: int = 60) -> List[Dict]:
        """
        Verificar alertas de falha de sensor
        
        Args:
            df: DataFrame com os dados
            parameters: Lista de parâmetros para verificar
            max_missing_minutes: Tempo máximo sem dados (em minutos)
            
        Returns:
            Lista de alertas gerados
        """
        alerts = []
        
        for param in parameters:
            if param not in df.columns:
                continue
                
            # Verificar se há dados faltantes recentes
            recent_data = df[param].tail(96)  # Últimas 24 horas (96 * 15min)
            missing_count = recent_data.isna().sum()
            
            if missing_count > max_missing_minutes / 15:  # Converter para número de observações
                alerts.append({
                    'timestamp': datetime.now(),
                    'parameter': param,
                    'value': missing_count,
                    'severity': 'high',
                    'type': 'sensor_failure',
                    'message': f'Possível falha no sensor de {param} - {missing_count} observações faltantes nas últimas 24h',
                    'missing_observations': int(missing_count)
                })
                
        return alerts
        
    def generate_alert_summary(self, alerts: List[Dict]) -> Dict:
        """
        Gerar resumo dos alertas
        
        Args:
            alerts: Lista de alertas
            
        Returns:
            Dicionário com resumo
        """
        if not alerts:
            return {
                'total_alerts': 0,
                'by_severity': {},
                'by_type': {},
                'by_parameter': {},
                'critical_count': 0
            }
            
        summary = {
            'total_alerts': len(alerts),
            'by_severity': {},
            'by_type': {},
            'by_parameter': {},
            'critical_count': 0
        }
        
        # Contar por severidade
        for alert in alerts:
            severity = alert['severity']
            summary['by_severity'][severity] = summary['by_severity'].get(severity, 0) + 1
            
            if severity == 'critical':
                summary['critical_count'] += 1
                
        # Contar por tipo
        for alert in alerts:
            alert_type = alert['type']
            summary['by_type'][alert_type] = summary['by_type'].get(alert_type, 0) + 1
            
        # Contar por parâmetro
        for alert in alerts:
            parameter = alert['parameter']
            summary['by_parameter'][parameter] = summary['by_parameter'].get(parameter, 0) + 1
            
        return summary
        
    def send_email_alert(self, alert: Dict, recipients: List[str],
                        smtp_config: Dict) -> bool:
        """
        Enviar alerta por email
        
        Args:
            alert: Dicionário com dados do alerta
            recipients: Lista de destinatários
            smtp_config: Configurações SMTP
            
        Returns:
            True se enviado com sucesso
        """
        try:
            msg = MIMEMultipart()
            msg['From'] = smtp_config['sender']
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"ALERTA {alert['severity'].upper()} - Sistema de Monitoramento"
            
            # Corpo do email
            body = f"""
            <html>
            <body>
                <h2>🚨 Alerta do Sistema de Monitoramento</h2>
                <p><strong>Severidade:</strong> {alert['severity'].upper()}</p>
                <p><strong>Parâmetro:</strong> {alert['parameter']}</p>
                <p><strong>Valor:</strong> {alert['value']}</p>
                <p><strong>Mensagem:</strong> {alert['message']}</p>
                <p><strong>Timestamp:</strong> {alert['timestamp']}</p>
                <hr>
                <p><em>Este é um alerta automático do sistema de monitoramento.</em></p>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            # Enviar email
            server = smtplib.SMTP(smtp_config['host'], smtp_config['port'])
            server.starttls()
            server.login(smtp_config['username'], smtp_config['password'])
            text = msg.as_string()
            server.sendmail(smtp_config['sender'], recipients, text)
            server.quit()
            
            return True
            
        except Exception as e:
            st.error(f"Erro ao enviar email: {str(e)}")
            return False
            
    def send_slack_alert(self, alert: Dict, webhook_url: str) -> bool:
        """
        Enviar alerta para Slack
        
        Args:
            alert: Dicionário com dados do alerta
            webhook_url: URL do webhook do Slack
            
        Returns:
            True se enviado com sucesso
        """
        try:
            # Cores por severidade
            colors = {
                'low': '#36a64f',
                'medium': '#ff9800',
                'high': '#ff5722',
                'critical': '#f44336'
            }
            
            payload = {
                "attachments": [
                    {
                        "color": colors.get(alert['severity'], '#36a64f'),
                        "title": f"🚨 Alerta {alert['severity'].upper()}",
                        "fields": [
                            {
                                "title": "Parâmetro",
                                "value": alert['parameter'],
                                "short": True
                            },
                            {
                                "title": "Valor",
                                "value": str(alert['value']),
                                "short": True
                            },
                            {
                                "title": "Mensagem",
                                "value": alert['message'],
                                "short": False
                            },
                            {
                                "title": "Timestamp",
                                "value": alert['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                                "short": True
                            }
                        ],
                        "footer": "Sistema de Monitoramento",
                        "ts": int(alert['timestamp'].timestamp())
                    }
                ]
            }
            
            response = requests.post(webhook_url, json=payload)
            return response.status_code == 200
            
        except Exception as e:
            st.error(f"Erro ao enviar para Slack: {str(e)}")
            return False
            
    def send_teams_alert(self, alert: Dict, webhook_url: str) -> bool:
        """
        Enviar alerta para Microsoft Teams
        
        Args:
            alert: Dicionário com dados do alerta
            webhook_url: URL do webhook do Teams
            
        Returns:
            True se enviado com sucesso
        """
        try:
            # Cores por severidade
            colors = {
                'low': '00ff00',
                'medium': 'ffaa00',
                'high': 'ff6600',
                'critical': 'ff0000'
            }
            
            payload = {
                "@type": "MessageCard",
                "@context": "http://schema.org/extensions",
                "themeColor": colors.get(alert['severity'], '00ff00'),
                "summary": f"Alerta {alert['severity'].upper()}",
                "sections": [
                    {
                        "activityTitle": f"🚨 Alerta {alert['severity'].upper()}",
                        "activitySubtitle": "Sistema de Monitoramento",
                        "facts": [
                            {
                                "name": "Parâmetro",
                                "value": alert['parameter']
                            },
                            {
                                "name": "Valor",
                                "value": str(alert['value'])
                            },
                            {
                                "name": "Mensagem",
                                "value": alert['message']
                            },
                            {
                                "name": "Timestamp",
                                "value": alert['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
                            }
                        ]
                    }
                ]
            }
            
            response = requests.post(webhook_url, json=payload)
            return response.status_code == 200
            
        except Exception as e:
            st.error(f"Erro ao enviar para Teams: {str(e)}")
            return False
            
    def process_alerts(self, df: pd.DataFrame, thresholds: Dict,
                      anomaly_columns: List[str] = None,
                      notification_config: Dict = None) -> List[Dict]:
        """
        Processar todos os tipos de alertas
        
        Args:
            df: DataFrame com os dados
            thresholds: Dicionário com limites
            anomaly_columns: Lista de colunas de anomalias
            notification_config: Configurações de notificação
            
        Returns:
            Lista de alertas processados
        """
        all_alerts = []
        
        # Verificar alertas de limite
        threshold_alerts = self.check_threshold_alerts(df, thresholds)
        all_alerts.extend(threshold_alerts)
        
        # Verificar alertas de anomalia
        if anomaly_columns:
            anomaly_alerts = self.check_anomaly_alerts(df, anomaly_columns)
            all_alerts.extend(anomaly_alerts)
            
        # Verificar alertas de tendência
        parameters = list(thresholds.keys())
        trend_alerts = self.check_trend_alerts(df, parameters)
        all_alerts.extend(trend_alerts)
        
        # Verificar alertas de falha de sensor
        sensor_alerts = self.check_sensor_failure_alerts(df, parameters)
        all_alerts.extend(sensor_alerts)
        
        # Enviar notificações se configurado
        if notification_config and all_alerts:
            self._send_notifications(all_alerts, notification_config)
            
        # Adicionar ao histórico
        self.alert_history.extend(all_alerts)
        
        return all_alerts
        
    def _send_notifications(self, alerts: List[Dict], config: Dict) -> None:
        """
        Enviar notificações para alertas
        
        Args:
            alerts: Lista de alertas
            config: Configurações de notificação
        """
        for alert in alerts:
            # Filtrar por severidade mínima
            min_severity = config.get('min_severity', 'low')
            severity_levels = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
            
            if severity_levels.get(alert['severity'], 0) < severity_levels.get(min_severity, 1):
                continue
                
            # Enviar email
            if config.get('email', {}).get('enabled', False):
                self.send_email_alert(alert, config['email']['recipients'], config['email']['smtp'])
                
            # Enviar Slack
            if config.get('slack', {}).get('enabled', False):
                self.send_slack_alert(alert, config['slack']['webhook_url'])
                
            # Enviar Teams
            if config.get('teams', {}).get('enabled', False):
                self.send_teams_alert(alert, config['teams']['webhook_url'])
                
    def get_alert_history(self, hours: int = 24) -> pd.DataFrame:
        """
        Obter histórico de alertas
        
        Args:
            hours: Número de horas para retornar
            
        Returns:
            DataFrame com histórico de alertas
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [alert for alert in self.alert_history 
                        if alert['timestamp'] >= cutoff_time]
        
        if not recent_alerts:
            return pd.DataFrame()
            
        return pd.DataFrame(recent_alerts)
        
    def clear_old_alerts(self, days: int = 7) -> None:
        """
        Limpar alertas antigos do histórico
        
        Args:
            days: Número de dias para manter
        """
        cutoff_time = datetime.now() - timedelta(days=days)
        self.alert_history = [alert for alert in self.alert_history 
                             if alert['timestamp'] >= cutoff_time]
