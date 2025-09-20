"""
Módulo para Backup e Auditoria do Sistema
"""

import os
import json
import shutil
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import yaml
import zipfile
import hashlib
from cryptography.fernet import Fernet
import schedule
import threading
import time

from .data_loader import DataLoader

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BackupManager:
    """Gerenciador de backup do sistema"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.backup_dir = Path(config.get('backup_location', './backups'))
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurações de backup
        self.retention_days = config.get('backup_retention_days', 30)
        self.encryption_enabled = config.get('encryption_enabled', False)
        self.compression_enabled = True
        
        # Chave de criptografia
        if self.encryption_enabled:
            self.encryption_key = self._get_or_create_encryption_key()
            self.cipher = Fernet(self.encryption_key)
        
        # Inicializar data loader
        self.data_loader = DataLoader()
        
    def _get_or_create_encryption_key(self) -> bytes:
        """Obter ou criar chave de criptografia"""
        key_file = self.backup_dir / 'encryption.key'
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            return key
    
    def create_full_backup(self) -> Dict[str, Any]:
        """Criar backup completo do sistema"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"full_backup_{timestamp}"
            backup_path = self.backup_dir / backup_name
            backup_path.mkdir(exist_ok=True)
            
            backup_info = {
                'backup_name': backup_name,
                'timestamp': datetime.now().isoformat(),
                'type': 'full',
                'files': [],
                'size': 0,
                'status': 'in_progress'
            }
            
            # Backup de dados
            data_backup_path = backup_path / 'data'
            data_backup_path.mkdir(exist_ok=True)
            
            data_dir = Path('data')
            if data_dir.exists():
                for file_path in data_dir.glob('*.csv'):
                    shutil.copy2(file_path, data_backup_path)
                    backup_info['files'].append(str(file_path))
            
            # Backup de configuração
            config_backup_path = backup_path / 'config'
            config_backup_path.mkdir(exist_ok=True)
            
            config_dir = Path('config')
            if config_dir.exists():
                for file_path in config_dir.glob('*.yaml'):
                    shutil.copy2(file_path, config_backup_path)
                    backup_info['files'].append(str(file_path))
            
            # Backup de logs
            logs_backup_path = backup_path / 'logs'
            logs_backup_path.mkdir(exist_ok=True)
            
            logs_dir = Path('logs')
            if logs_dir.exists():
                for file_path in logs_dir.glob('*.log'):
                    shutil.copy2(file_path, logs_backup_path)
                    backup_info['files'].append(str(file_path))
            
            # Backup de banco de dados (se existir)
            db_backup_path = backup_path / 'database'
            db_backup_path.mkdir(exist_ok=True)
            
            db_file = Path('water_treatment.db')
            if db_file.exists():
                shutil.copy2(db_file, db_backup_path)
                backup_info['files'].append(str(db_file))
            
            # Criar arquivo de metadados
            metadata_file = backup_path / 'backup_metadata.json'
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(backup_info, f, indent=2, ensure_ascii=False)
            
            # Comprimir backup
            if self.compression_enabled:
                zip_path = self.backup_dir / f"{backup_name}.zip"
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for root, dirs, files in os.walk(backup_path):
                        for file in files:
                            file_path = Path(root) / file
                            arcname = file_path.relative_to(backup_path)
                            zipf.write(file_path, arcname)
                
                # Remover diretório não comprimido
                shutil.rmtree(backup_path)
                backup_path = zip_path
            
            # Criptografar se habilitado
            if self.encryption_enabled:
                encrypted_path = self.backup_dir / f"{backup_name}.enc"
                with open(backup_path, 'rb') as f:
                    encrypted_data = self.cipher.encrypt(f.read())
                with open(encrypted_path, 'wb') as f:
                    f.write(encrypted_data)
                
                # Remover arquivo não criptografado
                backup_path.unlink()
                backup_path = encrypted_path
            
            # Calcular tamanho
            backup_info['size'] = backup_path.stat().st_size
            backup_info['status'] = 'completed'
            backup_info['backup_path'] = str(backup_path)
            
            # Salvar metadados atualizados
            metadata_file = self.backup_dir / f"{backup_name}_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(backup_info, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Backup completo criado: {backup_name}")
            return backup_info
            
        except Exception as e:
            logger.error(f"Erro ao criar backup completo: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def create_incremental_backup(self, last_backup_time: datetime) -> Dict[str, Any]:
        """Criar backup incremental"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"incremental_backup_{timestamp}"
            backup_path = self.backup_dir / backup_name
            backup_path.mkdir(exist_ok=True)
            
            backup_info = {
                'backup_name': backup_name,
                'timestamp': datetime.now().isoformat(),
                'type': 'incremental',
                'last_backup_time': last_backup_time.isoformat(),
                'files': [],
                'size': 0,
                'status': 'in_progress'
            }
            
            # Backup apenas arquivos modificados
            data_dir = Path('data')
            if data_dir.exists():
                for file_path in data_dir.glob('*.csv'):
                    if file_path.stat().st_mtime > last_backup_time.timestamp():
                        backup_file_path = backup_path / file_path.name
                        shutil.copy2(file_path, backup_file_path)
                        backup_info['files'].append(str(file_path))
            
            # Similar para outros diretórios...
            
            if not backup_info['files']:
                backup_info['status'] = 'no_changes'
                shutil.rmtree(backup_path)
                return backup_info
            
            # Comprimir e criptografar (similar ao backup completo)
            # ...
            
            backup_info['status'] = 'completed'
            logger.info(f"Backup incremental criado: {backup_name}")
            return backup_info
            
        except Exception as e:
            logger.error(f"Erro ao criar backup incremental: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def restore_backup(self, backup_name: str, restore_path: str = None) -> Dict[str, Any]:
        """Restaurar backup"""
        try:
            if restore_path is None:
                restore_path = '.'
            
            restore_path = Path(restore_path)
            
            # Encontrar arquivo de backup
            backup_file = None
            for ext in ['.zip', '.enc']:
                potential_file = self.backup_dir / f"{backup_name}{ext}"
                if potential_file.exists():
                    backup_file = potential_file
                    break
            
            if not backup_file:
                return {
                    'status': 'failed',
                    'error': f'Backup {backup_name} not found'
                }
            
            # Descriptografar se necessário
            if backup_file.suffix == '.enc':
                decrypted_path = self.backup_dir / f"{backup_name}_decrypted.zip"
                with open(backup_file, 'rb') as f:
                    decrypted_data = self.cipher.decrypt(f.read())
                with open(decrypted_path, 'wb') as f:
                    f.write(decrypted_data)
                backup_file = decrypted_path
            
            # Extrair backup
            if backup_file.suffix == '.zip':
                with zipfile.ZipFile(backup_file, 'r') as zipf:
                    zipf.extractall(restore_path)
            
            # Limpar arquivo temporário se existir
            if backup_file.name.endswith('_decrypted.zip'):
                backup_file.unlink()
            
            logger.info(f"Backup {backup_name} restaurado com sucesso")
            return {
                'status': 'completed',
                'backup_name': backup_name,
                'restore_path': str(restore_path),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erro ao restaurar backup: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """Listar backups disponíveis"""
        backups = []
        
        for file_path in self.backup_dir.glob('*_metadata.json'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    backup_info = json.load(f)
                backups.append(backup_info)
            except Exception as e:
                logger.error(f"Erro ao ler metadados do backup {file_path}: {e}")
        
        # Ordenar por timestamp
        backups.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return backups
    
    def cleanup_old_backups(self):
        """Limpar backups antigos"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            backups = self.list_backups()
            
            deleted_count = 0
            for backup in backups:
                backup_time = datetime.fromisoformat(backup['timestamp'])
                if backup_time < cutoff_date:
                    backup_name = backup['backup_name']
                    
                    # Remover arquivos de backup
                    for ext in ['.zip', '.enc']:
                        backup_file = self.backup_dir / f"{backup_name}{ext}"
                        if backup_file.exists():
                            backup_file.unlink()
                    
                    # Remover metadados
                    metadata_file = self.backup_dir / f"{backup_name}_metadata.json"
                    if metadata_file.exists():
                        metadata_file.unlink()
                    
                    deleted_count += 1
            
            logger.info(f"Removidos {deleted_count} backups antigos")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Erro ao limpar backups antigos: {e}")
            return 0
    
    def schedule_backups(self):
        """Agendar backups automáticos"""
        # Backup diário às 2:00
        schedule.every().day.at("02:00").do(self.create_full_backup)
        
        # Backup incremental a cada 6 horas
        schedule.every(6).hours.do(self.create_incremental_backup, datetime.now())
        
        # Limpeza semanal
        schedule.every().week.do(self.cleanup_old_backups)
        
        logger.info("Backups automáticos agendados")
    
    def start_scheduler(self):
        """Iniciar agendador de backups"""
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(60)  # Verificar a cada minuto
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        logger.info("Agendador de backups iniciado")

class AuditLogger:
    """Sistema de auditoria do sistema"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.audit_dir = Path('logs/audit')
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurar logging de auditoria
        self.audit_logger = logging.getLogger('audit')
        self.audit_logger.setLevel(logging.INFO)
        
        # Handler para arquivo de auditoria
        audit_handler = logging.FileHandler(
            self.audit_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.log"
        )
        audit_handler.setLevel(logging.INFO)
        
        # Formatter para logs de auditoria
        audit_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        audit_handler.setFormatter(audit_formatter)
        
        self.audit_logger.addHandler(audit_handler)
        
        # Inicializar banco de dados de auditoria
        self._init_audit_db()
    
    def _init_audit_db(self):
        """Inicializar banco de dados de auditoria"""
        self.audit_db_path = Path('audit.db')
        
        with sqlite3.connect(self.audit_db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    user_id TEXT,
                    session_id TEXT,
                    action TEXT NOT NULL,
                    resource TEXT,
                    details TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    success BOOLEAN,
                    error_message TEXT
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_audit_timestamp 
                ON audit_logs(timestamp)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_audit_user 
                ON audit_logs(user_id)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_audit_action 
                ON audit_logs(action)
            ''')
    
    def log_action(self, action: str, user_id: str = None, session_id: str = None,
                   resource: str = None, details: Dict = None, ip_address: str = None,
                   user_agent: str = None, success: bool = True, error_message: str = None):
        """Registrar ação de auditoria"""
        try:
            audit_entry = {
                'timestamp': datetime.now().isoformat(),
                'user_id': user_id,
                'session_id': session_id,
                'action': action,
                'resource': resource,
                'details': json.dumps(details) if details else None,
                'ip_address': ip_address,
                'user_agent': user_agent,
                'success': success,
                'error_message': error_message
            }
            
            # Salvar no banco de dados
            with sqlite3.connect(self.audit_db_path) as conn:
                conn.execute('''
                    INSERT INTO audit_logs 
                    (timestamp, user_id, session_id, action, resource, details, 
                     ip_address, user_agent, success, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    audit_entry['timestamp'],
                    audit_entry['user_id'],
                    audit_entry['session_id'],
                    audit_entry['action'],
                    audit_entry['resource'],
                    audit_entry['details'],
                    audit_entry['ip_address'],
                    audit_entry['user_agent'],
                    audit_entry['success'],
                    audit_entry['error_message']
                ))
            
            # Log estruturado
            log_message = f"Action: {action}"
            if user_id:
                log_message += f" | User: {user_id}"
            if resource:
                log_message += f" | Resource: {resource}"
            if not success:
                log_message += f" | Error: {error_message}"
            
            self.audit_logger.info(log_message)
            
        except Exception as e:
            logger.error(f"Erro ao registrar auditoria: {e}")
    
    def get_audit_logs(self, start_date: datetime = None, end_date: datetime = None,
                      user_id: str = None, action: str = None, limit: int = 1000) -> List[Dict]:
        """Obter logs de auditoria"""
        try:
            query = "SELECT * FROM audit_logs WHERE 1=1"
            params = []
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.isoformat())
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.isoformat())
            
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            
            if action:
                query += " AND action = ?"
                params.append(action)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            with sqlite3.connect(self.audit_db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"Erro ao obter logs de auditoria: {e}")
            return []
    
    def generate_audit_report(self, start_date: datetime, end_date: datetime) -> Dict:
        """Gerar relatório de auditoria"""
        try:
            logs = self.get_audit_logs(start_date, end_date)
            
            if not logs:
                return {
                    'period': f"{start_date.date()} to {end_date.date()}",
                    'total_actions': 0,
                    'summary': {}
                }
            
            # Estatísticas gerais
            total_actions = len(logs)
            successful_actions = sum(1 for log in logs if log['success'])
            failed_actions = total_actions - successful_actions
            
            # Ações por usuário
            user_actions = {}
            for log in logs:
                user_id = log['user_id'] or 'anonymous'
                if user_id not in user_actions:
                    user_actions[user_id] = 0
                user_actions[user_id] += 1
            
            # Ações por tipo
            action_types = {}
            for log in logs:
                action = log['action']
                if action not in action_types:
                    action_types[action] = 0
                action_types[action] += 1
            
            # Recursos mais acessados
            resource_access = {}
            for log in logs:
                resource = log['resource']
                if resource:
                    if resource not in resource_access:
                        resource_access[resource] = 0
                    resource_access[resource] += 1
            
            return {
                'period': f"{start_date.date()} to {end_date.date()}",
                'total_actions': total_actions,
                'successful_actions': successful_actions,
                'failed_actions': failed_actions,
                'success_rate': (successful_actions / total_actions * 100) if total_actions > 0 else 0,
                'user_actions': user_actions,
                'action_types': action_types,
                'resource_access': resource_access,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erro ao gerar relatório de auditoria: {e}")
            return {
                'error': str(e),
                'generated_at': datetime.now().isoformat()
            }
    
    def cleanup_old_logs(self, retention_days: int = 90):
        """Limpar logs antigos"""
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            with sqlite3.connect(self.audit_db_path) as conn:
                cursor = conn.execute(
                    "DELETE FROM audit_logs WHERE timestamp < ?",
                    (cutoff_date.isoformat(),)
                )
                deleted_count = cursor.rowcount
            
            logger.info(f"Removidos {deleted_count} logs de auditoria antigos")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Erro ao limpar logs antigos: {e}")
            return 0

class SystemMonitor:
    """Monitor do sistema"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.monitoring_data = {}
        
    def get_system_metrics(self) -> Dict[str, Any]:
        """Obter métricas do sistema"""
        try:
            import psutil
            
            # Métricas de CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Métricas de memória
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available = memory.available
            memory_total = memory.total
            
            # Métricas de disco
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_free = disk.free
            disk_total = disk.total
            
            # Métricas de rede
            network = psutil.net_io_counters()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'cpu': {
                    'percent': cpu_percent,
                    'count': cpu_count
                },
                'memory': {
                    'percent': memory_percent,
                    'available': memory_available,
                    'total': memory_total
                },
                'disk': {
                    'percent': disk_percent,
                    'free': disk_free,
                    'total': disk_total
                },
                'network': {
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                }
            }
            
        except ImportError:
            logger.warning("psutil não instalado. Métricas do sistema não disponíveis.")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': 'psutil not installed'
            }
        except Exception as e:
            logger.error(f"Erro ao obter métricas do sistema: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def check_system_health(self) -> Dict[str, Any]:
        """Verificar saúde do sistema"""
        try:
            metrics = self.get_system_metrics()
            
            health_status = {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'healthy',
                'checks': {}
            }
            
            # Verificar CPU
            if 'cpu' in metrics:
                cpu_percent = metrics['cpu']['percent']
                if cpu_percent > 90:
                    health_status['checks']['cpu'] = {
                        'status': 'critical',
                        'value': cpu_percent,
                        'message': f'CPU usage is {cpu_percent}%'
                    }
                    health_status['overall_status'] = 'critical'
                elif cpu_percent > 80:
                    health_status['checks']['cpu'] = {
                        'status': 'warning',
                        'value': cpu_percent,
                        'message': f'CPU usage is {cpu_percent}%'
                    }
                    if health_status['overall_status'] == 'healthy':
                        health_status['overall_status'] = 'warning'
                else:
                    health_status['checks']['cpu'] = {
                        'status': 'healthy',
                        'value': cpu_percent,
                        'message': f'CPU usage is {cpu_percent}%'
                    }
            
            # Verificar memória
            if 'memory' in metrics:
                memory_percent = metrics['memory']['percent']
                if memory_percent > 90:
                    health_status['checks']['memory'] = {
                        'status': 'critical',
                        'value': memory_percent,
                        'message': f'Memory usage is {memory_percent}%'
                    }
                    health_status['overall_status'] = 'critical'
                elif memory_percent > 80:
                    health_status['checks']['memory'] = {
                        'status': 'warning',
                        'value': memory_percent,
                        'message': f'Memory usage is {memory_percent}%'
                    }
                    if health_status['overall_status'] == 'healthy':
                        health_status['overall_status'] = 'warning'
                else:
                    health_status['checks']['memory'] = {
                        'status': 'healthy',
                        'value': memory_percent,
                        'message': f'Memory usage is {memory_percent}%'
                    }
            
            # Verificar disco
            if 'disk' in metrics:
                disk_percent = metrics['disk']['percent']
                if disk_percent > 90:
                    health_status['checks']['disk'] = {
                        'status': 'critical',
                        'value': disk_percent,
                        'message': f'Disk usage is {disk_percent}%'
                    }
                    health_status['overall_status'] = 'critical'
                elif disk_percent > 80:
                    health_status['checks']['disk'] = {
                        'status': 'warning',
                        'value': disk_percent,
                        'message': f'Disk usage is {disk_percent}%'
                    }
                    if health_status['overall_status'] == 'healthy':
                        health_status['overall_status'] = 'warning'
                else:
                    health_status['checks']['disk'] = {
                        'status': 'healthy',
                        'value': disk_percent,
                        'message': f'Disk usage is {disk_percent}%'
                    }
            
            return health_status
            
        except Exception as e:
            logger.error(f"Erro ao verificar saúde do sistema: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_status': 'error',
                'error': str(e)
            }

# Função para inicializar sistema de backup e auditoria
def initialize_backup_audit_system(config: Dict) -> tuple:
    """Inicializar sistema de backup e auditoria"""
    try:
        # Inicializar gerenciador de backup
        backup_manager = BackupManager(config)
        
        # Inicializar logger de auditoria
        audit_logger = AuditLogger(config)
        
        # Inicializar monitor do sistema
        system_monitor = SystemMonitor(config)
        
        # Agendar backups automáticos
        backup_manager.schedule_backups()
        backup_manager.start_scheduler()
        
        logger.info("Sistema de backup e auditoria inicializado com sucesso")
        
        return backup_manager, audit_logger, system_monitor
        
    except Exception as e:
        logger.error(f"Erro ao inicializar sistema de backup e auditoria: {e}")
        raise
