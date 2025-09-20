"""
Módulo para detecção de anomalias em dados de estações de tratamento
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import streamlit as st

# Imports para algoritmos de detecção de anomalias
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.cluster import DBSCAN
    from sklearn.svm import OneClassSVM
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning("scikit-learn não disponível. Algumas funcionalidades de detecção de anomalias serão limitadas.")

# Imports para PyOD (Python Outlier Detection)
try:
    from pyod.models.lof import LOF
    from pyod.models.copod import COPOD
    from pyod.models.ecod import ECOD
    from pyod.models.abod import ABOD
    PYOD_AVAILABLE = True
except ImportError:
    PYOD_AVAILABLE = False

# Imports para análise estatística
try:
    from scipy import stats
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class AnomalyDetector:
    """Classe para detecção de anomalias"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        
    def detect_outliers_statistical(self, df: pd.DataFrame, parameters: List[str],
                                   method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        Detecção de outliers usando métodos estatísticos
        
        Args:
            df: DataFrame com os dados
            parameters: Lista de parâmetros para analisar
            method: Método de detecção ('iqr', 'zscore', 'modified_zscore', 'grubbs')
            threshold: Limiar para detecção
            
        Returns:
            DataFrame com colunas de anomalias
        """
        df_anomalies = df.copy()
        
        for param in parameters:
            if param not in df_anomalies.columns:
                continue
                
            param_data = df_anomalies[param].dropna()
            
            if method == 'iqr':
                Q1 = param_data.quantile(0.25)
                Q3 = param_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                df_anomalies[f'{param}_anomaly'] = (
                    (df_anomalies[param] < lower_bound) | 
                    (df_anomalies[param] > upper_bound)
                ).astype(int)
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(param_data))
                df_anomalies[f'{param}_anomaly'] = (z_scores > threshold).astype(int)
                
            elif method == 'modified_zscore':
                median = param_data.median()
                mad = np.median(np.abs(param_data - median))
                modified_z_scores = 0.6745 * (param_data - median) / mad
                df_anomalies[f'{param}_anomaly'] = (np.abs(modified_z_scores) > threshold).astype(int)
                
            elif method == 'grubbs' and SCIPY_AVAILABLE:
                # Teste de Grubbs para outliers
                outliers = []
                data_copy = param_data.copy()
                
                while len(data_copy) > 3:
                    n = len(data_copy)
                    mean_val = data_copy.mean()
                    std_val = data_copy.std()
                    
                    # Calcular estatística G
                    g_stat = np.max(np.abs(data_copy - mean_val)) / std_val
                    
                    # Valor crítico (aproximado)
                    t_critical = stats.t.ppf(1 - 0.05 / (2 * n), n - 2)
                    g_critical = ((n - 1) / np.sqrt(n)) * np.sqrt(t_critical**2 / (n - 2 + t_critical**2))
                    
                    if g_stat > g_critical:
                        outlier_idx = np.argmax(np.abs(data_copy - mean_val))
                        outliers.append(data_copy.index[outlier_idx])
                        data_copy = data_copy.drop(data_copy.index[outlier_idx])
                    else:
                        break
                        
                df_anomalies[f'{param}_anomaly'] = df_anomalies.index.isin(outliers).astype(int)
                
        return df_anomalies
        
    def detect_anomalies_isolation_forest(self, df: pd.DataFrame, parameters: List[str],
                                        contamination: float = 0.1) -> pd.DataFrame:
        """
        Detecção de anomalias usando Isolation Forest
        
        Args:
            df: DataFrame com os dados
            parameters: Lista de parâmetros para analisar
            contamination: Proporção esperada de anomalias
            
        Returns:
            DataFrame com colunas de anomalias
        """
        if not SKLEARN_AVAILABLE:
            st.error("scikit-learn não disponível para Isolation Forest")
            return df
            
        df_anomalies = df.copy()
        
        # Preparar dados
        data_matrix = df[parameters].dropna()
        
        if len(data_matrix) == 0:
            return df_anomalies
            
        # Escalar dados
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_matrix)
        
        # Treinar modelo
        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        predictions = model.fit_predict(data_scaled)
        
        # Adicionar coluna de anomalias
        anomaly_mask = predictions == -1
        df_anomalies['isolation_forest_anomaly'] = 0
        df_anomalies.loc[data_matrix.index, 'isolation_forest_anomaly'] = anomaly_mask.astype(int)
        
        # Adicionar scores de anomalia
        anomaly_scores = model.decision_function(data_scaled)
        df_anomalies['isolation_forest_score'] = 0.0
        df_anomalies.loc[data_matrix.index, 'isolation_forest_score'] = anomaly_scores
        
        return df_anomalies
        
    def detect_anomalies_dbscan(self, df: pd.DataFrame, parameters: List[str],
                               eps: float = 0.5, min_samples: int = 5) -> pd.DataFrame:
        """
        Detecção de anomalias usando DBSCAN
        
        Args:
            df: DataFrame com os dados
            parameters: Lista de parâmetros para analisar
            eps: Distância máxima entre amostras
            min_samples: Número mínimo de amostras em um cluster
            
        Returns:
            DataFrame com colunas de anomalias
        """
        if not SKLEARN_AVAILABLE:
            st.error("scikit-learn não disponível para DBSCAN")
            return df
            
        df_anomalies = df.copy()
        
        # Preparar dados
        data_matrix = df[parameters].dropna()
        
        if len(data_matrix) == 0:
            return df_anomalies
            
        # Escalar dados
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_matrix)
        
        # Treinar modelo
        model = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = model.fit_predict(data_scaled)
        
        # Anomalias são pontos com label -1
        anomaly_mask = cluster_labels == -1
        df_anomalies['dbscan_anomaly'] = 0
        df_anomalies.loc[data_matrix.index, 'dbscan_anomaly'] = anomaly_mask.astype(int)
        
        return df_anomalies
        
    def detect_anomalies_one_class_svm(self, df: pd.DataFrame, parameters: List[str],
                                     nu: float = 0.1) -> pd.DataFrame:
        """
        Detecção de anomalias usando One-Class SVM
        
        Args:
            df: DataFrame com os dados
            parameters: Lista de parâmetros para analisar
            nu: Proporção de outliers esperados
            
        Returns:
            DataFrame com colunas de anomalias
        """
        if not SKLEARN_AVAILABLE:
            st.error("scikit-learn não disponível para One-Class SVM")
            return df
            
        df_anomalies = df.copy()
        
        # Preparar dados
        data_matrix = df[parameters].dropna()
        
        if len(data_matrix) == 0:
            return df_anomalies
            
        # Escalar dados
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_matrix)
        
        # Treinar modelo
        model = OneClassSVM(nu=nu, kernel='rbf', gamma='scale')
        predictions = model.fit_predict(data_scaled)
        
        # Anomalias são pontos com label -1
        anomaly_mask = predictions == -1
        df_anomalies['one_class_svm_anomaly'] = 0
        df_anomalies.loc[data_matrix.index, 'one_class_svm_anomaly'] = anomaly_mask.astype(int)
        
        # Adicionar scores de anomalia
        anomaly_scores = model.decision_function(data_scaled)
        df_anomalies['one_class_svm_score'] = 0.0
        df_anomalies.loc[data_matrix.index, 'one_class_svm_score'] = anomaly_scores
        
        return df_anomalies
        
    def detect_anomalies_lstm_autoencoder(self, df: pd.DataFrame, parameters: List[str],
                                        sequence_length: int = 24, threshold: float = 0.1) -> pd.DataFrame:
        """
        Detecção de anomalias usando LSTM Autoencoder (simplificado)
        
        Args:
            df: DataFrame com os dados
            parameters: Lista de parâmetros para analisar
            sequence_length: Comprimento da sequência
            threshold: Limiar para detecção de anomalias
            
        Returns:
            DataFrame com colunas de anomalias
        """
        df_anomalies = df.copy()
        
        # Preparar dados
        data_matrix = df[parameters].dropna()
        
        if len(data_matrix) < sequence_length:
            return df_anomalies
            
        # Escalar dados
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_matrix)
        
        # Criar sequências
        sequences = []
        for i in range(len(data_scaled) - sequence_length + 1):
            sequences.append(data_scaled[i:i + sequence_length])
            
        sequences = np.array(sequences)
        
        # Simular autoencoder simples (média móvel como proxy)
        reconstructed = []
        for seq in sequences:
            # Reconstrução simples usando média móvel
            recon = np.mean(seq, axis=0)
            reconstructed.append(recon)
            
        reconstructed = np.array(reconstructed)
        
        # Calcular erro de reconstrução
        reconstruction_errors = []
        for i, seq in enumerate(sequences):
            error = np.mean((seq[-1] - reconstructed[i]) ** 2)
            reconstruction_errors.append(error)
            
        # Detectar anomalias
        threshold_value = np.percentile(reconstruction_errors, (1 - threshold) * 100)
        anomaly_mask = np.array(reconstruction_errors) > threshold_value
        
        # Adicionar colunas de anomalias
        df_anomalies['lstm_autoencoder_anomaly'] = 0
        df_anomalies['lstm_autoencoder_score'] = 0.0
        
        # Mapear anomalias para índices originais
        for i, is_anomaly in enumerate(anomaly_mask):
            original_idx = data_matrix.index[i + sequence_length - 1]
            df_anomalies.loc[original_idx, 'lstm_autoencoder_anomaly'] = int(is_anomaly)
            df_anomalies.loc[original_idx, 'lstm_autoencoder_score'] = reconstruction_errors[i]
            
        return df_anomalies
        
    def detect_anomalies_pyod(self, df: pd.DataFrame, parameters: List[str],
                             method: str = 'lof', contamination: float = 0.1) -> pd.DataFrame:
        """
        Detecção de anomalias usando PyOD
        
        Args:
            df: DataFrame com os dados
            parameters: Lista de parâmetros para analisar
            method: Método PyOD ('lof', 'copod', 'ecod', 'abod')
            contamination: Proporção esperada de anomalias
            
        Returns:
            DataFrame com colunas de anomalias
        """
        if not PYOD_AVAILABLE:
            st.error("PyOD não disponível")
            return df
            
        df_anomalies = df.copy()
        
        # Preparar dados
        data_matrix = df[parameters].dropna()
        
        if len(data_matrix) == 0:
            return df_anomalies
            
        # Escalar dados
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_matrix)
        
        # Selecionar modelo
        if method == 'lof':
            model = LOF(contamination=contamination)
        elif method == 'copod':
            model = COPOD(contamination=contamination)
        elif method == 'ecod':
            model = ECOD(contamination=contamination)
        elif method == 'abod':
            model = ABOD(contamination=contamination)
        else:
            st.error(f"Método {method} não suportado")
            return df_anomalies
            
        # Treinar modelo
        model.fit(data_scaled)
        
        # Fazer predições
        predictions = model.predict(data_scaled)
        scores = model.decision_function(data_scaled)
        
        # Adicionar colunas de anomalias
        anomaly_mask = predictions == 1
        df_anomalies[f'{method}_anomaly'] = 0
        df_anomalies.loc[data_matrix.index, f'{method}_anomaly'] = anomaly_mask.astype(int)
        
        df_anomalies[f'{method}_score'] = 0.0
        df_anomalies.loc[data_matrix.index, f'{method}_score'] = scores
        
        return df_anomalies
        
    def detect_anomalies_ensemble(self, df: pd.DataFrame, parameters: List[str],
                                 methods: List[str] = None, voting_threshold: float = 0.5) -> pd.DataFrame:
        """
        Detecção de anomalias usando ensemble de métodos
        
        Args:
            df: DataFrame com os dados
            parameters: Lista de parâmetros para analisar
            methods: Lista de métodos para usar
            voting_threshold: Limiar para votação
            
        Returns:
            DataFrame com colunas de anomalias
        """
        if methods is None:
            methods = ['isolation_forest', 'one_class_svm', 'dbscan']
            
        df_ensemble = df.copy()
        
        # Aplicar cada método
        anomaly_columns = []
        
        for method in methods:
            if method == 'isolation_forest' and SKLEARN_AVAILABLE:
                df_ensemble = self.detect_anomalies_isolation_forest(df_ensemble, parameters)
                anomaly_columns.append('isolation_forest_anomaly')
            elif method == 'one_class_svm' and SKLEARN_AVAILABLE:
                df_ensemble = self.detect_anomalies_one_class_svm(df_ensemble, parameters)
                anomaly_columns.append('one_class_svm_anomaly')
            elif method == 'dbscan' and SKLEARN_AVAILABLE:
                df_ensemble = self.detect_anomalies_dbscan(df_ensemble, parameters)
                anomaly_columns.append('dbscan_anomaly')
            elif method == 'iqr':
                df_ensemble = self.detect_outliers_statistical(df_ensemble, parameters, 'iqr')
                anomaly_columns.append(f'{parameters[0]}_anomaly')  # Assumindo primeiro parâmetro
                
        # Calcular votação
        if anomaly_columns:
            df_ensemble['ensemble_anomaly'] = (
                df_ensemble[anomaly_columns].sum(axis=1) >= len(anomaly_columns) * voting_threshold
            ).astype(int)
            
            # Calcular score médio
            score_columns = [col.replace('_anomaly', '_score') for col in anomaly_columns 
                           if col.replace('_anomaly', '_score') in df_ensemble.columns]
            if score_columns:
                df_ensemble['ensemble_score'] = df_ensemble[score_columns].mean(axis=1)
                
        return df_ensemble
        
    def classify_anomaly_severity(self, df: pd.DataFrame, parameters: List[str],
                                 thresholds: Dict) -> pd.DataFrame:
        """
        Classificar severidade das anomalias baseado em limites de controle
        
        Args:
            df: DataFrame com os dados
            parameters: Lista de parâmetros para analisar
            thresholds: Dicionário com limites por parâmetro
            
        Returns:
            DataFrame com classificação de severidade
        """
        df_severity = df.copy()
        
        # Inicializar coluna de severidade
        df_severity['anomaly_severity'] = 'normal'
        
        for param in parameters:
            if param not in df_severity.columns or param not in thresholds:
                continue
                
            param_thresholds = thresholds[param]
            param_data = df_severity[param]
            
            # Classificar por severidade
            for idx, value in param_data.items():
                if pd.isna(value):
                    continue
                    
                severity = 'normal'
                
                # Verificar limites críticos
                if 'critical_min' in param_thresholds and value < param_thresholds['critical_min']:
                    severity = 'critical'
                elif 'critical_max' in param_thresholds and value > param_thresholds['critical_max']:
                    severity = 'critical'
                # Verificar limites normais
                elif 'min' in param_thresholds and value < param_thresholds['min']:
                    severity = 'high'
                elif 'max' in param_thresholds and value > param_thresholds['max']:
                    severity = 'high'
                # Verificar se é anomalia detectada por algoritmo
                elif any(col in df_severity.columns and df_severity.loc[idx, col] == 1 
                        for col in df_severity.columns if 'anomaly' in col):
                    severity = 'medium'
                    
                # Atualizar severidade se for mais crítica
                current_severity = df_severity.loc[idx, 'anomaly_severity']
                severity_order = {'normal': 0, 'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
                
                if severity_order[severity] > severity_order[current_severity]:
                    df_severity.loc[idx, 'anomaly_severity'] = severity
                    
        return df_severity
        
    def generate_anomaly_report(self, df: pd.DataFrame) -> Dict:
        """
        Gerar relatório de anomalias
        
        Args:
            df: DataFrame com dados e anomalias
            
        Returns:
            Dicionário com relatório
        """
        report = {
            'total_records': len(df),
            'anomaly_summary': {},
            'severity_distribution': {},
            'parameter_analysis': {}
        }
        
        # Resumo de anomalias
        anomaly_columns = [col for col in df.columns if 'anomaly' in col and col != 'anomaly_severity']
        
        for col in anomaly_columns:
            anomaly_count = df[col].sum()
            report['anomaly_summary'][col] = {
                'count': int(anomaly_count),
                'percentage': round(anomaly_count / len(df) * 100, 2)
            }
            
        # Distribuição de severidade
        if 'anomaly_severity' in df.columns:
            severity_counts = df['anomaly_severity'].value_counts()
            report['severity_distribution'] = severity_counts.to_dict()
            
        # Análise por parâmetro
        parameters = [col for col in df.columns if col not in ['timestamp', 'anomaly_severity'] 
                     and not col.endswith('_anomaly') and not col.endswith('_score')]
        
        for param in parameters:
            if param in df.columns:
                param_anomalies = df[df[f'{param}_anomaly'] == 1] if f'{param}_anomaly' in df.columns else pd.DataFrame()
                
                report['parameter_analysis'][param] = {
                    'total_measurements': len(df[param].dropna()),
                    'anomaly_count': len(param_anomalies),
                    'anomaly_rate': round(len(param_anomalies) / len(df[param].dropna()) * 100, 2) if len(df[param].dropna()) > 0 else 0,
                    'mean_value': round(df[param].mean(), 3),
                    'std_value': round(df[param].std(), 3)
                }
                
        return report
