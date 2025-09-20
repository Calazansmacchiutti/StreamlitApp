"""
Módulo para pré-processamento de dados das estações de tratamento
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
import streamlit as st
from typing import Dict, List, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Classe para pré-processamento de dados"""
    
    def __init__(self):
        self.scalers = {}
        self.imputers = {}
        self.missing_strategy = 'interpolate'
        self.outlier_method = 'iqr'
        self.outlier_threshold = 1.5
        
    def clean_data(self, df: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
        """
        Limpar dados removendo outliers e preenchendo valores faltantes
        
        Args:
            df: DataFrame com os dados
            method: Método de limpeza ('interpolate', 'drop', 'mean', 'median')
            
        Returns:
            DataFrame limpo
        """
        df_clean = df.copy()
        
        # Remover duplicatas
        df_clean = df_clean.drop_duplicates()
        
        # Tratar valores faltantes
        if method == 'interpolate':
            df_clean = df_clean.interpolate(method='linear', limit_direction='both')
        elif method == 'drop':
            df_clean = df_clean.dropna()
        elif method == 'mean':
            numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_columns] = df_clean[numeric_columns].fillna(df_clean[numeric_columns].mean())
        elif method == 'median':
            numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
            df_clean[numeric_columns] = df_clean[numeric_columns].fillna(df_clean[numeric_columns].median())
            
        return df_clean
        
    def remove_outliers(self, df: pd.DataFrame, method: str = 'iqr', 
                       parameters: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Remover outliers dos dados
        
        Args:
            df: DataFrame com os dados
            method: Método de detecção ('iqr', 'zscore', 'isolation')
            parameters: Lista de parâmetros para processar
            
        Returns:
            DataFrame sem outliers
        """
        df_clean = df.copy()
        
        if parameters is None:
            parameters = [col for col in df.columns if col != 'timestamp' and df[col].dtype in ['float64', 'int64']]
            
        for param in parameters:
            if param not in df_clean.columns:
                continue
                
            if method == 'iqr':
                Q1 = df_clean[param].quantile(0.25)
                Q3 = df_clean[param].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                mask = (df_clean[param] >= lower_bound) & (df_clean[param] <= upper_bound)
                df_clean = df_clean[mask]
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(df_clean[param].dropna()))
                threshold = 3
                mask = z_scores < threshold
                df_clean = df_clean[mask]
                
        return df_clean
        
    def smooth_data(self, df: pd.DataFrame, method: str = 'moving_average', 
                   window_size: int = 5, parameters: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Suavizar dados usando diferentes métodos
        
        Args:
            df: DataFrame com os dados
            method: Método de suavização ('moving_average', 'savgol', 'exponential')
            window_size: Tamanho da janela
            parameters: Lista de parâmetros para processar
            
        Returns:
            DataFrame suavizado
        """
        df_smooth = df.copy()
        
        if parameters is None:
            parameters = [col for col in df.columns if col != 'timestamp' and df[col].dtype in ['float64', 'int64']]
            
        for param in parameters:
            if param not in df_smooth.columns:
                continue
                
            if method == 'moving_average':
                df_smooth[param] = df_smooth[param].rolling(window=window_size, center=True).mean()
                
            elif method == 'savgol':
                if len(df_smooth[param].dropna()) > window_size:
                    df_smooth[param] = savgol_filter(df_smooth[param].fillna(method='ffill'), 
                                                   window_size, 3)
                    
            elif method == 'exponential':
                df_smooth[param] = df_smooth[param].ewm(span=window_size).mean()
                
        return df_smooth
        
    def normalize_data(self, df: pd.DataFrame, method: str = 'standard', 
                      parameters: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Normalizar dados
        
        Args:
            df: DataFrame com os dados
            method: Método de normalização ('standard', 'minmax', 'robust')
            parameters: Lista de parâmetros para processar
            
        Returns:
            DataFrame normalizado
        """
        df_norm = df.copy()
        
        if parameters is None:
            parameters = [col for col in df.columns if col != 'timestamp' and df[col].dtype in ['float64', 'int64']]
            
        for param in parameters:
            if param not in df_norm.columns:
                continue
                
            if method == 'standard':
                if param not in self.scalers:
                    self.scalers[param] = StandardScaler()
                df_norm[param] = self.scalers[param].fit_transform(df_norm[[param]]).flatten()
                
            elif method == 'minmax':
                scaler = MinMaxScaler()
                df_norm[param] = scaler.fit_transform(df_norm[[param]]).flatten()
                
            elif method == 'robust':
                median = df_norm[param].median()
                mad = np.median(np.abs(df_norm[param] - median))
                df_norm[param] = (df_norm[param] - median) / mad
                
        return df_norm
        
    def create_features(self, df: pd.DataFrame, parameters: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Criar features derivadas dos dados
        
        Args:
            df: DataFrame com os dados
            parameters: Lista de parâmetros para processar
            
        Returns:
            DataFrame com features adicionais
        """
        df_features = df.copy()
        
        if parameters is None:
            parameters = [col for col in df.columns if col != 'timestamp' and df[col].dtype in ['float64', 'int64']]
            
        # Features temporais
        if 'timestamp' in df_features.columns:
            df_features['hour'] = df_features['timestamp'].dt.hour
            df_features['day_of_week'] = df_features['timestamp'].dt.dayofweek
            df_features['month'] = df_features['timestamp'].dt.month
            df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6]).astype(int)
            
        # Features de lag
        for param in parameters:
            if param in df_features.columns:
                # Lag de 1, 4, 12, 24, 96 períodos (15min, 1h, 3h, 6h, 1d)
                for lag in [1, 4, 12, 24, 96]:
                    df_features[f'{param}_lag_{lag}'] = df_features[param].shift(lag)
                    
                # Médias móveis
                for window in [4, 12, 24, 96]:  # 1h, 3h, 6h, 1d
                    df_features[f'{param}_ma_{window}'] = df_features[param].rolling(window=window).mean()
                    
                # Desvio padrão móvel
                for window in [12, 24, 96]:  # 3h, 6h, 1d
                    df_features[f'{param}_std_{window}'] = df_features[param].rolling(window=window).std()
                    
                # Diferenças
                df_features[f'{param}_diff_1'] = df_features[param].diff(1)
                df_features[f'{param}_diff_4'] = df_features[param].diff(4)
                
        return df_features
        
    def detect_anomalies_simple(self, df: pd.DataFrame, parameters: Optional[List[str]] = None,
                               method: str = 'iqr') -> pd.DataFrame:
        """
        Detecção simples de anomalias
        
        Args:
            df: DataFrame com os dados
            parameters: Lista de parâmetros para analisar
            method: Método de detecção ('iqr', 'zscore', 'modified_zscore')
            
        Returns:
            DataFrame com colunas de anomalias
        """
        df_anomalies = df.copy()
        
        if parameters is None:
            parameters = [col for col in df.columns if col != 'timestamp' and df[col].dtype in ['float64', 'int64']]
            
        for param in parameters:
            if param not in df_anomalies.columns:
                continue
                
            if method == 'iqr':
                Q1 = df_anomalies[param].quantile(0.25)
                Q3 = df_anomalies[param].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                df_anomalies[f'{param}_anomaly'] = (
                    (df_anomalies[param] < lower_bound) | 
                    (df_anomalies[param] > upper_bound)
                ).astype(int)
                
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(df_anomalies[param].dropna()))
                threshold = 3
                df_anomalies[f'{param}_anomaly'] = (z_scores > threshold).astype(int)
                
            elif method == 'modified_zscore':
                median = df_anomalies[param].median()
                mad = np.median(np.abs(df_anomalies[param] - median))
                modified_z_scores = 0.6745 * (df_anomalies[param] - median) / mad
                df_anomalies[f'{param}_anomaly'] = (np.abs(modified_z_scores) > 3.5).astype(int)
                
        return df_anomalies
        
    def calculate_statistics(self, df: pd.DataFrame, parameters: Optional[List[str]] = None) -> Dict:
        """
        Calcular estatísticas descritivas
        
        Args:
            df: DataFrame com os dados
            parameters: Lista de parâmetros para analisar
            
        Returns:
            Dicionário com estatísticas
        """
        if parameters is None:
            parameters = [col for col in df.columns if col != 'timestamp' and df[col].dtype in ['float64', 'int64']]
            
        stats_dict = {}
        
        for param in parameters:
            if param not in df.columns:
                continue
                
            param_data = df[param].dropna()
            
            stats_dict[param] = {
                'count': len(param_data),
                'mean': param_data.mean(),
                'std': param_data.std(),
                'min': param_data.min(),
                'max': param_data.max(),
                'median': param_data.median(),
                'q25': param_data.quantile(0.25),
                'q75': param_data.quantile(0.75),
                'skewness': stats.skew(param_data),
                'kurtosis': stats.kurtosis(param_data),
                'cv': param_data.std() / param_data.mean() if param_data.mean() != 0 else 0
            }
            
        return stats_dict
        
    def validate_thresholds(self, df: pd.DataFrame, thresholds: Dict, 
                           parameters: Optional[List[str]] = None) -> Dict:
        """
        Validar dados contra limites de controle
        
        Args:
            df: DataFrame com os dados
            thresholds: Dicionário com limites por parâmetro
            parameters: Lista de parâmetros para validar
            
        Returns:
            Dicionário com resultados da validação
        """
        if parameters is None:
            parameters = [col for col in df.columns if col != 'timestamp' and df[col].dtype in ['float64', 'int64']]
            
        validation_results = {}
        
        for param in parameters:
            if param not in df.columns or param not in thresholds:
                continue
                
            param_data = df[param].dropna()
            param_thresholds = thresholds[param]
            
            violations = {
                'total_measurements': len(param_data),
                'within_limits': 0,
                'outside_limits': 0,
                'critical_violations': 0,
                'violation_rate': 0
            }
            
            for value in param_data:
                within_limits = True
                is_critical = False
                
                # Verificar limites normais
                if 'min' in param_thresholds and value < param_thresholds['min']:
                    within_limits = False
                if 'max' in param_thresholds and value > param_thresholds['max']:
                    within_limits = False
                    
                # Verificar limites críticos
                if 'critical_min' in param_thresholds and value < param_thresholds['critical_min']:
                    within_limits = False
                    is_critical = True
                if 'critical_max' in param_thresholds and value > param_thresholds['critical_max']:
                    within_limits = False
                    is_critical = True
                    
                if within_limits:
                    violations['within_limits'] += 1
                else:
                    violations['outside_limits'] += 1
                    if is_critical:
                        violations['critical_violations'] += 1
                        
            violations['violation_rate'] = violations['outside_limits'] / violations['total_measurements'] * 100
            validation_results[param] = violations
            
        return validation_results
        
    def handle_missing_values(self, df: pd.DataFrame, 
                            strategy: str = 'interpolate',
                            columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Tratar valores ausentes com múltiplas estratégias"""
        df = df.copy()
        columns = columns or df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            if strategy == 'interpolate':
                df[col] = df[col].interpolate(method='linear', limit_direction='both')
            elif strategy == 'forward_fill':
                df[col] = df[col].fillna(method='ffill')
            elif strategy == 'backward_fill':
                df[col] = df[col].fillna(method='bfill')
            elif strategy == 'mean':
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == 'median':
                df[col] = df[col].fillna(df[col].median())
            elif strategy == 'knn':
                imputer = KNNImputer(n_neighbors=5)
                df[col] = imputer.fit_transform(df[[col]]).flatten()
            elif strategy == 'drop':
                df = df.dropna(subset=[col])
                
        return df
        
    def create_lag_features(self, df: pd.DataFrame, 
                          target_col: str,
                          lags: List[int]) -> pd.DataFrame:
        """Criar features de lag para modelagem"""
        df = df.copy()
        
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
            
        return df
        
    def create_rolling_features(self, df: pd.DataFrame,
                              target_col: str,
                              windows: List[int],
                              functions: List[str] = ['mean', 'std', 'min', 'max']) -> pd.DataFrame:
        """Criar features de janela móvel"""
        df = df.copy()
        
        for window in windows:
            for func in functions:
                if func == 'mean':
                    df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window).mean()
                elif func == 'std':
                    df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window).std()
                elif func == 'min':
                    df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window).min()
                elif func == 'max':
                    df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window).max()
                elif func == 'median':
                    df[f'{target_col}_rolling_median_{window}'] = df[target_col].rolling(window).median()
                elif func == 'quantile_25':
                    df[f'{target_col}_rolling_q25_{window}'] = df[target_col].rolling(window).quantile(0.25)
                elif func == 'quantile_75':
                    df[f'{target_col}_rolling_q75_{window}'] = df[target_col].rolling(window).quantile(0.75)
                    
        return df
        
    def create_time_features(self, df: pd.DataFrame, 
                           timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """Criar features temporais avançadas"""
        df = df.copy()
        
        # Features básicas
        df['hour'] = df[timestamp_col].dt.hour
        df['day'] = df[timestamp_col].dt.day
        df['dayofweek'] = df[timestamp_col].dt.dayofweek
        df['month'] = df[timestamp_col].dt.month
        df['quarter'] = df[timestamp_col].dt.quarter
        df['year'] = df[timestamp_col].dt.year
        df['dayofyear'] = df[timestamp_col].dt.dayofyear
        df['weekofyear'] = df[timestamp_col].dt.isocalendar().week
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        df['is_month_start'] = df[timestamp_col].dt.is_month_start.astype(int)
        df['is_month_end'] = df[timestamp_col].dt.is_month_end.astype(int)
        df['is_quarter_start'] = df[timestamp_col].dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = df[timestamp_col].dt.is_quarter_end.astype(int)
        df['is_year_start'] = df[timestamp_col].dt.is_year_start.astype(int)
        df['is_year_end'] = df[timestamp_col].dt.is_year_end.astype(int)
        
        # Features cíclicas
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        
        return df
        
    def create_interaction_features(self, df: pd.DataFrame, 
                                  columns: List[str]) -> pd.DataFrame:
        """Criar features de interação entre variáveis"""
        df = df.copy()
        
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                # Multiplicação
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                # Divisão (evitar divisão por zero)
                df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-8)
                # Diferença
                df[f'{col1}_minus_{col2}'] = df[col1] - df[col2]
                # Soma
                df[f'{col1}_plus_{col2}'] = df[col1] + df[col2]
                
        return df
        
    def apply_pca(self, df: pd.DataFrame, 
                 columns: List[str],
                 n_components: Optional[int] = None,
                 explained_variance_threshold: float = 0.95) -> Tuple[pd.DataFrame, PCA]:
        """Aplicar PCA para redução de dimensionalidade"""
        df = df.copy()
        
        # Selecionar apenas colunas numéricas
        numeric_data = df[columns].select_dtypes(include=[np.number])
        
        # Normalizar dados
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(numeric_data)
        
        # Aplicar PCA
        if n_components is None:
            pca = PCA()
            pca.fit(data_scaled)
            
            # Encontrar número de componentes para threshold de variância
            cumsum = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= explained_variance_threshold) + 1
            
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(data_scaled)
        
        # Adicionar componentes ao DataFrame
        for i in range(n_components):
            df[f'PC_{i+1}'] = pca_result[:, i]
            
        return df, pca
        
    def select_features(self, df: pd.DataFrame,
                       target_col: str,
                       k: int = 10,
                       method: str = 'f_regression') -> List[str]:
        """Seleção de features usando métodos estatísticos"""
        
        # Separar features e target
        feature_cols = [col for col in df.columns if col != target_col and df[col].dtype in ['float64', 'int64']]
        X = df[feature_cols].fillna(0)
        y = df[target_col].fillna(0)
        
        # Seleção de features
        if method == 'f_regression':
            selector = SelectKBest(score_func=f_regression, k=k)
        else:
            raise ValueError(f"Método {method} não suportado")
            
        selector.fit(X, y)
        
        # Obter features selecionadas
        selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
        
        return selected_features
        
    def detect_drift(self, df1: pd.DataFrame, df2: pd.DataFrame,
                    columns: List[str]) -> Dict[str, float]:
        """Detectar drift entre dois conjuntos de dados"""
        drift_scores = {}
        
        for col in columns:
            if col in df1.columns and col in df2.columns:
                # Kolmogorov-Smirnov test
                ks_stat, ks_pvalue = stats.ks_2samp(df1[col].dropna(), df2[col].dropna())
                
                # Wasserstein distance
                from scipy.stats import wasserstein_distance
                wasserstein_dist = wasserstein_distance(df1[col].dropna(), df2[col].dropna())
                
                # Population Stability Index (PSI)
                psi = self.calculate_psi(df1[col], df2[col])
                
                drift_scores[col] = {
                    'ks_statistic': ks_stat,
                    'ks_pvalue': ks_pvalue,
                    'wasserstein_distance': wasserstein_dist,
                    'psi': psi,
                    'drift_detected': ks_pvalue < 0.05 or psi > 0.2
                }
                
        return drift_scores
        
    def calculate_psi(self, expected: pd.Series, actual: pd.Series, 
                     bins: int = 10) -> float:
        """Calcular Population Stability Index"""
        # Criar bins
        breakpoints = np.linspace(min(expected.min(), actual.min()),
                                max(expected.max(), actual.max()),
                                bins + 1)
        
        # Calcular distribuições
        expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)
        
        # Evitar divisão por zero
        expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
        actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
        
        # Calcular PSI
        psi = np.sum((actual_percents - expected_percents) * 
                    np.log(actual_percents / expected_percents))
        
        return psi
        
    def get_preprocessing_summary(self, df: pd.DataFrame) -> Dict:
        """Obter resumo do pré-processamento"""
        summary = {
            'total_records': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'duplicate_records': df.duplicated().sum(),
            'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(df.select_dtypes(include=['object']).columns),
            'datetime_columns': list(df.select_dtypes(include=['datetime64']).columns)
        }
        
        # Estatísticas para colunas numéricas
        numeric_cols = summary['numeric_columns']
        if numeric_cols:
            summary['numeric_stats'] = df[numeric_cols].describe().to_dict()
            
        return summary
