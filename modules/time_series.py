"""
Módulo para análise de séries temporais e modelagem preditiva
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import streamlit as st

# Imports para análise de séries temporais
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    st.warning("statsmodels não disponível. Algumas funcionalidades de séries temporais serão limitadas.")

# Imports para Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    st.warning("Prophet não disponível. Instale com: pip install prophet")

# Imports para machine learning
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning("scikit-learn não disponível. Funcionalidades de ML serão limitadas.")

# Imports para XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class TimeSeriesAnalyzer:
    """Classe para análise de séries temporais"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        
    def check_stationarity(self, series: pd.Series, method: str = 'adf') -> Dict:
        """
        Verificar estacionariedade da série temporal
        
        Args:
            series: Série temporal
            method: Método de teste ('adf', 'kpss')
            
        Returns:
            Dicionário com resultados do teste
        """
        if not STATSMODELS_AVAILABLE:
            return {'error': 'statsmodels não disponível'}
            
        try:
            if method == 'adf':
                result = adfuller(series.dropna())
                return {
                    'test_statistic': result[0],
                    'p_value': result[1],
                    'critical_values': result[4],
                    'is_stationary': result[1] < 0.05,
                    'method': 'Augmented Dickey-Fuller'
                }
            elif method == 'kpss':
                result = kpss(series.dropna(), regression='c')
                return {
                    'test_statistic': result[0],
                    'p_value': result[1],
                    'critical_values': result[3],
                    'is_stationary': result[1] > 0.05,
                    'method': 'KPSS'
                }
        except Exception as e:
            return {'error': str(e)}
            
    def seasonal_decomposition(self, series: pd.Series, model: str = 'additive', 
                             period: int = 96) -> Dict:
        """
        Decomposição sazonal da série temporal
        
        Args:
            series: Série temporal
            model: Modelo ('additive', 'multiplicative')
            period: Período sazonal
            
        Returns:
            Dicionário com componentes da decomposição
        """
        if not STATSMODELS_AVAILABLE:
            return {'error': 'statsmodels não disponível'}
            
        try:
            decomposition = seasonal_decompose(
                series.dropna(), 
                model=model, 
                period=period,
                extrapolate_trend='freq'
            )
            
            return {
                'original': series,
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid,
                'model': model,
                'period': period
            }
        except Exception as e:
            return {'error': str(e)}
            
    def fit_arima(self, series: pd.Series, order: Tuple[int, int, int] = (1, 1, 1)) -> Dict:
        """
        Ajustar modelo ARIMA
        
        Args:
            series: Série temporal
            order: Ordem do modelo (p, d, q)
            
        Returns:
            Dicionário com modelo e métricas
        """
        if not STATSMODELS_AVAILABLE:
            return {'error': 'statsmodels não disponível'}
            
        try:
            model = ARIMA(series.dropna(), order=order)
            model_fit = model.fit()
            
            return {
                'model': model_fit,
                'aic': model_fit.aic,
                'bic': model_fit.bic,
                'summary': str(model_fit.summary()),
                'residuals': model_fit.resid,
                'fitted_values': model_fit.fittedvalues
            }
        except Exception as e:
            return {'error': str(e)}
            
    def fit_prophet(self, df: pd.DataFrame, param: str) -> Dict:
        """
        Ajustar modelo Prophet
        
        Args:
            df: DataFrame com dados
            param: Nome do parâmetro
            
        Returns:
            Dicionário com modelo e previsões
        """
        if not PROPHET_AVAILABLE:
            return {'error': 'Prophet não disponível'}
            
        try:
            # Preparar dados para Prophet
            prophet_df = df[['timestamp', param]].copy()
            prophet_df.columns = ['ds', 'y']
            prophet_df = prophet_df.dropna()
            
            # Configurar modelo
            model = Prophet(
                changepoint_prior_scale=0.05,
                seasonality_mode='multiplicative',
                interval_width=0.95
            )
            
            # Adicionar sazonalidades
            model.add_seasonality(name='daily', period=1, fourier_order=15)
            model.add_seasonality(name='weekly', period=7, fourier_order=10)
            
            # Treinar modelo
            model.fit(prophet_df)
            
            # Fazer previsões
            future = model.make_future_dataframe(periods=96, freq='15T')  # 24 horas
            forecast = model.predict(future)
            
            return {
                'model': model,
                'forecast': forecast,
                'components': model.predict_components(future)
            }
        except Exception as e:
            return {'error': str(e)}
            
    def fit_ml_model(self, df: pd.DataFrame, param: str, model_type: str = 'random_forest') -> Dict:
        """
        Ajustar modelo de machine learning
        
        Args:
            df: DataFrame com dados
            param: Nome do parâmetro
            model_type: Tipo do modelo ('random_forest', 'gradient_boosting', 'linear', 'ridge', 'lasso', 'xgboost')
            
        Returns:
            Dicionário com modelo e métricas
        """
        if not SKLEARN_AVAILABLE and model_type != 'xgboost':
            return {'error': 'scikit-learn não disponível'}
            
        try:
            # Criar features
            df_features = self._create_ml_features(df, param)
            
            # Separar features e target
            feature_cols = [col for col in df_features.columns if col not in ['timestamp', param]]
            X = df_features[feature_cols].dropna()
            y = df_features[param].loc[X.index]
            
            if len(X) == 0:
                return {'error': 'Dados insuficientes para treinamento'}
                
            # Dividir dados
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Escalar features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Selecionar modelo
            if model_type == 'random_forest':
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            elif model_type == 'gradient_boosting':
                model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            elif model_type == 'linear':
                model = LinearRegression()
            elif model_type == 'ridge':
                model = Ridge(alpha=1.0)
            elif model_type == 'lasso':
                model = Lasso(alpha=1.0)
            elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
                model = xgb.XGBRegressor(n_estimators=100, random_state=42)
            else:
                return {'error': f'Modelo {model_type} não disponível'}
                
            # Treinar modelo
            if model_type == 'xgboost':
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            else:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
            # Calcular métricas
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            return {
                'model': model,
                'scaler': scaler,
                'feature_columns': feature_cols,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'rmse': np.sqrt(mse),
                'predictions': y_pred,
                'actual': y_test
            }
        except Exception as e:
            return {'error': str(e)}
            
    def _create_ml_features(self, df: pd.DataFrame, param: str) -> pd.DataFrame:
        """
        Criar features para modelos de machine learning
        
        Args:
            df: DataFrame com dados
            param: Nome do parâmetro
            
        Returns:
            DataFrame com features
        """
        df_features = df.copy()
        
        # Features temporais
        if 'timestamp' in df_features.columns:
            df_features['hour'] = df_features['timestamp'].dt.hour
            df_features['day_of_week'] = df_features['timestamp'].dt.dayofweek
            df_features['month'] = df_features['timestamp'].dt.month
            df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6]).astype(int)
            
        # Features de lag
        lags = [1, 4, 12, 24, 96]  # 15min, 1h, 3h, 6h, 1d
        for lag in lags:
            df_features[f'{param}_lag_{lag}'] = df_features[param].shift(lag)
            
        # Médias móveis
        windows = [4, 12, 24, 96]  # 1h, 3h, 6h, 1d
        for window in windows:
            df_features[f'{param}_ma_{window}'] = df_features[param].rolling(window=window).mean()
            
        # Desvio padrão móvel
        for window in [12, 24, 96]:
            df_features[f'{param}_std_{window}'] = df_features[param].rolling(window=window).std()
            
        # Diferenças
        df_features[f'{param}_diff_1'] = df_features[param].diff(1)
        df_features[f'{param}_diff_4'] = df_features[param].diff(4)
        
        return df_features
        
    def forecast_arima(self, model, periods: int = 24) -> pd.Series:
        """
        Fazer previsões com modelo ARIMA
        
        Args:
            model: Modelo ARIMA treinado
            periods: Número de períodos para prever
            
        Returns:
            Série com previsões
        """
        try:
            forecast = model.forecast(steps=periods)
            return forecast
        except Exception as e:
            return pd.Series([], name='forecast')
            
    def forecast_ml(self, model, scaler, feature_columns: List[str], 
                   last_values: pd.Series, periods: int = 24) -> List[float]:
        """
        Fazer previsões com modelo de ML
        
        Args:
            model: Modelo treinado
            scaler: Scaler usado no treinamento
            feature_columns: Lista de colunas de features
            last_values: Últimos valores da série
            periods: Número de períodos para prever
            
        Returns:
            Lista com previsões
        """
        try:
            predictions = []
            current_values = last_values.copy()
            
            for _ in range(periods):
                # Criar features para próxima previsão
                features = []
                
                for col in feature_columns:
                    if col.startswith('hour'):
                        features.append(datetime.now().hour)
                    elif col.startswith('day_of_week'):
                        features.append(datetime.now().weekday())
                    elif col.startswith('month'):
                        features.append(datetime.now().month)
                    elif col.startswith('is_weekend'):
                        features.append(1 if datetime.now().weekday() in [5, 6] else 0)
                    elif col.endswith('_lag_1'):
                        features.append(current_values.iloc[-1])
                    elif col.endswith('_lag_4'):
                        features.append(current_values.iloc[-4] if len(current_values) >= 4 else current_values.iloc[-1])
                    elif col.endswith('_lag_12'):
                        features.append(current_values.iloc[-12] if len(current_values) >= 12 else current_values.iloc[-1])
                    elif col.endswith('_lag_24'):
                        features.append(current_values.iloc[-24] if len(current_values) >= 24 else current_values.iloc[-1])
                    elif col.endswith('_lag_96'):
                        features.append(current_values.iloc[-96] if len(current_values) >= 96 else current_values.iloc[-1])
                    elif col.endswith('_ma_4'):
                        features.append(current_values.tail(4).mean())
                    elif col.endswith('_ma_12'):
                        features.append(current_values.tail(12).mean())
                    elif col.endswith('_ma_24'):
                        features.append(current_values.tail(24).mean())
                    elif col.endswith('_ma_96'):
                        features.append(current_values.tail(96).mean())
                    elif col.endswith('_std_12'):
                        features.append(current_values.tail(12).std())
                    elif col.endswith('_std_24'):
                        features.append(current_values.tail(24).std())
                    elif col.endswith('_std_96'):
                        features.append(current_values.tail(96).std())
                    elif col.endswith('_diff_1'):
                        features.append(current_values.iloc[-1] - current_values.iloc[-2] if len(current_values) >= 2 else 0)
                    elif col.endswith('_diff_4'):
                        features.append(current_values.iloc[-1] - current_values.iloc[-5] if len(current_values) >= 5 else 0)
                    else:
                        features.append(0)
                        
                # Fazer previsão
                if hasattr(model, 'predict'):
                    if 'XGBRegressor' in str(type(model)):
                        pred = model.predict([features])[0]
                    else:
                        features_scaled = scaler.transform([features])
                        pred = model.predict(features_scaled)[0]
                else:
                    pred = 0
                    
                predictions.append(pred)
                current_values = pd.concat([current_values, pd.Series([pred])])
                
            return predictions
        except Exception as e:
            return [0] * periods
            
    def calculate_forecast_metrics(self, actual: pd.Series, predicted: pd.Series) -> Dict:
        """
        Calcular métricas de avaliação de previsões
        
        Args:
            actual: Valores reais
            predicted: Valores previstos
            
        Returns:
            Dicionário com métricas
        """
        try:
            # Alinhar séries
            min_len = min(len(actual), len(predicted))
            actual_aligned = actual.iloc[:min_len]
            predicted_aligned = predicted.iloc[:min_len]
            
            mse = mean_squared_error(actual_aligned, predicted_aligned)
            mae = mean_absolute_error(actual_aligned, predicted_aligned)
            rmse = np.sqrt(mse)
            
            # MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((actual_aligned - predicted_aligned) / actual_aligned)) * 100
            
            # R²
            r2 = r2_score(actual_aligned, predicted_aligned)
            
            return {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'r2': r2
            }
        except Exception as e:
            return {'error': str(e)}
            
    def cross_validation(self, df: pd.DataFrame, param: str, model_type: str = 'arima',
                        n_splits: int = 5) -> Dict:
        """
        Validação cruzada para séries temporais
        
        Args:
            df: DataFrame com dados
            param: Nome do parâmetro
            model_type: Tipo do modelo
            n_splits: Número de divisões
            
        Returns:
            Dicionário com resultados da validação cruzada
        """
        try:
            series = df[param].dropna()
            split_size = len(series) // n_splits
            
            metrics = []
            
            for i in range(n_splits):
                # Dividir dados
                train_end = (i + 1) * split_size
                test_start = train_end
                test_end = min(test_start + split_size, len(series))
                
                train_data = series.iloc[:train_end]
                test_data = series.iloc[test_start:test_end]
                
                if len(test_data) == 0:
                    continue
                    
                # Treinar modelo
                if model_type == 'arima' and STATSMODELS_AVAILABLE:
                    model = ARIMA(train_data, order=(1, 1, 1))
                    model_fit = model.fit()
                    predictions = model_fit.forecast(steps=len(test_data))
                else:
                    # Modelo simples (média móvel)
                    predictions = [train_data.tail(24).mean()] * len(test_data)
                    
                # Calcular métricas
                split_metrics = self.calculate_forecast_metrics(test_data, pd.Series(predictions))
                if 'error' not in split_metrics:
                    metrics.append(split_metrics)
                    
            if metrics:
                # Calcular métricas médias
                avg_metrics = {}
                for key in metrics[0].keys():
                    avg_metrics[key] = np.mean([m[key] for m in metrics])
                    
                return avg_metrics
            else:
                return {'error': 'Não foi possível calcular métricas'}
                
        except Exception as e:
            return {'error': str(e)}
