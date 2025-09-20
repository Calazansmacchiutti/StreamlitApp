"""
Módulo para visualizações e gráficos do dashboard
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


class DashboardVisualizer:
    """Classe para criar visualizações do dashboard"""
    
    def __init__(self):
        self.color_palette = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
        
    def create_realtime_chart(self, df: pd.DataFrame, parameters: List[str], 
                            title: str = "Monitoramento em Tempo Real") -> go.Figure:
        """
        Criar gráfico de monitoramento em tempo real
        
        Args:
            df: DataFrame com os dados
            parameters: Lista de parâmetros para plotar
            title: Título do gráfico
            
        Returns:
            Figura do Plotly
        """
        fig = make_subplots(
            rows=len(parameters),
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=parameters
        )
        
        colors = px.colors.qualitative.Set1
        
        for i, param in enumerate(parameters):
            if param in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df[param],
                        mode='lines',
                        name=param,
                        line=dict(width=2, color=colors[i % len(colors)]),
                        hovertemplate=f'<b>{param}</b><br>' +
                                    'Tempo: %{x}<br>' +
                                    'Valor: %{y:.2f}<extra></extra>'
                    ),
                    row=i+1, col=1
                )
                
        fig.update_layout(
            title=title,
            height=200*len(parameters),
            showlegend=False,
            hovermode='x unified'
        )
        
        fig.update_xaxes(title_text="Tempo")
        
        return fig
        
    def create_threshold_chart(self, df: pd.DataFrame, param: str, 
                             thresholds: Dict, title: str = None) -> go.Figure:
        """
        Criar gráfico com limites de controle
        
        Args:
            df: DataFrame com os dados
            param: Parâmetro para plotar
            thresholds: Dicionário com limites
            title: Título do gráfico
            
        Returns:
            Figura do Plotly
        """
        if title is None:
            title = f"Monitoramento - {param}"
            
        fig = go.Figure()
        
        # Dados principais
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df[param],
            mode='lines',
            name=param,
            line=dict(width=2, color=self.color_palette['primary']),
            hovertemplate=f'<b>{param}</b><br>' +
                         'Tempo: %{x}<br>' +
                         'Valor: %{y:.2f}<extra></extra>'
        ))
        
        # Limites de controle
        if 'min' in thresholds:
            fig.add_hline(
                y=thresholds['min'],
                line_dash="dash",
                line_color=self.color_palette['warning'],
                annotation_text="Limite Mín",
                annotation_position="bottom right"
            )
            
        if 'max' in thresholds:
            fig.add_hline(
                y=thresholds['max'],
                line_dash="dash",
                line_color=self.color_palette['warning'],
                annotation_text="Limite Máx",
                annotation_position="top right"
            )
            
        if 'critical_min' in thresholds:
            fig.add_hline(
                y=thresholds['critical_min'],
                line_dash="dot",
                line_color=self.color_palette['danger'],
                annotation_text="Limite Crítico Mín",
                annotation_position="bottom left"
            )
            
        if 'critical_max' in thresholds:
            fig.add_hline(
                y=thresholds['critical_max'],
                line_dash="dot",
                line_color=self.color_palette['danger'],
                annotation_text="Limite Crítico Máx",
                annotation_position="top left"
            )
            
        fig.update_layout(
            title=title,
            xaxis_title="Tempo",
            yaxis_title=param,
            height=400,
            hovermode='x unified'
        )
        
        return fig
        
    def create_correlation_heatmap(self, df: pd.DataFrame, parameters: List[str],
                                 title: str = "Matriz de Correlação") -> go.Figure:
        """
        Criar mapa de calor de correlação
        
        Args:
            df: DataFrame com os dados
            parameters: Lista de parâmetros
            title: Título do gráfico
            
        Returns:
            Figura do Plotly
        """
        # Calcular matriz de correlação
        correlation_matrix = df[parameters].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlação")
        ))
        
        fig.update_layout(
            title=title,
            height=400,
            xaxis_title="Parâmetros",
            yaxis_title="Parâmetros"
        )
        
        return fig
        
    def create_distribution_plot(self, df: pd.DataFrame, param: str,
                               title: str = None) -> go.Figure:
        """
        Criar gráfico de distribuição
        
        Args:
            df: DataFrame com os dados
            param: Parâmetro para analisar
            title: Título do gráfico
            
        Returns:
            Figura do Plotly
        """
        if title is None:
            title = f"Distribuição - {param}"
            
        fig = go.Figure()
        
        # Histograma
        fig.add_trace(go.Histogram(
            x=df[param],
            nbinsx=30,
            name='Frequência',
            marker_color=self.color_palette['primary'],
            opacity=0.7
        ))
        
        # Curva de densidade (KDE)
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(df[param].dropna())
            x_range = np.linspace(df[param].min(), df[param].max(), 100)
            density = kde(x_range)
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=density * len(df[param]) * (df[param].max() - df[param].min()) / 30,
                mode='lines',
                name='Densidade',
                line=dict(color=self.color_palette['danger'], width=2),
                yaxis='y2'
            ))
            
            fig.update_layout(yaxis2=dict(
                title="Densidade",
                overlaying="y",
                side="right"
            ))
        except ImportError:
            pass
            
        fig.update_layout(
            title=title,
            xaxis_title=param,
            yaxis_title="Frequência",
            height=400,
            barmode='overlay'
        )
        
        return fig
        
    def create_box_plot(self, df: pd.DataFrame, parameters: List[str],
                       title: str = "Box Plot dos Parâmetros") -> go.Figure:
        """
        Criar box plot dos parâmetros
        
        Args:
            df: DataFrame com os dados
            parameters: Lista de parâmetros
            title: Título do gráfico
            
        Returns:
            Figura do Plotly
        """
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        
        for i, param in enumerate(parameters):
            if param in df.columns:
                fig.add_trace(go.Box(
                    y=df[param],
                    name=param,
                    boxpoints='outliers',
                    marker_color=colors[i % len(colors)],
                    line_color='darkblue'
                ))
                
        fig.update_layout(
            title=title,
            yaxis_title="Valores",
            height=400,
            showlegend=True
        )
        
        return fig
        
    def create_anomaly_timeline(self, df: pd.DataFrame, param: str,
                              anomaly_column: str = None, title: str = None) -> go.Figure:
        """
        Criar linha do tempo de anomalias
        
        Args:
            df: DataFrame com os dados
            param: Parâmetro para plotar
            anomaly_column: Coluna com indicador de anomalia
            title: Título do gráfico
            
        Returns:
            Figura do Plotly
        """
        if title is None:
            title = f"Anomalias Detectadas - {param}"
            
        fig = go.Figure()
        
        # Dados normais
        normal_data = df[df.get(anomaly_column, 0) == 0] if anomaly_column else df
        fig.add_trace(go.Scatter(
            x=normal_data['timestamp'],
            y=normal_data[param],
            mode='lines',
            name='Normal',
            line=dict(color=self.color_palette['primary'], width=1),
            opacity=0.7
        ))
        
        # Anomalias
        if anomaly_column and anomaly_column in df.columns:
            anomaly_data = df[df[anomaly_column] == 1]
            if len(anomaly_data) > 0:
                fig.add_trace(go.Scatter(
                    x=anomaly_data['timestamp'],
                    y=anomaly_data[param],
                    mode='markers',
                    name='Anomalias',
                    marker=dict(
                        color=self.color_palette['danger'],
                        size=10,
                        symbol='x'
                    )
                ))
                
        fig.update_layout(
            title=title,
            xaxis_title="Tempo",
            yaxis_title=param,
            height=400,
            hovermode='x unified'
        )
        
        return fig
        
    def create_forecast_chart(self, df: pd.DataFrame, param: str,
                            forecast_data: pd.DataFrame, title: str = None) -> go.Figure:
        """
        Criar gráfico de previsões
        
        Args:
            df: DataFrame com dados históricos
            param: Parâmetro previsto
            forecast_data: DataFrame com previsões
            title: Título do gráfico
            
        Returns:
            Figura do Plotly
        """
        if title is None:
            title = f"Previsões - {param}"
            
        fig = go.Figure()
        
        # Dados históricos
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df[param],
            mode='lines',
            name='Dados Históricos',
            line=dict(color=self.color_palette['primary'], width=2)
        ))
        
        # Previsões
        if 'yhat' in forecast_data.columns:
            fig.add_trace(go.Scatter(
                x=forecast_data['ds'],
                y=forecast_data['yhat'],
                mode='lines',
                name='Previsão',
                line=dict(color=self.color_palette['danger'], width=2)
            ))
            
            # Intervalo de confiança
            if 'yhat_lower' in forecast_data.columns and 'yhat_upper' in forecast_data.columns:
                fig.add_trace(go.Scatter(
                    x=forecast_data['ds'].tolist() + forecast_data['ds'].tolist()[::-1],
                    y=forecast_data['yhat_upper'].tolist() + forecast_data['yhat_lower'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(255, 0, 0, 0.2)',
                    line=dict(color='rgba(255, 255, 255, 0)'),
                    name='Intervalo de Confiança',
                    showlegend=True
                ))
                
        fig.update_layout(
            title=title,
            xaxis_title="Tempo",
            yaxis_title=param,
            height=500,
            hovermode='x unified'
        )
        
        return fig
        
    def create_seasonal_decomposition_chart(self, decomposition: Dict,
                                          title: str = "Decomposição Sazonal") -> go.Figure:
        """
        Criar gráfico de decomposição sazonal
        
        Args:
            decomposition: Dicionário com componentes da decomposição
            title: Título do gráfico
            
        Returns:
            Figura do Plotly
        """
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Dados Originais', 'Tendência', 'Sazonalidade', 'Resíduos')
        )
        
        # Dados originais
        fig.add_trace(
            go.Scatter(
                x=decomposition['original'].index,
                y=decomposition['original'].values,
                mode='lines',
                name='Original',
                line=dict(color=self.color_palette['primary'])
            ),
            row=1, col=1
        )
        
        # Tendência
        fig.add_trace(
            go.Scatter(
                x=decomposition['trend'].index,
                y=decomposition['trend'].values,
                mode='lines',
                name='Tendência',
                line=dict(color=self.color_palette['danger'])
            ),
            row=2, col=1
        )
        
        # Sazonalidade
        fig.add_trace(
            go.Scatter(
                x=decomposition['seasonal'].index,
                y=decomposition['seasonal'].values,
                mode='lines',
                name='Sazonalidade',
                line=dict(color=self.color_palette['success'])
            ),
            row=3, col=1
        )
        
        # Resíduos
        fig.add_trace(
            go.Scatter(
                x=decomposition['residual'].index,
                y=decomposition['residual'].values,
                mode='markers',
                name='Resíduos',
                marker=dict(size=3, color=self.color_palette['info'])
            ),
            row=4, col=1
        )
        
        fig.update_layout(
            title=title,
            height=800,
            showlegend=False
        )
        
        return fig
        
    def create_alert_summary_chart(self, alert_data: pd.DataFrame,
                                 title: str = "Resumo de Alertas") -> go.Figure:
        """
        Criar gráfico de resumo de alertas
        
        Args:
            alert_data: DataFrame com dados de alertas
            title: Título do gráfico
            
        Returns:
            Figura do Plotly
        """
        # Agrupar alertas por severidade
        severity_counts = alert_data['severity'].value_counts()
        
        colors = {
            'critical': self.color_palette['danger'],
            'high': '#ff9800',
            'medium': self.color_palette['warning'],
            'low': self.color_palette['success']
        }
        
        fig = go.Figure(data=[
            go.Bar(
                x=severity_counts.index,
                y=severity_counts.values,
                marker_color=[colors.get(level, self.color_palette['info']) 
                             for level in severity_counts.index],
                text=severity_counts.values,
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Nível de Severidade",
            yaxis_title="Número de Alertas",
            height=400
        )
        
        return fig
        
    def create_trend_analysis_chart(self, df: pd.DataFrame, param: str,
                                  title: str = None) -> go.Figure:
        """
        Criar gráfico de análise de tendências
        
        Args:
            df: DataFrame com os dados
            param: Parâmetro para analisar
            title: Título do gráfico
            
        Returns:
            Figura do Plotly
        """
        if title is None:
            title = f"Análise de Tendências - {param}"
            
        fig = go.Figure()
        
        # Dados originais
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df[param],
            mode='lines',
            name='Dados Originais',
            line=dict(color='lightgray', width=1),
            opacity=0.5
        ))
        
        # Médias móveis
        df['MA_7'] = df[param].rolling(window=7*96).mean()  # 7 dias
        df['MA_30'] = df[param].rolling(window=30*96).mean()  # 30 dias
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['MA_7'],
            mode='lines',
            name='Média Móvel 7 dias',
            line=dict(color=self.color_palette['primary'], width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['MA_30'],
            mode='lines',
            name='Média Móvel 30 dias',
            line=dict(color=self.color_palette['danger'], width=2)
        ))
        
        # Regressão linear
        try:
            from sklearn.linear_model import LinearRegression
            X = np.arange(len(df)).reshape(-1, 1)
            y = df[param].values
            model = LinearRegression()
            model.fit(X, y)
            trend_line = model.predict(X)
            
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=trend_line,
                mode='lines',
                name='Tendência Linear',
                line=dict(color=self.color_palette['success'], width=2, dash='dash')
            ))
        except ImportError:
            pass
            
        fig.update_layout(
            title=title,
            xaxis_title="Tempo",
            yaxis_title=param,
            height=400,
            hovermode='x unified'
        )
        
        return fig
        
    def create_heatmap_2d(self, df: pd.DataFrame, param: str,
                         title: str = None) -> go.Figure:
        """
        Criar mapa de calor 2D (tempo vs valor)
        
        Args:
            df: DataFrame com os dados
            param: Parâmetro para plotar
            title: Título do gráfico
            
        Returns:
            Figura do Plotly
        """
        if title is None:
            title = f"Mapa de Calor - {param}"
            
        # Preparar dados para heatmap
        df_heatmap = df.copy()
        df_heatmap['hour'] = df_heatmap['timestamp'].dt.hour
        df_heatmap['day'] = df_heatmap['timestamp'].dt.date
        
        # Pivotar dados
        heatmap_data = df_heatmap.pivot_table(
            values=param,
            index='day',
            columns='hour',
            aggfunc='mean'
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='Viridis',
            colorbar=dict(title=param)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Hora do Dia",
            yaxis_title="Data",
            height=500
        )
        
        return fig
        
    def create_gauge_chart(self, value: float, min_val: float, max_val: float,
                          title: str = "Medição Atual") -> go.Figure:
        """
        Criar gráfico de gauge (medidor)
        
        Args:
            value: Valor atual
            min_val: Valor mínimo
            max_val: Valor máximo
            title: Título do gráfico
            
        Returns:
            Figura do Plotly
        """
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title},
            delta={'reference': (min_val + max_val) / 2},
            gauge={
                'axis': {'range': [min_val, max_val]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [min_val, (min_val + max_val) * 0.5], 'color': "lightgray"},
                    {'range': [(min_val + max_val) * 0.5, max_val], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': max_val * 0.9
                }
            }
        ))
        
        fig.update_layout(height=300)
        
        return fig
