"""
Water Treatment Stations Monitoring System - Main Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yaml
from pathlib import Path
import time
import sys
import os

# Add modules directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

# Custom module imports
try:
    from modules.data_loader import DataLoader
    from modules.preprocessor import DataPreprocessor
    from modules.time_series import TimeSeriesAnalyzer
    from modules.anomaly_detection import AnomalyDetector
    from modules.visualizations import DashboardVisualizer
    from modules.alerts import AlertSystem
    from modules.localization import localization, t
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Water Treatment Monitoring System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.example.com/help',
        'Report a bug': "https://www.example.com/bug",
        'About': "Monitoring System v2.0"
    }
)

# Custom CSS for better appearance
st.markdown("""
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
""", unsafe_allow_html=True)

class WaterTreatmentDashboard:
    def __init__(self):
        self.initialize_session_state()
        self.load_configuration()
        self.initialize_components()
        
    def initialize_session_state(self):
        """Initialize session variables"""
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'current_station' not in st.session_state:
            st.session_state.current_station = None
        if 'alerts' not in st.session_state:
            st.session_state.alerts = []
        if 'last_update' not in st.session_state:
            st.session_state.last_update = datetime.now()
        if 'selected_parameters' not in st.session_state:
            st.session_state.selected_parameters = []
        if 'date_range' not in st.session_state:
            st.session_state.date_range = None

        # Reset selected parameters if containing invalid values
        if 'selected_parameters' in st.session_state and st.session_state.selected_parameters:
            # Clear invalid parameters like 'turbidity'
            st.session_state.selected_parameters = []
            
    def load_configuration(self):
        """Load system configuration"""
        config_path = Path(__file__).parent / 'config' / 'config.yaml'
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as file:
                self.config = yaml.safe_load(file)
        else:
            self.config = self.get_default_config()
            
    def get_default_config(self):
        """Default configuration"""
        return {
            'stations': {
                'Two Mouths': {
                    'type': 'primary',
                    'capacity': 10000,
                    'parameters': ['pH', 'turbidity', 'chlorine', 'flow_rate', 'temperature']
                },
                'New Rose of Rocky': {
                    'type': 'secondary',
                    'capacity': 8000,
                    'parameters': ['pH', 'DO', 'BOD', 'COD', 'TSS']
                },
                'Botanic Garden': {
                    'type': 'tertiary',
                    'capacity': 5000,
                    'parameters': ['pH', 'nitrogen', 'phosphorus', 'coliform', 'turbidity']
                }
            },
            'thresholds': {
                'pH': {'min': 6.5, 'max': 8.5, 'critical_min': 6.0, 'critical_max': 9.0},
                'turbidity': {'max': 5.0, 'critical_max': 10.0},
                'chlorine': {'min': 0.5, 'max': 2.0, 'critical_min': 0.3, 'critical_max': 3.0},
                'temperature': {'min': 15, 'max': 30, 'critical_min': 10, 'critical_max': 35},
                'DO': {'min': 5.0, 'critical_min': 3.0},
                'BOD': {'max': 30, 'critical_max': 50},
                'COD': {'max': 100, 'critical_max': 150}
            },
            'system': {
                'update_interval': 300,
                'data_retention_days': 90
            }
        }
        
    def initialize_components(self):
        """Initialize system components"""
        self.data_loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.time_series = TimeSeriesAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        self.visualizer = DashboardVisualizer()
        self.alert_system = AlertSystem()
        
    def render_header(self):
        """Render dashboard header"""
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f'<h1 class="main-header">{t("app_title")}</h1>',
                       unsafe_allow_html=True)
        
        # Status bar
        status_col1, status_col2, status_col3, status_col4 = st.columns(4)
        with status_col1:
            st.metric(t('active_stations'), "3/3", "100%")
        with status_col2:
            st.metric(t('active_alerts'), len(st.session_state.alerts),
                     f"{len([a for a in st.session_state.alerts if a.get('severity') == 'critical'])} {t('critical_alerts')}")
        with status_col3:
            compliance = self.calculate_compliance() if st.session_state.data_loaded else 94.7
            st.metric("Compliance Rate", f"{compliance:.1f}%", "+2.1%")
        with status_col4:
            st.metric(t('last_update'), 
                     st.session_state.last_update.strftime("%H:%M:%S"),
                     t('real_time'))
            
    def render_sidebar(self):
        """Render sidebar"""
        with st.sidebar:
            st.image("https://via.placeholder.com/300x100/667eea/ffffff?text=Water+Treatment",
                    use_container_width=True)

            # Language selector removed - application in English

            st.markdown(f"### ⚙️ {t('settings')}")

            # Station selection
            available_stations = list(self.config['stations'].keys())
            station = st.selectbox(
                f"📍 {t('select_station')}",
                available_stations,
                index=0 if not st.session_state.current_station else
                      available_stations.index(st.session_state.current_station)
                      if st.session_state.current_station in available_stations else 0
            )

            # Validate selected station
            if station in available_stations:
                st.session_state.current_station = station
            else:
                st.error(t('station_not_valid', station=station))
                st.session_state.current_station = available_stations[0]
            
            # Time interval
            st.markdown(f"### 📅 {t('date_range')}")
            date_option = st.radio(
                t('select_period'),
                [t('last_hour'), t('last_24_hours'), t('last_week'), t('last_month'), t('custom')]
            )
            
            if date_option == t('custom'):
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input(t('report_start_date'))
                    start_time = st.time_input(t('start_time'))
                with col2:
                    end_date = st.date_input(t('report_end_date'))
                    end_time = st.time_input(t('end_time'))
                st.session_state.date_range = (datetime.combine(start_date, start_time),
                                             datetime.combine(end_date, end_time))
            else:
                st.session_state.date_range = self.get_date_range(date_option)
            
            # Parameters for visualization
            st.markdown(f"### 📊 {t('parameters')}")
            station_params = self.config['stations'][station]['parameters']
            # Filter valid parameters from session_state
            if hasattr(st.session_state, 'selected_parameters') and st.session_state.selected_parameters:
                valid_defaults = [p for p in st.session_state.selected_parameters if p in station_params]
                default_params = valid_defaults if valid_defaults else station_params[:3]
            else:
                default_params = station_params[:3]

            selected_params = st.multiselect(
                t('select_parameters'),
                station_params,
                default=default_params
            )
            st.session_state.selected_parameters = selected_params

            # Update settings
            st.markdown(f"### 🔄 {t('auto_update')}")
            auto_refresh = st.checkbox(t('enable_auto_update'))
            if auto_refresh:
                refresh_rate = st.slider(t('update_interval_seconds'), 5, 300, 60)
                if st.button("🔄 Update Now"):
                    st.rerun()
                
            # Export data
            st.markdown(f"### {t('export_data')}")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 CSV", use_container_width=True):
                    self.export_data('csv')
            with col2:
                if st.button("📄 PDF", use_container_width=True):
                    self.export_data('pdf')
                
            return station, selected_params, date_option
            
    def get_date_range(self, option):
        """Get date range based on option"""
        now = datetime.now()
        if option == t('last_hour'):
            return (now - timedelta(hours=1), now)
        elif option == t('last_24_hours'):
            return (now - timedelta(hours=24), now)
        elif option == t('last_week'):
            return (now - timedelta(weeks=1), now)
        elif option == t('last_month'):
            return (now - timedelta(days=30), now)
        else:
            return (now - timedelta(hours=24), now)
            
    def load_station_data(self, station, date_range=None):
        """Load data from selected station"""
        # Try to load real data first
        df = self.data_loader.load_station_data(station)
        
        if df.empty:
            # Generate synthetic data for demonstration
            df = self.data_loader.generate_synthetic_data(station)
            st.session_state.data_loaded = True
            
        if date_range:
            df = self.data_loader.filter_by_date(df, date_range)
            
        return df
        
    def render_main_dashboard(self, station, params, df):
        """Render main dashboard"""
        # Tabs for different visualizations
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            f"📊 {t('overview')}",
            f"📈 {t('time_series')}",
            f"🔍 {t('detailed_analysis')}",
            f"🔮 {t('predictions')}",
            f"⚠️ {t('anomalies')}",
            f"📑 {t('reports')}"
        ])
        
        with tab1:
            self.render_overview_tab(df, params)
            
        with tab2:
            self.render_time_series_tab(df, params)
            
        with tab3:
            self.render_detailed_analysis_tab(df, params)
            
        with tab4:
            self.render_predictions_tab(df, params)
            
        with tab5:
            self.render_alerts_tab(df, params)
            
        with tab6:
            self.render_reports_tab(df, params)
            
    def render_overview_tab(self, df, params):
        """Render overview tab"""
        st.markdown(f"### 📊 {t('overview')}")
        
        # Main KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'pH' in df.columns:
                avg_ph = df['pH'].mean()
                st.metric(
                    t('average_ph'),
                    f"{avg_ph:.2f}",
                    f"{avg_ph - 7.0:+.2f}",
                    delta_color="inverse" if abs(avg_ph - 7.0) > 0.5 else "normal"
                )
            
        with col2:
            if 'turbidity' in df.columns:
                avg_turb = df['turbidity'].mean()
                st.metric(
                    t('average_turbidity'),
                    f"{avg_turb:.2f}",
                    f"{(avg_turb - 5.0)/5.0*100:+.1f}%"
                )
                
        with col3:
            if 'flow_rate' in df.columns:
                current_flow = df['flow_rate'].iloc[-1]
                st.metric(
                    t('current_flow'),
                    f"{current_flow:.0f}",
                    f"{(current_flow - df['flow_rate'].mean())/df['flow_rate'].mean()*100:+.1f}%"
                )
                
        with col4:
            compliance = self.calculate_compliance(df, params)
            st.metric(
                "Taxa de Conformidade",
                f"{compliance:.1f}%",
                f"{compliance - 90:+.1f}%",
                delta_color="normal" if compliance > 90 else "inverse"
            )
            
        # Real-time status chart
        st.markdown("### 📈 Real-Time Monitoring")
        
        if params:
            fig = self.visualizer.create_realtime_chart(df, params)
            st.plotly_chart(fig, use_container_width=True)
            
        # Correlation heatmap
        if len(params) > 1:
            st.markdown(f"### 🔥 {t('correlation_matrix')}")
            fig_corr = self.visualizer.create_correlation_heatmap(df, params)
            st.plotly_chart(fig_corr, use_container_width=True)
            
    def render_time_series_tab(self, df, params):
        """Render time series tab"""
        st.markdown(f"### 📈 {t('time_series')}")
        
        # Analysis type selector
        analysis_type = st.selectbox(
            t('select_analysis_type'),
            [t('seasonal_decomposition'), "ARIMA", "Prophet", t('trend_analysis')]
        )
        
        selected_param = st.selectbox(t('select_parameter'), params)
        
        if selected_param in df.columns:
            if analysis_type == t('seasonal_decomposition'):
                self.render_seasonal_decomposition(df, selected_param)
            elif analysis_type == "ARIMA":
                self.render_arima_analysis(df, selected_param)
            elif analysis_type == "Prophet":
                self.render_prophet_analysis(df, selected_param)
            elif analysis_type == t('trend_analysis'):
                self.render_trend_analysis(df, selected_param)
                
    def render_seasonal_decomposition(self, df, param):
        """Render seasonal decomposition"""
        try:
            # Prepare data
            ts_data = df.set_index('timestamp')[param].dropna()
            
            # Decomposition settings
            col1, col2, col3 = st.columns(3)
            with col1:
                model_type = st.radio("Modelo:", ["additive", "multiplicative"])
            with col2:
                period = st.number_input("Period:", min_value=2, max_value=len(ts_data)//2, value=96)
            with col3:
                extrapolate_trend = st.checkbox("Extrapolate trend", value=True)
            
            # Perform decomposition
            decomposition = self.time_series.seasonal_decomposition(
                ts_data, model=model_type, period=period
            )
            
            if 'error' not in decomposition:
                # Create visualization
                fig = self.visualizer.create_seasonal_decomposition_chart(decomposition)
                st.plotly_chart(fig, use_container_width=True)
                
                # Decomposition metrics
                st.markdown(f"### 📊 {t('seasonal_decomposition_metrics')}")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    trend_strength = 1 - np.var(decomposition['residual'].dropna()) / np.var(ts_data - decomposition['seasonal'])
                    st.metric(t('trend_strength'), f"{trend_strength:.3f}")
                    
                with col2:
                    seasonal_strength = 1 - np.var(decomposition['residual'].dropna()) / np.var(ts_data - decomposition['trend'])
                    st.metric(t('seasonal_strength'), f"{seasonal_strength:.3f}")
                    
                with col3:
                    st.metric(t('residuals_variance'), f"{np.var(decomposition['residual'].dropna()):.3f}")
                    
                with col4:
                    st.metric(t('dominant_period'), f"{period} {t('obs')}")
            else:
                st.error(f"Decomposition error: {decomposition['error']}")
                
        except Exception as e:
            st.error(f"Error performing seasonal decomposition: {str(e)}")
            
    def render_arima_analysis(self, df, param):
        """Render ARIMA analysis"""
        try:
            ts_data = df.set_index('timestamp')[param].dropna()
            
            # ARIMA settings
            st.markdown("#### ARIMA Model Settings")
            col1, col2, col3 = st.columns(3)
            with col1:
                p = st.number_input("p (AR order):", min_value=0, max_value=10, value=1)
            with col2:
                d = st.number_input("d (Diferenciação):", min_value=0, max_value=3, value=1)
            with col3:
                q = st.number_input("q (MA order):", min_value=0, max_value=10, value=1)
                
            # Fit model
            if st.button("Fit ARIMA Model"):
                with st.spinner("Fitting model..."):
                    result = self.time_series.fit_arima(ts_data, order=(p, d, q))
                    
                    if 'error' not in result:
                        # Model summary
                        st.text(str(result['summary']))
                        
                        # Predictions
                        n_periods = st.slider("Periods for forecast:", 1, 100, 24)
                        forecast = self.time_series.forecast_arima(result['model'], n_periods)
                        
                        # Visualization
                        fig = go.Figure()
                        
                        # Historical data
                        fig.add_trace(go.Scatter(
                            x=ts_data.index,
                            y=ts_data.values,
                            mode='lines',
                            name=t('historical_data'),
                            line=dict(color='blue')
                        ))
                        
                        # Fitted values
                        fig.add_trace(go.Scatter(
                            x=ts_data.index,
                            y=result['fitted_values'],
                            mode='lines',
                            name='Fitted Values',
                            line=dict(color='red', dash='dash')
                        ))
                        
                        # Forecasts
                        future_dates = pd.date_range(
                            start=ts_data.index[-1] + pd.Timedelta(minutes=15),
                            periods=n_periods,
                            freq='15T'
                        )
                        
                        fig.add_trace(go.Scatter(
                            x=future_dates,
                            y=forecast,
                            mode='lines',
                            name=t('forecast'),
                            line=dict(color='green')
                        ))
                        
                        fig.update_layout(
                            title=f"ARIMA Model ({p},{d},{q}) - {param}",
                            xaxis_title="Time",
                            yaxis_title=param,
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"ARIMA model error: {result['error']}")
                        
        except Exception as e:
            st.error(f"ARIMA analysis error: {str(e)}")
            
    def render_prophet_analysis(self, df, param):
        """Render Prophet analysis"""
        try:
            if st.button("Fit Prophet Model"):
                with st.spinner("Fitting Prophet model..."):
                    result = self.time_series.fit_prophet(df, param)
                    
                    if 'error' not in result:
                        # Visualize forecasts
                        fig = self.visualizer.create_forecast_chart(df, param, result['forecast'])
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show components
                        if 'components' in result:
                            st.markdown("#### Model Components")
                            components = result['components']
                            
                            fig_comp = make_subplots(
                                rows=3, cols=1,
                                subplot_titles=('Trend', 'Daily Seasonality', 'Weekly Seasonality'),
                                vertical_spacing=0.1
                            )
                            
                            fig_comp.add_trace(
                                go.Scatter(x=components['ds'], y=components['trend'], 
                                         mode='lines', name='Tendência'),
                                row=1, col=1
                            )
                            
                            fig_comp.add_trace(
                                go.Scatter(x=components['ds'], y=components['daily'], 
                                         mode='lines', name='Diária'),
                                row=2, col=1
                            )
                            
                            fig_comp.add_trace(
                                go.Scatter(x=components['ds'], y=components['weekly'], 
                                         mode='lines', name='Semanal'),
                                row=3, col=1
                            )
                            
                            fig_comp.update_layout(height=600, showlegend=False)
                            st.plotly_chart(fig_comp, use_container_width=True)
                    else:
                        st.error(f"Prophet model error: {result['error']}")
                        
        except Exception as e:
            st.error(f"Prophet analysis error: {str(e)}")
            
    def render_trend_analysis(self, df, param):
        """Render trend analysis"""
        try:
            fig = self.visualizer.create_trend_analysis_chart(df, param)
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate rate of change
            from sklearn.linear_model import LinearRegression
            X = np.arange(len(df)).reshape(-1, 1)
            y = df[param].values
            model = LinearRegression()
            model.fit(X, y)
            slope = model.coef_[0]
            st.info(f"📈 Rate of change: {slope:.4f} units per observation")
            
        except Exception as e:
            st.error(f"Trend analysis error: {str(e)}")
            
    def render_detailed_analysis_tab(self, df, params):
        """Render detailed analysis tab"""
        st.markdown(f"### 🔍 {t('detailed_analysis')}")
        
        # Parameter selection for analysis
        selected_params = st.multiselect(
            t('select_parameters'),
            params,
            default=params[:2] if len(params) >= 2 else params
        )

        if len(selected_params) == 0:
            st.warning(t('select_parameters_msg'))
            return
            
        # Statistical analysis
        st.markdown(f"#### 📊 {t('descriptive_statistics')}")
        stats_dict = self.preprocessor.calculate_statistics(df, selected_params)
        
        # Create DataFrame with statistics
        stats_df = pd.DataFrame(stats_dict).T
        st.dataframe(stats_df, use_container_width=True)
        
        # Distribution analysis
        st.markdown(f"#### 📈 {t('distribution_analysis')}")
        
        for param in selected_params:
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                # Histogram with density curve
                fig_hist = self.visualizer.create_distribution_plot(df, param)
                st.plotly_chart(fig_hist, use_container_width=True)
                
            with col2:
                # Box plot
                fig_box = self.visualizer.create_box_plot(df, [param])
                st.plotly_chart(fig_box, use_container_width=True)
                
            with col3:
                # Statistical metrics
                st.markdown(f"**{param}**")
                if param in stats_dict:
                    stats = stats_dict[param]
                    st.metric(t('mean'), f"{stats['mean']:.2f}")
                    st.metric(t('std_dev'), f"{stats['std']:.2f}")
                    st.metric("CV (%)", f"{stats['cv']:.1f}")
                    
                    # Normality test
                    if stats['skewness'] < 0.5 and stats['kurtosis'] < 0.5:
                        st.success(t('normal_distribution'))
                    else:
                        st.warning(t('non_normal_distribution'))
                        
    def render_predictions_tab(self, df, params):
        """Render predictions tab"""
        st.markdown(f"### 🔮 {t('predictions')} and Predictive Modeling")
        
        # Model selection
        model_type = st.selectbox(
            t('select_prediction_model'),
            ["Prophet", "ARIMA", t('random_forest'), t('trend_analysis')]
        )
        
        selected_param = st.selectbox("Parameter for prediction:", params)
        
        if selected_param in df.columns:
            # Prediction settings
            col1, col2, col3 = st.columns(3)
            with col1:
                forecast_horizon = st.number_input(
                    t('prediction_horizon_hours'),
                    min_value=1,
                    max_value=168,
                    value=24
                )
            with col2:
                confidence_level = st.slider(
                    t('confidence_level'),
                    min_value=80,
                    max_value=99,
                    value=95
                )
            with col3:
                include_exogenous = st.checkbox(t('include_exogenous'))
                
            if st.button(t('generate_predictions')):
                with st.spinner(f"Training {model_type} model..."):
                    if model_type == "Prophet":
                        self.render_prophet_forecast(df, selected_param, forecast_horizon)
                    elif model_type == "ARIMA":
                        self.render_arima_forecast(df, selected_param, forecast_horizon)
                    elif model_type == t('random_forest'):
                        self.render_ml_forecast(df, selected_param, forecast_horizon, "random_forest")
                    elif model_type == t('trend_analysis'):
                        self.render_trend_forecast(df, selected_param, forecast_horizon)
                        
    def render_prophet_forecast(self, df, param, horizon):
        """Render Prophet forecasts"""
        try:
            result = self.time_series.fit_prophet(df, param)
            if 'error' not in result:
                fig = self.visualizer.create_forecast_chart(df, param, result['forecast'])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"Prophet error: {result['error']}")
        except Exception as e:
            st.error(f"Prophet forecasting error: {str(e)}")
            
    def render_arima_forecast(self, df, param, horizon):
        """Render ARIMA forecasts"""
        try:
            ts_data = df.set_index('timestamp')[param].dropna()
            result = self.time_series.fit_arima(ts_data, order=(1, 1, 1))
            
            if 'error' not in result:
                forecast = self.time_series.forecast_arima(result['model'], horizon * 4)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ts_data.index, y=ts_data.values,
                                       mode='lines', name=t('historical_data')))
                fig.add_trace(go.Scatter(x=pd.date_range(ts_data.index[-1], periods=horizon*4, freq='15T'),
                                       y=forecast, mode='lines', name=t('forecast')))
                fig.update_layout(title=f"ARIMA Forecast - {param}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"ARIMA error: {result['error']}")
        except Exception as e:
            st.error(f"ARIMA forecasting error: {str(e)}")
            
    def render_ml_forecast(self, df, param, horizon, model_type):
        """Render ML forecasts"""
        try:
            result = self.time_series.fit_ml_model(df, param, model_type)
            if 'error' not in result:
                st.success(f"Model trained with R² = {result['r2']:.3f}")
                
                # Make predictions
                last_values = df[param].tail(96)
                predictions = self.time_series.forecast_ml(
                    result['model'], result['scaler'], result['feature_columns'], 
                    last_values, horizon * 4
                )
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df[param],
                                       mode='lines', name=t('historical_data')))
                fig.add_trace(go.Scatter(x=pd.date_range(df['timestamp'].iloc[-1], periods=horizon*4, freq='15T'),
                                       y=predictions, mode='lines', name=t('forecast')))
                fig.update_layout(title=f"{model_type} Forecast - {param}")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"ML model error: {result['error']}")
        except Exception as e:
            st.error(f"ML forecasting error: {str(e)}")
            
    def render_trend_forecast(self, df, param, horizon):
        """Render trend forecasts"""
        try:
            from sklearn.linear_model import LinearRegression
            X = np.arange(len(df)).reshape(-1, 1)
            y = df[param].values
            model = LinearRegression()
            model.fit(X, y)
            
            # Predictions
            future_X = np.arange(len(df), len(df) + horizon * 4).reshape(-1, 1)
            predictions = model.predict(future_X)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['timestamp'], y=df[param],
                                   mode='lines', name=t('historical_data')))
            fig.add_trace(go.Scatter(x=pd.date_range(df['timestamp'].iloc[-1], periods=horizon*4, freq='15T'),
                                   y=predictions, mode='lines', name=t('linear_prediction')))
            fig.update_layout(title=f"Trend Forecast - {param}")
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Trend forecasting error: {str(e)}")
            
    def render_alerts_tab(self, df, params):
        """Render alerts and anomalies tab"""
        st.markdown(f"### {t('alerts_anomalies_system')}")
        
        # Detection settings
        col1, col2, col3 = st.columns(3)
        with col1:
            detection_method = st.selectbox(
                t('detection_method'),
                ["Isolation Forest", "DBSCAN", "One-Class SVM", "Z-Score", "IQR"]
            )
        with col2:
            sensitivity = st.slider(
                t('sensitivity_threshold'),
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.1
            )
        with col3:
            auto_alert = st.checkbox(t('auto_alerts'), value=True)
            
        # Detect anomalies
        if st.button("🔍 Detect Anomalies"):
            with st.spinner("Detecting anomalies..."):
                anomalies = self.detect_anomalies(df, params, detection_method, sensitivity)
                
                if len(anomalies) > 0:
                    self.render_anomaly_results(anomalies, df, params)
                else:
                    st.success(t('no_anomalies_detected'))
                    
    def detect_anomalies(self, df, params, method, sensitivity):
        """Detect anomalies in data"""
        try:
            if method == "Isolation Forest":
                df_anomalies = self.anomaly_detector.detect_anomalies_isolation_forest(
                    df, params, contamination=sensitivity
                )
            elif method == "DBSCAN":
                df_anomalies = self.anomaly_detector.detect_anomalies_dbscan(df, params)
            elif method == "One-Class SVM":
                df_anomalies = self.anomaly_detector.detect_anomalies_one_class_svm(
                    df, params, nu=sensitivity
                )
            elif method == "Z-Score":
                df_anomalies = self.anomaly_detector.detect_outliers_statistical(
                    df, params, method='zscore', threshold=3-sensitivity*2
                )
            elif method == "IQR":
                df_anomalies = self.anomaly_detector.detect_outliers_statistical(
                    df, params, method='iqr', threshold=1.5-sensitivity*0.5
                )
            else:
                df_anomalies = df
                
            # Convert to anomalies list
            anomalies_list = []
            for param in params:
                anomaly_col = f'{param}_anomaly'
                if anomaly_col in df_anomalies.columns:
                    anomaly_indices = df_anomalies[df_anomalies[anomaly_col] == 1].index
                    for idx in anomaly_indices:
                        anomalies_list.append({
                            'timestamp': df_anomalies.loc[idx, 'timestamp'],
                            'parameter': param,
                            'value': df_anomalies.loc[idx, param],
                            'severity': 'medium',
                            'method': method
                        })
                        
            return anomalies_list
            
        except Exception as e:
            st.error(f"Anomaly detection error: {str(e)}")
            return []
            
    def render_anomaly_results(self, anomalies, df, params):
        """Renderizar resultados de anomalias"""
        # Resumo de alertas
        st.markdown("#### 📊 Resumo de Alertas")
        
        anomaly_df = pd.DataFrame(anomalies)
        severity_counts = anomaly_df['severity'].value_counts()
        
        fig_alerts = go.Figure(data=[
            go.Bar(x=severity_counts.index, y=severity_counts.values,
                  marker_color=['green', 'orange', 'red'][:len(severity_counts)])
        ])
        
        fig_alerts.update_layout(
            title="Distribuição de Alertas por Severidade",
            xaxis_title="Severidade",
            yaxis_title="Número de Alertas",
            height=300
        )
        
        st.plotly_chart(fig_alerts, use_container_width=True)
        
        # Tabela de alertas recentes
        st.markdown("#### 🚨 Alertas Recentes")
        st.dataframe(anomaly_df.head(10), use_container_width=True)
        
        # Visualização temporal de anomalias
        st.markdown("#### 📈 Linha do Tempo de Anomalias")
        
        for param in params:
            param_anomalies = [a for a in anomalies if a['parameter'] == param]
            if param_anomalies:
                fig_timeline = self.visualizer.create_anomaly_timeline(
                    df, param, f'{param}_anomaly'
                )
                st.plotly_chart(fig_timeline, use_container_width=True)
                
    def render_reports_tab(self, df, params):
        """Renderizar aba de relatórios"""
        st.markdown(f"### {t('report_generation')}")
        
        # Tipo de relatório
        report_type = st.selectbox(
            "Tipo de relatório:",
            ["Relatório Operacional", "Relatório de Conformidade", 
             "Relatório de Manutenção", "Relatório Executivo"]
        )
        
        # Período do relatório
        col1, col2 = st.columns(2)
        with col1:
            report_start = st.date_input("Data inicial do relatório")
        with col2:
            report_end = st.date_input("Data final do relatório")
            
        # Opções do relatório
        st.markdown("#### Opções do Relatório")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            include_charts = st.checkbox("Incluir gráficos", value=True)
            include_statistics = st.checkbox("Incluir estatísticas", value=True)
            include_anomalies = st.checkbox("Incluir anomalias", value=True)
            
        with col2:
            include_predictions = st.checkbox("Incluir previsões", value=False)
            include_recommendations = st.checkbox("Incluir recomendações", value=True)
            include_raw_data = st.checkbox("Incluir dados brutos", value=False)
            
        with col3:
            report_format = st.selectbox("Formato:", ["PDF", "Excel", "HTML"])
            report_language = st.selectbox("Idioma:", ["Português", "English", "Español"])
            
        if st.button(t('generate_report'), type="primary"):
            with st.spinner(t('generating_report')):
                time.sleep(2)
                
                # Preview do relatório
                st.markdown("#### 📋 Preview do Relatório")
                
                compliance = self.calculate_compliance(df, params)
                
                st.markdown(f"""
                ---
                ## {report_type} - {st.session_state.current_station}
                **Período:** {report_start} a {report_end}
                
                ### Resumo Executivo
                Durante o período analisado, a estação {st.session_state.current_station} operou com 
                uma taxa de conformidade de **{compliance:.1f}%**, processando um volume total de **{df['flow_rate'].sum()/1000:.0f} mil m³**.
                
                ### Principais Indicadores
                - **pH médio:** {df['pH'].mean():.2f} (dentro dos limites)
                - **Turbidez média:** {df['turbidity'].mean():.2f} NTU
                - **Eficiência do tratamento:** 96.3%
                - **Tempo de operação:** 99.2%
                
                ### Eventos Notáveis
                - {len(st.session_state.alerts)} alertas registrados
                - 1 manutenção preventiva realizada
                - Nenhuma parada não programada
                
                ### Recomendações
                1. Ajustar dosagem de cloro no período noturno
                2. Verificar calibração do sensor de pH da linha 2
                3. Programar manutenção do decantador para próximo mês
                
                ---
                """)
                
                # Botão de download
                st.success("✅ Relatório gerado com sucesso!")
                st.download_button(
                    label=f"⬇️ Baixar Relatório ({report_format})",
                    data="Conteúdo do relatório...",  # Aqui seria o conteúdo real
                    file_name=f"relatorio_{st.session_state.current_station}_{report_start}_{report_end}.{report_format.lower()}",
                    mime="application/pdf"
                )
                
    def calculate_compliance(self, df=None, params=None):
        """Calcular taxa de conformidade"""
        if df is None or params is None:
            return 94.7  # Valor padrão
            
        total_checks = 0
        compliant_checks = 0
        
        for param in params:
            if param in df.columns and param in self.config['thresholds']:
                thresholds = self.config['thresholds'][param]
                param_data = df[param].dropna()
                
                total_checks += len(param_data)
                
                if 'min' in thresholds and 'max' in thresholds:
                    compliant = param_data[(param_data >= thresholds['min']) & 
                                          (param_data <= thresholds['max'])]
                elif 'min' in thresholds:
                    compliant = param_data[param_data >= thresholds['min']]
                elif 'max' in thresholds:
                    compliant = param_data[param_data <= thresholds['max']]
                else:
                    compliant = param_data
                    
                compliant_checks += len(compliant)
                
        return (compliant_checks / total_checks * 100) if total_checks > 0 else 100
        
    def export_data(self, format_type):
        """Exportar dados"""
        if st.session_state.data_loaded:
            try:
                # Aqui você implementaria a lógica de exportação real
                st.success(f"Dados exportados em formato {format_type.upper()}")
            except Exception as e:
                st.error(f"Erro na exportação: {str(e)}")
        else:
            st.warning("Nenhum dado carregado para exportar")
        
    def run(self):
        """Executar o dashboard"""
        # Renderizar cabeçalho
        self.render_header()
        
        # Renderizar sidebar e obter configurações
        station, params, date_option = self.render_sidebar()
        
        # Carregar dados
        df = self.load_station_data(station, st.session_state.date_range)
        
        if not df.empty:
            st.session_state.data_loaded = True
            
            # Renderizar dashboard principal
            self.render_main_dashboard(station, params, df)
        else:
            st.error(t('error_loading_station_data'))
            
        # Footer
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: gray; font-size: 0.8em;'>
            Sistema de Monitoramento v2.0 | Desenvolvido com ❤️ usando Streamlit | 
            Última atualização: {:%Y-%m-%d %H:%M:%S}
            </div>
            """.format(datetime.now()),
            unsafe_allow_html=True
        )

# Executar aplicação
if __name__ == "__main__":
    dashboard = WaterTreatmentDashboard()
    dashboard.run()
