import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import numpy as np
import scipy.stats as stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

# Configurar o título e o ícone da página
st.set_page_config(
    page_title='Data Dashboard',
    page_icon=':earth_americas:',  # Pode ser um URL também.
)

# Função para carregar e processar os dados
@st.cache_data
def load_data(filename):
    """Carregar e processar dados do arquivo CSV."""
    DATA_FILENAME = Path(__file__).parent / filename
    raw_df = pd.read_csv(DATA_FILENAME)

    # Exibir colunas para depuração
    st.write(f"Columns in the dataset {filename}:", raw_df.columns.tolist())
    
    return raw_df

# Seleção do dataset
dataset_options = {
    'Two Mouths': 'data/Two Mouths.csv',
    'New Rose of Rocky': 'data/New Rose of Rocky.csv',
    'Botanic Garden': 'data/Botanic Garden.csv'
}

# List of allowed datasets (security validation)
allowed_datasets = set(dataset_options.keys())

selected_dataset = st.selectbox(
    'Choose the dataset below',
    list(allowed_datasets)
)

# Validate selection
if selected_dataset not in allowed_datasets:
    st.error(f"Dataset '{selected_dataset}' is not allowed")
    st.stop()

data_file = dataset_options[selected_dataset]
df = load_data(data_file)

# If DataFrame is empty, stop execution
if df.empty:
    st.stop()

# Select datetime column if 'timestamp' is not present
date_column = None
if 'timestamp' not in df.columns:
    date_column = st.selectbox(
        "Verify if DateTime is selected below",
        df.columns
    )
    df['timestamp'] = pd.to_datetime(df[date_column])
else:
    df['timestamp'] = pd.to_datetime(df['timestamp'])

# Selection of variables for Y-axis
st.markdown("## Select variables to plot")
excluded_columns = ['timestamp']
if date_column:
    excluded_columns.append(date_column)
columns = [col for col in df.columns if col not in excluded_columns]
selected_columns = st.multiselect(
    'Which columns would you like to select?',
    columns
)

# If no column is selected, stop execution
if not selected_columns:
    st.warning("Please choose at least one column.")
    st.stop()

# Time interval selection
st.markdown("## Select time interval")
min_date = df['timestamp'].min().to_pydatetime()
max_date = df['timestamp'].max().to_pydatetime()

date_range = st.slider(
    'Select time interval',
    min_value=min_date,
    max_value=max_date,
    value=[min_date, max_date],
    format="MM/DD/YYYY"
)

# Filter data based on selected time interval
filtered_df = df[(df['timestamp'] >= date_range[0]) & (df['timestamp'] <= date_range[1])]

# Chart selection for display
st.markdown("## Choose chart for display")
selected_plot = st.selectbox(
    'Choose the chart you want to display',
    ['Line X vs Y', 'Correlation Matrix', 'Box Plot', 'Histogram', 'Covariance over time', 'ARIMA Decomposition']
)

# Function to plot line chart
def plot_line_chart():
    fig = px.line()

    for column in selected_columns:
        fig.add_scatter(x=filtered_df['timestamp'], y=filtered_df[column], mode='lines', name=column)

    # Add additional Y axes if more than 6 variables are selected
    if len(selected_columns) > 6:
        for i, column in enumerate(selected_columns[6:], start=1):
            fig.update_traces(yaxis=f"y{i+1}", selector=dict(name=column))
            fig.add_scatter(x=filtered_df['timestamp'], y=filtered_df[column], mode='lines', name=column, yaxis=f"y{i+1}")

        # Update layout for additional Y axes
        fig.update_layout(
            yaxis2=dict(title=selected_columns[3], overlaying='y', side='right'),
            yaxis3=dict(title=selected_columns[4], overlaying='y', side='right', anchor='free', position=0.95),
            yaxis4=dict(title=selected_columns[5], overlaying='y', side='right', anchor='free', position=0.90),
            yaxis5=dict(title=selected_columns[6], overlaying='y', side='right', anchor='free', position=0.85),
        )

    st.plotly_chart(fig)

# Function to plot correlation matrix
def plot_correlation_matrix():
    if len(selected_columns) > 1:
        correlation_matrix = filtered_df[selected_columns].corr()
        fig_corr = px.imshow(correlation_matrix,
                             text_auto=True,
                             aspect="auto",
                             color_continuous_scale='RdBu_r',
                             zmin=-1, zmax=1)
        st.plotly_chart(fig_corr)
    else:
        st.warning("Select more than one column to view the correlation matrix.")

# Function to plot box plot
def plot_box_plot():
    if len(selected_columns) > 0:
        # Sample data if the dataset is too large
        sample_size = min(100000, len(filtered_df))
        sampled_df = filtered_df.sample(n=sample_size, random_state=42)
        
        melted_df = sampled_df.melt(id_vars=['timestamp'], value_vars=selected_columns)
        fig_box = px.box(melted_df, x='variable', y='value', points=False)  # Removing individual points for optimization
        st.plotly_chart(fig_box)
    else:
        st.warning("Select at least one column to view the box plot.")

# Function to plot histogram
def plot_histogram():
    if len(selected_columns) > 0:
        for column in selected_columns:
            fig_hist = px.histogram(filtered_df, x=column, nbins=30, title=f'Histograma de {column}')
            st.plotly_chart(fig_hist)

            # Calculation of confidence intervals
            data = filtered_df[column].dropna()
            mean = np.mean(data)
            std_dev = np.std(data)
            conf_interval = stats.norm.interval(0.95, loc=mean, scale=std_dev/np.sqrt(len(data)))

            st.write(f"**{column}**: 95% confidence interval: {conf_interval}")
    else:
        st.warning("Select at least one column to view the histogram.")

# Function to plot covariance over time
def plot_time_covariance():
    if len(selected_columns) < 2:
        st.warning("Select at least two columns to view the covariance.")
        return

    # Selection of window size for moving covariance
    window_size = st.slider("Select window size for moving covariance:", min_value=2, max_value=50, value=10)

    # Calculating moving covariance
    cov_df = pd.DataFrame()
    for i in range(len(selected_columns)):
        for j in range(i + 1, len(selected_columns)):
            cov_col_name = f"{selected_columns[i]}-{selected_columns[j]}"
            cov_df[cov_col_name] = filtered_df[selected_columns[i]].rolling(window=window_size).cov(filtered_df[selected_columns[j]])

    cov_df['timestamp'] = filtered_df['timestamp']

    # Plot covariance over time
    fig_cov = px.scatter(cov_df, x='timestamp', y=cov_df.columns[:-1], title="Covariance over time")

    st.plotly_chart(fig_cov)

# Function to decompose time series using ARIMA
def plot_arima_decomposition():
    if len(selected_columns) != 1:
        st.warning("Select exactly one column to decompose the time series.")
        return

    column = selected_columns[0]
    ts_data = filtered_df.set_index('timestamp')[column].dropna()

    # Ask user to enter period and ARIMA model orders p, d, q
    period = st.number_input('Enter the time series period:', min_value=1, max_value=len(ts_data), value=max(12, min(len(ts_data) // 2, 2016)))
    p = st.number_input('Autoregressive component order (p):', min_value=0, max_value=10, value=1)
    d = st.number_input('Differencing order (d):', min_value=0, max_value=5, value=1)
    q = st.number_input('Moving average order (q):', min_value=0, max_value=10, value=1)

    # Fit ARIMA model
    model = ARIMA(ts_data, order=(p, d, q))
    fitted_model = model.fit()

    # Decompose time series
    decomposition = seasonal_decompose(ts_data, model='additive', period=period)

    # Create subplots for each decomposition component
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                        subplot_titles=("Original data", "Trend", "Seasonality", "Residuals"))

    fig.add_trace(go.Scatter(x=ts_data.index, y=ts_data, mode='lines', name='Original data'), row=1, col=1)
    fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, mode='lines', name='Trend'), row=2, col=1)
    fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, mode='lines', name='Seasonality'), row=3, col=1)
    fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, mode='lines', name='Residuals'), row=4, col=1)

    fig.update_layout(height=800, title_text="Time series decomposition")

    st.plotly_chart(fig)

# Display selected chart
if selected_plot == 'Line X vs Y':
    plot_line_chart()
elif selected_plot == 'Correlation Matrix':
    plot_correlation_matrix()
elif selected_plot == 'Box Plot':
    plot_box_plot()
elif selected_plot == 'Histogram':
    plot_histogram()
elif selected_plot == 'Covariance over time':
    plot_time_covariance()
elif selected_plot == 'ARIMA Decomposition':
    plot_arima_decomposition()   
