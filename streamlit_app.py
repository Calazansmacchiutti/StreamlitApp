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

selected_dataset = st.selectbox(
    'Escolha o dataset abaixo',
    list(dataset_options.keys())
)

data_file = dataset_options[selected_dataset]
df = load_data(data_file)

# Se o DataFrame estiver vazio, parar a execução
if df.empty:
    st.stop()

# Selecionar a coluna de data/hora se 'timestamp' não estiver presente
if 'timestamp' not in df.columns:
    date_column = st.selectbox(
        "Verifique se esta selecionado DateTime abaixo",
        df.columns
    )
    df['timestamp'] = pd.to_datetime(df[date_column])
else:
    df['timestamp'] = pd.to_datetime(df['timestamp'])

# Seleção das variáveis para o eixo Y
st.markdown("## Selecione as variáveis para graficar")
columns = [col for col in df.columns if col not in ['timestamp', date_column]]
selected_columns = st.multiselect(
    'Quais colunas você gostaria de selecionar?',
    columns
)

# Se nenhuma coluna for selecionada, parar a execução
if not selected_columns:
    st.warning("Por favor, escolha ao menos uma coluna.")
    st.stop()

# Seleção do intervalo de tempo
st.markdown("## Selecione o intervalo de tempo")
min_date = df['timestamp'].min().to_pydatetime()
max_date = df['timestamp'].max().to_pydatetime()

date_range = st.slider(
    'Selecione o intervalo de tempo',
    min_value=min_date,
    max_value=max_date,
    value=[min_date, max_date],
    format="MM/DD/YYYY"
)

# Filtrar os dados com base no intervalo de tempo selecionado
filtered_df = df[(df['timestamp'] >= date_range[0]) & (df['timestamp'] <= date_range[1])]

# Seleção do gráfico para exibição
st.markdown("## Escolha o gráfico para exibição")
selected_plot = st.selectbox(
    'Escolha o gráfico que deseja exibir',
    ['Linha X vs Y', 'Matriz de Correlação', 'Box Plot', 'Histograma', 'Covariância ao longo do tempo', 'ARIMA Decomposition']
)

# Função para plotar gráfico de linha
def plot_line_chart():
    fig = px.line()

    for column in selected_columns:
        fig.add_scatter(x=filtered_df['timestamp'], y=filtered_df[column], mode='lines', name=column)

    # Adicionar eixos Y adicionais se houver mais de 6 variáveis selecionadas
    if len(selected_columns) > 6:
        for i, column in enumerate(selected_columns[6:], start=1):
            fig.update_traces(yaxis=f"y{i+1}", selector=dict(name=column))
            fig.add_scatter(x=filtered_df['timestamp'], y=filtered_df[column], mode='lines', name=column, yaxis=f"y{i+1}")

        # Atualizar layout para eixos Y adicionais
        fig.update_layout(
            yaxis2=dict(title=selected_columns[3], overlaying='y', side='right'),
            yaxis3=dict(title=selected_columns[4], overlaying='y', side='right', anchor='free', position=0.95),
            yaxis4=dict(title=selected_columns[5], overlaying='y', side='right', anchor='free', position=0.90),
            yaxis5=dict(title=selected_columns[6], overlaying='y', side='right', anchor='free', position=0.85),
        )

    st.plotly_chart(fig)

# Função para plotar matriz de correlação
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
        st.warning("Selecione mais de uma coluna para visualizar a matriz de correlação.")

# Função para plotar box plot
def plot_box_plot():
    if len(selected_columns) > 0:
        # Amostrar os dados se o conjunto for muito grande
        sample_size = min(100000, len(filtered_df))
        sampled_df = filtered_df.sample(n=sample_size, random_state=42)
        
        melted_df = sampled_df.melt(id_vars=['timestamp'], value_vars=selected_columns)
        fig_box = px.box(melted_df, x='variable', y='value', points=False)  # Removendo pontos individuais para otimização
        st.plotly_chart(fig_box)
    else:
        st.warning("Selecione pelo menos uma coluna para visualizar o box plot.")

# Função para plotar histograma
def plot_histogram():
    if len(selected_columns) > 0:
        for column in selected_columns:
            fig_hist = px.histogram(filtered_df, x=column, nbins=30, title=f'Histograma de {column}')
            st.plotly_chart(fig_hist)

            # Cálculo dos intervalos de confiança
            data = filtered_df[column].dropna()
            mean = np.mean(data)
            std_dev = np.std(data)
            conf_interval = stats.norm.interval(0.95, loc=mean, scale=std_dev/np.sqrt(len(data)))

            st.write(f"**{column}**: Intervalo de confiança de 95%: {conf_interval}")
    else:
        st.warning("Selecione pelo menos uma coluna para visualizar o histograma.")

# Função para plotar covariância ao longo do tempo
def plot_time_covariance():
    if len(selected_columns) < 2:
        st.warning("Selecione pelo menos duas colunas para visualizar a covariância.")
        return

    # Seleção do tamanho da janela para a covariância móvel
    window_size = st.slider("Selecione o tamanho da janela para a covariância móvel:", min_value=2, max_value=50, value=10)

    # Calculando a covariância móvel
    cov_df = pd.DataFrame()
    for i in range(len(selected_columns)):
        for j in range(i + 1, len(selected_columns)):
            cov_col_name = f"{selected_columns[i]}-{selected_columns[j]}"
            cov_df[cov_col_name] = filtered_df[selected_columns[i]].rolling(window=window_size).cov(filtered_df[selected_columns[j]])

    cov_df['timestamp'] = filtered_df['timestamp']

    # Plotar covariância ao longo do tempo
    fig_cov = px.scatter(cov_df, x='timestamp', y=cov_df.columns[:-1], title="Covariância ao longo do tempo")

    st.plotly_chart(fig_cov)

# Função para decompor a série temporal usando ARIMA
def plot_arima_decomposition():
    if len(selected_columns) != 1:
        st.warning("Selecione exatamente uma coluna para decompor a série temporal.")
        return

    column = selected_columns[0]
    ts_data = filtered_df.set_index('timestamp')[column].dropna()

    # Pedir ao usuário para inserir o período e ordens p, d, q do modelo ARIMA
    period = st.number_input('Insira o período da série temporal:', min_value=1, max_value=len(ts_data), value=max(12, min(len(ts_data) // 2, 365)))
    p = st.number_input('Ordem do componente autoregressivo (p):', min_value=0, max_value=10, value=1)
    d = st.number_input('Ordem de diferenciação (d):', min_value=0, max_value=5, value=1)
    q = st.number_input('Ordem da média móvel (q):', min_value=0, max_value=10, value=1)

    # Ajustar o modelo ARIMA
    model = ARIMA(ts_data, order=(p, d, q))
    fitted_model = model.fit()

    # Decompor a série temporal
    decomposition = seasonal_decompose(ts_data, model='additive', period=period)

    # Criação de subplots para cada componente da decomposição
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                        subplot_titles=("Original", "Trend", "Seasonal", "Residual"))

    fig.add_trace(go.Scatter(x=ts_data.index, y=ts_data, mode='lines', name='Original'), row=1, col=1)
    fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, mode='lines', name='Trend'), row=2, col=1)
    fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, mode='lines', name='Seasonal'), row=3, col=1)
    fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, mode='lines', name='Residual'), row=4, col=1)

    fig.update_layout(height=800, title_text="ARIMA Decomposition")

    st.plotly_chart(fig)

# Exibir o gráfico selecionado
if selected_plot == 'Linha X vs Y':
    plot_line_chart()
elif selected_plot == 'Matriz de Correlação':
    plot_correlation_matrix()
elif selected_plot == 'Box Plot':
    plot_box_plot()
elif selected_plot == 'Histograma':
    plot_histogram()
elif selected_plot == 'Covariância ao longo do tempo':
    plot_time_covariance()
elif selected_plot == 'ARIMA Decomposition':
    plot_arima_decomposition()   
