import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

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
    'Quais colunas você gostaria de selcionar?',
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

# Plotar o gráfico de linha com plotly express
st.markdown("## Data Plot")
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

# Exibir a matriz de correlação
if len(selected_columns) > 1:
    st.markdown("## Matriz de correlação")
    correlation_matrix = filtered_df[selected_columns].corr()
    fig_corr = px.imshow(correlation_matrix,
                         text_auto=True,
                         aspect="auto",
                         color_continuous_scale='RdBu_r',
                         zmin=-1, zmax=1)
    st.plotly_chart(fig_corr)
else:
    st.warning("Selecione mais de uma coluna para visualizar a matriz de correlação.")

# Exibir o box plot
if len(selected_columns) > 0:
    st.markdown("## Box Plot")
    melted_df = filtered_df.melt(id_vars=['timestamp'], value_vars=selected_columns)
    fig_box = px.box(melted_df, x='variable', y='value', points="all")
    st.plotly_chart(fig_box)
else:
    st.warning("Selecione pelo menos uma coluna para visualizar o box plot.")
