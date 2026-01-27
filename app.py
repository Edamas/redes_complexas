# ============================================================
# IMPORTS
# ============================================================

import streamlit as st
import random
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# ============================================================
# TÍTULO E CABEÇALHO
# ============================================================

st.title("Simulação de Redes Complexas")
st.write("Aplicação baseada na Aula 1 (26/1/2026) de Introdução às Redes Complexas - EACH-USP")
st.write("Autor: Elysio Damasceno da Silva Neto")

# ============================================================
# PARÂMETROS DA SIMULAÇÃO (EDITE AQUI)
# ============================================================

st.sidebar.header("Parâmetros da Simulação")

NUM_NODES = st.sidebar.slider(
    "Número de Nós",
    min_value=10,
    max_value=2000,
    value=100,
    step=10
)
EDGE_PROBABILITY = st.sidebar.slider(
    "Probabilidade de Conexão",
    min_value=0.0,
    max_value=1.0,
    value=0.1,
    step=0.01
)

st.sidebar.header("Layout / Figura")

NET_HEIGHT = st.sidebar.slider(
    "Altura do Gráfico da Rede (px)",
    min_value=300,
    max_value=1200,
    value=600,
    step=50
)
HIST_HEIGHT = st.sidebar.slider(
    "Altura do Gráfico do Histograma (px)",
    min_value=200,
    max_value=800,
    value=400,
    step=50
)
COLUMN_RATIO = st.sidebar.slider(
    "Proporção da Largura (Rede / Histograma)",
    min_value=10,
    max_value=90,
    value=60,
    step=5
)


st.sidebar.header("Rede 3D")

col1, col2 = st.sidebar.columns(2)
with col1:
    NODE_COLOR = st.color_picker(
        "Cor do Nó",
        value="#1f77b4"
    )
with col2:
    EDGE_COLOR = st.color_picker(
        "Cor da Aresta",
        value="#888888"
    )

NODE_OPACITY = st.sidebar.slider(
    "Opacidade do Nó",
    min_value=0.0,
    max_value=1.0,
    value=0.8,
    step=0.05
)
EDGE_OPACITY = st.sidebar.slider(
    "Opacidade da Aresta",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
)

NODE_SIZE = st.sidebar.slider(
    "Tamanho do Nó",
    min_value=1,
    max_value=20,
    value=8,
    step=1
)
EDGE_WIDTH = st.sidebar.slider(
    "Largura da Aresta",
    min_value=1,
    max_value=10,
    value=2,
    step=1
)
FONT_SIZE = st.sidebar.slider(
    "Tamanho da Fonte Principal",
    min_value=8,
    max_value=24,
    value=14,
    step=1
)

NODE_FONT_SIZE = st.sidebar.slider(
    "Tamanho da Fonte dos Nomes dos Nós",
    min_value=5,
    max_value=20,
    value=10,
    step=1
)

SHOW_NODE_NAMES = st.sidebar.checkbox(
    "Exibir Nomes dos Nós",
    value=True
)

NODE_PREFIX = st.sidebar.text_input(
    "Prefixo para Nomes dos Nós",
    value="Nó "
)

AUTONUMBER_NODES = st.sidebar.checkbox(
    "Adicionar Autonumeração aos Nós",
    value=True
)

st.sidebar.header("Histograma")

HIST_BINS = st.sidebar.slider(
    "Número de Bins do Histograma",
    min_value=5,
    max_value=50,
    value=10,
    step=1
)
HIST_COLOR = st.sidebar.color_picker(
    "Cor do Histograma",
    value="#ff7f0e"
)

# ============================================================
# GERAÇÃO DA REDE
# ============================================================

@st.cache_data
def generate_network(num_nodes, edge_probability):
    nodes = list(range(num_nodes))
    edges = [
        (i, j)
        for i in nodes
        for j in nodes
        if i < j and random.random() < edge_probability
    ]

    # Garantir que cada nó tenha ao menos uma conexão
    connected_nodes = set(n for e in edges for n in e)
    for i in nodes:
        if i not in connected_nodes:
            j = random.choice([n for n in nodes if n != i])
            edges.append((i, j))

    return nodes, edges

nodes, edges = generate_network(NUM_NODES, EDGE_PROBABILITY)

# ============================================================
# POSIÇÕES 3D
# ============================================================

pos = np.random.randn(NUM_NODES, 3)

# ============================================================
# GRAUS DOS NÓS
# ============================================================

degrees = [0] * NUM_NODES
for i, j in edges:
    degrees[i] += 1
    degrees[j] += 1

# ============================================================
# TRACES DA REDE
# ============================================================

edge_x, edge_y, edge_z = [], [], []
for i, j in edges:
    edge_x += [pos[i, 0], pos[j, 0], None]
    edge_y += [pos[i, 1], pos[j, 1], None]
    edge_z += [pos[i, 2], pos[j, 2], None]

edge_trace = go.Scatter3d(
    x=edge_x,
    y=edge_y,
    z=edge_z,
    mode="lines",
    opacity=EDGE_OPACITY,
    line=dict(color=EDGE_COLOR, width=EDGE_WIDTH),
    hoverinfo="none"
)

node_mode = "markers+text" if SHOW_NODE_NAMES else "markers"
if SHOW_NODE_NAMES:
    if AUTONUMBER_NODES:
        node_text = [f"{NODE_PREFIX}{i}" for i in nodes]
    else:
        node_text = [NODE_PREFIX] * NUM_NODES
else:
    node_text = None


node_trace = go.Scatter3d(
    x=pos[:, 0],
    y=pos[:, 1],
    z=pos[:, 2],
    mode=node_mode,
    opacity=NODE_OPACITY,
    marker=dict(size=NODE_SIZE, color=NODE_COLOR),
    text=node_text,
    textposition="top center",
    textfont=dict(size=NODE_FONT_SIZE)
)

# ============================================================
# HISTOGRAMA
# ============================================================

hist_trace = go.Histogram(
    x=degrees,
    nbinsx=HIST_BINS,
    marker=dict(color=HIST_COLOR)
)

# ============================================================
# FIGURA DA REDE
# ============================================================
fig_net = go.Figure(data=[edge_trace, node_trace])

fig_net.update_layout(
    title="Rede Complexa 3D",
    height=NET_HEIGHT,
    margin=dict(l=10, r=10, t=40, b=10),
    showlegend=False,
    scene=dict(
        aspectmode="cube",
        xaxis_visible=False,
        yaxis_visible=False,
        zaxis_visible=False
    )
)

# ============================================================
# FIGURA DO HISTOGRAMA
# ============================================================
fig_hist = go.Figure(data=[hist_trace])

fig_hist.update_layout(
    title="Distribuição de Graus",
    height=HIST_HEIGHT,
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis_title="Grau",
    yaxis_title="Frequência",
    font=dict(size=FONT_SIZE)
)


# ============================================================
# EXIBIÇÃO DOS GRÁFICOS
# ============================================================
col_net, col_hist = st.columns([COLUMN_RATIO, 100 - COLUMN_RATIO])

with col_net:
    st.plotly_chart(fig_net, width='stretch')

with col_hist:
    st.plotly_chart(fig_hist, width='stretch')


# ============================================================
# DATAFRAME COM DADOS DOS NÓS
# ============================================================

st.subheader("Dados Detalhados dos Nós")

# Generate node display names for the DataFrame
if AUTONUMBER_NODES:
    node_display_names_for_df = [f"{NODE_PREFIX}{i}" for i in nodes]
else:
    node_display_names_for_df = [NODE_PREFIX] * NUM_NODES

node_data = {
    "Nó ID": nodes,
    "Nome do Nó": node_display_names_for_df,
    "Grau": degrees,
    "Posição X": pos[:, 0],
    "Posição Y": pos[:, 1],
    "Posição Z": pos[:, 2],
}
df = pd.DataFrame(node_data)
df = df.set_index("Nó ID")
df.index.name = "ID do Nó" # Set index name

st.dataframe(df)

# ============================================================
# ESTATÍSTICAS DA REDE
# ============================================================

st.subheader("Estatísticas da Distribuição de Graus")

if degrees:
    desc_stats = stats.describe(degrees)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Contagem de Nós", f"{desc_stats.nobs}")
        st.metric("Grau Médio", f"{desc_stats.mean:.2f}")
    with col2:
        st.metric("Grau Mínimo", f"{desc_stats.minmax[0]}")
        st.metric("Grau Máximo", f"{desc_stats.minmax[1]}")
    with col3:
        st.metric("Variância", f"{desc_stats.variance:.2f}")
        st.metric("Desvio Padrão", f"{np.sqrt(desc_stats.variance):.2f}")
    with col4:
        st.metric("Assimetria (Skewness)", f"{desc_stats.skewness:.2f}")
        st.metric("Curtose (Kurtosis)", f"{desc_stats.kurtosis:.2f}")
else:
    st.warning("Não há dados de grau para calcular as estatísticas.")