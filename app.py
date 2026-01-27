# ============================================================
# IMPORTS
# ============================================================

import streamlit as st
import random
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
import networkx as nx

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
    step=10,
    key="num_nodes"
)
EDGE_PROBABILITY = st.sidebar.slider(
    "Probabilidade de Conexão",
    min_value=0.0,
    max_value=1.0,
    value=0.1,
    step=0.01,
    key="edge_prob"
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
# FUNÇÕES DE GERAÇÃO E ANÁLISE DA REDE
# ============================================================

@st.cache_data
def generate_network(num_nodes, edge_probability):
    """Gera nós e arestas para uma rede aleatória e garante conectividade mínima."""
    nodes = list(range(num_nodes))
    if edge_probability == 0:
        return nodes, []
        
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

def build_graph_from_data(nodes, edges):
    """Constrói um objeto de grafo NetworkX."""
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G

def calcular_caminho_minimo_medio(G):
    """Calcula o caminho mínimo médio da rede. Retorna '∞' para grafos desconexos."""
    if nx.is_connected(G):
        return nx.average_shortest_path_length(G)
    else:
        return float('inf')

def calcular_coeficiente_aglomeracao_medio(G):
    """Calcula o coeficiente de aglomeração médio da rede."""
    if not G.edges:
        return 0.0
    return nx.average_clustering(G)

# ============================================================
# GERAÇÃO E CÁLCULOS PRINCIPAIS
# ============================================================

# 1. Geração da Rede
nodes, edges = generate_network(NUM_NODES, EDGE_PROBABILITY)
G = build_graph_from_data(nodes, edges)

# 2. Posições para visualização
pos = np.random.randn(NUM_NODES, 3)

# 3. Métricas Estruturais
L = calcular_caminho_minimo_medio(G)
C = calcular_coeficiente_aglomeracao_medio(G)

# 4. Métricas de Grau
degrees = [d for n, d in G.degree()]
avg_degree = np.mean(degrees) if degrees else 0

# ============================================================
# VISUALIZAÇÃO (GRÁFICOS)
# ============================================================
st.header("Visualização da Rede")

# --- Trace da Rede ---
edge_x, edge_y, edge_z = [], [], []
for i, j in edges:
    edge_x.extend([pos[i, 0], pos[j, 0], None])
    edge_y.extend([pos[i, 1], pos[j, 1], None])
    edge_z.extend([pos[i, 2], pos[j, 2], None])

edge_trace = go.Scatter3d(
    x=edge_x, y=edge_y, z=edge_z, mode="lines", opacity=EDGE_OPACITY,
    line=dict(color=EDGE_COLOR, width=EDGE_WIDTH), hoverinfo="none"
)

node_mode = "markers+text" if SHOW_NODE_NAMES else "markers"
if SHOW_NODE_NAMES:
    node_text = [f"{NODE_PREFIX}{i}" for i in nodes] if AUTONUMBER_NODES else [NODE_PREFIX] * NUM_NODES
else:
    node_text = None

node_trace = go.Scatter3d(
    x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], mode=node_mode, opacity=NODE_OPACITY,
    marker=dict(size=NODE_SIZE, color=NODE_COLOR), text=node_text,
    textposition="top center", textfont=dict(size=NODE_FONT_SIZE)
)

# --- Trace do Histograma ---
hist_trace = go.Histogram(x=degrees, nbinsx=HIST_BINS, marker=dict(color=HIST_COLOR))

# --- Figura da Rede ---
fig_net = go.Figure(data=[edge_trace, node_trace])
fig_net.update_layout(
    title="Rede Complexa 3D", height=NET_HEIGHT, margin=dict(l=10, r=10, t=40, b=10),
    showlegend=False, scene=dict(aspectmode="cube", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
)

# --- Figura do Histograma ---
fig_hist = go.Figure(data=[hist_trace])
fig_hist.update_layout(
    title="Distribuição de Graus", height=HIST_HEIGHT, margin=dict(l=40, r=20, t=40, b=40),
    xaxis_title="Grau", yaxis_title="Frequência", font=dict(size=FONT_SIZE)
)

# --- Exibição ---
col_net, col_hist = st.columns([COLUMN_RATIO, 100 - COLUMN_RATIO])
with col_net:
    st.plotly_chart(fig_net, use_container_width=True)
with col_hist:
    st.plotly_chart(fig_hist, use_container_width=True)

# ============================================================
# ANÁLISE ESTATÍSTICA E ESTRUTURAL
# ============================================================
st.header("Análise Estrutural da Rede")

# --- Diagnóstico Automático (Mundo Pequeno) ---
st.subheader("Diagnóstico da Rede")

is_connected = L != float('inf')
if is_connected and avg_degree > 0:
    C_rand = avg_degree / NUM_NODES
    L_rand = np.log(NUM_NODES) / np.log(avg_degree)
    
    # Condições para ser mundo pequeno
    is_highly_clustered = C > C_rand * 2 # Heurística: C é pelo menos 2x maior
    is_short_path = L < L_rand * 2 # Heurística: L é no máximo 2x maior

    if is_highly_clustered and is_short_path:
        st.success("✅ **Diagnóstico:** A rede exibe fortes características de **Mundo Pequeno (Small-World)**.")
        st.info("Isso significa que ela possui alta aglomeração local (seus amigos se conhecem) e, ao mesmo tempo, um caminho médio curto entre quaisquer dois nós (você está a poucos 'passos' de qualquer pessoa), similar a redes sociais como Facebook ou LinkedIn.")
    elif is_highly_clustered:
        st.warning("⚠️ **Diagnóstico:** A rede é **altamente clusterizada**, mas não necessariamente possui um caminho médio curto. Pode se assemelhar a uma rede regular ou de treliça.")
    else:
        st.warning("⚠️ **Diagnóstico:** A rede se assemelha a um **Grafo Aleatório (Erdos-Renyi)**, com baixa clusterização.")
else:
    st.error("❌ **Diagnóstico:** A rede está **desconexa ou é muito esparsa** para uma análise de mundo pequeno.")

st.markdown("---")

# --- Apresentação das Métricas ---
st.subheader("Métricas Detalhadas")
m_col1, m_col2 = st.columns(2)

with m_col1:
    st.metric(
        label="Coeficiente de Aglomeração Médio (C)",
        value=f"{C:.4f}"
    )
    with st.expander("O que isso significa?"):
        st.markdown("""
        O **Coeficiente de Aglomeração (ou Clustering)** mede a tendência dos nós em uma rede de formarem "grupinhos" ou "clusters".
        - **Autoria:** Proposto por Duncan J. Watts e Steven Strogatz (1998).
        - **Fórmula:** É a média dos coeficientes de aglomeração locais de todos os nós. O coeficiente local de um nó é a fração de conexões existentes entre seus vizinhos, dividida pelo número máximo de conexões possíveis entre eles.
        - **Interpretação:** Um valor próximo de **1** indica que a vizinhança de um nó médio é quase um "clique" (todos se conhecem). Um valor próximo de **0** indica que os vizinhos de um nó raramente se conectam entre si.
        - **Nesta Rede:** O valor de **`{:.4f}`** sugere que a rede tem uma tendência de clusterização {}.
        """.format(C, "**relativamente alta**" if C > 0.5 else ("**moderada**" if C > 0.1 else "**baixa**")))

with m_col2:
    st.metric(
        label="Caminho Mínimo Médio (L)",
        value=f"{L:.4f}" if L != float('inf') else "∞"
    )
    with st.expander("O que isso significa?"):
        st.markdown(f"""
        Mede a distância média (número de "pulos") entre todos os pares de nós na rede. É uma medida de eficiência da rede em transportar informação.
        - **Fórmula:** A média dos comprimentos dos caminhos mais curtos para todos os pares de nós.
        - **Interpretação:** Um valor **baixo** indica uma rede altamente conectada e eficiente, onde se chega rapidamente de um ponto a outro. É a base dos "seis graus de separação". Redes desconexas têm um caminho infinito.
        - **Nesta Rede:** Um caminho médio de **`{'%.4f' % L if L != float('inf') else '∞'}`** indica que, em média, são necessários cerca de **`{L:.1f}`** passos para ir de um nó a qualquer outro. Isso é considerado um caminho {}.
        """.format(L, "**muito curto**" if L < np.log(NUM_NODES) else "**relativamente longo**" if not is_connected else ""))

# --- Métricas de Grau ---
st.subheader("Estatísticas da Distribuição de Graus")
if degrees:
    desc_stats = stats.describe(degrees)
    d_col1, d_col2, d_col3 = st.columns(3)
    with d_col1:
        st.metric("Grau Médio", f"{desc_stats.mean:.2f}")
    with d_col2:
        st.metric("Grau Máximo", f"{desc_stats.minmax[1]}")
    with d_col3:
        st.metric("Variância do Grau", f"{desc_stats.variance:.2f}")
    
    with st.expander("Entenda as Estatísticas de Grau"):
        st.markdown("""
        A **distribuição de graus** é uma das propriedades mais fundamentais de uma rede.
        - **Grau de um Nó:** O número de conexões que ele possui.
        - **Grau Médio:** A média dos graus de todos os nós. Indica a densidade geral de conexões.
        - **Grau Máximo:** O grau do nó mais conectado (o "hub" principal).
        - **Variância do Grau:** Mede a dispersão dos graus. Um valor alto sugere uma rede heterogênea, com muitos nós de baixo grau e alguns hubs de alto grau (típico de redes "livres de escala" - Scale-Free). Um valor baixo indica que a maioria dos nós tem um número similar de conexões.
        """)
else:
    st.warning("Não há dados de grau para calcular as estatísticas.")

# ============================================================
# DATAFRAME COM DADOS DOS NÓS
# ============================================================
st.subheader("Dados Detalhados dos Nós")

if AUTONUMBER_NODES:
    node_display_names_for_df = [f"{NODE_PREFIX}{i}" for i in nodes]
else:
    node_display_names_for_df = [NODE_PREFIX] * NUM_NODES

node_data = {
    "Nome do Nó": node_display_names_for_df,
    "Grau": degrees,
    "Posição X": pos[:, 0],
    "Posição Y": pos[:, 1],
    "Posição Z": pos[:, 2],
}
df = pd.DataFrame(node_data)
df.index.name = "ID do Nó"

st.dataframe(df)
