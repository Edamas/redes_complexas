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

st.set_page_config(layout="wide")
st.title("Simulador Interativo de Redes Complexas")
st.write("Autor: Elysio Damasceno da Silva Neto (Baseado na Aula 1 de Introdução às Redes Complexas - EACH-USP)")

# ============================================================ 
# SIDEBAR - CONTROLES DE SIMULAÇÃO 
# ============================================================ 

st.sidebar.header("1. Modelo da Rede")
model_type = st.sidebar.selectbox(
    "Escolha o modelo de geração da rede:",
    ("Erdos-Renyi", "Watts-Strogatz"),
    help="**Erdos-Renyi (ER):** Um modelo simples onde cada aresta é formada com uma probabilidade uniforme. Tende a gerar redes homogêneas.\n\n**Watts-Strogatz (WS):** Gera redes do tipo 'Mundo Pequeno' (Small-World), com alta clusterização local e baixo caminho médio global, como em redes sociais.")

st.sidebar.header("2. Parâmetros do Modelo")

# Parâmetros condicionais ao modelo
if model_type == "Erdos-Renyi":
    N = st.sidebar.slider("Número de Nós (N)", 5, 1000, 100, 5, key="er_n")
    P_ER = st.sidebar.slider("Probabilidade de Conexão (p)", 0.0, 1.0, 0.1, 0.01, key="er_p")
    G = nx.erdos_renyi_graph(N, P_ER)
else: # Watts-Strogatz
    N = st.sidebar.slider("Número de Nós (eNG)", 5, 1000, 20, 5, key="ws_n")
    K_WS = st.sidebar.slider("Nº de Vizinhos Próximos (k)", 2, 20, 4, 2, help="Cada nó se conecta aos 'k' vizinhos mais próximos no anel inicial. Deve ser um número par.")
    P_WS = st.sidebar.slider("Probabilidade de Reconexão (ePG)", 0.0, 1.0, 0.2, 0.01, help="Probabilidade de 'religar' uma aresta, introduzindo aleatoriedade.")
    # Garante que k seja par e menor que n para evitar erros
    k_ws_safe = max(2, K_WS if K_WS % 2 == 0 else K_WS - 1)
    if k_ws_safe >= N: k_ws_safe = max(2, N - 2)
    G = nx.watts_strogatz_graph(N, k_ws_safe, P_WS)


st.sidebar.header("3. Parâmetros de Visualização")
layout_dim = st.sidebar.selectbox("Dimensão", ("3D", "2D"))
layout_type = st.sidebar.selectbox(
    "Algoritmo de Layout",
    ("Spring (Física)", "Circular/Esférico", "Shell", "Aleatório")
)
if layout_type in ["Circular/Esférico", "Shell"] and layout_dim == "3D":
    layout_dist = st.sidebar.selectbox("Distribuição", ("Superfície", "Volume"))
else:
    layout_dist = "Superfície" # Padrão para outros casos

# Controles de aparência
st.sidebar.subheader("Aparência dos Nós")
col1, col2 = st.sidebar.columns(2)
NODE_COLOR = col1.color_picker("Cor", value="#1f77b4", key="node_color")
NODE_OPACITY = col2.slider("Opacidade", 0.0, 1.0, 0.9, 0.05, key="node_opacity")
NODE_SIZE = st.sidebar.slider("Tamanho do Nó", 1, 30, 8, 1)

st.sidebar.subheader("Aparência das Arestas")
col3, col4 = st.sidebar.columns(2)
EDGE_COLOR = col3.color_picker("Cor", value="#888888", key="edge_color")
EDGE_OPACITY = col4.slider("Opacidade", 0.0, 1.0, 0.5, 0.05, key="edge_opacity")
EDGE_WIDTH = st.sidebar.slider("Largura da Aresta", 1, 10, 2, 1)

st.sidebar.subheader("Rótulos dos Nós")
SHOW_NODE_NAMES = st.sidebar.checkbox("Exibir Rótulos", value=True)
if SHOW_NODE_NAMES:
    NODE_FONT_SIZE = st.sidebar.slider("Tamanho da Fonte", 5, 20, 10, 1)
    NODE_PREFIX = st.sidebar.text_input("Prefixo", value="Nó ")
    AUTONUMBER_NODES = st.sidebar.checkbox("Autonumerar", value=True)

# ============================================================ 
# FUNÇÕES DE GERAÇÃO DE LAYOUT E DESENHO 
# ============================================================ 

@st.cache_data
def generate_layout(_G, dim, l_type, l_dist):
    """Gera as posições dos nós com base nos parâmetros de layout."""
    n = len(_G.nodes)
    if n == 0: return {}

    d = int(dim[0])

    if l_type == "Aleatório":
        return nx.random_layout(_G, dim=d)
    
    if l_type == "Circular/Esférico":
        if dim == "3D":
            if l_dist == "Volume":
                vec = np.random.randn(3, n)
                vec /= np.linalg.norm(vec, axis=0)
                r = np.random.rand(n) ** (1/3)
                pos_arr = (vec * r).T
                return {i: pos_arr[i] for i in range(n)}
            else: # Superfície
                return nx.spherical_layout(_G) # FIX: chamada correta da função
        else: # 2D
             return nx.circular_layout(_G)

    if l_type == "Shell":
        n_shells = int(np.ceil(np.sqrt(n/4)))
        shells_list = [list(range(sum(2**i for i in range(j)), sum(2**i for i in range(j+1)))) for j in range(n_shells)]
        if shells_list:
            shells_list[-1].extend(range(max(shells_list[-1] or [0]) + 1, n))
        else:
            shells_list = [list(range(n))]
        return nx.shell_layout(_G, nlist=shells_list) # FIX: usar nlist em vez de shells

    if l_type == "Spring (Física)":
        return nx.spring_layout(_G, dim=d)

    return nx.random_layout(_G, dim=d) # Fallback

def draw_network(G, pos, dim):
    """Desenha a rede em 2D ou 3D usando Plotly."""
    if dim == "3D":
        edge_x, edge_y, edge_z = [], [], []
        for u, v in G.edges():
            edge_x.extend([pos[u][0], pos[v][0], None])
            edge_y.extend([pos[u][1], pos[v][1], None])
            edge_z.extend([pos[u][2], pos[v][2], None])
        
        edge_trace = go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines', line=dict(color=EDGE_COLOR, width=EDGE_WIDTH), opacity=EDGE_OPACITY, hoverinfo='none')
        
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_z = [pos[node][2] for node in G.nodes()]

        node_trace = go.Scatter3d(x=node_x, y=node_y, z=node_z, mode='markers+text' if SHOW_NODE_NAMES else 'markers',
                                  marker=dict(size=NODE_SIZE, color=NODE_COLOR, opacity=NODE_OPACITY), hoverinfo='text')
    else: # 2D
        edge_x, edge_y = [], []
        for u, v in G.edges():
            edge_x.extend([pos[u][0], pos[v][0], None])
            edge_y.extend([pos[u][1], pos[v][1], None])

        edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color=EDGE_COLOR, width=EDGE_WIDTH), opacity=EDGE_OPACITY, hoverinfo='none')

        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text' if SHOW_NODE_NAMES else 'markers',
                                marker=dict(size=NODE_SIZE, color=NODE_COLOR, opacity=NODE_OPACITY), hoverinfo='text')
    
    if SHOW_NODE_NAMES:
        node_trace.text = [f"{NODE_PREFIX}{i}" for i in G.nodes()] if AUTONUMBER_NODES else [NODE_PREFIX] * len(G.nodes())
        node_trace.textfont = dict(size=NODE_FONT_SIZE)
        node_trace.hovertext = [f"Nó: {node}<br>Grau: {G.degree(node)}" for node in G.nodes()]
    
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
    
    # FIX: Garantir proporção (Aspect Ratio)
    if dim == "3D":
        fig.update_scenes(aspectmode='cube')
    else:
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
    
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, visible=False)
    
    return fig

# ============================================================ 
# CÁLCULOS E VISUALIZAÇÃO PRINCIPAL 
# ============================================================ 

# 1. Posições para visualização
pos = generate_layout(G, layout_dim, layout_type, layout_dist)

# 2. Métricas
nodes = list(G.nodes())
degrees = [d for n, d in G.degree()]
avg_degree = np.mean(degrees) if degrees else 0
is_connected = nx.is_connected(G)
L = nx.average_shortest_path_length(G) if is_connected else float('inf')
C = nx.average_clustering(G) if nodes else 0.0

# --- Desenho da Rede e Histograma ---
st.header("Visualização da Rede")
col_vis1, col_vis2 = st.columns([7, 3]) 

with col_vis1:
    st.subheader(f"Modelo: {model_type} ({layout_dim}, {layout_type})")
    fig_net = draw_network(G, pos, layout_dim)
    st.plotly_chart(fig_net, use_container_width=True)

with col_vis2:
    st.subheader("Distribuição de Graus")
    fig_hist = go.Figure(go.Histogram(x=degrees, marker_color=NODE_COLOR))
    fig_hist.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_hist, use_container_width=True)

# ============================================================ 
# ANÁLISE ESTATÍSTICA E ESTRUTURAL 
# ============================================================ 
st.markdown("---")
st.header("Análise Estrutural da Rede")

# --- Diagnóstico ---
if G.number_of_nodes() > 2 and G.number_of_edges() > 1 and avg_degree > 1:
    try:
        C_rand = avg_degree / N
        L_rand = np.log(N) / np.log(avg_degree)
        if is_connected and C > C_rand * 2 and L < L_rand * 2:
            st.success("✅ **Diagnóstico:** A rede exibe fortes características de **Mundo Pequeno (Small-World)**.")
        elif C > C_rand * 2:
            st.warning("⚠️ **Diagnóstico:** A rede é **altamente clusterizada**, mas não necessariamente um 'Mundo Pequeno'.")
        else:
             st.info("ℹ️ **Diagnóstico:** A rede se assemelha a um **Grafo Aleatório**, com baixa clusterização.")
    except (ZeroDivisionError, ValueError):
        st.error("❌ **Diagnóstico:** Não foi possível analisar as propriedades de Mundo Pequeno (grau médio baixo).")
else:
    st.error("❌ **Diagnóstico:** Rede vazia ou trivial demais para análise.")

# --- Métricas ---
st.subheader("Métricas Detalhadas")
m_col1, m_col2 = st.columns(2)
with m_col1:
    st.metric("Coeficiente de Aglomeração Médio (C)", f"{C:.4f}")
    with st.expander("Entenda o Coeficiente de Aglomeração (C)"):
        st.markdown("- **O que é?** Mede a tendência dos nós de formarem 'panelinhas'.\n- **Interpretação:** Um valor alto (próximo de 1) significa que os vizinhos de um nó tendem a ser vizinhos entre si. Um valor baixo (próximo de 0) indica o contrário.")
with m_col2:
    st.metric("Caminho Mínimo Médio (L)", f"{L:.4f}" if is_connected else "∞")
    with st.expander("Entenda o Caminho Mínimo Médio (L)"):
        st.markdown("- **O que é?** Mede a eficiência da rede, ou o número médio de 'passos' para ir de um nó a qualquer outro.\n- **Interpretação:** Um valor baixo indica uma rede eficiente. Um valor '∞' indica que a rede está quebrada em partes desconexas.")

# FIX: Restaurar métricas com expanders individuais
desc_stats = stats.describe(degrees) if degrees else None
if desc_stats:
    st.subheader("Estatísticas da Distribuição de Graus")
    d_col1, d_col2, d_col3, d_col4, d_col5 = st.columns(5)
    with d_col1:
        st.metric("Nº de Nós", f"{desc_stats.nobs}")
        with st.expander("Sobre 'Nº de Nós'"):
            st.markdown("O número total de vértices (pontos) na sua rede.")
    with d_col2:
        st.metric("Grau Médio", f"{desc_stats.mean:.2f}")
        with st.expander("Sobre 'Grau Médio'"):
            st.markdown("O número médio de conexões (arestas) que cada nó possui. Uma medida de densidade.")
    with d_col3:
        st.metric("Grau (Min-Max)", f"{desc_stats.minmax[0]}-{desc_stats.minmax[1]}")
        with st.expander("Sobre 'Grau (Min-Max)'"):
            st.markdown("Mostra o grau do nó menos conectado e do nó mais conectado (o 'hub' principal).")
    with d_col4:
        st.metric("Assimetria (Skewness)", f"{desc_stats.skewness:.2f}")
        with st.expander("Sobre 'Assimetria'"):
            st.markdown("Mede a assimetria da distribuição de graus. **> 0:** A maioria dos nós tem grau baixo. **< 0:** A maioria dos nós tem grau alto.")
    with d_col5:
        st.metric("Curtose (Kurtosis)", f"{desc_stats.kurtosis:.2f}")
        with st.expander("Sobre 'Curtose'"):
            st.markdown("Mede o quão 'pesada' é a cauda da distribuição. Um valor alto indica uma grande presença de outliers (hubs muito fortes).")

# ============================================================ 
# DATAFRAME 
# ============================================================ 
st.subheader("Dados Detalhados dos Nós")
if G.number_of_nodes() > 0:
    df = pd.DataFrame({
        "Grau": dict(G.degree()),
        "Coef. Aglomeração": nx.clustering(G),
    })
    df.index.name = "ID do Nó"
    st.dataframe(df)
else:
    st.warning("Rede vazia. Não há dados para exibir.")
