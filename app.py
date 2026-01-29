# ============================================================ 
# IMPORTS 
# ============================================================ 

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
import networkx as nx

# ============================================================ 
# CONFIGURAÇÃO DA PÁGINA E ESTADO DA SESSÃO 
# ============================================================ 

st.set_page_config(layout="wide", page_title="Simulação de Redes")

def initialize_session_state():
    """Define os valores padrão para todos os parâmetros no estado da sessão na primeira execução."""
    defaults = {
        'model_type': 'Erdos-Renyi',
        'N': 100,
        'P_ER': 0.1,
        'K_WS': 4,
        'P_WS': 0.2,
        'layout_dim': '3D',
        'layout_type': 'Spring (Física)',
        'layout_dist': 'Superfície',
        'NODE_COLOR': '#1f77b4',
        'NODE_OPACITY': 0.9,
        'NODE_SIZE': 8,
        'EDGE_COLOR': '#888888',
        'EDGE_OPACITY': 0.5,
        'EDGE_WIDTH': 2,
        'SHOW_NODE_NAMES': True,
        'NODE_FONT_SIZE': 10,
        'NODE_PREFIX': 'Nó ',
        'AUTONUMBER_NODES': True,
        # Checkboxes de Métricas
        'show_n_nodes': True, 'show_n_edges': True, 'show_density': True,
        'show_L': True, 'show_C': True, 'show_avg_degree': True,
        'show_degree_minmax': True, 'show_degree_skew': True, 'show_degree_kurt': True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# Adiciona um link para a página de configurações na sidebar
st.sidebar.header("Navegação")
st.sidebar.info("Navegue entre as páginas (Simulação, Configurações, Sobre) usando o menu da barra lateral.")



# ============================================================ 
# FUNÇÕES DE GERAÇÃO E ANÁLISE 
# ============================================================ 
# As funções de geração de layout e desenho são movidas para cá para usar o estado da sessão

@st.cache_data
def generate_graph(model, n, p_er, k_ws, p_ws):
    """Gera um grafo NetworkX com base no modelo e parâmetros selecionados."""
    if model == "Erdos-Renyi":
        return nx.erdos_renyi_graph(n, p_er)
    elif model == "Watts-Strogatz":
        k_ws_safe = max(2, k_ws if k_ws % 2 == 0 else k_ws - 1)
        if k_ws_safe >= n: k_ws_safe = max(2, n - 2 if n > 2 else 0)
        return nx.watts_strogatz_graph(n, k_ws_safe, p_ws)
    return nx.Graph()

@st.cache_data
def generate_layout(_G, dim, l_type, l_dist):
    """Gera as posições dos nós com base nos parâmetros de layout."""
    n = len(_G.nodes)
    if n == 0: return {}
    d = int(dim[0])
    if l_type == "Aleatório": return nx.random_layout(_G, dim=d)
    if l_type == "Circular/Esférico":
        if dim == "3D":
            if l_dist == "Volume":
                vec = np.random.randn(3, n); vec /= np.linalg.norm(vec, axis=0)
                r = np.random.rand(n) ** (1/3)
                pos_arr = (vec * r).T; return {i: pos_arr[i] for i in range(n)}
            else: return nx.spherical_layout(_G)
        else: return nx.circular_layout(_G)
    if l_type == "Shell":
        n_shells = int(np.ceil(np.sqrt(n/4))); shells_list = [list(range(sum(2**i for i in range(j)), sum(2**i for i in range(j+1)))) for j in range(n_shells)]
        if shells_list: shells_list[-1].extend(range(max(shells_list[-1] or [0]) + 1, n))
        else: shells_list = [list(range(n))]
        return nx.shell_layout(_G, nlist=shells_list)
    if l_type == "Spring (Física)": return nx.spring_layout(_G, dim=d)
    return nx.random_layout(_G, dim=d)

def draw_network(G, pos, dim):
    """Desenha a rede em 2D ou 3D usando Plotly."""
    s = st.session_state
    if dim == "3D":
        edge_x, edge_y, edge_z = [], [], []; node_x, node_y, node_z = [], [], []
        for u, v in G.edges(): edge_x.extend([pos[u][0], pos[v][0], None]); edge_y.extend([pos[u][1], pos[v][1], None]); edge_z.extend([pos[u][2], pos[v][2], None])
        for node in G.nodes(): node_x.append(pos[node][0]); node_y.append(pos[node][1]); node_z.append(pos[node][2])
        edge_trace = go.Scatter3d(x=edge_x, y=edge_y, z=edge_z, mode='lines', line=dict(color=s.EDGE_COLOR, width=s.EDGE_WIDTH), opacity=s.EDGE_OPACITY, hoverinfo='none')
        node_trace = go.Scatter3d(x=node_x, y=node_y, z=node_z, mode='markers' if not s.SHOW_NODE_NAMES else 'markers+text', marker=dict(size=s.NODE_SIZE, color=s.NODE_COLOR, opacity=s.NODE_OPACITY), hoverinfo='text')
    else:
        edge_x, edge_y = [], []; node_x, node_y = [], []
        for u, v in G.edges(): edge_x.extend([pos[u][0], pos[v][0], None]); edge_y.extend([pos[u][1], pos[v][1], None])
        for node in G.nodes(): node_x.append(pos[node][0]); node_y.append(pos[node][1])
        edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color=s.EDGE_COLOR, width=s.EDGE_WIDTH), opacity=s.EDGE_OPACITY, hoverinfo='none')
        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers' if not s.SHOW_NODE_NAMES else 'markers+text', marker=dict(size=s.NODE_SIZE, color=s.NODE_COLOR, opacity=s.NODE_OPACITY), hoverinfo='text')
    
    if s.SHOW_NODE_NAMES:
        node_trace.text = [f"{s.NODE_PREFIX}{i}" for i in G.nodes()] if s.AUTONUMBER_NODES else [s.NODE_PREFIX] * len(G.nodes())
        node_trace.textfont = dict(size=s.NODE_FONT_SIZE); node_trace.hovertext = [f"Nó: {node}<br>Grau: {G.degree(node)}" for node in G.nodes()]
    
    fig = go.Figure(data=[edge_trace, node_trace]); fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
    if dim == "3D": fig.update_scenes(aspectmode='cube')
    else: fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, visible=False); fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, visible=False)
    return fig

# ============================================================ 
# PÁGINA PRINCIPAL: SIMULAÇÃO E ANÁLISE 
# ============================================================ 
st.title("Simulação e Análise da Rede")
st.info("Esta página exibe a rede gerada com os parâmetros definidos na página 'Configurações'.")

# 1. Geração e Cálculos
s = st.session_state
G = generate_graph(s.model_type, s.N, s.get('P_ER'), s.get('K_WS'), s.get('P_WS'))
pos = generate_layout(G, s.layout_dim, s.layout_type, s.layout_dist)
degrees = [d for n, d in G.degree()]

# 2. Visualização
st.header("Visualização da Rede")
col_vis1, col_vis2 = st.columns([7, 3])
with col_vis1:
    st.subheader(f"Modelo: {s.model_type} ({s.layout_dim}, {s.layout_type})")
    fig_net = draw_network(G, pos, s.layout_dim)
    st.plotly_chart(fig_net, use_container_width=True)
with col_vis2:
    st.subheader("Distribuição de Graus")
    fig_hist = go.Figure(go.Histogram(x=degrees, marker_color=s.NODE_COLOR))
    fig_hist.update_layout(margin=dict(l=10, r=10, t=40, b=10)); st.plotly_chart(fig_hist, use_container_width=True)

# 3. Análise Estrutural
st.markdown("---")
st.header("Análise Estrutural da Rede")

# --- Diagnóstico ---
avg_degree = np.mean(degrees) if degrees else 0
if G.number_of_nodes() > 2 and G.number_of_edges() > 1 and avg_degree > 1:
    is_connected = nx.is_connected(G); C = nx.average_clustering(G)
    try:
        C_rand = avg_degree / s.N
        if is_connected: L = nx.average_shortest_path_length(G); L_rand = np.log(s.N) / np.log(avg_degree)
        if is_connected and C > C_rand * 2 and L < L_rand * 2: st.success("✅ **Diagnóstico:** A rede exibe fortes características de **Mundo Pequeno (Small-World)**.")
        elif C > C_rand * 2: st.warning("⚠️ **Diagnóstico:** A rede é **altamente clusterizada**, mas não necessariamente um 'Mundo Pequeno'.")
        else: st.info("ℹ️ **Diagnóstico:** A rede se assemelha a um **Grafo Aleatório**, com baixa clusterização.")
    except (ZeroDivisionError, ValueError): st.error("❌ **Diagnóstico:** Não foi possível analisar as propriedades de Mundo Pequeno.")
else:
    st.error("❌ **Diagnóstico:** Rede vazia ou trivial demais para análise.")

# --- Métricas Detalhadas ---
st.subheader("Métricas Detalhadas")

# Categoria: Conectividade
if any(s.get(k, True) for k in ['show_n_nodes', 'show_n_edges', 'show_density', 'show_L']):
    st.markdown("##### Conectividade")
    c_cols = st.columns(4)
    if s.get('show_n_nodes', True): c_cols[0].metric("Nº de Nós", G.number_of_nodes())
    if s.get('show_n_edges', True): c_cols[1].metric("Nº de Arestas", G.number_of_edges())
    if s.get('show_density', True): c_cols[2].metric("Densidade", f"{nx.density(G):.4f}")
    if s.get('show_L', True):
        L = nx.average_shortest_path_length(G) if is_connected else float('inf')
        c_cols[3].metric("Caminho Mínimo Médio (L)", f"{L:.4f}" if is_connected else "∞")

# Categoria: Agrupamento
if s.get('show_C', True):
    st.markdown("##### Agrupamento e Modularidade")
    C = nx.average_clustering(G) if G.nodes else 0.0
    st.metric("Coeficiente de Aglomeração Médio (C)", f"{C:.4f}")
    with st.expander("Entenda o Coeficiente de Aglomeração (C)"):
        st.markdown("- **O que é?** Mede a tendência dos nós de formarem 'panelinhas'.\n- **Interpretação:** Um valor alto (próximo de 1) significa que os vizinhos de um nó tendem a ser vizinhos entre si. Um valor baixo (próximo de 0) indica o contrário.")

# Categoria: Distribuição de Graus
if any(s.get(k, True) for k in ['show_avg_degree', 'show_degree_minmax', 'show_degree_skew', 'show_degree_kurt']):
    st.markdown("##### Distribuição de Graus")
    desc_stats = stats.describe(degrees) if degrees else None
    if desc_stats:
        d_cols = st.columns(4)
        if s.get('show_avg_degree', True): d_cols[0].metric("Grau Médio", f"{desc_stats.mean:.2f}")
        if s.get('show_degree_minmax', True): d_cols[1].metric("Grau (Min-Max)", f"{desc_stats.minmax[0]}-{desc_stats.minmax[1]}")
        if s.get('show_degree_skew', True): d_cols[2].metric("Assimetria", f"{desc_stats.skewness:.2f}")
        if s.get('show_degree_kurt', True): d_cols[3].metric("Curtose", f"{desc_stats.kurtosis:.2f}")

# Categoria: DataFrame
st.subheader("Dados Detalhados dos Nós")
if G.number_of_nodes() > 0:
    df = pd.DataFrame({"Grau": dict(G.degree()), "Coef. Aglomeração": nx.clustering(G)})
    df.index.name = "ID do Nó"; st.dataframe(df, use_container_width=True)
else:
    st.warning("Rede vazia. Não há dados para exibir.")