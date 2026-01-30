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
# CONFIGURAÇÃO GERAL E ESTADO DA SESSÃO
# ============================================================
st.set_page_config(layout="wide", page_title="Simulador de Redes Complexas")

def initialize_session_state():
    """Define os valores padrão para todos os parâmetros no estado da sessão na primeira execução."""
    defaults = {
        'model_type': 'Erdos-Renyi', 'N': 100, 'P_ER': 0.1, 'K_WS': 4, 'P_WS': 0.2,
        'layout_dim': '3D', 'layout_type': 'Spring (Física)', 'layout_dist': 'Superfície',
        'NODE_COLOR': '#1f77b4', 'NODE_OPACITY': 0.9, 'NODE_SIZE': 8,
        'EDGE_COLOR': '#888888', 'EDGE_OPACITY': 0.5, 'EDGE_WIDTH': 2,
        'SHOW_NODE_NAMES': True, 'NODE_FONT_SIZE': 10, 'NODE_PREFIX': 'Nó ', 'AUTONUMBER_NODES': True,
        'show_n_nodes': True, 'show_n_edges': True, 'show_density': True, 'show_L': True,
        'show_C': True, 'show_avg_degree': True, 'show_degree_minmax': True,
        'show_degree_skew': True, 'show_degree_kurt': True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()

# ============================================================
# FUNÇÕES DE GERAÇÃO E ANÁLISE DE REDES (Compartilhadas)
# ============================================================
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
    
    fig = go.Figure(data=[edge_trace, node_trace]); fig.update_layout(showlegend=False, margin=dict(l=0,r=0,t=0,b=0));
    if dim == "3D": fig.update_scenes(aspectmode='cube')
    else: fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, visible=False); fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, visible=False)
    return fig

# ============================================================
# DEFINIÇÃO DAS PÁGINAS (FUNÇÕES)
# ============================================================

def simulation_page():
    st.title("Simulação e Análise da Rede")
    st.info("Esta página exibe a rede gerada com os parâmetros definidos na página 'Configurações'.")

    s = st.session_state
    
    # --- Geração e Análise ---
    G = generate_graph(s.model_type, s.N, s.P_ER, s.K_WS, s.P_WS)
    pos = generate_layout(G, s.layout_dim, s.layout_type, s.layout_dist)
    degrees = [d for n, d in G.degree()]

    # --- Visualização ---
    st.header("Visualização da Rede")
    vcol1, vcol2 = st.columns([3, 1])
    with vcol1:
        st.subheader(f"Modelo: {s.model_type} ({s.layout_dim}, {s.layout_type})")
        fig_net = draw_network(G, pos, s.layout_dim)
        st.plotly_chart(fig_net, use_container_width=True)
    with vcol2:
        st.subheader("Distribuição de Graus")
        fig_hist = go.Figure(go.Histogram(x=degrees, marker_color=s.NODE_COLOR))
        fig_hist.update_layout(margin=dict(l=10, r=10, t=0, b=0)); st.plotly_chart(fig_hist, use_container_width=True)

    # --- Análise ---
    st.markdown("---")
    st.header("Análise da Rede")

    # Diagnóstico
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

    # Métricas Detalhadas
    st.subheader("Métricas Detalhadas")

    # Categoria: Conectividade
    st.markdown("##### Conectividade")
    c_cols = st.columns(4)
    if s.show_n_nodes: c_cols[0].metric("Nós",G.number_of_nodes())
    if s.show_n_edges: c_cols[1].metric("Arestas",G.number_of_edges())
    if s.show_density: c_cols[2].metric("Densidade",f"{nx.density(G):.4f}")
    if s.show_L: L=nx.average_shortest_path_length(G) if is_connected else float('inf'); c_cols[3].metric("Caminho Médio (L)",f"{L:.4f}" if is_connected else "∞")

    st.markdown("##### Agrupamento")
    if s.show_C: C=nx.average_clustering(G); st.metric("Aglomeração (C)",f"{C:.4f}")
    
    st.markdown("##### Distribuição de Graus")
    desc_stats = stats.describe(degrees) if degrees else None
    if desc_stats:
        d_cols = st.columns(4)
        if s.show_avg_degree: d_cols[0].metric("Grau Médio", f"{desc_stats.mean:.2f}")
        if s.show_degree_minmax: d_cols[1].metric("Grau (Min-Max)",f"{desc_stats.minmax[0]}-{desc_stats.minmax[1]}")
        if s.show_degree_skew: d_cols[2].metric("Assimetria",f"{desc_stats.skewness:.2f}")
        if s.show_degree_kurt: d_cols[3].metric("Curtose", f"{desc_stats.kurtosis:.2f}")
            
    # Dados por Nó
    st.subheader("Dados Detalhados dos Nós")
    if G.number_of_nodes() > 0:
        df = pd.DataFrame({"Grau":dict(G.degree()), "Coef. Aglomeração":nx.clustering(G)})
        df.index.name = "ID do Nó"; st.dataframe(df, use_container_width=True)
    else:
        st.warning("Rede vazia. Não há dados para exibir.")

def config_page():
    st.header("Configurações da Simulação")
    st.write("Ajuste todos os parâmetros da sua simulação aqui. As mudanças serão refletidas na página 'Simulação'.")

    # --- 1. Modelo da Rede ---
    st.subheader("1. Modelo da Rede")
    st.info("""
    **Erdos-Renyi (ER):** Um modelo fundamental onde cada par de nós tem uma probabilidade uniforme de se conectar. Gera redes homogêneas e é um bom ponto de partida para comparações.

    **Watts-Strogatz (WS):** Um modelo que gera redes do tipo 'Mundo Pequeno' (Small-World). Começa com uma rede regular e depois 'reconecta' aleatoriamente algumas arestas.
    """)
    st.radio("Escolha o modelo de geração:", ("Erdos-Renyi", "Watts-Strogatz"), key='model_type', horizontal=True)

    # --- 2. Parâmetros do Modelo ---
    st.subheader("2. Parâmetros do Modelo")
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.model_type == "Erdos-Renyi":
            st.slider("Número de Nós (N)", 5, 1000, key="N")
            st.slider("Probabilidade de Conexão (p)", 0.0, 1.0, step=0.01, key="P_ER")
        else: # Watts-Strogatz
            st.slider("Número de Nós (eNG)", 5, 1000, key="N")
            st.slider("Nº de Vizinhos Próximos (k)", 2, 20, step=2, help="Cada nó se conecta aos 'k' vizinhos mais próximos.", key="K_WS")
            st.slider("Probabilidade de Reconexão (ePG)", 0.0, 1.0, step=0.01, help="Probabilidade de 'religar' uma aresta.", key="P_WS")

    # --- 3. Parâmetros de Visualização ---
    with col2:
        st.selectbox("Dimensão", ("3D", "2D"), key="layout_dim")
        st.selectbox("Algoritmo de Layout", ("Spring (Física)", "Circular/Esférico", "Shell", "Aleatório"), key="layout_type")
        if st.session_state.layout_type in ["Circular/Esférico", "Shell"] and st.session_state.layout_dim == "3D":
            st.selectbox("Distribuição", ("Superfície", "Volume"), key="layout_dist")
    
    # --- 4. Aparência e Rótulos ---
    st.subheader("3. Aparência e Rótulos")
    acol1, acol2 = st.columns(2)
    with acol1:
        st.markdown("##### Nós")
        c1,c2 = st.columns(2); c1.color_picker("Cor", key="NODE_COLOR"); c2.slider("Opacidade", 0.0, 1.0, key="NODE_OPACITY")
        st.slider("Tamanho", 1, 30, key="NODE_SIZE")
    with acol2:
        st.markdown("##### Arestas")
        c3,c4 = st.columns(2); c3.color_picker("Cor", key="EDGE_COLOR"); c4.slider("Opacidade", 0.0, 1.0, key="EDGE_OPACITY")
        st.slider("Largura", 1, 10, key="EDGE_WIDTH")
    
    st.checkbox("Exibir Rótulos", key="SHOW_NODE_NAMES")
    if st.session_state.SHOW_NODE_NAMES:
        rcol1, rcol2, rcol3 = st.columns(3)
        rcol1.slider("Tamanho da Fonte", 5, 20, key="NODE_FONT_SIZE")
        rcol2.text_input("Prefixo", key="NODE_PREFIX")
        rcol3.checkbox("Autonumerar", key="AUTONUMBER_NODES")

    # --- 5. Métricas a Exibir ---
    st.subheader("4. Métricas a Exibir na Simulação")
    with st.expander("Selecionar Métricas"):
        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            st.markdown("##### Conectividade")
            st.checkbox("Nº de Nós", key="show_n_nodes"); st.checkbox("Nº de Arestas", key="show_n_edges")
            st.checkbox("Densidade", key="show_density"); st.checkbox("Caminho Mínimo Médio (L)", key="show_L")
        with mcol2:
            st.markdown("##### Agrupamento")
            st.checkbox("Coef. de Aglomeração (C)", key="show_C")
        with mcol3:
            st.markdown("##### Distribuição de Graus")
            st.checkbox("Grau Médio", key="show_avg_degree"); st.checkbox("Grau (Min-Max)", key="show_degree_minmax")
            st.checkbox("Assimetria", key="show_degree_skew"); st.checkbox("Curtose", key="show_degree_kurt")

def about_page():
    st.header("Sobre o Simulador Interativo de Redes Complexas")
    st.markdown("""
    Esta aplicação foi desenvolvida como uma ferramenta educacional e de exploração para os conceitos fundamentais da ciência de redes complexas.
    O projeto foi inspirado e baseado nos conceitos apresentados na **Aula 1 do curso de "Introdução às Redes Complexas, com aplicações, utilizando Python e IA-LLM"**, ministrado na Escola de Artes, Ciências e Humanidades (EACH) da Universidade de São Paulo (USP).

    ### Autor
    - **Elysio Damasceno da Silva Neto**

    ### Tecnologias
    - **Interface:** [Streamlit](https://streamlit.io/)
    - **Análise de Redes:** [NetworkX](https://networkx.org/)
    - **Visualização:** [Plotly](https://plotly.com/python/)
    - **Cálculos:** [NumPy](https://numpy.org/) e [SciPy](https://scipy.org/)

    ### Repositório
    [https://github.com/Edamas/redes_complexas](https://github.com/Edamas/redes_complexas)
    """)

# ============================================================
# NAVEGAÇÃO PRINCIPAL E EXECUÇÃO DA PÁGINA SELECIONADA
# ============================================================
PAGES = {
    "Simulação": simulation_page,
    "Configurações": config_page,
    "Sobre": about_page
}

st.sidebar.header("Navegação")
selection = st.sidebar.radio("Escolha uma página:", list(PAGES.keys()))

# Executar a página selecionada
page = PAGES[selection]
page()