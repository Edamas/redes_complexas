import streamlit as st

st.set_page_config(layout="wide", page_title="Configurações")

st.sidebar.header("Navegação")
st.header("Configurações da Simulação")
st.write("Ajuste todos os parâmetros da sua simulação aqui. As mudanças serão refletidas na página 'Simulação'.")

# ============================================================
# INICIALIZAÇÃO DO ESTADO DA SESSÃO
# ============================================================
# Usado para garantir que as variáveis existam na primeira execução
# A inicialização completa com valores padrão acontece na página principal (app.py)

def init_session_var(key, value):
    if key not in st.session_state:
        st.session_state[key] = value

# ============================================================
# CONTROLES DE SIMULAÇÃO
# ============================================================

# --- 1. Modelo da Rede ---
st.subheader("1. Modelo da Rede")
st.info("""
**Erdos-Renyi (ER):** Um modelo fundamental onde cada par de nós tem uma probabilidade uniforme de se conectar. Gera redes homogêneas e é um bom ponto de partida para comparações.

**Watts-Strogatz (WS):** Um modelo que gera redes do tipo 'Mundo Pequeno' (Small-World). Começa com uma rede regular (um anel onde cada nó se conecta a `k` vizinhos) e depois 'reconecta' aleatoriamente algumas arestas. O resultado é uma rede com alta clusterização local (típica de redes regulares) e baixo caminho médio global (típico de redes aleatórias).
""")
model_type = st.radio(
    "Escolha o modelo de geração da rede:",
    ("Erdos-Renyi", "Watts-Strogatz"),
    key='model_type',
    horizontal=True
)

# --- 2. Parâmetros do Modelo ---
st.subheader("2. Parâmetros do Modelo")
col1, col2 = st.columns(2)

with col1:
    if st.session_state.model_type == "Erdos-Renyi":
        st.slider("Número de Nós (N)", 5, 1000, 100, 5, key="N")
        st.slider("Probabilidade de Conexão (p)", 0.0, 1.0, 0.1, 0.01, key="P_ER")
    else: # Watts-Strogatz
        st.slider("Número de Nós (eNG)", 5, 1000, 20, 5, key="N")
        st.slider("Nº de Vizinhos Próximos (k)", 2, 20, 4, 2, help="Cada nó se conecta aos 'k' vizinhos mais próximos no anel inicial. Deve ser um número par.", key="K_WS")
        st.slider("Probabilidade de Reconexão (ePG)", 0.0, 1.0, 0.2, 0.01, help="Probabilidade de 'religar' uma aresta, introduzindo aleatoriedade.", key="P_WS")

# --- 3. Parâmetros de Visualização ---
with col2:
    st.selectbox("Dimensão", ("3D", "2D"), key="layout_dim")
    st.selectbox(
        "Algoritmo de Layout",
        ("Spring (Física)", "Circular/Esférico", "Shell", "Aleatório"),
        key="layout_type"
    )
    if st.session_state.layout_type in ["Circular/Esférico", "Shell"] and st.session_state.layout_dim == "3D":
        st.selectbox("Distribuição", ("Superfície", "Volume"), key="layout_dist")
    else:
        init_session_var('layout_dist', 'Superfície')


# --- 4. Aparência ---
st.subheader("3. Aparência da Rede")
acol1, acol2 = st.columns(2)
with acol1:
    st.markdown("##### Aparência dos Nós")
    cc1, cc2 = st.columns(2)
    cc1.color_picker("Cor", value="#1f77b4", key="NODE_COLOR")
    cc2.slider("Opacidade", 0.0, 1.0, 0.9, 0.05, key="NODE_OPACITY")
    st.slider("Tamanho do Nó", 1, 30, 8, 1, key="NODE_SIZE")
with acol2:
    st.markdown("##### Aparência das Arestas")
    cc3, cc4 = st.columns(2)
    cc3.color_picker("Cor", value="#888888", key="EDGE_COLOR")
    cc4.slider("Opacidade", 0.0, 1.0, 0.5, 0.05, key="EDGE_OPACITY")
    st.slider("Largura da Aresta", 1, 10, 2, 1, key="EDGE_WIDTH")

# --- 5. Rótulos ---
st.subheader("4. Rótulos dos Nós")
st.checkbox("Exibir Rótulos", value=True, key="SHOW_NODE_NAMES")
if st.session_state.get('SHOW_NODE_NAMES', True):
    rcol1, rcol2, rcol3 = st.columns(3)
    rcol1.slider("Tamanho da Fonte", 5, 20, 10, 1, key="NODE_FONT_SIZE")
    rcol2.text_input("Prefixo", value="Nó ", key="NODE_PREFIX")
    rcol3.checkbox("Autonumerar", value=True, key="AUTONUMBER_NODES")

# --- 6. Métricas a Exibir ---
st.subheader("5. Métricas a Exibir na Simulação")
st.info("Marque as métricas que você deseja visualizar na página 'Simulação'.")

exp_metrics = st.expander("Selecionar Métricas", expanded=False)
with exp_metrics:
    mcol1, mcol2, mcol3 = st.columns(3)
    with mcol1:
        st.markdown("##### Conectividade")
        st.checkbox("Nº de Nós", value=True, key="show_n_nodes")
        st.checkbox("Nº de Arestas", value=True, key="show_n_edges")
        st.checkbox("Densidade da Rede", value=True, key="show_density")
        st.checkbox("Caminho Mínimo Médio (L)", value=True, key="show_L")
        
        st.markdown("##### Distribuição de Graus")
        st.checkbox("Grau Médio", value=True, key="show_avg_degree")
        st.checkbox("Grau (Min-Max)", value=True, key="show_degree_minmax")
        st.checkbox("Assimetria do Grau", value=True, key="show_degree_skew")
        st.checkbox("Curtose do Grau", value=True, key="show_degree_kurt")

    with mcol2:
        st.markdown("##### Agrupamento")
        st.checkbox("Coef. de Aglomeração Médio (C)", value=True, key="show_C")

    with mcol3:
        st.markdown("##### Centralidade (Em breve)")
        st.markdown("##### Espectro (Em breve)")
        st.markdown("##### Resiliência (Em breve)")

# Workaround to force rerun when model changes
if 'previous_model_type' not in st.session_state:
    st.session_state.previous_model_type = model_type
if st.session_state.previous_model_type != model_type:
    st.session_state.previous_model_type = model_type
    st.experimental_rerun()
