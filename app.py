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
    """
    Define os valores padrão para o dicionário 'configuracoes' e garante que todas as chaves existam.
    """
    defaults = {
        'model_type': 'Erdos-Renyi', 'N': 100, 'P_ER': 0.1, 'K_WS': 4, 'P_WS': 0.2,
        'layout_dim': '3D', 'layout_type': 'Spring (Física)', 'layout_dist': 'Superfície',
        'NODE_COLOR': '#1f77b4', 'NODE_OPACITY': 0.9, 'NODE_SIZE': 8,
        'EDGE_COLOR': '#888888', 'EDGE_OPACITY': 0.5, 'EDGE_WIDTH': 2,
        'SHOW_NODE_NAMES': True, 'NODE_FONT_SIZE': 10, 'NODE_PREFIX': 'Nó ', 'AUTONUMBER_NODES': True,
        'show_hist': True, 'hist_bins': 20,
        'show_adjacency': True,
        'show_degree_rank': True, 'degree_rank_plot_type': 'Scatter',
        'show_n_nodes': True, 'show_n_edges': True, 'show_density': True, 'show_L': True,
        'show_C': True, 'show_avg_degree': True, 'show_degree_minmax': True,
        'show_degree_skew': True, 'show_degree_kurt': True,
        'show_data_frame': True,
    }
    
    if 'configuracoes' not in st.session_state:
        st.session_state.configuracoes = defaults.copy()
    else:
        for key, value in defaults.items():
            if key not in st.session_state.configuracoes:
                st.session_state.configuracoes[key] = value

initialize_session_state()

# ============================================================
# FUNÇÕES DE GERAÇÃO E ANÁLISE (Compartilhadas)
# ============================================================
def generate_graph(configs):
    if configs['model_type'] == "Erdos-Renyi": return nx.erdos_renyi_graph(configs['N'], configs['P_ER'])
    elif configs['model_type'] == "Watts-Strogatz":
        k_ws_safe = max(2, configs['K_WS'] if configs['K_WS'] % 2 == 0 else configs['K_WS'] - 1)
        if k_ws_safe >= configs['N']: k_ws_safe = max(2, configs['N'] - 2 if configs['N'] > 2 else 0)
        return nx.watts_strogatz_graph(configs['N'], k_ws_safe, configs['P_WS'])
    return nx.Graph()

def generate_layout(_G, configs):
    n=len(_G.nodes); d=int(configs['layout_dim'][0]);
    if n==0: return {}
    l_type, l_dist = configs['layout_type'], configs.get('layout_dist', 'Superfície')
    if l_type=="Aleatório": return nx.random_layout(_G,dim=d)
    if l_type=="Circular/Esférico":
        if configs['layout_dim']=="3D":
            if l_dist=="Volume": vec=np.random.randn(3,n); vec/=np.linalg.norm(vec,axis=0); r=np.random.rand(n)**(1/3); pos_arr=(vec * r).T; return {i: pos_arr[i] for i in range(n)}
            else: return nx.spherical_layout(_G)
        else: return nx.circular_layout(_G)
    if l_type=="Shell":
        n_s=int(np.ceil(np.sqrt(n/4))); shells_list=[list(range(sum(2**i for i in range(j)),sum(2**i for i in range(j+1)))) for j in range(n_s)]
        if shells_list and shells_list[-1]: shells_list[-1].extend(range(max(shells_list[-1])+1,n))
        else: shells_list=[list(range(n))]
        return nx.shell_layout(_G,nlist=shells_list)
    if l_type=="Spring (Física)": return nx.spring_layout(_G,dim=d)
    return nx.random_layout(_G,dim=d)

def draw_network(G, pos, configs):
    dim = configs['layout_dim']
    if dim=="3D":
        ex,ey,ez,nx_pos,ny_pos,nz_pos = [[] for _ in range(6)]
        for u,v in G.edges(): ex.extend([pos[u][0],pos[v][0],None]); ey.extend([pos[u][1],pos[v][1],None]); ez.extend([pos[u][2],pos[v][2],None])
        for node in G.nodes(): nx_pos.append(pos[node][0]); ny_pos.append(pos[node][1]); nz_pos.append(pos[node][2])
        et=go.Scatter3d(x=ex,y=ey,z=ez,mode='lines',line=dict(color=configs['EDGE_COLOR'],width=configs['EDGE_WIDTH']),opacity=configs['EDGE_OPACITY'],hoverinfo='none')
        nt=go.Scatter3d(x=nx_pos,y=ny_pos,z=nz_pos,mode='markers' if not configs['SHOW_NODE_NAMES'] else 'markers+text',marker=dict(size=configs['NODE_SIZE'],color=configs['NODE_COLOR'],opacity=configs['NODE_OPACITY']),hoverinfo='text')
    else:
        ex,ey,nx_pos,ny_pos = [[] for _ in range(4)]
        for u,v in G.edges(): ex.extend([pos[u][0],pos[v][0],None]); ey.extend([pos[u][1],pos[v][1],None])
        for node in G.nodes(): nx_pos.append(pos[node][0]); ny_pos.append(pos[node][1])
        et=go.Scatter(x=ex,y=ey,mode='lines',line=dict(color=configs['EDGE_COLOR'],width=configs['EDGE_WIDTH']),opacity=configs['EDGE_OPACITY'],hoverinfo='none')
        nt=go.Scatter(x=nx_pos,y=ny_pos,mode='markers' if not configs['SHOW_NODE_NAMES'] else 'markers+text',marker=dict(size=configs['NODE_SIZE'],color=configs['NODE_COLOR'],opacity=configs['NODE_OPACITY']),hoverinfo='text')
    if configs['SHOW_NODE_NAMES']: nt.text=[f"{configs['NODE_PREFIX']}{i}" for i in G.nodes()] if configs['AUTONUMBER_NODES'] else [configs['NODE_PREFIX']]*len(G.nodes()); nt.textfont=dict(size=configs['NODE_FONT_SIZE']); nt.hovertext=[f"Nó: {n}<br>Grau: {d}" for n,d in G.degree()]
    fig=go.Figure(data=[et,nt]); fig.update_layout(showlegend=False,margin=dict(l=0,r=0,t=0,b=0));
    if dim=="3D": fig.update_scenes(aspectmode='cube')
    else: fig.update_yaxes(scaleanchor="x",scaleratio=1)
    fig.update_xaxes(showticklabels=False,showgrid=False,zeroline=False,visible=False); fig.update_yaxes(showticklabels=False,showgrid=False,zeroline=False,visible=False)
    return fig

# ============================================================
# DEFINIÇÃO DAS PÁGINAS (COMO FUNÇÕES)
# ============================================================

def simulation_page():
    st.title("Simulação e Análise da Rede")
    configs = st.session_state.configuracoes
    G = generate_graph(configs)
    pos = generate_layout(G, configs)
    degrees = [d for n, d in G.degree()]

    st.header("Visualização da Rede")
    main_vis_cols = st.columns([3, 1] if configs['show_hist'] else [1])
    with main_vis_cols[0]:
        st.subheader(f"Modelo: {configs['model_type']} ({configs['layout_dim']}, {configs['layout_type']})")
        fig_net = draw_network(G, pos, configs)
        st.plotly_chart(fig_net, use_container_width=True)
    if configs['show_hist'] and len(main_vis_cols) > 1:
        with main_vis_cols[1]:
            st.subheader("Distribuição de Graus")
            fig_hist = go.Figure(go.Histogram(x=degrees, nbinsx=configs['hist_bins'], marker_color=configs['NODE_COLOR']));
            fig_hist.update_layout(margin=dict(l=10, r=10, t=20, b=20)); st.plotly_chart(fig_hist, use_container_width=True)
    
    plots_to_show = [p for p in ['adjacency', 'degree_rank'] if configs[f'show_{p}']]
    if plots_to_show:
        st.markdown("---"); st.header("Análises Visuais Adicionais")
        add_vis_cols = st.columns(len(plots_to_show))
        col_idx = 0
        if 'adjacency' in plots_to_show:
            with add_vis_cols[col_idx]:
                st.subheader("Matriz de Adjacência"); 
                if G.number_of_nodes()>150: st.info("Matriz não exibida para >150 nós.")
                elif G.number_of_nodes()>0: st.plotly_chart(go.Figure(go.Heatmap(z=nx.to_numpy_array(G),colorscale='Viridis')).update_layout(yaxis_autorange='reversed'),use_container_width=True)
                else: st.info("Rede vazia.")
            col_idx+=1
        if 'degree_rank' in plots_to_show:
            with add_vis_cols[col_idx]:
                st.subheader("Gráfico de Grau por Nó"); 
                if degrees:
                    trace_map={'Scatter':go.Scatter(mode='markers'),'Linhas':go.Scatter(mode='lines'),'Área':go.Scatter(fill='tozeroy'),'Barras':go.Bar()}
                    trace=trace_map[configs['degree_rank_plot_type']]; trace.x=list(range(len(degrees))); trace.y=sorted(degrees,reverse=True); trace.marker.color=configs['NODE_COLOR']
                    st.plotly_chart(go.Figure(trace).update_layout(xaxis_title="Rank do Nó",yaxis_title="Grau"),use_container_width=True)
                else: st.info("Rede vazia.")

    st.markdown("---"); st.header("Análise da Rede")
    avg_degree = np.mean(degrees) if degrees else 0
    if G.number_of_nodes()>2 and G.number_of_edges()>0 and avg_degree>0:
        is_connected = nx.is_connected(G); C = nx.average_clustering(G)
        try:
            if is_connected:
                L = nx.average_shortest_path_length(G)
                if avg_degree > 1:
                    L_rand = np.log(configs['N'])/np.log(avg_degree); C_rand = avg_degree/configs['N']
                    if C > C_rand * 2 and L < L_rand * 2: st.success("✅ **Diagnóstico:** A rede exibe fortes características de **Mundo Pequeno (Small-World)**.")
                    else: st.info("ℹ️ **Diagnóstico:** A rede é conectada, mas não se classifica como 'Mundo Pequeno'.")
                else: st.info("ℹ️ **Diagnóstico:** A rede é conectada, mas trivial (como uma linha).")
            else: st.warning("⚠️ **Diagnóstico:** A rede está **desconexa**.")
        except (ZeroDivisionError, ValueError): st.error("❌ **Diagnóstico:** Não foi possível analisar as propriedades de Mundo Pequeno.")
    else: st.error("❌ **Diagnóstico:** Rede vazia ou trivial demais para análise.")

    st.subheader("Métricas Detalhadas")
    if any(configs.get(k) for k in ['show_n_nodes', 'show_n_edges', 'show_density', 'show_L']):
        st.markdown("##### Conectividade")
        c_cols = st.columns(4)
        if configs['show_n_nodes']: c_cols[0].metric("Nós",G.number_of_nodes())
        if configs['show_n_edges']: c_cols[1].metric("Arestas",G.number_of_edges())
        if configs['show_density']: c_cols[2].metric("Densidade",f"{nx.density(G):.4f}")
        if configs['show_L']: L=nx.average_shortest_path_length(G) if is_connected else float('inf'); c_cols[3].metric("Caminho Médio (L)",f"{L:.4f}" if is_connected else "∞")
    if configs['show_C']:
        st.markdown("##### Agrupamento"); st.metric("Aglomeração (C)",f"{nx.average_clustering(G) if G.nodes() else 0.0:.4f}")
    if any(configs.get(k) for k in ['show_avg_degree', 'show_degree_minmax', 'show_degree_skew', 'show_degree_kurt']):
        st.markdown("##### Distribuição de Graus")
        if degrees:
            desc_stats = stats.describe(degrees)
            d_cols = st.columns(4)
            if configs['show_avg_degree']: d_cols[0].metric("Grau Médio",f"{desc_stats.mean:.2f}")
            if configs['show_degree_minmax']: d_cols[1].metric("Grau (Min-Max)",f"{desc_stats.minmax[0]}-{desc_stats.minmax[1]}")
            if configs['show_degree_skew']: d_cols[2].metric("Assimetria",f"{desc_stats.skewness:.2f}")
            if configs['show_degree_kurt']: d_cols[3].metric("Curtose",f"{desc_stats.kurtosis:.2f}")
    if configs['show_data_frame']:
        st.subheader("Dados Detalhados dos Nós")
        if G.number_of_nodes()>0: st.dataframe(pd.DataFrame({"Grau":dict(G.degree()),"Coef. Aglomeração":nx.clustering(G)}).rename_axis("ID do Nó"),use_container_width=True)
        else: st.warning("Rede vazia.")

def config_page():
    st.header("Configurações da Simulação")
    c = st.session_state.configuracoes.copy()
    col_main, col_json = st.columns([2, 1])
    with col_main:
        if st.button("✅ Aplicar Configurações",type="primary",use_container_width=True): st.session_state.configuracoes=c; st.success("Configurações aplicadas!")
        tabs=st.tabs(["Modelo","Gráfico da Rede","Histograma","Matriz de Adjacência","Gráfico Grau por Nó","Métricas","Tabela de Dados"])
        with tabs[0]:
            c['model_type']=st.radio("Modelo:",("Erdos-Renyi","Watts-Strogatz"),index=0 if c['model_type']=="Erdos-Renyi" else 1,horizontal=True, key="config_model_type")
            if c['model_type']=="Erdos-Renyi": c['N']=st.slider("Nós",5,1000,c['N'],key="config_N_er"); c['P_ER']=st.slider("Prob. Conexão",0.0,1.0,c['P_ER'],step=0.01,key="config_P_ER")
            else: c['N']=st.slider("Nós",5,1000,c['N'],key="config_N_ws"); c['K_WS']=st.slider("Vizinhos",2,20,c['K_WS'],step=2,key="config_K_WS"); c['P_WS']=st.slider("Prob. Reconexão",0.0,1.0,c['P_WS'],step=0.01,key="config_P_WS")
        with tabs[1]:
            c['layout_dim']=st.selectbox("Dimensão",("3D","2D"),index=["3D","2D"].index(c['layout_dim'])); c['layout_type']=st.selectbox("Layout",("Spring (Física)","Circular/Esférico","Shell","Aleatório"),index=["Spring (Física)","Circular/Esférico","Shell","Aleatório"].index(c['layout_type']))
            if c['layout_type'] in ["Circular/Esférico","Shell"] and c['layout_dim']=="3D": c['layout_dist']=st.selectbox("Distribuição",("Superfície","Volume"),index=["Superfície","Volume"].index(c['layout_dist']))
            else: c['layout_dist']="Superfície"
            ac1,ac2=st.columns(2)
            with ac1: c['NODE_COLOR']=st.color_picker("Cor Nó",c['NODE_COLOR']); c['NODE_OPACITY']=st.slider("Opacidade Nó",0.0,1.0,c['NODE_OPACITY']); c['NODE_SIZE']=st.slider("Tamanho Nó",1,30,c['NODE_SIZE'])
            with ac2: c['EDGE_COLOR']=st.color_picker("Cor Aresta",c['EDGE_COLOR']); c['EDGE_OPACITY']=st.slider("Opacidade Aresta",0.0,1.0,c['EDGE_OPACITY']); c['EDGE_WIDTH']=st.slider("Largura Aresta",1,10,c['EDGE_WIDTH'])
            c['SHOW_NODE_NAMES']=st.checkbox("Exibir Rótulos",c['SHOW_NODE_NAMES'])
            if c['SHOW_NODE_NAMES']: r1,r2,r3=st.columns(3); c['NODE_FONT_SIZE']=r1.slider("Fonte",5,20,c['NODE_FONT_SIZE']); c['NODE_PREFIX']=r2.text_input("Prefixo",c['NODE_PREFIX']); c['AUTONUMBER_NODES']=r3.checkbox("Autonumerar",c['AUTONUMBER_NODES'])
        with tabs[2]: c['show_hist']=st.checkbox("Exibir Histograma",c['show_hist'],key="cfg_show_hist"); c['hist_bins']=st.slider("Nº de Bins",5,50,c['hist_bins'],key="cfg_hist_bins")
        with tabs[3]: c['show_adjacency']=st.checkbox("Exibir Matriz de Adjacência",c['show_adjacency'],key="cfg_show_adj")
        with tabs[4]: c['show_degree_rank']=st.checkbox("Exibir Gráfico de Grau",c['show_degree_rank'],key="cfg_show_rank"); c['degree_rank_plot_type']=st.selectbox("Tipo de Gráfico",('Scatter','Barras','Linhas','Área'),index=['Scatter','Barras','Linhas','Área'].index(c['degree_rank_plot_type']),key="cfg_rank_type")
        with tabs[5]:
            c['show_n_nodes']=st.checkbox("Nº de Nós",c['show_n_nodes']); c['show_n_edges']=st.checkbox("Nº de Arestas",c['show_n_edges']); c['show_density']=st.checkbox("Densidade",c['show_density']); c['show_L']=st.checkbox("Caminho Médio (L)",c['show_L']); c['show_C']=st.checkbox("Aglomeração (C)",c['show_C']); c['show_avg_degree']=st.checkbox("Grau Médio",c['show_avg_degree']); c['show_degree_minmax']=st.checkbox("Grau (Min-Max)",c['show_degree_minmax']); c['show_degree_skew']=st.checkbox("Assimetria",c['show_degree_skew']); c['show_degree_kurt']=st.checkbox("Curtose",c['show_degree_kurt'])
        with tabs[6]: c['show_data_frame']=st.checkbox("Exibir Tabela de Dados",c['show_data_frame'],key="cfg_show_df")
    with col_json: st.subheader("Configs Atuais"); st.json(st.session_state.configuracoes)

def about_page():
    st.header("Sobre o Simulador"); st.markdown("""Ferramenta educacional para explorar conceitos de redes complexas. **Autor:** Elysio D. S. Neto. **Tecnologias:** Streamlit, NetworkX, Plotly. **Repo:** [GitHub](https://github.com/Edamas/redes_complexas)""")

# ============================================================
# NAVEGADOR PRINCIPAL
# ============================================================
pg = st.navigation([
    st.Page(config_page, title="Configurações", icon="⚙️", default=True),
    st.Page(simulation_page, title="Simulação", icon="📈"),
    st.Page(about_page, title="Sobre", icon="ℹ️"),
])
pg.run()