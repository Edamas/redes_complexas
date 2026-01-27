# Simulador de Redes Complexas com Streamlit

Uma aplicação web interativa construída com Streamlit para gerar, visualizar e analisar redes complexas em 3D. O projeto foi inspirado e baseado nos conceitos apresentados na Aula 1 do curso de "Introdução às Redes Complexas, com aplicações, utilizando Python e IA-LLM" da EACH-USP.

![Captura de Tela da Aplicação](https://i.imgur.com/YOUR_IMAGE_URL.png) 
*(Depois de subir o projeto, você pode tirar uma captura de tela da aplicação, fazer o upload em um site como o [Imgur](https://imgur.com/upload) e colar o link aqui para ter uma bela pré-visualização).*

## Funcionalidades

*   **Geração de Redes:** Crie redes do tipo Erdos-Renyi ajustando o número de nós e a probabilidade de conexão.
*   **Visualização 3D:** Explore a topologia da rede em um gráfico 3D interativo.
*   **Análise de Graus:** Visualize a distribuição de graus da rede em um histograma.
*   **Customização:** Ajuste em tempo real parâmetros visuais como cores, tamanhos, opacidade e nomes dos nós.
*   **Análise de Dados:** Exiba e explore os dados de cada nó (grau, posição) em uma tabela.
*   **Estatísticas:** Veja as principais métricas da distribuição de graus (média, variância, assimetria, etc.).

## Instalação

1.  **Clone o repositório:**
    ```bash
    git clone <URL_DO_SEU_REPOSITÓRIO_AQUI>
    cd simulador-redes-complexas-streamlit
    ```

2.  **Crie e ative um ambiente virtual (recomendado):**
    ```bash
    # Para Windows
    python -m venv venv
    .\venv\Scripts\activate

    # Para macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

## Como Executar

Com o ambiente virtual ativado e as dependências instaladas, execute o seguinte comando:

```bash
streamlit run "Aula 1 - 26-01-2026/redes_complexas_1.py"
```

A aplicação será aberta automaticamente no seu navegador.

## Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.
