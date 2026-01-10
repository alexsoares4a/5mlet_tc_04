# 📈 Tech Challenge - Fase 4 - Machine Learning Engineering

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-005571?logo=fastapi)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?logo=docker)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🧾 Descrição do Projeto

Este projeto é parte do **Tech Challenge - Fase 4** do curso de Pós-Graduação em **Machine Learning Engineering** da **Faculdade de Informática e Administração Paulista - FIAP**.

O objetivo principal é desenvolver um **modelo preditivo baseado em LSTM (Long Short-Term Memory)** para prever o preço de fechamento diário da ação **Itaú Unibanco (ITUB4.SA)** listada na B3, e disponibilizá-lo via **API RESTful** com **containerização em Docker**.

O modelo foi treinado exclusivamente com a série histórica de preços de fechamento (**abordagem univariada**), obtendo desempenho robusto sem depender de variáveis exógenas. A solução segue boas práticas de **MLOps**, com separação clara entre experimentação (notebook) e produção (`train.py` + `api.py`).

A arquitetura foi desenhada para ser simples, robusta e reprodutível, garantindo que o modelo treinado possa ser consumido em produção com facilidade.

## ✨ Funcionalidades Principais

* **Coleta Automática:** Integração com Yahoo Finance para download de dados históricos atualizados.
* **Deep Learning (LSTM):** Modelo de arquitetura Long Short-Term Memory, ideal para capturar padrões sequenciais em séries temporais.
* **API RESTful:** Endpoints rápidos (FastAPI) para inferência de preços futuros.
* **Experiment Tracking:** Rastreamento completo de execuções de treino, métricas (RMSE) e artefatos via MLflow.
* **Dockerizado:** Aplicação empacotada em container para rodar em qualquer ambiente.

## 🧩 Tecnologias Utilizadas

| Componente | Tecnologia | Motivação |
|:----------|:-----------|:----------|
| Linguagem | Python 3.12 | Padrão da indústria para Data Science e ML. |
| API Framework | FastAPI | Alta performance, validação automática de dados (Pydantic) e documentação nativa (Swagger). |
| Deep Learning | TensorFlow / Keras | Biblioteca robusta e escalável para construção e treinamento de redes neurais complexas. |
| Rastreamento | MLflow | Gerenciamento do ciclo de vida de ML (parâmetros, métricas e versionamento de modelos). |
| Coleta de Dados | yfinance | Acesso simplificado e gratuito aos dados históricos do mercado financeiro. |
| Processamento | Pandas / Scikit-learn | Manipulação de séries temporais e normalização de dados (MinMaxScaler). |
| Container | Docker | Isolamento da aplicação e garantia de reprodutibilidade em diferentes ambientes. |
| Servidor Web | Uvicorn | Servidor ASGI leve e rápido para produção. |

---

## 🛠️ Como Rodar a Aplicação Localmente

### 🔧 Requisitos

* Python 3.12 ou superior
* `pip` (gerenciador de pacotes)
* Docker (opcional, para rodar via container)

### 📦 Passo a passo (Modo Tradicional)

1.  **Clone o repositório**
    ```bash
    git clone https://github.com/seu-usuario/tech-challenge-fase4.git
    cd tech-challenge-fase4
    ```

2.  **Instale as dependências**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Treine o Modelo (Geração de Artefatos)**
    Antes de subir a API, é necessário treinar o modelo e gerar os arquivos `.keras` e `.pkl`.
    ```bash
    python src/train.py
    ```
    > *Isso criará a pasta `models/` com os arquivos necessários e a pasta `mlruns/` com os logs do MLflow.*

4.  **Inicie a API**
    ```bash
    uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
    ```

5.  **Acesse a documentação**

    A API estará disponível em http://localhost:8000.  
    A documentação (Swagger UI) está em http://localhost/docs.

    > *Optando pela execução tradicional você pode partir para a seção '**Como Testar os Endpoints no Swagger UI**'*
---


## 🐳 Como Rodar com Docker

Para evitar conflitos de dependências, recomenda-se o uso do Docker.

1.  **Construa a imagem**
    ```bash
    docker build -t lstm-itau-api .
    ```

2.  **Execute o container**
    ```bash
    docker run -p 8000:8000 lstm-itau-api
    ```

3.  **Acesse a documentação**

    A API estará disponível em http://localhost:8000.  
    A documentação (Swagger UI) está em http://localhost/docs.

---

## 📊 Monitoramento com MLflow

O treinamento do modelo é registrado automaticamente no **MLflow**, uma plataforma open-source para rastreamento de experimentos de Machine Learning. Isso permite acompanhar métricas (MAE, RMSE, R², MAPE), hiperparâmetros e artefatos gerados.

### Como visualizar os resultados

1. **Treine o modelo** (se ainda não fez):
   ```bash
   python src/train.py
   ```
   > *Isso gera a pasta `mlruns/` na raiz do projeto.*

2. **Inicie o MLflow UI**:
   ```bash
   mlflow ui
   ```
3. **Acesse a interface no navegador**:
   
   > Por padrão, o servidor roda em http://localhost:5000

4. **Navegue pelos experimentos:**:
- Clique no experimento `ITUB4_SA_LSTM`;
- Selecione a run mais recente (ex: `lstm_itau_60d`);
- Na aba **"Metrics"**, veja os valores de:
    - `MAE`: Erro absoluto médio (R$)
    - `RMSE`: Raiz do erro quadrático médio (R$)
    - `R2`: Coeficiente de determinação (quanto mais próximo de 1, melhor)
    - `MAPE`: Erro percentual médio absoluto (%)
- Na aba **"Params"**, confira os hiperparâmetros usados (janela de 60 dias, dropout=0.2, etc.);
- Na aba **"Artifacts"**, você verá o modelo salvo (`model/`).

>💡 Dica: Mantenha o terminal com `mlflow ui` aberto enquanto testa diferentes versões do modelo - ele atualiza automaticamente!

---

## 🧪 Como Testar os Endpoints no Swagger UI

Acesse `http://localhost:8000/docs` para ver a interface interativa do **Swagger UI**. Ela permite testar todos os endpoints sem precisar de ferramentas externas.

### 1. Health Check (`/health`)
**Finalidade**: Verifica se a API está online e se os artefatos do modelo foram carregados corretamente.  
**Como testar**:
- Clique em **`/health`** → **"Try it out"** → **"Execute"**  
- **Resposta esperada**:  
  ```json
    { 
        "status": "healthy", 
        "model_loaded": true 
    }

---

### 2. Obter Dados Reais (`/data/latest`)

**Finalidade**: Fornece automaticamente os últimos 60 preços de fechamento reais da ação ITUB4.SA, facilitando testes manuais.
**Como testar**:
- Clique em `/data/latest` → **"Try it out"** → **"Execute"** 
- **Resposta esperada**:  
  ```json
    {
        "asset": "ITUB4.SA",
        "window_size": 60,
        "prices": [
            34.20015335083008,
            34.346893310546875,
            34.346893310546875,
            34.26435089111328, 
            ...,
            39.849998474121094
        ] // ← lista de 60 números  
    }

---

### 3. Fazer Previsão (`/predict`)

**Finalidade**: Recebe uma lista de 60 preços históricos e retorna a previsão do próximo fechamento.
**Como testar (usando os dados reais)**:

1. Execute primeiro o endpoint `/data/latest` (passo 2 acima);
2. Copie o resultado da resposta;
3. Vá para `/predict` → **"Try it out"**;
4. Cole o resultado da resposta no corpo da requisição:

  ```json
    {
        "asset": "ITUB4.SA",
        "window_size": 60,
        "prices": [
            34.20015335083008,
            34.346893310546875,
            34.346893310546875,
            34.26435089111328, 
            ...,
            39.849998474121094
        ] // ← lista de 60 números  
    }
  ```

5. Clique em **"Execute"**

  - **Resposta esperada**:  
  ```json
    {
        "predicted_close": 38.35,
        "asset": "ITUB4.SA",
        "window_size": 60,
        "last_known_price": 39.85
    }
  ```
---

> 💡 Dica: Você também pode digitar manualmente 60 valores, mas usar `/data/latest` é mais prático e realista!

---

## 📁 Estrutura de Pastas e Módulos

```
5mlet_tc_04/
├── src/
│   ├── train.py         # Script de treinamento com MLflow
│   └── api.py           # API FastAPI com endpoints /predict, /data/latest, /health
├── models/              # Artefatos gerados pelo train.py
│   ├── lstm_itau.keras
│   └── scaler_itau.pkl
├── video/
│   └── previsao_itau.mp4 # Vídeo Explicativo
├── 5mlet_tc_04.ipynb    # Análise exploratória e modelagem (relatório técnico)
├── requirements.txt     # Dependências do projeto
├── Dockerfile           # Imagem Docker para produção
└── README.md
```

---

## ⚠️ Disclaimer

Este projeto tem fins estritamente **educacionais** e acadêmicos. As previsões geradas pelo modelo de Inteligência Artificial **não constituem recomendação de investimento**. O mercado financeiro é volátil e envolve riscos; consulte sempre um profissional qualificado antes de tomar decisões financeiras.

---

## 📬 Contato

* **Nome:** Alex Soares da Silva
* **Curso:** Pós-Tech Data Analytics & Machine Learning Engineering - FIAP