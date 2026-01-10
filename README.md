# 📈 Tech Challenge - Fase 4 - Machine Learning Engineering

[![Python 3.12](https://img.shields.io/badge/Python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-005571?logo=fastapi)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?logo=docker)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow)](https://mlflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🧾 Descrição do Projeto

Este projeto compõe a entrega do **Tech Challenge - Fase 4** do curso de Pós-Graduação em **Machine Learning Engineering** da **FIAP**.

O objetivo central é desenvolver uma solução completa de Deep Learning para a **previsão de preços de ações** (focando no ativo **ITUB4.SA - Itaú Unibanco**). O projeto abrange todo o ciclo de vida de ML:
1.  **Coleta e Preparação:** Extração de dados históricos da B3 via `yfinance`.
2.  **Modelagem:** Treinamento de uma Rede Neural Recorrente (**LSTM**) para séries temporais.
3.  **Experiment Tracking:** Monitoramento de métricas e parâmetros com **MLflow**.
4.  **Deploy:** Disponibilização do modelo através de uma **API RESTful** containerizada com **Docker**.

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
| Linguagem | Python 3.9+ | Padrão da indústria para Data Science e ML. |
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

* Python 3.9 ou superior
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
    A API estará disponível em: `http://localhost:8000/docs`

---

## 🐳 Como Rodar com Docker

Para evitar conflitos de dependências, recomenda-se o uso do Docker.

1.  **Construa a imagem**
    ```bash
    docker build -t tc4-api-invest .
    ```

2.  **Execute o container**
    ```bash
    docker run -p 8000:8000 tc4-api-invest
    ```

A API estará disponível em `http://localhost:8000`.

---

## 🧪 Como Testar os Endpoints no Swagger UI

Acesse `http://localhost:8000/docs` para ver a interface interativa do **Swagger UI**. Ela permite testar todos os endpoints sem precisar de ferramentas externas.

### 1. Health Check (`/health`)
**Finalidade**: Verifica se a API está online e se os artefatos do modelo foram carregados corretamente.  
**Como testar**:
- Clique em **`/health`** → **"Try it out"** → **"Execute"**  
- **Resposta esperada**:  
  ```json
  { "status": "healthy", "model_loaded": true }

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
        "prices": [32.50, 32.80, ..., 29.50]  // ← lista de 60 números
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
        "prices": [32.50, 32.80, ..., 29.50]  // ← lista de 60 números
    }
  ```

5. Clique em **"Execute"**

  - **Resposta esperada**:  
  ```json
    {
        "predicted_close": 29.65,
        "asset": "ITUB4.SA",
        "window_size": 60,
        "last_known_price": 29.50
    }
  ```
---

> 💡 Dica: Você também pode digitar manualmente 60 valores, mas usar /data/latest é mais prático e realista!

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

## 📈 Monitoramento e Observabilidade

* **Treinamento:** Utilizamos o **MLflow** para registrar cada execução do `train.py`. Isso garante que saibamos exatamente quais hiperparâmetros (epochs, time_steps) geraram qual RMSE.
    * *Comando:* `mlflow ui` (para visualizar o dashboard).
---

## ⚠️ Disclaimer

Este projeto tem fins estritamente **educacionais** e acadêmicos. As previsões geradas pelo modelo de Inteligência Artificial **não constituem recomendação de investimento**. O mercado financeiro é volátil e envolve riscos; consulte sempre um profissional qualificado antes de tomar decisões financeiras.

---

## 📬 Contato

* **Nome:** Alex Soares da Silva
* **Curso:** Pós-Tech Data Analytics & Machine Learning Engineering - FIAP