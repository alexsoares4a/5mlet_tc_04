"""
API RESTful para previsão do preço de fechamento da ação ITUB4.SA
usando um modelo LSTM treinado previamente.

Endpoints:
- POST /predict: recebe lista de 60 preços históricos e retorna previsão
- GET /data/latest: obtém os últimos 60 preços reais (para copiar/colar)
- GET /health: verifica se a API está ativa
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import joblib
import yfinance as yf
from tensorflow.keras.models import load_model
import os

# Verificação de artefatos
if not os.path.exists('models/lstm_itau.keras'):
    raise RuntimeError("Modelo 'lstm_itau.keras' não encontrado em ./models/")
if not os.path.exists('models/scaler_itau.pkl'):
    raise RuntimeError("Scaler 'scaler_itau.pkl' não encontrado em ./models/")

# Carregar artefatos
model = load_model('models/lstm_itau.keras')
scaler = joblib.load('models/scaler_itau.pkl')

app = FastAPI(
    title="Previsão de Ação – Itaú Unibanco (ITUB4.SA)",
    description="API para prever o próximo preço de fechamento com base nos últimos 60 dias.",
    version="1.0"
)

class PriceInput(BaseModel):
    prices: List[float]

@app.get("/health", summary="Health Check")
def health():
    """Verifica se a API está ativa e os artefatos foram carregados."""
    return {"status": "healthy", "model_loaded": True}

@app.get("/data/latest", summary="Obter últimos 60 preços reais")
def get_latest_prices():
    """
    Retorna os últimos 60 preços de fechamento da ITUB4.SA como uma lista JSON.
    Ideal para copiar e colar manualmente no endpoint /predict.
    """
    try:
        df = yf.download("ITUB4.SA", period="90d")
        if df.empty:
            raise HTTPException(500, "Falha ao coletar dados da ITUB4.SA.")
        
        close_series = df['Close'].dropna().tail(60)
        if len(close_series) < 60:
            raise HTTPException(500, "Menos de 60 preços disponíveis após limpeza.")
        
        # Garantir lista plana de floats
        prices = close_series.values.flatten().tolist()
        if not all(isinstance(x, (int, float, np.number)) for x in prices):
            raise HTTPException(500, "Dados de preço inválidos.")

        return {
            "asset": "ITUB4.SA",
            "window_size": 60,
            "prices": prices  # Lista pronta para copiar/colar
        }
    except Exception as e:
        raise HTTPException(500, f"Erro ao obter dados: {str(e)}")

@app.post("/predict", summary="Prever com dados fornecidos")
def predict(payload: PriceInput):
    """
    Recebe uma lista de 60 preços de fechamento históricos e retorna a previsão do próximo fechamento.
    """
    prices = payload.prices
    if len(prices) != 60:
        raise HTTPException(400, "Envie exatamente 60 preços (float).")

    try:
        data = np.array(prices).reshape(-1, 1)
        scaled = scaler.transform(data)
        X = scaled.reshape(1, 60, 1)
        pred_scaled = model.predict(X, verbose=0)
        pred_price = scaler.inverse_transform(pred_scaled)[0][0]
        return {
            "predicted_close": round(float(pred_price), 2),
            "asset": "ITUB4.SA",
            "window_size": 60,
            "last_known_price": round(float(prices[-1]), 2)
        }
    except Exception as e:
        raise HTTPException(500, f"Erro na previsão: {str(e)}")

@app.get("/", include_in_schema=False)
def root():
    return {
        "message": "API de previsão de ITUB4.SA está ativa.",
        "endpoints": ["/health", "/data/latest", "/predict"],
        "docs": "/docs"
    }