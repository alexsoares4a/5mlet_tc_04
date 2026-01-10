# train.py
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import joblib
import os 
import mlflow
import mlflow.keras

# Garantir que a pasta 'models' exista
os.makedirs('models', exist_ok=True) 

# -----------------------------
# 1. Coleta de dados (10 anos)
# -----------------------------
print("Coletando dados de ITUB4.SA...")
df = yf.download('ITUB4.SA', start='2014-01-01', end='2025-07-20')
df = df[['Close']].dropna()
print(f"Dados coletados: {len(df)} dias")

# -----------------------------
# 2. Pré-processamento
# -----------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Função para criar sequências
def create_sequences(data, seq_len=60):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len, 0])  # prever próximo Close
    return np.array(X), np.array(y)

SEQ_LENGTH = 60
X, y = create_sequences(scaled_data, SEQ_LENGTH)

# Divisão treino/teste (80/20, sem shuffle!)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# -----------------------------
# 3. Modelo LSTM
# -----------------------------
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
print("Treinando o modelo...")
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=20,
    validation_split=0.1,
    verbose=1
)

# -----------------------------
# 4. Avaliação + MLflow
# -----------------------------
# Iniciar experimento
mlflow.set_experiment("ITUB4_SA_LSTM")

with mlflow.start_run(run_name="lstm_itau_60d"):
    y_pred = model.predict(X_test, verbose=0)
    
    # Reverter escala
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_inv = scaler.inverse_transform(y_pred).flatten()

    # Métricas
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    r2 = r2_score(y_test_inv, y_pred_inv)
    mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv)

    # Logar métricas
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("R2", r2)
    mlflow.log_metric("MAPE", mape)

    # Logar parâmetros
    mlflow.log_param("seq_length", SEQ_LENGTH)
    mlflow.log_param("lstm_units", 50)
    mlflow.log_param("dropout", 0.2)
    mlflow.log_param("epochs", 20)
    mlflow.log_param("batch_size", 32)

    # Salvar modelo no formato MLflow
    mlflow.keras.log_model(model, "model")

    print("\n=== Métricas no conjunto de teste ===")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f} ({r2*100:.2f}%)")
    print(f"MAPE: {mape:.2f}%")

# -----------------------------
# 5. Salvamento
# -----------------------------
model.save('models/lstm_itau.keras')
joblib.dump(scaler, 'models/scaler_itau.pkl')
print("\nModelo e scaler salvos em ./models/")