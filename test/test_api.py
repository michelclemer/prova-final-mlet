"""
Testes para a API de previsão.
"""
import os
import pytest
from fastapi.testclient import TestClient
import joblib
import numpy as np
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler

# Importar a aplicação FastAPI
from src.api.main import app

# Criar um cliente de teste
client = TestClient(app)


@pytest.fixture
def setup_test_environment():
    """
    Configura o ambiente de teste com um modelo e scaler fictícios.
    """
    # Criar diretórios necessários
    os.makedirs("models", exist_ok=True)

    # Criar um modelo simples para teste
    model = Sequential()
    model.add(Dense(10, input_shape=(60, 1), activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # Salvar o modelo
    model_path = "models/test_model.h5"
    save_model(model, model_path)

    # Criar e salvar um scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(np.array([[0], [100]]))  # Ajustar para escala de preços

    scaler_path = "models/scaler.pkl"
    joblib.dump(scaler, scaler_path)

    # Configurar o app para usar os caminhos de teste
    app.state.model = model
    app.state.scaler = scaler

    yield

    # Limpar após os testes
    if os.path.exists(model_path):
        os.remove(model_path)
    if os.path.exists(scaler_path):
        os.remove(scaler_path)


class TestAPI:
    """
    Testes para os endpoints da API.
    """

    def test_health(self, setup_test_environment):
        """
        Testa o endpoint de verificação de saúde.
        """
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["model_loaded"] == True

    def test_predict(self, setup_test_environment):
        """
        Testa o endpoint de previsão.
        """
        # Mock para o método download do yfinance que é usado no endpoint
        import yfinance as yf

        # Criar um DataFrame de exemplo para retorno do yf.download
        import pandas as pd
        from datetime import datetime, timedelta

        # Dados de exemplo para os últimos 60 dias
        dates = [datetime.now() - timedelta(days=i) for i in range(60, 0, -1)]
        prices = [100 + i * 0.5 for i in range(60)]  # Preços crescentes

        mock_data = pd.DataFrame({
            'Open': prices,
            'High': [p * 1.01 for p in prices],
            'Low': [p * 0.99 for p in prices],
            'Close': prices,
            'Volume': [1000000 for _ in prices]
        }, index=dates)

        # Patch da função download
        original_download = yf.download
        yf.download = lambda *args, **kwargs: mock_data

        try:
            # Fazer a requisição
            response = client.post(
                "/predict",
                json={"symbol": "AAPL", "days": 5}
            )

            # Verificar a resposta
            assert response.status_code == 200

            result = response.json()
            assert "symbol" in result
            assert result["symbol"] == "AAPL"
            assert "current_price" in result
            assert "predicted_prices" in result
            assert len(result["predicted_prices"]) == 5
            assert "prediction_dates" in result
            assert len(result["prediction_dates"]) == 5
            assert "confidence" in result

        finally:
            # Restaurar a função original
            yf.download = original_download

    def test_metrics(self, setup_test_environment):
        """
        Testa o endpoint de métricas.
        """
        # Criar um arquivo de métricas fictício
        os.makedirs("reports", exist_ok=True)

        metrics = {
            'mse': 10.5,
            'rmse': 3.24,
            'mae': 2.75,
            'r2': 0.85,
            'mape': 3.5,
            'last_update': '2023-01-01 12:00:00'
        }

        metrics_path = "reports/metrics.json"
        joblib.dump(metrics, metrics_path)

        try:
            # Fazer a requisição
            response = client.get("/metrics")

            # Verificar a resposta
            assert response.status_code == 200

            result = response.json()
            assert "mse" in result
            assert result["mse"] == metrics["mse"]
            assert "rmse" in result
            assert result["rmse"] == metrics["rmse"]
            assert "mae" in result
            assert result["mae"] == metrics["mae"]
            assert "r2" in result
            assert result["r2"] == metrics["r2"]
            assert "mape" in result
            assert result["mape"] == metrics["mape"]
            assert "last_update" in result

        finally:
            # Limpar após o teste
            if os.path.exists(metrics_path):
                os.remove(metrics_path)