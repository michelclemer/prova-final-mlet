"""
Módulo para avaliação do modelo de previsão de preços de ações.
"""
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

# Tente diferentes métodos de importação para load_model para compatibilidade
try:
    from tensorflow.keras.models import load_model
except ImportError:
    try:
        load_model = keras.models.load_model
    except ImportError:
        try:
            from keras.models import load_model
        except ImportError:
            raise ImportError("Não foi possível importar load_model. Verifique a instalação do TensorFlow ou Keras.")

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_model(
        model_path=None,
        scaler_path=None,
        X_test=None,
        y_test=None,
        test_dates_path=None,
        output_dir=None,
        symbol="AAPL"
):
    """
    Avalia o desempenho do modelo de previsão.

    Args:
        model_path (str): Caminho para o modelo treinado.
        scaler_path (str): Caminho para o objeto scaler.
        X_test (numpy.ndarray): Dados de teste (features).
        y_test (numpy.ndarray): Dados de teste (targets).
        test_dates_path (str): Caminho para as datas de teste.
        output_dir (str): Diretório para salvar os resultados da avaliação.
        symbol (str): Símbolo da ação.

    Returns:
        dict: Métricas de avaliação do modelo.
    """
    # Usar valores padrão se não fornecidos
    model_path = model_path or os.path.join("models", f"{symbol}_model.h5")
    scaler_path = scaler_path or os.path.join("models", "scaler.pkl")
    data_dir = os.path.join("data", "processed")
    output_dir = output_dir or os.path.join("reports", "figures")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    try:
        # Carregar o modelo
        logger.info(f"Carregando modelo de {model_path}")
        model = keras.models.load_model(model_path)

        # Carregar o scaler
        logger.info(f"Carregando scaler de {scaler_path}")
        scaler = joblib.load(scaler_path)

        # Carregar dados de teste se não fornecidos
        if X_test is None:
            X_test = np.load(f"{data_dir}/{symbol}_processed_X_test.npy")
            y_test = np.load(f"{data_dir}/{symbol}_processed_y_test.npy")

        # Carregar datas de teste
        test_dates_path = test_dates_path or f"{data_dir}/{symbol}_processed_test_dates.csv"
        if os.path.exists(test_dates_path):
            test_dates_df = pd.read_csv(test_dates_path)
            date_col = [col for col in test_dates_df.columns if 'date' in col.lower() or 'Date' in col][0]
            test_dates = test_dates_df[date_col].values
        else:
            # Se não houver arquivo de datas, criar datas genéricas
            test_dates = [f"Data {i+1}" for i in range(len(y_test))]

        # Garantir formato correto para LSTM
        timesteps = X_test.shape[1]
        features = X_test.shape[2] if len(X_test.shape) > 2 else 1
        X_test = X_test.reshape(X_test.shape[0], timesteps, features)

        # Fazer previsões
        logger.info("Realizando previsões no conjunto de teste")
        y_pred = model.predict(X_test)

        # Desnormalizar os resultados
        y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred_real = scaler.inverse_transform(y_pred)

        # Calcular métricas
        mse = mean_squared_error(y_test_real, y_pred_real)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_real, y_pred_real)
        r2 = r2_score(y_test_real, y_pred_real)

        # Calcular MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_test_real - y_pred_real) / y_test_real)) * 100

        logger.info(f"MSE: {mse:.4f}")
        logger.info(f"RMSE: {rmse:.4f}")
        logger.info(f"MAE: {mae:.4f}")
        logger.info(f"R²: {r2:.4f}")
        logger.info(f"MAPE: {mape:.4f}%")

        # Criar visualização dos resultados
        plt.figure(figsize=(12, 6))
        plt.plot(test_dates[-len(y_test_real):], y_test_real, label='Valor Real')
        plt.plot(test_dates[-len(y_pred_real):], y_pred_real, label='Previsão')
        plt.title(f'Previsão de Preço das Ações de {symbol}')
        plt.xlabel('Data')
        plt.ylabel('Preço ($)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Salvar figura
        fig_path = os.path.join(output_dir, f"{symbol}_prediction_vs_actual.png")
        plt.savefig(fig_path)
        logger.info(f"Figura salva em {fig_path}")

        # Salvar resultados em um arquivo CSV
        results_df = pd.DataFrame({
            'Date': test_dates[-len(y_test_real):],
            'Actual': y_test_real.flatten(),
            'Predicted': y_pred_real.flatten(),
            'Error': (y_test_real - y_pred_real).flatten(),
            'Percent_Error': ((y_test_real - y_pred_real) / y_test_real * 100).flatten()
        })

        results_path = os.path.join(output_dir, f"{symbol}_prediction_results.csv")
        results_df.to_csv(results_path, index=False)
        logger.info(f"Resultados salvos em {results_path}")

        # Salvar métricas em um arquivo JSON
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'mape': float(mape),
            'samples': int(len(y_test_real))
        }

        metrics_path = os.path.join("reports", f"{symbol}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Métricas salvas em {metrics_path}")

        return metrics

    except Exception as e:
        logger.error(f"Erro na avaliação do modelo: {e}")
        raise

def main():
    """Função principal para execução direta do script."""
    try:
        # Definir o símbolo da ação
        symbol = "AAPL"  # Apple como exemplo

        # Verificar se o modelo existe
        model_path = os.path.join("models", f"{symbol}_model.h5")
        if not os.path.exists(model_path):
            logger.error(f"Modelo não encontrado em {model_path}")
            logger.info("Execute o treinamento do modelo primeiro")
            return

        metrics = evaluate_model(symbol=symbol)
        logger.info("Avaliação do modelo concluída com sucesso!")
        logger.info(f"Métricas: {metrics}")
    except Exception as e:
        logger.error(f"Falha na avaliação do modelo: {e}")

if __name__ == "__main__":
    main()