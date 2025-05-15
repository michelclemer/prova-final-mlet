"""
Módulo para avaliação do modelo de previsão de preços de ações.
"""
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
import mlflow

# Carregar variáveis de ambiente
load_dotenv()

# Configurar logging
logger = logging.getLogger(__name__)

# Configurar MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", ""))
mlflow_experiment = os.getenv("MLFLOW_EXPERIMENT", "stock-prediction")
mlflow.set_experiment(mlflow_experiment)


def evaluate_model(
        model_path=None,
        scaler_path=None,
        X_test=None,
        y_test=None,
        test_dates_path=None,
        output_dir=None
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

    Returns:
        dict: Métricas de avaliação do modelo.
    """
    # Usar valores padrão se não fornecidos
    symbol = os.getenv("STOCK_SYMBOL", "AAPL")
    model_path = model_path or os.path.join("models", f"{symbol}_model.h5")
    scaler_path = scaler_path or os.path.join("models", "scaler.pkl")
    data_dir = os.path.join("data", "processed")
    output_dir = output_dir or os.path.join("reports", "figures")

    os.makedirs(output_dir, exist_ok=True)

    try:
        # Carregar o modelo
        logger.info(f"Carregando modelo de {model_path}")
        model = load_model(model_path)

        # Carregar o scaler
        logger.info(f"Carregando scaler de {scaler_path}")
        scaler = joblib.load(scaler_path)

        # Carregar dados de teste se não fornecidos
        if X_test is None:
            X_test = np.load(f"{data_dir}/{symbol}_processed_X_test.npy")
            y_test = np.load(f"{data_dir}/{symbol}_processed_y_test.npy")

        # Carregar datas de teste
        test_dates_path = test_dates_path or f"{data_dir}/{symbol}_processed_test_dates.csv"
        test_dates = pd.read_csv(test_dates_path)['Date'].values

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

        with mlflow.start_run():
            # Registrar métricas
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mape", mape)

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
            mlflow.log_artifact(fig_path)

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
            mlflow.log_artifact(results_path)

            # Retornar métricas
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape
            }

            return metrics

    except Exception as e:
        logger.error(f"Erro na avaliação do modelo: {e}")
        raise

def main():
    """Função principal para execução direta do script."""
    try:
        metrics = evaluate_model()
        logger.info("Avaliação do modelo concluída com sucesso!")
        logger.info(f"Métricas: {metrics}")
    except Exception as e:
        logger.error(f"Falha na avaliação do modelo: {e}")

if __name__ == "__main__":
    main()