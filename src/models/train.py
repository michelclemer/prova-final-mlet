"""
Módulo para treinamento do modelo LSTM para previsão de preços de ações.
"""
import os
import logging
import numpy as np
import mlflow
import mlflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Configurar logging
logger = logging.getLogger(__name__)

# Configurar MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", ""))
mlflow_experiment = os.getenv("MLFLOW_EXPERIMENT", "stock-prediction")
mlflow.set_experiment(mlflow_experiment)


def build_model(input_shape):
    """
    Constrói um modelo LSTM para previsão de séries temporais.

    Args:
        input_shape (tuple): Formato dos dados de entrada (timesteps, features).

    Returns:
        tensorflow.keras.models.Sequential: Modelo LSTM compilado.
    """
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_model(
        X_train=None,
        y_train=None,
        X_test=None,
        y_test=None,
        model_path=None,
        batch_size=32,
        epochs=100
):
    """
    Treina um modelo LSTM para previsão de preços de ações.

    Args:
        X_train (numpy.ndarray): Dados de treino (features).
        y_train (numpy.ndarray): Dados de treino (targets).
        X_test (numpy.ndarray): Dados de teste (features).
        y_test (numpy.ndarray): Dados de teste (targets).
        model_path (str): Caminho para salvar o modelo treinado.
        batch_size (int): Tamanho do batch para treinamento.
        epochs (int): Número máximo de épocas para treinamento.

    Returns:
        tensorflow.keras.models.Sequential: Modelo LSTM treinado.
    """
    # Usar valores padrão se não fornecidos
    symbol = os.getenv("STOCK_SYMBOL", "AAPL")
    data_dir = os.path.join("data", "processed")
    model_path = model_path or os.path.join("models", f"{symbol}_model.h5")

    try:
        # Carregar dados se não fornecidos
        if X_train is None:
            X_train = np.load(f"{data_dir}/{symbol}_processed_X_train.npy")
            y_train = np.load(f"{data_dir}/{symbol}_processed_y_train.npy")
            X_test = np.load(f"{data_dir}/{symbol}_processed_X_test.npy")
            y_test = np.load(f"{data_dir}/{symbol}_processed_y_test.npy")

        # Obter formato dos dados
        timesteps = X_train.shape[1]
        features = X_train.shape[2] if len(X_train.shape) > 2 else 1

        # Garantir o formato correto para o modelo LSTM
        X_train = X_train.reshape(X_train.shape[0], timesteps, features)
        X_test = X_test.reshape(X_test.shape[0], timesteps, features)

        # Criar diretório para salvar o modelo
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        with mlflow.start_run():
            # Registrar parâmetros
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("timesteps", timesteps)
            mlflow.log_param("features", features)
            mlflow.log_param("stock_symbol", symbol)

            # Construir o modelo
            model = build_model((timesteps, features))
            model.summary()

            # Callbacks para treinamento
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True)
            ]

            # Treinar o modelo
            logger.info(f"Iniciando treinamento do modelo para {symbol}")
            history = model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=1
            )

            # Avaliar o modelo no conjunto de teste
            test_loss = model.evaluate(X_test, y_test, verbose=0)
            logger.info(f"Loss no conjunto de teste: {test_loss}")

            # Registrar métricas
            mlflow.log_metric("test_loss", test_loss)
            for epoch, value in enumerate(history.history['loss']):
                mlflow.log_metric("train_loss", value, step=epoch)

            for epoch, value in enumerate(history.history['val_loss']):
                mlflow.log_metric("val_loss", value, step=epoch)

            # Registrar o modelo no MLflow
            mlflow.keras.log_model(model, "model")

            logger.info(f"Modelo salvo em {model_path}")

            return model

    except Exception as e:
        logger.error(f"Erro no treinamento do modelo: {e}")
        raise


def main():
    """Função principal para execução direta do script."""
    try:
        train_model()
        logger.info("Treinamento do modelo concluído com sucesso!")
    except Exception as e:
        logger.error(f"Falha no treinamento do modelo: {e}")


if __name__ == "__main__":
    main()