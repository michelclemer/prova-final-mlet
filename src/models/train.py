"""
Módulo para treinamento do modelo LSTM para previsão de preços de ações.
"""
import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime
import joblib

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
        epochs=100,
        symbol="AAPL"
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
        symbol (str): Símbolo da ação.

    Returns:
        tensorflow.keras.models.Sequential: Modelo LSTM treinado.
    """
    # Usar valores padrão se não fornecidos
    data_dir = os.path.join("data", "processed")

    if model_path is None:
        model_path = os.path.join("models", f"{symbol}_model.h5")

    try:
        # Certificar que o diretório models existe
        os.makedirs("models", exist_ok=True)

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

        # Salvar informações do treinamento
        training_info = {
            'symbol': symbol,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'epochs_trained': len(history.history['loss']),
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'test_loss': test_loss,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Salvar informações do treinamento
        info_path = os.path.join("models", f"{symbol}_training_info.pkl")
        joblib.dump(training_info, info_path)

        # Salvar histórico de treinamento
        history_path = os.path.join("models", f"{symbol}_history.pkl")
        joblib.dump(history.history, history_path)

        logger.info(f"Modelo salvo em {model_path}")
        logger.info(f"Informações do treinamento salvas em {info_path}")

        return model

    except Exception as e:
        logger.error(f"Erro no treinamento do modelo: {e}")
        raise


def main():
    """Função principal para execução direta do script."""
    try:
        # Definir o símbolo da ação
        symbol = "AAPL"  # Apple como exemplo

        # Verificar se existem dados processados
        processed_dir = os.path.join("data", "processed")
        X_train_path = f"{processed_dir}/{symbol}_processed_X_train.npy"

        if not os.path.exists(X_train_path):
            # Se os dados processados não existirem, executar pré-processamento
            from process import preprocess_data

            # Verificar se existem dados brutos
            raw_dir = "data/raw"
            if not os.path.exists(raw_dir) or len([f for f in os.listdir(raw_dir) if f.startswith(symbol)]) == 0:
                # Se os dados brutos não existirem, baixar dados
                from collect_data import download_stock_data
                download_stock_data(symbol)

            # Pré-processar dados
            preprocess_data(symbol=symbol)

        # Treinar modelo
        train_model(symbol=symbol)

        logger.info("Treinamento do modelo concluído com sucesso!")
    except Exception as e:
        logger.error(f"Falha no treinamento do modelo: {e}")


if __name__ == "__main__":
    main()
