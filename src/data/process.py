import pandas as pd
import numpy as np
import logging
import os
import joblib
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sequences(data, window_size):
    """
    Cria sequências para o modelo LSTM.

    Args:
        data: Array com os dados
        window_size: Tamanho da janela de tempo (número de dias passados para usar como features)

    Returns:
        X: Array com sequências de entrada
        y: Array com valores alvo
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])

    return np.array(X), np.array(y)


def preprocess_data(input_path=None, output_path=None, scaler_path=None,
                    test_size=0.2, feature_window=60, symbol="AAPL"):
    """
    Pré-processa os dados para treinamento do modelo LSTM.

    Args:
        input_path: Caminho para o arquivo CSV com os dados
        output_path: Prefixo para os arquivos de saída
        scaler_path: Caminho para salvar o scaler
        test_size: Proporção dos dados para teste
        feature_window: Tamanho da janela de tempo
        symbol: Símbolo da ação

    Returns:
        Dict com os dados processados
    """
    try:
        logger.info("Iniciando pré-processamento dos dados")

        # Se input_path não for fornecido, procurar dados brutos
        if input_path is None:
            raw_dir = "data/raw"
            if not os.path.exists(raw_dir):
                raise FileNotFoundError(f"Diretório {raw_dir} não encontrado")

            # Encontrar arquivo mais recente para o símbolo
            files = [f for f in os.listdir(raw_dir) if f.startswith(symbol.replace('.', '_')) and f.endswith('.csv')]

            if not files:
                # Se não encontrar, gerar dados simulados
                logger.info(f"Nenhum arquivo encontrado para {symbol} em {raw_dir}. Gerando dados simulados.")
                from src.data.simulate_data import generate_stock_data
                stock_data, input_path = generate_stock_data(symbol=symbol, days=500)
            else:
                # Usar o arquivo mais recente
                input_path = os.path.join(raw_dir, max(files, key=lambda x: os.path.getmtime(os.path.join(raw_dir, x))))
                logger.info(f"Carregando dados de {input_path}")
                stock_data = pd.read_csv(input_path)
        else:
            logger.info(f"Carregando dados de {input_path}")
            stock_data = pd.read_csv(input_path)

        # Verificar se há dados suficientes
        if len(stock_data) < feature_window + 10:
            logger.warning(f"Dados insuficientes ({len(stock_data)} registros). Gerando dados adicionais.")
            from src.data.simulate_data import generate_stock_data
            additional_data, _ = generate_stock_data(symbol=symbol, days=max(500, feature_window + 50))
            stock_data = pd.concat([stock_data, additional_data]).drop_duplicates(subset=['date']).reset_index(
                drop=True)
            logger.info(f"Dados aumentados para {len(stock_data)} registros.")

        # Certificar que os dados estão ordenados por data
        if 'date' in stock_data.columns:
            stock_data['date'] = pd.to_datetime(stock_data['date'])
            stock_data = stock_data.sort_values('date')
        elif 'Date' in stock_data.columns:
            stock_data['Date'] = pd.to_datetime(stock_data['Date'])
            stock_data = stock_data.sort_values('Date')

        # Selecionar apenas o preço de fechamento
        if 'close' in stock_data.columns:
            close_prices = stock_data['close'].values.reshape(-1, 1)
        elif 'Close' in stock_data.columns:
            close_prices = stock_data['Close'].values.reshape(-1, 1)
        else:
            # Se não encontrar coluna de fechamento, tentar identificar colunas numéricas
            numeric_cols = stock_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                logger.warning(f"Coluna de preço de fechamento não encontrada. Usando '{numeric_cols[0]}'.")
                close_prices = stock_data[numeric_cols[0]].values.reshape(-1, 1)
            else:
                raise ValueError("Nenhuma coluna numérica encontrada para usar como preço")

        # Criar diretório para dados processados
        processed_dir = "data/processed"
        os.makedirs(processed_dir, exist_ok=True)

        # Definir caminhos padrão se não fornecidos
        if output_path is None:
            output_path = os.path.join(processed_dir, f"{symbol}_processed")

        if scaler_path is None:
            model_dir = "models"
            os.makedirs(model_dir, exist_ok=True)
            scaler_path = os.path.join(model_dir, "scaler.pkl")

        # Normalizar os dados
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(close_prices)

        logger.info(f"Normalização aplicada. Dados entre {scaled_data.min()} e {scaled_data.max()}")

        # Criar sequências
        X, y = create_sequences(scaled_data, feature_window)

        logger.info(f"Sequências criadas. X shape: {X.shape}, y shape: {y.shape}")

        # Dividir em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )

        # Extrair datas para o conjunto de teste
        test_start_idx = len(X) - len(X_test)
        test_dates = None

        if 'date' in stock_data.columns:
            test_dates = stock_data['date'].iloc[test_start_idx + feature_window:].reset_index(drop=True)
        elif 'Date' in stock_data.columns:
            test_dates = stock_data['Date'].iloc[test_start_idx + feature_window:].reset_index(drop=True)

        # Criar última janela para previsões futuras
        last_window = scaled_data[-feature_window:].reshape(1, feature_window, 1)

        # Salvar os dados processados
        np.save(f"{output_path}_X_train.npy", X_train)
        np.save(f"{output_path}_y_train.npy", y_train)
        np.save(f"{output_path}_X_test.npy", X_test)
        np.save(f"{output_path}_y_test.npy", y_test)
        np.save(f"{output_path}_last_window.npy", last_window)

        # Salvar o scaler
        joblib.dump(scaler, scaler_path)

        # Salvar datas de teste
        if test_dates is not None:
            date_col = 'date' if 'date' in stock_data.columns else 'Date'
            test_dates_df = pd.DataFrame({date_col: test_dates})
            test_dates_df.to_csv(f"{output_path}_test_dates.csv", index=False)

        logger.info("Pré-processamento concluído com sucesso")

        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'scaler': scaler,
            'last_window': last_window
        }

    except Exception as e:
        logger.error(f"Erro no pré-processamento: {str(e)}")
        raise


if __name__ == "__main__":
    # Exemplo de uso
    try:
        symbol = "AAPL"  # Usar Apple como exemplo

        # Verificar se existe arquivo de dados
        raw_dir = "data/raw"
        if not os.path.exists(raw_dir) or len(os.listdir(raw_dir)) == 0:
            # Se não existir, baixar dados
            from collect_data import download_stock_data

            download_stock_data(symbol)

        # Pré-processar dados
        result = preprocess_data(symbol=symbol)

        print(f"Dados pré-processados com sucesso!")
        print(f"Conjunto de treino: {result['X_train'].shape[0]} amostras")
        print(f"Conjunto de teste: {result['X_test'].shape[0]} amostras")

    except Exception as e:
        print(f"Erro: {str(e)}")