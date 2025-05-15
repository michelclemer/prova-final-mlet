"""
Módulo para previsão utilizando o modelo treinado.
"""
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
import json
from typing import List, Dict, Any, Union

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_latest_model(symbol="AAPL"):
    """
    Carrega o modelo mais recente para um determinado ticker.

    Args:
        symbol: Símbolo da ação

    Returns:
        Modelo carregado e scaler
    """
    try:
        # Verificar se há modelo salvo
        model_path = os.path.join("models", f"{symbol}_model.h5")
        scaler_path = os.path.join("models", "scaler.pkl")

        if not os.path.exists(model_path):
            logger.error(f"Modelo não encontrado: {model_path}")
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

        if not os.path.exists(scaler_path):
            logger.error(f"Scaler não encontrado: {scaler_path}")
            raise FileNotFoundError(f"Scaler não encontrado: {scaler_path}")

        # Carregar modelo
        model = load_model(model_path)

        # Carregar scaler
        scaler = joblib.load(scaler_path)

        logger.info(f"Modelo e scaler carregados com sucesso")
        return model, scaler

    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {str(e)}")
        raise


def predict_next_days(model, scaler, last_window, days=7):
    """
    Faz previsões para os próximos dias utilizando o modelo carregado.

    Args:
        model: Modelo carregado
        scaler: Scaler para desnormalização
        last_window: Janela de dados mais recente
        days: Número de dias para prever

    Returns:
        DataFrame com previsões
    """
    try:
        logger.info(f"Fazendo previsões para os próximos {days} dias")

        # Garantir que a janela tem o formato correto
        current_window = last_window.copy()

        # Fazer previsões para os próximos dias
        predicted_prices = []
        predicted_dates = []

        # Data atual para começar as previsões
        current_date = datetime.now()

        for i in range(days):
            # Pular finais de semana
            while current_date.weekday() > 4:  # 5 = sábado, 6 = domingo
                current_date += timedelta(days=1)

            # Fazer previsão para o próximo dia
            prediction = model.predict(current_window)
            predicted_prices.append(prediction[0, 0])
            predicted_dates.append(current_date.strftime('%Y-%m-%d'))

            # Atualizar a janela para incluir a previsão feita
            current_window = np.append(current_window[:, 1:, :],
                                       np.array([[[prediction[0, 0]]]]),
                                       axis=1)

            # Avançar para o próximo dia útil
            current_date += timedelta(days=1)

        # Desnormalizar as previsões
        predicted_prices_real = scaler.inverse_transform(
            np.array(predicted_prices).reshape(-1, 1)
        ).flatten()

        # Criar DataFrame com as previsões
        result = pd.DataFrame({
            'date': predicted_dates,
            'predicted_price': predicted_prices_real
        })

        # Adicionar intervalo de confiança simples (± 5%)
        result['lower_bound'] = result['predicted_price'] * 0.95
        result['upper_bound'] = result['predicted_price'] * 1.05

        logger.info("Previsões concluídas com sucesso")
        return result

    except Exception as e:
        logger.error(f"Erro ao fazer previsões: {str(e)}")
        raise


def get_last_window(symbol="AAPL"):
    """
    Obtém a janela mais recente de dados para previsão.

    Args:
        symbol: Símbolo da ação

    Returns:
        Numpy array com a última janela de dados
    """
    try:
        # Verificar se existe arquivo da última janela
        last_window_path = os.path.join("data", "processed", f"{symbol}_processed_last_window.npy")

        if os.path.exists(last_window_path):
            logger.info(f"Carregando última janela de {last_window_path}")
            return np.load(last_window_path)

        # Se não existir, tentar processar dados recentes
        logger.info("Arquivo da última janela não encontrado. Tentando processar dados recentes...")

        # Importar funções necessárias
        from src.data.collect_data import download_stock_data
        from src.data.process import preprocess_data

        # Baixar dados recentes
        download_stock_data(symbol)

        # Pré-processar dados
        result = preprocess_data(symbol=symbol)

        return result['last_window']

    except Exception as e:
        logger.error(f"Erro ao obter última janela: {str(e)}")
        raise


def format_predictions(predictions):
    """
    Formata as previsões para um formato mais amigável para a API.

    Args:
        predictions: DataFrame com previsões

    Returns:
        Lista de dicionários com previsões formatadas
    """
    try:
        # Converter para formato de lista de dicionários
        result = []

        for _, row in predictions.iterrows():
            prediction = {
                'date': row['date'],
                'predicted_price': round(float(row['predicted_price']), 2),
                'lower_bound': round(float(row['lower_bound']), 2),
                'upper_bound': round(float(row['upper_bound']), 2)
            }
            result.append(prediction)

        return result

    except Exception as e:
        logger.error(f"Erro ao formatar previsões: {str(e)}")
        raise


def predict_specific_date(model, scaler, last_window, target_date):
    """
    Faz previsão para uma data específica.

    Args:
        model: Modelo carregado
        scaler: Scaler para desnormalização
        last_window: Última janela de dados
        target_date: Data alvo para previsão (datetime)

    Returns:
        Dicionário com a previsão
    """
    try:
        logger.info(f"Fazendo previsão para {target_date}")

        # Calcular quantos dias no futuro está a data alvo
        days_ahead = (target_date - datetime.now()).days + 1

        if days_ahead <= 0:
            raise ValueError("A data alvo deve estar no futuro")

        # Fazer previsões até a data alvo
        predictions = predict_next_days(model, scaler, last_window, days=days_ahead)

        # Filtrar a previsão específica para a data alvo
        target_date_str = target_date.strftime('%Y-%m-%d')
        target_prediction = predictions[predictions['date'] == target_date_str]

        if target_prediction.empty:
            raise ValueError(f"Não foi possível fazer previsão para {target_date_str}")

        # Formatar resultado
        prediction = {
            'date': target_date_str,
            'predicted_price': round(float(target_prediction['predicted_price'].iloc[0]), 2),
            'lower_bound': round(float(target_prediction['lower_bound'].iloc[0]), 2),
            'upper_bound': round(float(target_prediction['upper_bound'].iloc[0]), 2)
        }

        logger.info("Previsão para data específica concluída com sucesso")
        return prediction

    except Exception as e:
        logger.error(f"Erro ao fazer previsão para data específica: {str(e)}")
        raise


def get_model_info(symbol="AAPL"):
    """
    Obtém informações sobre o modelo mais recente.

    Args:
        symbol: Símbolo da ação

    Returns:
        Dicionário com informações do modelo
    """
    try:
        # Verificar se há modelo salvo
        model_path = os.path.join("models", f"{symbol}_model.h5")

        if not os.path.exists(model_path):
            logger.error(f"Modelo não encontrado: {model_path}")
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

        # Obter informações do arquivo
        model_stats = os.stat(model_path)
        creation_time = datetime.fromtimestamp(model_stats.st_ctime)

        # Verificar se há métricas salvas
        metrics_path = os.path.join("reports", f"{symbol}_metrics.json")
        metrics = {}

        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)

        # Tentar carregar informações de treinamento
        training_info_path = os.path.join("models", f"{symbol}_training_info.pkl")
        training_info = {}

        if os.path.exists(training_info_path):
            training_info = joblib.load(training_info_path)

        # Montar informações
        info = {
            'ticker': symbol,
            'model_path': model_path,
            'creation_date': creation_time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_age_days': (datetime.now() - creation_time).days,
            'metrics': metrics,
            'training_info': training_info
        }

        logger.info(f"Informações do modelo obtidas com sucesso")
        return info

    except Exception as e:
        logger.error(f"Erro ao obter informações do modelo: {str(e)}")
        raise


def predict_from_date(model, scaler, last_window, start_date, days=7):
    """
    Faz previsões a partir de uma data específica.

    Args:
        model: Modelo carregado
        scaler: Scaler para desnormalização
        last_window: Última janela de dados
        start_date: Data inicial para previsão
        days: Número de dias para prever

    Returns:
        DataFrame com previsões
    """
    try:
        logger.info(f"Fazendo previsões a partir de {start_date} para {days} dias")

        # Calcular quantos dias no futuro está a data inicial
        days_to_start = (start_date - datetime.now()).days + 1

        if days_to_start <= 0:
            raise ValueError("A data inicial deve estar no futuro")

        # Fazer previsões até a data final (início + número de dias)
        total_days = days_to_start + days - 1
        all_predictions = predict_next_days(model, scaler, last_window, days=total_days)

        # Filtrar apenas as previsões do período solicitado
        start_date_str = start_date.strftime('%Y-%m-%d')
        start_idx = all_predictions.index[all_predictions['date'] >= start_date_str].min()

        if pd.isna(start_idx):
            raise ValueError(f"Não foi possível fazer previsões a partir de {start_date_str}")

        result = all_predictions.iloc[start_idx:start_idx + days]

        logger.info("Previsões a partir de data específica concluídas com sucesso")
        return result

    except Exception as e:
        logger.error(f"Erro ao fazer previsões a partir de data: {str(e)}")
        raise


if __name__ == "__main__":
    # Exemplo de uso
    try:
        # Definir símbolo
        symbol = "AAPL"  # Apple como exemplo

        # Carregar modelo e scaler
        model, scaler = load_latest_model(symbol)

        # Obter última janela de dados
        last_window = get_last_window(symbol)

        # Fazer previsões para os próximos 7 dias
        predictions = predict_next_days(model, scaler, last_window, days=7)

        # Formatar previsões
        formatted_predictions = format_predictions(predictions)

        # Exibir previsões
        print("Previsões para os próximos 7 dias:")
        for pred in formatted_predictions:
            print(
                f"Data: {pred['date']} - Previsão: ${pred['predicted_price']} (Intervalo: ${pred['lower_bound']} a ${pred['upper_bound']})")

        # Obter informações do modelo
        model_info = get_model_info(symbol)
        print("\nInformações do modelo:")
        print(f"Ticker: {model_info['ticker']}")
        print(f"Caminho: {model_info['model_path']}")
        print(f"Data de criação: {model_info['creation_date']}")
        print(f"Idade do modelo (dias): {model_info['model_age_days']}")

        if model_info['metrics']:
            print("\nMétricas do modelo:")
            for key, value in model_info['metrics'].items():
                print(f"{key}: {value}")

    except Exception as e:
        logger.error(f"Erro: {str(e)}")