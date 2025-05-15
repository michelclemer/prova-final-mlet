import pandas as pd
import numpy as np
import logging
import os
import pickle
from datetime import datetime, timedelta
from typing import List, Dict, Any, Union

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_latest_model(ticker: str = 'stock') -> Any:
    """
    Carrega o modelo mais recente para um determinado ticker.

    Args:
        ticker: Ticker da ação

    Returns:
        Modelo carregado
    """
    try:
        # Verificar se há modelo salvo
        model_path = f"models/{ticker}_prophet_latest.pkl"

        if not os.path.exists(model_path):
            logger.error(f"Modelo não encontrado: {model_path}")
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

        # Carregar modelo
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        logger.info(f"Modelo carregado com sucesso: {model_path}")
        return model

    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {str(e)}")
        raise


def predict_next_days(model: Any, days: int = 7) -> pd.DataFrame:
    """
    Faz previsões para os próximos dias utilizando o modelo carregado.

    Args:
        model: Modelo carregado
        days: Número de dias para prever

    Returns:
        DataFrame com previsões
    """
    try:
        logger.info(f"Fazendo previsões para os próximos {days} dias")

        # Criar dataframe para previsões futuras
        future = model.make_future_dataframe(periods=days)

        # Fazer previsões
        forecast = model.predict(future)

        # Selecionar apenas as previsões futuras
        future_forecast = forecast.iloc[-days:]

        # Selecionar colunas relevantes
        result = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

        # Renomear colunas para melhor compreensão
        result = result.rename(columns={
            'ds': 'date',
            'yhat': 'predicted_price',
            'yhat_lower': 'lower_bound',
            'yhat_upper': 'upper_bound'
        })

        logger.info("Previsões concluídas com sucesso")
        return result

    except Exception as e:
        logger.error(f"Erro ao fazer previsões: {str(e)}")
        raise


def format_predictions(predictions: pd.DataFrame) -> List[Dict[str, Any]]:
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
                'date': row['date'].strftime('%Y-%m-%d'),
                'predicted_price': round(float(row['predicted_price']), 2),
                'lower_bound': round(float(row['lower_bound']), 2),
                'upper_bound': round(float(row['upper_bound']), 2)
            }
            result.append(prediction)

        return result

    except Exception as e:
        logger.error(f"Erro ao formatar previsões: {str(e)}")
        raise


def predict_from_date(model: Any, start_date: datetime, days: int = 7) -> pd.DataFrame:
    """
    Faz previsões a partir de uma data específica.

    Args:
        model: Modelo carregado
        start_date: Data inicial para previsão
        days: Número de dias para prever

    Returns:
        DataFrame com previsões
    """
    try:
        logger.info(f"Fazendo previsões a partir de {start_date} para {days} dias")

        # Criar dataframe com datas futuras
        dates = [start_date + timedelta(days=i) for i in range(days)]
        future = pd.DataFrame({'ds': dates})

        # Fazer previsões
        forecast = model.predict(future)

        # Selecionar colunas relevantes
        result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

        # Renomear colunas para melhor compreensão
        result = result.rename(columns={
            'ds': 'date',
            'yhat': 'predicted_price',
            'yhat_lower': 'lower_bound',
            'yhat_upper': 'upper_bound'
        })

        logger.info("Previsões a partir de data específica concluídas com sucesso")
        return result

    except Exception as e:
        logger.error(f"Erro ao fazer previsões a partir de data: {str(e)}")
        raise


def predict_specific_date(model: Any, target_date: datetime) -> Dict[str, Any]:
    """
    Faz previsão para uma data específica.

    Args:
        model: Modelo carregado
        target_date: Data alvo para previsão

    Returns:
        Dicionário com a previsão
    """
    try:
        logger.info(f"Fazendo previsão para {target_date}")

        # Criar dataframe com a data alvo
        future = pd.DataFrame({'ds': [target_date]})

        # Fazer previsão
        forecast = model.predict(future)

        # Formatar resultado
        prediction = {
            'date': target_date.strftime('%Y-%m-%d'),
            'predicted_price': round(float(forecast['yhat'].iloc[0]), 2),
            'lower_bound': round(float(forecast['yhat_lower'].iloc[0]), 2),
            'upper_bound': round(float(forecast['yhat_upper'].iloc[0]), 2)
        }

        logger.info("Previsão para data específica concluída com sucesso")
        return prediction

    except Exception as e:
        logger.error(f"Erro ao fazer previsão para data específica: {str(e)}")
        raise


def get_model_info(ticker: str = 'stock') -> Dict[str, Any]:
    """
    Obtém informações sobre o modelo mais recente.

    Args:
        ticker: Ticker da ação

    Returns:
        Dicionário com informações do modelo
    """
    try:
        # Verificar se há modelo salvo
        model_path = f"models/{ticker}_prophet_latest.pkl"

        if not os.path.exists(model_path):
            logger.error(f"Modelo não encontrado: {model_path}")
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

        # Obter informações do arquivo
        model_stats = os.stat(model_path)
        creation_time = datetime.fromtimestamp(model_stats.st_ctime)

        # Verificar se há métricas salvas
        metrics_path = None
        metrics_dir = "models"
        metrics_files = [f for f in os.listdir(metrics_dir) if
                         f.endswith('.pkl') and f.startswith(f"{ticker}_metrics_")]

        if metrics_files:
            # Usar o arquivo mais recente
            latest_metrics = max(metrics_files, key=lambda x: os.path.getmtime(os.path.join(metrics_dir, x)))
            metrics_path = os.path.join(metrics_dir, latest_metrics)

            # Carregar métricas
            with open(metrics_path, 'rb') as f:
                metrics = pickle.load(f)
        else:
            metrics = {}

        # Montar informações
        info = {
            'ticker': ticker,
            'model_path': model_path,
            'creation_date': creation_time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_age_days': (datetime.now() - creation_time).days,
            'metrics': metrics
        }

        logger.info(f"Informações do modelo obtidas com sucesso")
        return info

    except Exception as e:
        logger.error(f"Erro ao obter informações do modelo: {str(e)}")
        raise


if __name__ == "__main__":
    # Exemplo de uso
    try:
        # Definir ticker
        ticker = "PETR4"

        # Carregar modelo
        model = load_latest_model(ticker)

        # Fazer previsões para os próximos 7 dias
        predictions = predict_next_days(model, days=7)

        # Formatar previsões
        formatted_predictions = format_predictions(predictions)

        # Exibir previsões
        for pred in formatted_predictions:
            print(
                f"Data: {pred['date']} - Previsão: {pred['predicted_price']} (Intervalo: {pred['lower_bound']} a {pred['upper_bound']})")

        # Obter informações do modelo
        model_info = get_model_info(ticker)
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