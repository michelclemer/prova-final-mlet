"""
Endpoints adicionais para a API de previsão de ações.
"""
import os
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, Query
import joblib
from dotenv import load_dotenv
import mlflow
from pydantic import BaseModel

# Carregar variáveis de ambiente
load_dotenv()

# Configurar logging
logger = logging.getLogger(__name__)

# Criar router
router = APIRouter()


class ModelMetrics(BaseModel):
    mse: float
    rmse: float
    mae: float
    r2: float
    mape: float
    last_update: str


@router.get("/metrics", response_model=ModelMetrics)
async def get_metrics():
    """
    Retorna as métricas de desempenho do modelo.

    Returns:
        ModelMetrics: Métricas do modelo.
    """
    try:
        # Caminho para arquivo de métricas
        metrics_path = os.path.join("reports", "metrics.json")

        # Se o arquivo existir, carregá-lo
        if os.path.exists(metrics_path):
            metrics = joblib.load(metrics_path)
        else:
            # Caso contrário, obter as métricas mais recentes do MLflow
            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", ""))
            experiment_name = os.getenv("MLFLOW_EXPERIMENT", "stock-prediction")

            # Obter o ID do experimento
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                raise HTTPException(status_code=404, detail="Experimento não encontrado no MLflow")

            # Obter a execução mais recente
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id],
                                      order_by=["start_time DESC"])

            if runs.empty:
                raise HTTPException(status_code=404, detail="Nenhuma execução encontrada no MLflow")

            latest_run = runs.iloc[0]

            # Extrair métricas
            metrics = {
                'mse': float(latest_run['metrics.mse']),
                'rmse': float(latest_run['metrics.rmse']),
                'mae': float(latest_run['metrics.mae']),
                'r2': float(latest_run['metrics.r2']),
                'mape': float(latest_run['metrics.mape']),
                'last_update': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            # Salvar métricas para uso futuro
            os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
            joblib.dump(metrics, metrics_path)

        return ModelMetrics(**metrics)

    except Exception as e:
        logger.error(f"Erro ao obter métricas: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class HistoricalData(BaseModel):
    dates: list
    prices: list
    predictions: list = []


@router.get("/historical/{symbol}", response_model=HistoricalData)
async def get_historical_data(
        symbol: str,
        days: int = Query(30, description="Número de dias a serem exibidos")
):
    """
    Retorna dados históricos para um símbolo, com previsões se disponíveis.

    Args:
        symbol (str): Símbolo da ação.
        days (int): Número de dias de histórico a serem exibidos.

    Returns:
        HistoricalData: Dados históricos e previsões.
    """
    try:
        # Verificar se existem previsões salvas
        predictions_path = os.path.join("reports", "predictions", f"{symbol}_predictions.csv")

        # Calcular datas
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Carregar dados históricos
        import yfinance as yf
        stock_data = yf.download(symbol, start=start_date, end=end_date)

        if stock_data.empty:
            raise HTTPException(status_code=404, detail=f"Dados não encontrados para {symbol}")

        # Formatar dados para a resposta
        dates = stock_data.index.strftime("%Y-%m-%d").tolist()
        prices = stock_data['Close'].tolist()

        # Carregar previsões se existirem
        predictions = []
        if os.path.exists(predictions_path):
            pred_df = pd.read_csv(predictions_path)

            # Filtrar apenas as previsões que se aplicam ao período solicitado
            mask = pd.to_datetime(pred_df['date']).between(start_date, end_date + timedelta(days=7))
            if not pred_df[mask].empty:
                # Alinhar datas de previsão com dates
                for date in dates:
                    price = pred_df[pred_df['date'] == date]['predicted_price'].values
                    if len(price) > 0:
                        predictions.append(float(price[0]))
                    else:
                        predictions.append(None)

        return HistoricalData(
            dates=dates,
            prices=prices,
            predictions=predictions if predictions else []
        )

    except Exception as e:
        logger.error(f"Erro ao obter dados históricos: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrain")
async def retrain_model():
    """
    Endpoint para retreinamento do modelo.

    Returns:
        dict: Status do retreinamento.
    """
    try:
        # Importar funções necessárias
        from src.data.collect_data import collect_stock_data
        from src.data.process import preprocess_data
        from src.models.train import train_model
        from src.models.evaluate_model import evaluate_model

        # Executar pipeline de retreinamento
        logger.info("Iniciando retreinamento do modelo...")

        # 1. Coletar dados atualizados
        collect_stock_data()

        # 2. Pré-processar dados
        processed_data = preprocess_data()

        # 3. Treinar modelo
        model = train_model(
            X_train=processed_data['X_train'],
            y_train=processed_data['y_train'],
            X_test=processed_data['X_test'],
            y_test=processed_data['y_test']
        )

        # 4. Avaliar modelo
        metrics = evaluate_model(
            X_test=processed_data['X_test'],
            y_test=processed_data['y_test']
        )

        logger.info("Retreinamento concluído com sucesso!")

        return {
            "status": "success",
            "message": "Modelo retreinado com sucesso",
            "metrics": metrics
        }

    except Exception as e:
        logger.error(f"Erro no retreinamento do modelo: {e}")
        raise HTTPException(status_code=500, detail=str(e))