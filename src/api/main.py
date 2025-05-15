"""
API para previsão de preços de ações.
"""
import os
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Modelos para API
class PredictionResponse(BaseModel):
    date: str
    predicted_price: float
    lower_bound: float
    upper_bound: float


class PredictionRequest(BaseModel):
    symbol: str = Field(default="AAPL", description="Símbolo da ação")
    days: int = Field(default=7, ge=1, le=30, description="Número de dias para previsão")


class ModelInfo(BaseModel):
    ticker: str
    model_path: str
    creation_date: str
    model_age_days: int
    metrics: Dict[str, float] = {}


class DriftCheckResponse(BaseModel):
    status: str
    drift_detected: bool
    mape: float
    threshold_mape: float
    message: str
    timestamp: str


# Inicializar FastAPI
app = FastAPI(
    title="Stock Price Prediction API",
    description="API para previsão de preços de ações usando LSTM",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Carregar dependências apenas quando necessário
def get_prediction_module():
    from src.models.predict import load_latest_model, get_last_window, predict_next_days, format_predictions, \
        predict_specific_date, predict_from_date, get_model_info
    return {
        'load_latest_model': load_latest_model,
        'get_last_window': get_last_window,
        'predict_next_days': predict_next_days,
        'format_predictions': format_predictions,
        'predict_specific_date': predict_specific_date,
        'predict_from_date': predict_from_date,
        'get_model_info': get_model_info
    }


def get_monitoring_module():
    from src.monitoring.metrics import ModelMonitor
    return ModelMonitor


# Rotas da API
@app.get("/", tags=["Status"])
async def root():
    return {"message": "Stock Price Prediction API", "status": "active"}


@app.get("/health", tags=["Status"])
async def health_check():
    try:
        # Verificar se o modelo existe
        pred_module = get_prediction_module()
        model_info = pred_module['get_model_info']("AAPL")  # Usar Apple como padrão

        return {
            "status": "ok",
            "model_loaded": True,
            "model_age_days": model_info['model_age_days'],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Erro na verificação de saúde: {str(e)}")
        return {
            "status": "error",
            "model_loaded": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@app.get("/model/info", response_model=ModelInfo, tags=["Modelo"])
async def model_information(symbol: str = Query("AAPL", description="Símbolo da ação")):
    """
    Retorna informações sobre o modelo atual.
    """
    try:
        pred_module = get_prediction_module()
        info = pred_module['get_model_info'](symbol)
        return info
    except Exception as e:
        logger.error(f"Erro ao obter informações do modelo: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict/next", response_model=List[PredictionResponse], tags=["Previsão"])
async def predict_next(
        days: int = Query(7, ge=1, le=30, description="Número de dias para prever"),
        symbol: str = Query("AAPL", description="Símbolo da ação")
):
    """
    Retorna previsões para os próximos dias.
    """
    try:
        pred_module = get_prediction_module()

        # Carregar modelo e scaler
        model, scaler = pred_module['load_latest_model'](symbol)

        # Obter última janela de dados
        last_window = pred_module['get_last_window'](symbol)

        # Fazer previsões
        predictions = pred_module['predict_next_days'](model, scaler, last_window, days)

        # Formatar previsões
        formatted = pred_module['format_predictions'](predictions)

        # Salvar previsões para monitoramento futuro
        os.makedirs("reports", exist_ok=True)
        predictions.to_csv(f"reports/{symbol}_predictions.csv", index=False)

        return formatted
    except Exception as e:
        logger.error(f"Erro ao fazer previsões: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict/date", response_model=PredictionResponse, tags=["Previsão"])
async def predict_date(
        date: str = Query(..., description="Data para previsão (formato: YYYY-MM-DD)"),
        symbol: str = Query("AAPL", description="Símbolo da ação")
):
    """
    Retorna previsão para uma data específica.
    """
    try:
        target_date = datetime.strptime(date, "%Y-%m-%d")

        pred_module = get_prediction_module()

        # Carregar modelo e scaler
        model, scaler = pred_module['load_latest_model'](symbol)

        # Obter última janela de dados
        last_window = pred_module['get_last_window'](symbol)

        # Fazer previsão para a data específica
        prediction = pred_module['predict_specific_date'](model, scaler, last_window, target_date)

        return prediction
    except ValueError:
        raise HTTPException(status_code=400, detail="Formato de data inválido. Use YYYY-MM-DD")
    except Exception as e:
        logger.error(f"Erro ao fazer previsão para data específica: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict/range", response_model=List[PredictionResponse], tags=["Previsão"])
async def predict_range(
        start_date: str = Query(..., description="Data inicial (formato: YYYY-MM-DD)"),
        days: int = Query(7, ge=1, le=30, description="Número de dias para prever"),
        symbol: str = Query("AAPL", description="Símbolo da ação")
):
    """
    Retorna previsões para um intervalo de datas a partir de uma data inicial.
    """
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")

        pred_module = get_prediction_module()

        # Carregar modelo e scaler
        model, scaler = pred_module['load_latest_model'](symbol)

        # Obter última janela de dados
        last_window = pred_module['get_last_window'](symbol)

        # Fazer previsões para o intervalo
        predictions = pred_module['predict_from_date'](model, scaler, last_window, start, days)

        # Formatar previsões
        formatted = pred_module['format_predictions'](predictions)

        return formatted
    except ValueError:
        raise HTTPException(status_code=400, detail="Formato de data inválido. Use YYYY-MM-DD")
    except Exception as e:
        logger.error(f"Erro ao fazer previsões para intervalo: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=Dict[str, Any], tags=["Previsão"])
async def predict(request: PredictionRequest):
    """
    Faz previsões com base nos parâmetros fornecidos.
    """
    try:
        pred_module = get_prediction_module()

        # Carregar modelo e scaler
        model, scaler = pred_module['load_latest_model'](request.symbol)

        # Obter última janela de dados
        last_window = pred_module['get_last_window'](request.symbol)

        # Fazer previsões
        predictions = pred_module['predict_next_days'](model, scaler, last_window, request.days)

        # Formatar previsões
        formatted = pred_module['format_predictions'](predictions)

        # Obter preço atual
        from src.data.collect_data import get_latest_stock_data
        current_data = get_latest_stock_data(request.symbol, days=1)

        current_price = 0.0
        if not current_data.empty:
            if 'close' in current_data.columns:
                current_price = float(current_data['close'].iloc[-1])
            elif 'Close' in current_data.columns:
                current_price = float(current_data['Close'].iloc[-1])

        # Construir resposta
        response = {
            "symbol": request.symbol,
            "current_price": current_price,
            "prediction_date": datetime.now().strftime("%Y-%m-%d"),
            "predicted_prices": [p['predicted_price'] for p in formatted],
            "prediction_dates": [p['date'] for p in formatted],
            "lower_bounds": [p['lower_bound'] for p in formatted],
            "upper_bounds": [p['upper_bound'] for p in formatted],
            "confidence": 0.95  # Intervalo de confiança de 95%
        }

        # Salvar previsões para monitoramento futuro
        os.makedirs("reports", exist_ok=True)
        predictions.to_csv(f"reports/{request.symbol}_predictions.csv", index=False)

        return response
    except Exception as e:
        logger.error(f"Erro ao fazer previsões: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/monitoring/metrics", response_model=Dict[str, Any], tags=["Monitoramento"])
async def get_monitoring_metrics(symbol: str = Query("AAPL", description="Símbolo da ação")):
    """
    Retorna métricas de monitoramento do modelo.
    """
    try:
        monitor_cls = get_monitoring_module()
        monitor = monitor_cls(symbol)

        metrics = monitor.collect_metrics()

        if not metrics:
            return {"message": "Não há métricas de monitoramento disponíveis"}

        return metrics
    except Exception as e:
        logger.error(f"Erro ao obter métricas de monitoramento: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/monitoring/drift", response_model=DriftCheckResponse, tags=["Monitoramento"])
async def check_for_drift(
        symbol: str = Query("AAPL", description="Símbolo da ação"),
        threshold_mape: float = Query(10.0, description="Limiar de MAPE para considerar drift")
):
    """
    Verifica se há drift no desempenho do modelo.
    """
    try:
        monitor_cls = get_monitoring_module()
        monitor = monitor_cls(symbol)

        drift_result = monitor.check_for_drift(threshold_mape)

        return drift_result
    except Exception as e:
        logger.error(f"Erro ao verificar drift: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/monitoring/update", tags=["Monitoramento"])
async def update_monitoring(symbol: str = Query("AAPL", description="Símbolo da ação"),
                            days: int = Query(10, description="Número de dias para atualizar")):
    """
    Atualiza o monitoramento com dados reais recentes.
    """
    try:
        # Obter dados reais recentes
        from src.data.collect_data import get_latest_stock_data
        actual_data = get_latest_stock_data(symbol, days=days)

        if actual_data.empty:
            raise HTTPException(status_code=404, detail=f"Não foi possível obter dados recentes para {symbol}")

        # Atualizar monitoramento
        monitor_cls = get_monitoring_module()
        monitor = monitor_cls(symbol)

        count = monitor.update_with_actual_data(actual_data)

        if count == 0:
            return {"message": "Nenhum registro atualizado", "status": "warning"}

        return {"message": f"{count} registros atualizados com sucesso", "status": "success"}
    except Exception as e:
        logger.error(f"Erro ao atualizar monitoramento: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrain", tags=["Modelo"])
async def retrain_model(symbol: str = Query("AAPL", description="Símbolo da ação")):
    """
    Retreina o modelo com dados atualizados.
    """
    try:
        # Importar módulos necessários
        from src.data.collect_data import download_stock_data
        from src.data.process import preprocess_data
        from src.models.train import train_model
        from src.models.evaluate_model import evaluate_model

        # 1. Coletar dados atualizados
        logger.info(f"Coletando dados atualizados para {symbol}")
        download_stock_data(symbol)

        # 2. Pré-processar dados
        logger.info("Pré-processando dados")
        processed_data = preprocess_data(symbol=symbol)

        # 3. Treinar novo modelo
        logger.info("Treinando novo modelo")
        model = train_model(
            X_train=processed_data['X_train'],
            y_train=processed_data['y_train'],
            X_test=processed_data['X_test'],
            y_test=processed_data['y_test'],
            symbol=symbol
        )

        # 4. Avaliar novo modelo
        logger.info("Avaliando novo modelo")
        metrics = evaluate_model(
            symbol=symbol
        )

        return {
            "message": "Modelo retreinado com sucesso",
            "status": "success",
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"Erro ao retreinar modelo: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Tratamento de erros global
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Erro não tratado: {str(exc)}")

    return JSONResponse(
        status_code=500,
        content={
            "detail": str(exc),
            "status": "error"
        }
    )


# Ponto de entrada para uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)