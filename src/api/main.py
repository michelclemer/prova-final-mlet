from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import sys
import os

# Adicionar o diretório pai ao path para importar os módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importar módulos do projeto
from src.models.predict import (
    load_latest_model, predict_next_days, format_predictions,
    predict_from_date, predict_specific_date, get_model_info
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="Stock Price Prediction API",
    description="API para previsão de preços de ações usando Prophet",
    version="1.0.0"
)

# Configurar CORS para permitir todas as origens (para desenvolvimento)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Definir modelos de dados para a API
class PredictionResponse(BaseModel):
    date: str
    predicted_price: float
    lower_bound: float
    upper_bound: float

class ModelInfo(BaseModel):
    ticker: str
    model_path: str
    creation_date: str
    model_age_days: int
    metrics: Dict[str, float] = {}

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

# Função para carregar o modelo sob demanda
def get_model(ticker: str = "PETR4"):
    try:
        return load_latest_model(ticker)
    except Exception as e:
        logger.error(f"Erro ao carregar modelo para {ticker}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao carregar modelo: {str(e)}")

# Rotas da API
@app.get("/")
async def root():
    return {"message": "Stock Price Prediction API", "status": "active"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/model/info", response_model=ModelInfo)
async def model_information(ticker: str = Query("PETR4", description="Ticker da ação")):
    """
    Retorna informações sobre o modelo atual.
    """
    try:
        info = get_model_info(ticker)
        return info
    except Exception as e:
        logger.error(f"Erro ao obter informações do modelo: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/next", response_model=List[PredictionResponse])
async def predict_next(
    days: int = Query(7, ge=1, le=30, description="Número de dias para prever"),
    ticker: str = Query("PETR4", description="Ticker da ação"),
    model=Depends(get_model)
):
    """
    Retorna previsões para os próximos dias.
    """
    try:
        predictions = predict_next_days(model, days)
        formatted = format_predictions(predictions)
        return formatted
    except Exception as e:
        logger.error(f"Erro ao fazer previsões: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/date", response_model=PredictionResponse)
async def predict_date(
    date: str = Query(..., description="Data para previsão (formato: YYYY-MM-DD)"),
    ticker: str = Query("PETR4", description="Ticker da ação"),
    model=Depends(get_model)
):
    """
    Retorna previsão para uma data específica.
    """
    try:
        target_date = datetime.strptime(date, "%Y-%m-%d")
        prediction = predict_specific_date(model, target_date)
        return prediction
    except ValueError:
        raise HTTPException(status_code=400, detail="Formato de data inválido. Use YYYY-MM-DD")
    except Exception as e:
        logger.error(f"Erro ao fazer previsão para data específica: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/range", response_model=List[PredictionResponse])
async def predict_range(
    start_date: str = Query(..., description="Data inicial (formato: YYYY-MM-DD)"),
    days: int = Query(7, ge=1, le=30, description="Número de dias para prever"),
    ticker: str = Query("PETR4", description="Ticker da ação"),
    model=Depends(get_model)
):
    """
    Retorna previsões para um intervalo de datas a partir de uma data inicial.
    """
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        predictions = predict_from_date(model, start, days)
        formatted = format_predictions(predictions)
        return formatted
    except ValueError:
        raise HTTPException(status_code=400, detail="Formato de data inválido. Use YYYY-MM-DD")
    except Exception as e:
        logger.error(f"Erro ao fazer previsões para intervalo: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Tratamento de erros global
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Erro não tratado: {str(exc)}")

    error_response = ErrorResponse(
        error="Erro interno do servidor",
        detail=str(exc)
    )

    return JSONResponse(
        status_code=500,
        content=jsonable_encoder(error_response)
    )

# Entry point
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)