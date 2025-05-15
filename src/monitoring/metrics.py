import pandas as pd
import numpy as np
import logging
import os
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Diretório para armazenar métricas de monitoramento
MONITORING_DIR = "monitoring"
os.makedirs(MONITORING_DIR, exist_ok=True)

def calculate_metrics(actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
    """
    Calcula métricas de desempenho para um conjunto de valores reais e previstos.
    
    Args:
        actual: Série com valores reais
        predicted: Série com valores previstos
        
    Returns:
        Dicionário com métricas calculadas
    """
    if len(actual) != len(predicted):
        logger.error(f"Tamanhos incompatíveis: actual={len(actual)}, predicted={len(predicted)}")
        raise ValueError("Os conjuntos de valores reais e previstos devem ter o mesmo tamanho")
    
    # Calcular métricas básicas
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    
    # Calcular MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    # Calcular erro médio (ME) para verificar viés
    me = np.mean(predicted - actual)
    
    # Calcular desvio padrão dos erros
    std_error = np.std(predicted - actual)
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'me': me,
        'std_error': std_error,
        'n_samples': len(actual)
    }
    
    return metrics

def log_prediction_metrics(
    ticker: str,
    date: datetime,
    actual_value: float,
    predicted_value: float,
    prediction_date: datetime,
    metadata: Dict = None
) -> None:
    """
    Registra métricas de uma única previsão para monitoramento contínuo.
    
    Args:
        ticker: Ticker da ação
        date: Data da previsão
        actual_value: Valor real observado
        predicted_value: Valor previsto pelo modelo
        prediction_date: Data em que a previsão foi feita
        metadata: Metadados adicionais para registro
    """
    try:
        # Criar registro de previsão
        prediction_record = {
            'ticker': ticker,
            'date': date.strftime('%Y-%m-%d'),
            'actual_value': float(actual_value),
            'predicted_value': float(predicted_value),
            'prediction_date': prediction_date.strftime('%Y-%m-%d'),
            'prediction_age_days': (date - prediction_date).days,
            'error': float(actual_value - predicted_value),
            'abs_error': float(abs(actual_value - predicted_value)),
            'pct_error': float(abs(actual_value - predicted_value) / actual_value * 100),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Adicionar metadados se fornecidos
        if metadata:
            prediction_record.update(metadata)
        
        # Caminho para o arquivo de registro
        log_path = os.path.join(MONITORING_DIR, f"{ticker}_prediction_log.jsonl")
        
        # Adicionar registro ao arquivo
        with open(log_path, 'a') as f:
            f.write(json.dumps(prediction_record) + '\n')
        
        logger.info(f"Registro de previsão salvo para {ticker} em {date}")
    
    except Exception as e:
        logger.error(f"Erro ao registrar previsão: {str(e)}")

def get_prediction_logs(ticker: str, days: int = 30) -> pd.DataFrame:
    """
    Recupera logs de previsões para um ticker específico.
    
    Args:
        ticker: Ticker da ação
        days: Número de dias para recuperar (0 para todos)
        
    Returns:
        DataFrame com logs de previsões
    """
    try:
        log_path = os.path.join(MONITORING_DIR, f"{ticker}_prediction_log.jsonl")
        
        if not os.path.exists(log_path):
            logger.warning(f"Arquivo de log não encontrado: {log_path}")
            return pd.DataFrame()
        
        # Carregar logs como DataFrame
        logs = []
        with open(log_path, 'r') as f:
            for line in f:
                logs.append(json.loads(line.strip()))
        
        logs_df = pd.DataFrame(logs)
        
        # Converter colunas de data para datetime
        date_columns = ['date', 'prediction_date', 'timestamp']
        for col in date_columns:
            if col in logs_df.columns:
                logs_df[col] = pd.to_datetime(logs_df[col])
        
        # Filtrar por período, se necessário
        if days > 0:
            cutoff_date = datetime.now() - timedelta(days=days)
            logs_df = logs_df[logs_df['date'] >= cutoff_date]
        
        return logs_df
    
    except Exception as e:
        logger.error(f"Erro ao recuperar logs de previsão: {str(e)}")
        return pd.DataFrame()

def calculate_drift_metrics(ticker: str, window_days: int = 30) -> Dict[str, Any]:
    """
    Calcula métricas de drift para monitorar o desempenho do modelo ao longo do tempo.
    
    Args:
        ticker: Ticker da ação
        window_days: Janela de dias para análise
        
    Returns:
        Dicionário com métricas de drift
    """
    try:
        # Obter logs de previsão
        logs_df = get_prediction_logs(ticker, window_days)
        
        if logs_df.empty:
            logger.warning(f"Não há logs suficientes para calcular métricas de drift para {ticker}")
            return {}
        
        # Calcular métricas básicas
        metrics = calculate_metrics(logs_df['actual_value'], logs_df['predicted_value'])
        
        # Adicionar metadados
        metrics['ticker'] = ticker
        metrics['window_days'] = window_days
        metrics['start_date'] = logs_df['date'].min().strftime('%Y-%m-%d')
        metrics['end_date'] = logs_df['date'].max().strftime('%Y-%m-%d')
        metrics['calculation_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Calcular tendência de erro (correlação entre data e erro)
        if len(logs_df) >= 5:  # Precisamos de algumas amostras para uma correlação significativa
            logs_df['date_numeric'] = (logs_df['date'] - logs_df['date'].min()).dt.days
            error_trend = np.corrcoef(logs_df['date_numeric'], logs_df['error'])[0, 1]
            metrics['error_trend'] = float(error_trend)
        
        # Salvar métricas de drift
        drift_path = os.path.join(MONITORING_DIR, f"{ticker}_drift_metrics.json")
        with open(drift_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Métricas de drift calculadas e salvas para {ticker}")
        return metrics
    
    except Exception as e:
        logger.error(f"Erro ao calcular métricas de drift: {str(e)}")
        return {}

def check_model_performance(ticker: str, threshold_mape: float = 10.0) -> Dict[str, Any]:
    """
    Verifica o desempenho do modelo e determina se é necessário retreinar.
    
    Args:
        ticker: Ticker da ação
        threshold_mape: Limiar de MAPE para considerar retreino
        
    Returns:
        Dicionário com resultados da verificação
    """
    try:
        # Calcular métricas de drift
        drift_metrics = calculate_drift_metrics(ticker)
        
        if not drift_metrics:
            logger.warning(f"Não foi possível calcular métricas de drift para {ticker}")
            return {
                'status': 'unknown',
                'message': 'Não há dados suficientes para avaliar o desempenho do modelo',
                'retrain_recommended': True
            }
        
        # Verificar se MAPE está acima do limiar
        mape = drift_metrics.get('mape', 0)
        retrain_recommended = mape > threshold_mape
        
        # Verificar se há tendência significativa de erro (drift)
        error_trend = drift_metrics.get('error_trend', 0)
        drift_detected = abs(error_trend) > 0.3  # Correlação moderada entre data e erro
        
        # Determinar status do modelo
        if retrain_recommended and drift_detected:
            status = 'critical'
            message = f"Desempenho do modelo está crítico (MAPE={mape:.2f}%, tendência de erro={error_trend:.2f})"
        elif retrain_recommended:
            status = 'warning'
            message = f"Desempenho do modelo está abaixo do esperado (MAPE={mape:.2f}%)"
        elif drift_detected:
            status = 'warning'
            message = f"Possível drift detectado (tendência de erro={error_trend:.2f})"
        else:
            status = 'ok'
            message = f"Desempenho do modelo está adequado (MAPE={mape:.2f}%)"
        
        result = {
            'status': status,
            'message': message,
            'retrain_recommended': retrain_recommended or drift_detected,
            'metrics': drift_metrics
        }
        
        # Salvar resultado da verificação
        check_path = os.path.join(MONITORING_DIR, f"{ticker}_performance_check.json")
        with open(check_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Verificação de desempenho concluída para {ticker}: {status}")
        return result
    
    except Exception as e:
        logger.error(f"Erro ao verificar desempenho do modelo: {str(e)}")
        return {
            'status': 'error',
            'message': f"Erro ao verificar desempenho: {str(e)}",
            'retrain_recommended': True
        }

def update_model_monitoring(ticker: str, actual_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Atualiza o monitoramento do modelo com novos dados reais.
    
    Args:
        ticker: Ticker da ação
        actual_data: DataFrame com dados reais recentes (com colunas 'date' e 'close')
        
    Returns:
        Dicionário com resultados da atualização
    """
    try:
        logger.info(f"Atualizando monitoramento do modelo para {ticker}")
        
        # Verificar se há um modelo salvo
        model_path = f"models/{ticker}_prophet_latest.pkl"
        if not os.path.exists(model_path):
            logger.error(f"Modelo não encontrado: {model_path}")
            return {
                'status': 'error',
                'message': f"Modelo não encontrado: {model_path}"
            }
        
        # Carregar o modelo
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Verificar se há previsões anteriores salvas
        forecast_path = f"data/processed/{ticker}_future_forecast.csv"
        if not os.path.exists(forecast_path):
            logger.warning(f"Arquivo de previsões não encontrado: {forecast_path}")
            return {
                'status': 'error',
                'message': f"Arquivo de previsões não encontrado: {forecast_path}"
            }
        
        # Carregar previsões anteriores
        forecasts = pd.read_csv(forecast_path)
        forecasts['ds'] = pd.to_datetime(forecasts['ds'])
        
        # Processar dados reais
        actual_data = actual_data.copy()
        if 'date' in actual_data.columns:
            actual_data['date'] = pd.to_datetime(actual_data['date'])
        
        # Comparar previsões com valores reais
        merged_data = pd.merge(
            forecasts, 
            actual_data,
            left_on='ds',
            right_on='date',
            how='inner'
        )
        
        # Registrar métricas para cada dia
        prediction_date = datetime.fromtimestamp(os.path.getmtime(forecast_path))
        
        for _, row in merged_data.iterrows():
            log_prediction_metrics(
                ticker=ticker,
                date=row['ds'],
                actual_value=row['close'],
                predicted_value=row['yhat'],
                prediction_date=prediction_date,
                metadata={
                    'yhat_lower': float(row['yhat_lower']),
                    'yhat_upper': float(row['yhat_upper'])
                }
            )
        
        # Verificar desempenho do modelo
        performance = check_model_performance(ticker)
        
        # Determinar próximas ações
        next_actions = []
        if performance.get('retrain_recommended', False):
            next_actions.append("Retreinar o modelo com dados mais recentes")
        
        if performance.get('status') == 'critical':
            next_actions.append("Revisar parâmetros do modelo")
            next_actions.append("Avaliar a inclusão de features adicionais")
        
        result = {
            'status': 'success',
            'message': f"Monitoramento atualizado para {ticker}",
            'performance': performance,
            'metrics_updated': len(merged_data),
            'next_actions': next_actions
        }
        
        logger.info(f"Monitoramento atualizado para {ticker}: {performance.get('status', 'unknown')}")
        return result
    
    except Exception as e:
        logger.error(f"Erro ao atualizar monitoramento: {str(e)}")
        return {
            'status': 'error',
            'message': f"Erro ao atualizar monitoramento: {str(e)}"
        }

if __name__ == "__main__":
    # Exemplo de uso
    try:
        from src.data.collect import get_latest_stock_data
        
        # Definir ticker
        ticker = "PETR4"
        
        # Obter dados recentes
        actual_data = get_latest_stock_data(ticker, days=10)
        
        if not actual_data.empty:
            # Atualizar monitoramento
            result = update_model_monitoring(ticker, actual_data)
            
            print(f"Status: {result['status']}")
            print(f"Mensagem: {result['message']}")
            
            if 'performance' in result and result['performance']:
                print(f"\nDesempenho do modelo: {result['performance']['status']}")
                print(f"Retreino recomendado: {result['performance']['retrain_recommended']}")
                
                if 'metrics' in result['performance'] and result['performance']['metrics']:
                    metrics = result['performance']['metrics']
                    print(f"\nMétricas:")
                    print(f"MAPE: {metrics.get('mape', 0):.2f}%")
                    print(f"MAE: {metrics.get('mae', 0):.4f}")
                    print(f"RMSE: {metrics.get('rmse', 0):.4f}")
            
            if 'next_actions' in result and result['next_actions']:
                print("\nPróximas ações recomendadas:")
                for i, action in enumerate(result['next_actions'], 1):
                    print(f"{i}. {action}")
        else:
            print("Não foi possível obter dados recentes")
    
    except Exception as e:
        logger.error(f"Erro: {str(e)}")