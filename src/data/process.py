import pandas as pd
import numpy as np
import logging
import os
import pickle
from datetime import datetime
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_prophet_model(train_data: pd.DataFrame, params: dict = None) -> Prophet:
    """
    Treina um modelo Prophet com os dados fornecidos.

    Args:
        train_data: DataFrame com colunas 'ds' (data) e 'y' (valor a prever)
        params: Dicionário com parâmetros para o modelo Prophet

    Returns:
        Modelo Prophet treinado
    """
    if train_data.empty or 'ds' not in train_data.columns or 'y' not in train_data.columns:
        logger.error("Dados de treino inválidos")
        raise ValueError("Dados de treino devem conter colunas 'ds' e 'y'")

    # Parâmetros padrão
    default_params = {
        'changepoint_prior_scale': 0.05,
        'seasonality_prior_scale': 10.0,
        'holidays_prior_scale': 10.0,
        'seasonality_mode': 'multiplicative',
        'weekly_seasonality': True,
        'daily_seasonality': False,
        'yearly_seasonality': True
    }

    # Atualizar com parâmetros fornecidos
    if params:
        default_params.update(params)

    try:
        logger.info("Iniciando treinamento do modelo Prophet")
        logger.info(f"Parâmetros: {default_params}")

        # Criar e configurar o modelo
        model = Prophet(
            changepoint_prior_scale=default_params['changepoint_prior_scale'],
            seasonality_prior_scale=default_params['seasonality_prior_scale'],
            holidays_prior_scale=default_params['holidays_prior_scale'],
            seasonality_mode=default_params['seasonality_mode']
        )

        # Configurar sazonalidades
        if default_params['weekly_seasonality']:
            model.add_seasonality(name='weekly', period=7, fourier_order=3)

        if default_params['yearly_seasonality']:
            model.add_seasonality(name='yearly', period=365.25, fourier_order=10)

        # Treinar o modelo
        model.fit(train_data)

        logger.info("Modelo treinado com sucesso")
        return model

    except Exception as e:
        logger.error(f"Erro ao treinar modelo: {str(e)}")
        raise


def evaluate_prophet_model(model: Prophet, test_data: pd.DataFrame) -> dict:
    """
    Avalia o modelo Prophet nos dados de teste.

    Args:
        model: Modelo Prophet treinado
        test_data: DataFrame com dados de teste, contendo colunas 'ds' e 'y'

    Returns:
        Dicionário com métricas de avaliação
    """
    if test_data.empty or 'ds' not in test_data.columns or 'y' not in test_data.columns:
        logger.error("Dados de teste inválidos")
        raise ValueError("Dados de teste devem conter colunas 'ds' e 'y'")

    try:
        logger.info("Avaliando o modelo nos dados de teste")

        # Fazer previsões para as datas de teste
        forecast = model.predict(test_data[['ds']])

        # Mesclar as previsões com os valores reais
        evaluation_df = pd.merge(test_data, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')

        # Calcular métricas
        mae = mean_absolute_error(evaluation_df['y'], evaluation_df['yhat'])
        rmse = np.sqrt(mean_squared_error(evaluation_df['y'], evaluation_df['yhat']))
        r2 = r2_score(evaluation_df['y'], evaluation_df['yhat'])

        # Calcular MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((evaluation_df['y'] - evaluation_df['yhat']) / evaluation_df['y'])) * 100

        # Calcular intervalo de confiança
        coverage = np.mean((evaluation_df['y'] >= evaluation_df['yhat_lower']) &
                           (evaluation_df['y'] <= evaluation_df['yhat_upper']))

        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'coverage': coverage
        }

        logger.info(f"Métricas de avaliação: {metrics}")
        return metrics

    except Exception as e:
        logger.error(f"Erro ao avaliar modelo: {str(e)}")
        raise


def save_model(model: Prophet, metrics: dict, ticker: str = 'stock') -> str:
    """
    Salva o modelo treinado e suas métricas.

    Args:
        model: Modelo Prophet treinado
        metrics: Dicionário com métricas de avaliação
        ticker: Ticker da ação

    Returns:
        Caminho do arquivo do modelo salvo
    """
    try:
        # Criar diretório se não existir
        os.makedirs("models", exist_ok=True)

        # Salvar o modelo
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f"models/{ticker}_prophet_{timestamp}.pkl"

        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)

        # Salvar métricas
        metrics_filename = f"models/{ticker}_metrics_{timestamp}.pkl"
        with open(metrics_filename, 'wb') as f:
            pickle.dump(metrics, f)

        logger.info(f"Modelo salvo em: {model_filename}")
        logger.info(f"Métricas salvas em: {metrics_filename}")

        # Salvar como modelo mais recente (para uso na API)
        latest_model_filename = f"models/{ticker}_prophet_latest.pkl"
        with open(latest_model_filename, 'wb') as f:
            pickle.dump(model, f)

        logger.info(f"Modelo mais recente salvo em: {latest_model_filename}")

        return model_filename

    except Exception as e:
        logger.error(f"Erro ao salvar modelo: {str(e)}")
        raise


def load_model(model_path: str) -> Prophet:
    """
    Carrega um modelo Prophet salvo.

    Args:
        model_path: Caminho para o arquivo do modelo

    Returns:
        Modelo Prophet carregado
    """
    try:
        logger.info(f"Carregando modelo de: {model_path}")

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        logger.info("Modelo carregado com sucesso")
        return model

    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {str(e)}")
        raise


def predict_future(model: Prophet, days: int = 30) -> pd.DataFrame:
    """
    Faz previsões para os próximos dias.

    Args:
        model: Modelo Prophet treinado
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

        logger.info("Previsões feitas com sucesso")
        return future_forecast

    except Exception as e:
        logger.error(f"Erro ao fazer previsões: {str(e)}")
        raise


def run_prophet_workflow(train_data: pd.DataFrame, test_data: pd.DataFrame, ticker: str, params: dict = None) -> tuple:
    """
    Executa todo o fluxo de trabalho para treinar, avaliar e salvar um modelo Prophet.

    Args:
        train_data: DataFrame com dados de treino
        test_data: DataFrame com dados de teste
        ticker: Ticker da ação
        params: Parâmetros para o modelo Prophet

    Returns:
        Tuple com (model, metrics, model_path)
    """
    try:
        # Treinar modelo
        model = train_prophet_model(train_data, params)

        # Avaliar modelo
        metrics = evaluate_prophet_model(model, test_data)

        # Salvar modelo
        model_path = save_model(model, metrics, ticker)

        # Fazer previsões futuras
        future_forecast = predict_future(model)

        return model, metrics, model_path, future_forecast

    except Exception as e:
        logger.error(f"Erro no fluxo de trabalho: {str(e)}")
        raise


if __name__ == "__main__":
    # Exemplo de uso
    try:
        # Verificar se há dados processados
        processed_dir = "data/processed"

        if not os.path.exists(processed_dir):
            logger.error(f"Diretório {processed_dir} não encontrado")
            exit(1)

        # Encontrar arquivos de treino e teste
        train_files = [f for f in os.listdir(processed_dir) if f.endswith('.csv') and 'train_prophet' in f]
        test_files = [f for f in os.listdir(processed_dir) if f.endswith('.csv') and 'test_prophet' in f]

        if not train_files or not test_files:
            logger.error("Arquivos de treino ou teste não encontrados")
            exit(1)

        # Usar os arquivos mais recentes
        latest_train = max(train_files, key=lambda x: os.path.getmtime(os.path.join(processed_dir, x)))
        latest_test = max(test_files, key=lambda x: os.path.getmtime(os.path.join(processed_dir, x)))

        logger.info(f"Usando arquivo de treino: {latest_train}")
        logger.info(f"Usando arquivo de teste: {latest_test}")

        # Carregar dados
        train_data = pd.read_csv(os.path.join(processed_dir, latest_train))
        test_data = pd.read_csv(os.path.join(processed_dir, latest_test))

        # Extrair ticker do nome do arquivo
        ticker = latest_train.split('_')[0]

        # Converter datas para datetime
        train_data['ds'] = pd.to_datetime(train_data['ds'])
        test_data['ds'] = pd.to_datetime(test_data['ds'])

        # Executar fluxo de trabalho
        model, metrics, model_path, future_forecast = run_prophet_workflow(train_data, test_data, ticker)

        logger.info(f"Fluxo de trabalho concluído com sucesso")
        logger.info(f"Métricas finais: {metrics}")
        logger.info(f"Modelo salvo em: {model_path}")

        # Salvar previsões futuras
        future_filename = f"data/processed/{ticker}_future_forecast.csv"
        future_forecast.to_csv(future_filename, index=False)
        logger.info(f"Previsões futuras salvas em: {future_filename}")

    except Exception as e:
        logger.error(f"Erro: {str(e)}")
