import yfinance as yf
import pandas as pd
import logging
from datetime import datetime, timedelta

# Configurar logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_stock_data(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """
    Baixa dados históricos de ações do Yahoo Finance.
    
    Args:
        ticker: Símbolo da empresa (ex: 'PETR4.SA', 'VALE3.SA')
        period: Período de tempo (ex: '1d', '5d', '1mo', '3mo', '1y', '5y', 'max')
        interval: Intervalo de tempo (ex: '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
    
    Returns:
        DataFrame com os dados históricos
    """
    try:
        logger.info(f"Baixando dados para {ticker} com período {period} e intervalo {interval}")
        stock_data = yf.download(ticker, period=period, interval=interval)
        
        if stock_data.empty:
            logger.error(f"Nenhum dado encontrado para {ticker}")
            return pd.DataFrame()
        
        # Converte o índice para datetime se não for
        if not isinstance(stock_data.index, pd.DatetimeIndex):
            stock_data.index = pd.to_datetime(stock_data.index)
        
        # Resetar o índice para ter a data como coluna
        stock_data = stock_data.reset_index()
        
        # Renomear coluna de data para um nome padrão
        stock_data = stock_data.rename(columns={'Date': 'date', 'Datetime': 'date'})
        
        # Verificar colunas e padronizar nomes
        column_mapping = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adj_close',
            'Volume': 'volume'
        }
        
        stock_data = stock_data.rename(columns={col: column_mapping.get(col, col) for col in stock_data.columns})
        
        logger.info(f"Dados baixados com sucesso. Total de registros: {len(stock_data)}")
        
        # Salvar os dados brutos
        save_path = f"data/raw/{ticker.replace('.', '_')}_{datetime.now().strftime('%Y%m%d')}.csv"
        stock_data.to_csv(save_path, index=False)
        logger.info(f"Dados salvos em {save_path}")
        
        return stock_data
    
    except Exception as e:
        logger.error(f"Erro ao baixar dados: {str(e)}")
        raise

def get_latest_stock_data(ticker: str, days: int = 1) -> pd.DataFrame:
    """
    Obtém os dados mais recentes de uma ação.
    
    Args:
        ticker: Símbolo da empresa
        days: Número de dias para buscar
    
    Returns:
        DataFrame com os dados mais recentes
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    try:
        logger.info(f"Buscando dados recentes para {ticker} de {start_date.date()} até {end_date.date()}")
        stock_data = yf.download(ticker, start=start_date, end=end_date, interval="1d")
        
        if stock_data.empty:
            logger.warning(f"Nenhum dado recente encontrado para {ticker}")
            return pd.DataFrame()
        
        # Resetar o índice para ter a data como coluna
        stock_data = stock_data.reset_index()
        
        # Renomear coluna de data para um nome padrão
        stock_data = stock_data.rename(columns={'Date': 'date', 'Datetime': 'date'})
        
        # Verificar colunas e padronizar nomes
        column_mapping = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adj_close',
            'Volume': 'volume'
        }
        
        stock_data = stock_data.rename(columns={col: column_mapping.get(col, col) for col in stock_data.columns})
        
        logger.info(f"Dados recentes obtidos com sucesso. Total de registros: {len(stock_data)}")
        return stock_data
    
    except Exception as e:
        logger.error(f"Erro ao obter dados recentes: {str(e)}")
        raise


if __name__ == "__main__":
    # Exemplo de uso
    ticker = "PETR4.SA"  # Petrobras
    data = download_stock_data(ticker)
    print(data.head())
