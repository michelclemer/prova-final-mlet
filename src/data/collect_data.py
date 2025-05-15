import yfinance as yf
import pandas as pd
import logging
import os
import numpy as np
from datetime import datetime, timedelta

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Importar o gerador de dados simulados
try:
    from src.data.simulate_data import generate_stock_data
except ImportError:
    # Se estiver executando do mesmo diretório
    try:
        from simulate_data import generate_stock_data
    except ImportError:
        # Definir uma função simples para o caso de falha na importação
        def generate_stock_data(symbol="TEST", days=500, **kwargs):
            logger.warning("Função de simulação completa não disponível. Usando versão simples.")
            dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
            np.random.seed(42)
            price = 100.0
            prices = [price]
            for i in range(1, len(dates)):
                price = price * (1 + np.random.normal(0, 0.02))
                prices.append(price)

            df = pd.DataFrame({
                'date': dates,
                'open': prices,
                'high': [p * 1.01 for p in prices],
                'low': [p * 0.99 for p in prices],
                'close': prices,
                'adj_close': prices,
                'volume': [int(1e6) for _ in prices]
            })

            path = f"data/raw/{symbol.replace('.', '_')}_{datetime.now().strftime('%Y%m%d')}.csv"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_csv(path, index=False)
            return df, path


def download_stock_data(ticker: str, period: str = "2y", interval: str = "1d",
                        use_simulated: bool = False) -> pd.DataFrame:
    """
    Baixa dados históricos de ações do Yahoo Finance.

    Args:
        ticker: Símbolo da empresa (ex: 'PETR4.SA', 'VALE3.SA')
        period: Período de tempo (ex: '1d', '5d', '1mo', '3mo', '1y', '2y', 'max')
        interval: Intervalo de tempo (ex: '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        use_simulated: Se True, gera dados simulados diretamente

    Returns:
        DataFrame com os dados históricos
    """
    # Se solicitado dados simulados, gerar imediatamente
    if use_simulated:
        logger.info(f"Gerando dados simulados para {ticker} conforme solicitado")

        # Determinar número de dias com base no período
        days = 500  # padrão
        if period == "1d":
            days = 1
        elif period == "5d":
            days = 5
        elif period == "1mo":
            days = 30
        elif period == "3mo":
            days = 90
        elif period == "6mo":
            days = 180
        elif period == "1y":
            days = 365
        elif period == "2y":
            days = 730
        elif period == "5y":
            days = 1825
        elif period == "max":
            days = 2000

        # Gerar dados simulados
        df, path = generate_stock_data(symbol=ticker, days=days)
        logger.info(f"Dados simulados gerados com sucesso: {len(df)} registros")
        return df

    try:
        logger.info(f"Baixando dados para {ticker} com período {period} e intervalo {interval}")

        # Tente diferentes variações do ticker se o original falhar
        variants = [
            ticker,  # Ticker original
            f"{ticker}.SA" if ".SA" not in ticker else ticker,  # Adicionar .SA para tickers brasileiros
            ticker.replace(".SA", "")  # Remover .SA se já existir
        ]

        # Tente períodos diferentes se o original falhar
        periods = [period, "1y", "max"]

        stock_data = pd.DataFrame()
        error_msg = ""

        # Tentar diferentes combinações de ticker e período
        for variant in variants:
            if not stock_data.empty:
                break

            for p in periods:
                try:
                    logger.info(f"Tentando baixar dados para {variant} com período {p}")
                    data = yf.download(variant, period=p, interval=interval)

                    if not data.empty:
                        logger.info(f"Download bem-sucedido para {variant} com período {p}")
                        stock_data = data
                        break
                except Exception as e:
                    error_msg = str(e)
                    logger.warning(f"Falha ao tentar {variant} com período {p}: {error_msg}")

        # Se todas as tentativas falharem, gerar dados simulados
        if stock_data.empty:
            logger.warning(f"Nenhum dado encontrado para {ticker}. Gerando dados simulados como fallback.")

            # Determinar número de dias com base no período
            days = 500  # padrão
            if period == "1d":
                days = 1
            elif period == "5d":
                days = 5
            elif period == "1mo":
                days = 30
            elif period == "3mo":
                days = 90
            elif period == "6mo":
                days = 180
            elif period == "1y":
                days = 365
            elif period == "2y":
                days = 730
            elif period == "5y":
                days = 1825
            elif period == "max":
                days = 2000

            # Gerar dados simulados
            stock_data, _ = generate_stock_data(symbol=ticker, days=days)
            logger.info(f"Dados simulados gerados como fallback: {len(stock_data)} registros")

        # Converte o índice para datetime se não for
        if not isinstance(stock_data.index, pd.DatetimeIndex):
            stock_data.index = pd.to_datetime(stock_data.index)

        # Resetar o índice para ter a data como coluna
        if 'date' not in stock_data.columns:
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

        logger.info(f"Dados preparados com sucesso. Total de registros: {len(stock_data)}")

        # Certificar que o diretório existe
        os.makedirs("data/raw", exist_ok=True)

        # Salvar os dados brutos
        save_path = f"data/raw/{ticker.replace('.', '_')}_{datetime.now().strftime('%Y%m%d')}.csv"
        stock_data.to_csv(save_path, index=False)
        logger.info(f"Dados salvos em {save_path}")

        return stock_data

    except Exception as e:
        logger.error(f"Erro ao baixar/gerar dados: {str(e)}")
        raise


def get_latest_stock_data(ticker: str, days: int = 1, use_simulated: bool = False) -> pd.DataFrame:
    """
    Obtém os dados mais recentes de uma ação.

    Args:
        ticker: Símbolo da empresa
        days: Número de dias para buscar
        use_simulated: Se True, gera dados simulados

    Returns:
        DataFrame com os dados mais recentes
    """
    # Se solicitado dados simulados, gerar imediatamente
    if use_simulated:
        logger.info(f"Gerando dados simulados recentes para {ticker}")
        df, _ = generate_stock_data(symbol=ticker, days=days)
        return df

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    try:
        logger.info(f"Buscando dados recentes para {ticker} de {start_date.date()} até {end_date.date()}")
        stock_data = yf.download(ticker, start=start_date, end=end_date, interval="1d")

        if stock_data.empty:
            logger.warning(f"Nenhum dado recente encontrado para {ticker}. Gerando dados simulados.")
            stock_data, _ = generate_stock_data(symbol=ticker, days=days)
            return stock_data

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
        logger.warning(f"Erro ao obter dados recentes: {str(e)}. Gerando dados simulados.")
        stock_data, _ = generate_stock_data(symbol=ticker, days=days)
        return stock_data


def collect_stock_data(symbol: str = "PETR4.SA", start_date: str = None, end_date: str = None,
                       save_path: str = None, use_simulated: bool = False) -> pd.DataFrame:
    """
    Coleta dados históricos para um símbolo específico.

    Args:
        symbol: Símbolo da ação
        start_date: Data de início (formato 'YYYY-MM-DD')
        end_date: Data de fim (formato 'YYYY-MM-DD')
        save_path: Caminho para salvar os dados
        use_simulated: Se True, força o uso de dados simulados

    Returns:
        DataFrame com os dados históricos
    """
    try:
        # Se solicitado dados simulados, gerar imediatamente
        if use_simulated:
            logger.info(f"Gerando dados simulados para {symbol} conforme solicitado")

            # Determinar número de dias com base nas datas fornecidas
            days = 500  # padrão
            if start_date and end_date:
                start = datetime.strptime(start_date, "%Y-%m-%d")
                end = datetime.strptime(end_date, "%Y-%m-%d")
                days = (end - start).days

            # Gerar dados simulados
            stock_data, path = generate_stock_data(symbol=symbol, days=days)
            logger.info(f"Dados simulados gerados com sucesso: {len(stock_data)} registros")

            # Salvar se caminho fornecido
            if save_path and save_path != path:
                stock_data.to_csv(save_path, index=False)
                logger.info(f"Dados simulados salvos em {save_path}")

            return stock_data

        logger.info(f"Coletando dados históricos para {symbol}")

        # Se datas não fornecidas, usar valores padrão
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")

        if not start_date:
            # Por padrão, pegar 2 anos de dados
            start_date = (datetime.now() - timedelta(days=2 * 365)).strftime("%Y-%m-%d")

        logger.info(f"Período: {start_date} a {end_date}")

        # Baixar dados
        try:
            stock_data = yf.download(symbol, start=start_date, end=end_date)

            if stock_data.empty:
                logger.warning(f"Nenhum dado encontrado para {symbol}. Gerando dados simulados.")

                # Calcular número de dias
                start = datetime.strptime(start_date, "%Y-%m-%d")
                end = datetime.strptime(end_date, "%Y-%m-%d")
                days = (end - start).days

                # Gerar dados simulados
                stock_data, _ = generate_stock_data(symbol=symbol, days=days)
        except Exception as e:
            logger.warning(f"Erro ao baixar dados para {symbol}: {e}. Gerando dados simulados.")

            # Calcular número de dias
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            days = (end - start).days

            # Gerar dados simulados
            stock_data, _ = generate_stock_data(symbol=symbol, days=days)

        # Resetar o índice para ter a data como coluna
        stock_data = stock_data.reset_index()

        # Garantir que o diretório existe
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            stock_data.to_csv(save_path, index=False)
            logger.info(f"Dados salvos em {save_path}")

        logger.info(f"Dados coletados com sucesso. Total de registros: {len(stock_data)}")
        return stock_data

    except Exception as e:
        logger.error(f"Erro ao coletar dados: {str(e)}")

        # Último recurso: tentar gerar dados simulados
        try:
            logger.warning("Tentando gerar dados simulados como último recurso...")
            stock_data, _ = generate_stock_data(symbol=symbol, days=500)
            return stock_data
        except:
            raise  # Se mesmo a simulação falhar, propagar o erro originaldirname(save_path), exist_ok=True)
            stock_data.to_csv(save_path, index=False)
            logger.info(f"Dados salvos em {save_path}")

        logger.info(f"Dados coletados com sucesso. Total de registros: {len(stock_data)}")
        return stock_data

    except Exception as e:
        logger.error(f"Erro ao coletar dados: {str(e)}")
        raise


if __name__ == "__main__":
    # Exemplo de uso
    ticker = "AAPL"  # Apple
    data = download_stock_data(ticker)
    print(data.head())