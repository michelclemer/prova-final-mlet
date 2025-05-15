"""
Ponto de entrada principal da aplicação.
Este script inicializa e executa todos os componentes necessários.
"""
import os
import logging
import threading
import time
import argparse
import uvicorn
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Configurar logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler()
    ]
)

# Criar diretório de logs se não existir
os.makedirs("logs", exist_ok=True)

logger = logging.getLogger(__name__)


def start_monitoring():
    """Inicia o monitoramento do modelo em um thread separado."""
    try:
        from src.monitoring.metrics import ModelMonitor

        if os.getenv("ENABLE_MONITORING", "False").lower() == "true":
            logger.info("Iniciando monitoramento do modelo...")

            # Definir ticker
            symbol = os.getenv("STOCK_SYMBOL", "AAPL")
            monitor = ModelMonitor(symbol)

            interval = int(os.getenv("MONITORING_INTERVAL", 3600))  # Padrão: 1 hora

            while True:
                try:
                    # Coletar métricas
                    monitor.collect_metrics()

                    # Verificar drift
                    drift_result = monitor.check_for_drift()
                    logger.info(f"Status do monitoramento: {drift_result.get('status', 'unknown')}")

                    # Se detectar drift, talvez queira disparar retreinamento
                    if drift_result.get('drift_detected', False):
                        logger.warning(f"Drift detectado! {drift_result.get('message', '')}")

                        # Se configurado para retreinamento automático
                        if os.getenv("AUTO_RETRAIN", "False").lower() == "true":
                            try:
                                logger.info("Iniciando retreinamento automático...")
                                # Importar sob demanda para evitar problemas de circular import
                                from src.data.collect_data import download_stock_data
                                from src.data.process import preprocess_data
                                from src.models.train import train_model
                                from src.models.evaluate_model import evaluate_model

                                # Retreinar
                                download_stock_data(symbol)
                                preprocess_data(symbol=symbol)
                                train_model(symbol=symbol)
                                evaluate_model(symbol=symbol)

                                logger.info("Retreinamento automático concluído com sucesso!")
                            except Exception as e:
                                logger.error(f"Erro no retreinamento automático: {e}")

                    # Aguardar até o próximo ciclo
                    time.sleep(interval)
                except Exception as e:
                    logger.error(f"Erro no monitoramento: {e}")
                    time.sleep(60)  # Espera 1 minuto antes de tentar novamente

    except Exception as e:
        logger.error(f"Erro ao iniciar monitoramento: {e}")


def setup_model():
    """Configura o modelo inicial se necessário."""
    try:
        # Verificar se existe um modelo
        symbol = os.getenv("STOCK_SYMBOL", "AAPL")
        model_path = os.path.join("models", f"{symbol}_model.h5")

        if not os.path.exists(model_path):
            logger.info(f"Modelo não encontrado. Iniciando pipeline de treinamento para {symbol}...")

            # Importar sob demanda para evitar problemas de circular import
            from src.data.collect_data import download_stock_data
            from src.data.process import preprocess_data
            from src.models.train import train_model
            from src.models.evaluate_model import evaluate_model

            # Executar pipeline completo
            download_stock_data(symbol)
            preprocess_data(symbol=symbol)
            train_model(symbol=symbol)
            evaluate_model(symbol=symbol)

            logger.info("Pipeline de treinamento concluído com sucesso!")
        else:
            logger.info(f"Modelo encontrado em {model_path}")

    except Exception as e:
        logger.error(f"Erro na configuração do modelo: {e}")


def main():
    """Função principal para inicializar a aplicação."""
    logger.info("Iniciando aplicação de previsão de ações...")

    # Parsear argumentos da linha de comando
    parser = argparse.ArgumentParser(description='Stock Price Prediction API')
    parser.add_argument('--setup', action='store_true', help='Configurar o modelo antes de iniciar a API')
    parser.add_argument('--train', action='store_true', help='Forçar treinamento do modelo antes de iniciar a API')
    parser.add_argument('--monitoring', action='store_true', help='Habilitar monitoramento do modelo')
    args = parser.parse_args()

    # Se solicitado, configurar o modelo
    if args.setup or args.train:
        setup_model()

    # Se solicitado, definir monitoramento
    if args.monitoring:
        os.environ["ENABLE_MONITORING"] = "True"

    # Iniciar monitoramento em um thread separado se estiver habilitado
    if os.getenv("ENABLE_MONITORING", "False").lower() == "true":
        monitoring_thread = threading.Thread(target=start_monitoring, daemon=True)
        monitoring_thread.start()
        logger.info("Thread de monitoramento iniciado")

    # Iniciar a API
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))

    logger.info(f"Iniciando API em {host}:{port}")
    uvicorn.run("src.api.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()