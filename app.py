"""
Ponto de entrada principal da aplicação.
Este script inicializa e executa todos os componentes necessários.
"""
import os
import logging
from dotenv import load_dotenv
import uvicorn
import threading
import time

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

logger = logging.getLogger(__name__)


def start_monitoring():
    """Inicia o monitoramento do modelo em um thread separado."""
    from src.monitoring.metrics import ModelMonitor

    if os.getenv("ENABLE_MONITORING", "False").lower() == "true":
        logger.info("Iniciando monitoramento do modelo...")
        monitor = ModelMonitor()
        interval = int(os.getenv("MONITORING_INTERVAL", 3600))  # Padrão: 1 hora

        while True:
            try:
                monitor.collect_metrics()
                monitor.check_for_drift()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Erro no monitoramento: {e}")
                time.sleep(60)  # Espera 1 minuto antes de tentar novamente


def main():
    """Função principal para inicializar a aplicação."""
    logger.info("Iniciando aplicação de previsão de ações...")

    # Iniciar monitoramento em um thread separado se estiver habilitado
    if os.getenv("ENABLE_MONITORING", "False").lower() == "true":
        monitoring_thread = threading.Thread(target=start_monitoring, daemon=True)
        monitoring_thread.start()

    # Iniciar a API
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))

    logger.info(f"Iniciando API em {host}:{port}")
    uvicorn.run("src.api.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()