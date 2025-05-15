"""
Script para configuração inicial do projeto.
"""
import os
import sys
import logging
import argparse
import subprocess

from src.data.collect_data import generate_stock_data

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("setup")


def create_directories():
    """Cria os diretórios necessários para o projeto."""
    directories = [
        "data/raw",
        "data/processed",
        "models",
        "monitoring",
        "reports/figures",
        "logs"
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Diretório criado ou verificado: {directory}")


def setup_environment():
    """Configura o ambiente virtual e instala dependências."""
    try:
        # Instalar dependências
        logger.info("Instalando dependências...")
        subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)

        # Criar arquivo .env se não existir
        if not os.path.exists(".env"):
            if os.path.exists(".env.example"):
                with open(".env.example", "r") as source:
                    with open(".env", "w") as target:
                        target.write(source.read())
                logger.info("Arquivo .env criado. Por favor, verifique as configurações.")
            else:
                logger.warning("Arquivo .env.example não encontrado. Criando .env vazio.")
                with open(".env", "w") as f:
                    f.write("# Configurações da aplicação\n")
                    f.write("API_HOST=0.0.0.0\n")
                    f.write("API_PORT=8000\n")
                    f.write("LOG_LEVEL=INFO\n")
                    f.write("STOCK_SYMBOL=AAPL\n")
                    f.write("ENABLE_MONITORING=True\n")
                    f.write("MONITORING_INTERVAL=3600\n")
                    f.write("AUTO_RETRAIN=False\n")

    except subprocess.CalledProcessError as e:
        logger.error(f"Erro ao configurar ambiente: {e}")
        sys.exit(1)


def train_initial_model(symbol):
    """Treina o modelo inicial para um símbolo específico."""
    try:
        model_path = os.path.join("models", f"{symbol}_model.h5")

        if not os.path.exists(model_path):
            logger.info(f"Modelo não encontrado. Iniciando treinamento inicial para {symbol}...")

            # Coleta de dados
            logger.info("Coletando dados históricos...")
            from src.data.collect_data import download_stock_data
            # Usar dados simulados para garantir que funcione
            stock_data = download_stock_data(symbol, use_simulated=True)

            # Pré-processamento
            logger.info("Pré-processando dados...")
            from src.data.process import preprocess_data
            processed_data = preprocess_data(symbol=symbol)

            # Treinamento
            logger.info("Treinando modelo...")
            from src.models.train import train_model
            train_model(symbol=symbol)

            # Avaliação
            logger.info("Avaliando modelo...")
            from src.models.evaluate_model import evaluate_model
            metrics = evaluate_model(symbol=symbol)

            logger.info(f"Treinamento concluído! Métricas: {metrics}")
        else:
            logger.info(f"Modelo encontrado para {symbol}. Pulando treinamento inicial.")

    except Exception as e:
        logger.error(f"Erro ao treinar modelo inicial: {e}")
        # Usar dados simulados como último recurso
        try:
            logger.warning("Tentando novamente com dados 100% simulados...")


            # Gerar dados simulados
            stock_data, path = generate_stock_data(symbol=symbol, days=500)
            logger.info(f"Dados simulados gerados com sucesso: {len(stock_data)} registros")

            # Pré-processamento
            logger.info("Pré-processando dados simulados...")
            from src.data.process import preprocess_data
            processed_data = preprocess_data(input_path=path, symbol=symbol)

            # Treinamento
            logger.info("Treinando modelo com dados simulados...")
            from src.models.train import train_model
            train_model(symbol=symbol)

            # Avaliação
            logger.info("Avaliando modelo com dados simulados...")
            from src.models.evaluate_model import evaluate_model
            metrics = evaluate_model(symbol=symbol)

            logger.info(f"Treinamento com dados simulados concluído! Métricas: {metrics}")

        except Exception as e2:
            logger.error(f"Erro final ao treinar modelo com dados simulados: {e2}")
            sys.exit(1)


def main():
    """Função principal para configuração do projeto."""
    parser = argparse.ArgumentParser(description="Configuração inicial do projeto")
    parser.add_argument("symbol", nargs="?", default="AAPL", help="Símbolo da ação para treinamento inicial")
    parser.add_argument("--no-train", action="store_true", help="Não realizar treinamento inicial")

    args = parser.parse_args()

    logger.info("Iniciando configuração do projeto...")

    # Criar diretórios
    create_directories()

    # Configurar ambiente
    setup_environment()

    # Treinar modelo inicial (se necessário)
    if not args.no_train:
        train_initial_model(args.symbol)

    logger.info("Configuração concluída! Para iniciar a API, execute: python app.py")


if __name__ == "__main__":
    main()