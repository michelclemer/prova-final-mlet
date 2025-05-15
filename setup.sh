#!/bin/bash
# Script para inicializar o projeto

# Verificar dependências
echo "Verificando dependências..."
pip install -r requirements.txt

# Criar diretórios necessários
echo "Criando diretórios..."
mkdir -p data/raw data/processed models monitoring reports/figures logs

# Configurar variáveis de ambiente
echo "Configurando ambiente..."
if [ ! -f .env ]; then
  cp .env.example .env
  echo "Arquivo .env criado. Por favor, verifique as configurações."
fi

# Treinar modelo inicial se necessário
echo "Verificando modelo..."
SYMBOL=${1:-AAPL}  # Usar o primeiro argumento ou AAPL como padrão

if [ ! -f "models/${SYMBOL}_model.h5" ]; then
  echo "Modelo não encontrado. Iniciando treinamento inicial para ${SYMBOL}..."
  python -m src.data.collect_data ${SYMBOL}
  python -m src.data.process --symbol ${SYMBOL}
  python -m src.models.train --symbol ${SYMBOL}
  python -m src.models.evaluate_model --symbol ${SYMBOL}
  echo "Treinamento inicial concluído!"
else
  echo "Modelo encontrado para ${SYMBOL}. Pulando treinamento inicial."
fi

# Iniciar API
echo "Configuração concluída! Para iniciar a API, execute:"
echo "python app.py"