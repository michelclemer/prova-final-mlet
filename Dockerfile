# Imagem base do Python
FROM python:3.9-slim

# Definir variáveis de ambiente
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000 \
    HOST=0.0.0.0 \
    STOCK_SYMBOL=PETR4.SA \
    ENABLE_MONITORING=True \
    TZ=America/Sao_Paulo

# Configurar timezone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Diretório de trabalho
WORKDIR /app

# Instalar dependências para compilação
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Atualizar pip
RUN pip install --no-cache-dir --upgrade pip

# Copiar apenas o arquivo de requisitos primeiro (para cache de camadas)
COPY requirements.txt .

# Instalar dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o código do projeto
COPY . .

# Criar diretórios necessários
RUN mkdir -p data/raw data/processed models monitoring reports/figures logs

# Executar setup inicial para treinar um modelo com dados simulados
RUN python setup.py --no-train || echo "Continuando mesmo se o setup falhar"

# Expor a porta
EXPOSE $PORT

# Script de inicialização
RUN echo '#!/bin/bash\n\
echo "Iniciando setup inicial..."\n\
python setup.py\n\
echo "Iniciando a API..."\n\
python app.py\n\
' > /app/docker-entrypoint.sh \
&& chmod +x /app/docker-entrypoint.sh

# Comando para iniciar a aplicação
ENTRYPOINT ["/app/docker-entrypoint.sh"]