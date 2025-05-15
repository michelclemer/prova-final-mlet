# Imagem base do Python
FROM python:3.9-slim

# Diretório de trabalho
WORKDIR /app

# Atualizar pip
RUN pip install --no-cache-dir --upgrade pip

# Instalar dependências necessárias para compilação
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copiar arquivo de requisitos
COPY requirements.txt .

# Instalar dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o código do projeto
COPY . .

# Criar diretórios necessários
RUN mkdir -p data/raw data/processed models monitoring

# Expor a porta
EXPOSE 8000

# Comando para iniciar a aplicação
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]