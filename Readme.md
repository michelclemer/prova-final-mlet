Projeto de Previsão de Ações - MLOps
Este projeto implementa um sistema de previsão de preços de ações usando Prophet, seguindo as melhores práticas de MLOps para o deployment do modelo.
Estrutura do Projeto
stock-prediction/
├── data/                      # Armazenamento de dados
│   ├── raw/                   # Dados brutos
│   └── processed/             # Dados processados
├── models/                    # Modelos serializados
├── notebooks/                 # Notebooks de análise
│   └── exploratory.ipynb      # Análise exploratória
├── src/                       # Código fonte
│   ├── data/                  # Scripts de coleta e processamento de dados
│   │   ├── collect.py         # Coleta dados da API
│   │   └── process.py         # Processa os dados
│   ├── models/                # Scripts de modelagem
│   │   ├── train.py           # Treina o modelo
│   │   └── predict.py         # Faz previsões
│   ├── api/                   # API de serviço
│   │   ├── main.py            # Ponto de entrada da API
│   │   └── routers/           # Rotas da API
│   └── monitoring/            # Scripts de monitoramento
│       └── metrics.py         # Métricas de desempenho
├── tests/                     # Testes
├── .gitignore                 # Arquivos ignorados pelo Git
├── requirements.txt           # Dependências
├── Dockerfile                 # Para containerização
├── docker-compose.yml         # Para orquestração
└── README.md                  # Documentação principal
Tecnologias Utilizadas

Python 3.9: Linguagem de programação
Prophet: Algoritmo de séries temporais
FastAPI: Framework para API
Docker: Containerização
Pandas/NumPy: Manipulação de dados
yfinance: API para dados de ações
scikit-learn: Métricas de avaliação

Funcionalidades

Coleta de dados históricos de ações
Processamento e preparação de dados
Treinamento de modelo preditivo
API RESTful para acesso às previsões
Monitoramento de desempenho do modelo
Containerização para deploy simplificado

Fluxo de MLOps Implementado
O projeto segue as melhores práticas de MLOps:

Coleta de Dados: Automatizada usando a API do Yahoo Finance
Processamento de Dados: Tratamento, feature engineering e divisão em treino/teste
Treinamento do Modelo: Utilização do Prophet para previsão de séries temporais
Avaliação de Modelo: Métricas como MAE, RMSE, MAPE para validação
Serialização: Modelo salvo em formato pickle
Deploy: Exposição via API RESTful usando FastAPI
Monitoramento: Acompanhamento contínuo do desempenho do modelo
Containerização: Docker para simplificar o deploy e replicabilidade

Principais Endpoints da API

GET /: Status da API
GET /health: Verificação de saúde da API
GET /model/info: Informações sobre o modelo atual
GET /predict/next: Previsões para os próximos dias
GET /predict/date: Previsão para uma data específica
GET /predict/range: Previsões para um intervalo de datas

Como Executar
Requisitos

Python 3.9+
Docker e Docker Compose (opcional)

Instalação Local

Clone o repositório:
bashgit clone https://github.com/seu-usuario/stock-prediction.git
cd stock-prediction

Instale as dependências:
bashpip install -r requirements.txt

Execute o fluxo completo:
bash# Coletar dados
python -m src.data.collect

# Processar dados
python -m src.data.process

# Treinar modelo
python -m src.models.train

# Iniciar API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload


Execução com Docker

Construa e inicie o container:
bashdocker-compose up -d

Acesse a API em http://localhost:8000
Para encerrar:
bashdocker-compose down


Monitoramento
O monitoramento do modelo é realizado através de logs de previsões e métricas de desempenho. Os principais indicadores monitorados são:

MAPE (Mean Absolute Percentage Error): Mede a acurácia percentual das previsões
MAE (Mean Absolute Error): Mede o erro absoluto médio
RMSE (Root Mean Squared Error): Mede o erro quadrático médio
Drift Detection: Detecta mudanças no desempenho do modelo ao longo do tempo

Futuros Melhoramentos

Implementação de alertas para degradação de modelo
Pipeline de retreinamento automático
Interface web para visualização de previsões
Suporte a múltiplos modelos e comparação
Testes automatizados mais abrangentes
CI/CD para automação completa

Licença
Este projeto está licenciado sob a MIT License.
Autor
Este projeto foi desenvolvido como parte da avaliação da pós-graduação em Machine Learning Engineering.