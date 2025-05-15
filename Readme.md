# Modelo de Previsão da Bolsa de Valores - MLOps

Este projeto implementa um modelo preditivo para previsão de preços de ações na bolsa de valores, seguindo boas práticas de MLOps.

## Visão Geral

O sistema usa um modelo LSTM (Long Short-Term Memory) para prever o preço de fechamento de ações com base em dados históricos. A aplicação inclui uma API para servir as previsões, monitoramento do desempenho do modelo em produção e recursos para retreinamento automático quando necessário.

## Funcionalidades

- Coleta de dados históricos da bolsa de valores usando a API do Yahoo Finance
- Pré-processamento de dados para treinar o modelo LSTM
- Treinamento e avaliação do modelo com métricas relevantes
- API REST para servir previsões (implementada com FastAPI)
- Monitoramento do desempenho do modelo em produção
- Detecção de drift e retreinamento automático quando necessário
- Containerização com Docker para facilitar a implantação

## Estrutura do Projeto

```
.
├── app.py                  # Ponto de entrada principal da aplicação
├── data/                   # Diretório para armazenar dados
│   ├── raw/                # Dados brutos coletados
│   └── processed/          # Dados processados para o modelo
├── models/                 # Diretório para armazenar modelos treinados
├── monitoring/             # Diretório para armazenar dados de monitoramento
├── reports/                # Relatórios, gráficos e métricas
│   └── figures/            # Visualizações geradas
├── src/                    # Código fonte da aplicação
│   ├── api/                # Módulos da API REST
│   │   ├── main.py         # Implementação da API FastAPI
│   │   └── routers.py      # Rotas adicionais
│   ├── data/               # Módulos para processamento de dados
│   │   ├── collect_data.py # Coleta de dados
│   │   └── process.py      # Pré-processamento
│   ├── models/             # Módulos para modelo de ML
│   │   ├── train.py        # Treinamento do modelo
│   │   ├── predict.py      # Previsão com modelo treinado
│   │   └── evaluate_model.py  # Avaliação do modelo
│   └── monitoring/         # Módulos para monitoramento
│       └── metrics.py      # Monitoramento e métricas
├── test/                   # Testes unitários e de integração
├── logs/                   # Logs da aplicação
├── Dockerfile              # Dockerfile para containerização
├── docker-compose.yml      # Configuração Docker Compose
├── requirements.txt        # Dependências do projeto
├── .env                    # Variáveis de ambiente
└── README.md               # Este arquivo
```

## Requisitos

- Python 3.9+
- Bibliotecas listadas em requirements.txt

## Instalação

1. Clone o repositório:
```
git clone https://github.com/seu-usuario/stock-prediction-mlops.git
cd stock-prediction-mlops
```

2. Crie um ambiente virtual e instale as dependências:
```
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Configure as variáveis de ambiente:
```
cp .env.example .env
# Edite o arquivo .env conforme necessário
```

## Uso

### Treinamento do Modelo

Para treinar o modelo com dados históricos de uma ação específica:

```
python -m src.models.train --symbol AAPL
```

### Iniciando a API

Para iniciar a API de previsão:

```
python app.py
```

Ou, para configurar o modelo antes de iniciar a API:

```
python app.py --setup
```

Para forçar o treinamento do modelo antes de iniciar a API:

```
python app.py --train
```

Para habilitar o monitoramento em tempo real:

```
python app.py --monitoring
```

### Docker

Para iniciar o sistema usando Docker:

```
docker-compose up -d
```

## API Endpoints

A API oferece os seguintes endpoints:

- `GET /`: Informações básicas da API
- `GET /health`: Verificação de status da API
- `GET /model/info`: Informações sobre o modelo atual
- `GET /predict/next`: Previsões para os próximos dias
- `GET /predict/date`: Previsão para uma data específica
- `GET /predict/range`: Previsões para um intervalo de datas
- `POST /predict`: Previsões com parâmetros personalizados
- `GET /monitoring/metrics`: Métricas de monitoramento
- `GET /monitoring/drift`: Verificação de drift no modelo
- `POST /monitoring/update`: Atualiza monitoramento com dados recentes
- `POST /retrain`: Retreina o modelo com dados atualizados

## Monitoramento

O sistema de monitoramento acompanha:

- Precisão das previsões comparadas com valores reais
- Detecção de drift no modelo
- Métricas de desempenho (MAPE, RMSE, MAE, etc.)

## MLOps

A implementação segue as melhores práticas de MLOps, incluindo:

- Versionamento de código e modelos
- Automação do pipeline de dados e treinamento
- API para servir modelos
- Monitoramento de modelos em produção
- Retreinamento automático
- Containerização para implantação reproduzível

## Modelo LSTM

O modelo LSTM (Long Short-Term Memory) é uma arquitetura de rede neural recorrente especialmente adequada para prever séries temporais, como preços de ações. A implementação usa:

- 60 dias de histórico como janela de tempo para prever o próximo dia
- Duas camadas LSTM com 50 unidades cada, com dropout para prevenir overfitting
- Camadas densas para a saída
- Normalização de dados para melhorar o treinamento
- Divisão treino/teste de 80/20
- Early stopping para prevenir overfitting

## Avaliação do Modelo

O modelo é avaliado usando várias métricas:

- MSE (Erro Quadrático Médio)
- RMSE (Raiz do Erro Quadrático Médio)
- MAE (Erro Absoluto Médio)
- MAPE (Erro Percentual Absoluto Médio)
- R² (Coeficiente de Determinação)

## Exemplo de Resposta da API

Quando você faz uma solicitação para `/predict/next`, a resposta tem o seguinte formato:

```json
[
  {
    "date": "2025-05-16",
    "predicted_price": 178.45,
    "lower_bound": 169.53,
    "upper_bound": 187.37
  },
  {
    "date": "2025-05-19",
    "predicted_price": 180.23,
    "lower_bound": 171.22,
    "upper_bound": 189.24
  },
  ...
]
```

## Troubleshooting

Se você encontrar problemas, verifique:

1. Logs da aplicação em `logs/app.log`
2. Certifique-se de que todas as dependências estão instaladas
3. Confirme se todos os diretórios necessários existem
4. Verifique se a API do Yahoo Finance está funcionando corretamente

## Contribuindo

Contribuições são bem-vindas! Por favor, abra uma issue para discutir suas ideias ou envie um pull request.

## Licença

Este projeto está licenciado sob a licença MIT.