# Instruções para Entrega da Tarefa

## Resumo do Projeto

Você desenvolveu um sistema de previsão de preços de ações usando Machine Learning e MLOps. O sistema usa um modelo LSTM para prever o preço de fechamento das ações da Apple (AAPL), mas pode ser facilmente adaptado para outras ações.

### Principais Componentes Implementados:

1. **Coleta de Dados**: Usando a biblioteca `yfinance` para obter dados históricos da bolsa.
2. **Pré-processamento**: Transformação dos dados para uso pelo modelo LSTM.
3. **Treinamento do Modelo**: Usando TensorFlow/Keras para criar e treinar um modelo LSTM.
4. **Avaliação do Modelo**: Cálculo de métricas como RMSE, MAE, MAPE para avaliar a qualidade do modelo.
5. **API de Previsão**: Uma API REST implementada com FastAPI para servir as previsões.
6. **Monitoramento**: Sistema para monitorar o desempenho do modelo em produção.
7. **Containerização**: Configuração Docker para facilitar a implantação.

## Requisitos da Tarefa Atendidos

- ✅ Escolha de empresa listada na bolsa (Apple - AAPL)
- ✅ Algoritmo de ML para séries temporais (LSTM)
- ✅ Avaliação do modelo com métricas relevantes
- ✅ Serialização do modelo
- ✅ Ambiente virtualizado (Docker)
- ✅ API para receber requisições
- ✅ Monitoramento do modelo em produção
- ✅ Documentação completa do projeto

## Preparação para Entrega

### 1. Fazer upload para o GitHub

1. Crie um novo repositório no GitHub.
2. Inicialize o repositório localmente e faça o primeiro commit:

```bash
git init
git add .
git commit -m "Projeto de previsão de ações com MLOps"
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/NOME_DO_REPOSITORIO.git
git push -u origin main
```

### 2. Preparar a API para Demonstração

1. Treine o modelo e inicie a API:

```bash
# Configurar o projeto
python setup.py

# Iniciar a API
python app.py
```

2. Teste a API localmente acessando:
```
http://localhost:8000/docs
```

3. Faça algumas chamadas de exemplo para verificar o funcionamento.

### 3. Criar Vídeo de Demonstração

Grave um vídeo de 5 minutos (no mínimo) explicando:

1. **Introdução** (30s): Apresente o projeto e seu objetivo.
2. **Arquitetura** (1min): Explique os componentes do sistema.
3. **Modelo LSTM** (1min): Descreva o modelo escolhido e suas características.
4. **Pipeline de MLOps** (1min): Mostre como você implementou as práticas de MLOps.
5. **Demonstração da API** (1min): Demonstre o funcionamento da API.
6. **Monitoramento** (30s): Mostre o sistema de monitoramento.

### 4. Documentos para a Entrega

1. Link do repositório GitHub com o código completo
2. Link da API (se hospedada online) ou instruções para execução local
3. Vídeo explicativo de pelo menos 5 minutos

## Dicas para Apresentação

- **Foque na Estratégia de MLOps**: Enfatize como você implementou o ciclo completo de MLOps, desde a coleta de dados até o monitoramento em produção.
- **Destaque a Arquitetura**: Explique os componentes e como eles se integram.
- **Mostre Exemplos Reais**: Demonstre previsões com dados reais e compare com valores históricos.
- **Explique o Monitoramento**: Mostre como o sistema detecta drift e como isso ajuda a manter o modelo atualizado.

## Checklist Final

- [ ] Código completo e organizado no GitHub
- [ ] Arquivo README detalhado
- [ ] Documentação da API
- [ ] Modelo treinado e serializado
- [ ] API funcionando corretamente
- [ ] Sistema de monitoramento implementado
- [ ] Docker configurado
- [ ] Vídeo explicativo gravado