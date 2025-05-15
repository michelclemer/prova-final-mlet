"""
Módulo para monitoramento do modelo em produção.
"""
import os
import logging
import numpy as np
import pandas as pd
import joblib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Diretório para armazenar métricas de monitoramento
MONITORING_DIR = "monitoring"
os.makedirs(MONITORING_DIR, exist_ok=True)


class ModelMonitor:
    """
    Classe para monitoramento do modelo em produção.
    """

    def __init__(self, symbol="AAPL"):
        """
        Inicializa o monitor do modelo.

        Args:
            symbol: Símbolo da ação
        """
        self.symbol = symbol
        self.log_path = os.path.join(MONITORING_DIR, f"{symbol}_prediction_log.jsonl")
        self.metrics_path = os.path.join(MONITORING_DIR, f"{symbol}_drift_metrics.json")

        # Criar arquivos vazios se não existirem
        for path in [self.log_path, self.metrics_path]:
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path), exist_ok=True)

            if not os.path.exists(path):
                with open(path, 'w') as f:
                    if path.endswith('.json'):
                        f.write('{}')
                    else:
                        pass  # Arquivo vazio para .jsonl

    def calculate_metrics(self, actual, predicted):
        """
        Calcula métricas de desempenho para um conjunto de valores reais e previstos.

        Args:
            actual: Série ou array com valores reais
            predicted: Série ou array com valores previstos

        Returns:
            Dicionário com métricas calculadas
        """
        if len(actual) != len(predicted):
            logger.error(f"Tamanhos incompatíveis: actual={len(actual)}, predicted={len(predicted)}")
            raise ValueError("Os conjuntos de valores reais e previstos devem ter o mesmo tamanho")

        # Calcular métricas básicas
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        r2 = r2_score(actual, predicted)

        # Calcular MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100

        # Calcular erro médio (ME) para verificar viés
        me = np.mean(predicted - actual)

        # Calcular desvio padrão dos erros
        std_error = np.std(predicted - actual)

        metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'mape': float(mape),
            'me': float(me),
            'std_error': float(std_error),
            'n_samples': int(len(actual))
        }

        return metrics

    def log_prediction(self, date, actual_value, predicted_value, prediction_date=None, metadata=None):
        """
        Registra uma previsão e seu valor real para monitoramento.

        Args:
            date: Data da previsão (string ou datetime)
            actual_value: Valor real observado
            predicted_value: Valor previsto pelo modelo
            prediction_date: Data em que a previsão foi feita (opcional)
            metadata: Metadados adicionais para registro
        """
        try:
            # Converter datas para string se forem datetime
            if isinstance(date, datetime):
                date = date.strftime('%Y-%m-%d')

            if prediction_date is None:
                prediction_date = datetime.now().strftime('%Y-%m-%d')
            elif isinstance(prediction_date, datetime):
                prediction_date = prediction_date.strftime('%Y-%m-%d')

            # Criar registro de previsão
            prediction_record = {
                'ticker': self.symbol,
                'date': date,
                'actual_value': float(actual_value),
                'predicted_value': float(predicted_value),
                'prediction_date': prediction_date,
                'error': float(actual_value - predicted_value),
                'abs_error': float(abs(actual_value - predicted_value)),
                'pct_error': float(abs(actual_value - predicted_value) / actual_value * 100),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            # Adicionar metadados se fornecidos
            if metadata:
                prediction_record.update(metadata)

            # Adicionar registro ao arquivo
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(prediction_record) + '\n')

            logger.info(f"Registro de previsão salvo para {self.symbol} em {date}")

        except Exception as e:
            logger.error(f"Erro ao registrar previsão: {str(e)}")

    def get_prediction_logs(self, days=30):
        """
        Recupera logs de previsões para um período específico.

        Args:
            days: Número de dias para recuperar (0 para todos)

        Returns:
            DataFrame com logs de previsões
        """
        try:
            if not os.path.exists(self.log_path):
                logger.warning(f"Arquivo de log não encontrado: {self.log_path}")
                return pd.DataFrame()

            # Carregar logs como DataFrame
            logs = []
            with open(self.log_path, 'r') as f:
                for line in f:
                    if line.strip():  # Ignorar linhas vazias
                        logs.append(json.loads(line.strip()))

            if not logs:
                return pd.DataFrame()

            logs_df = pd.DataFrame(logs)

            # Converter colunas de data para datetime
            date_columns = ['date', 'prediction_date', 'timestamp']
            for col in date_columns:
                if col in logs_df.columns:
                    logs_df[col] = pd.to_datetime(logs_df[col])

            # Filtrar por período, se necessário
            if days > 0:
                cutoff_date = datetime.now() - timedelta(days=days)
                logs_df = logs_df[logs_df['date'] >= cutoff_date]

            return logs_df

        except Exception as e:
            logger.error(f"Erro ao recuperar logs de previsão: {str(e)}")
            return pd.DataFrame()

    def collect_metrics(self):
        """
        Coleta métricas de desempenho do modelo com base nos logs de previsão.

        Returns:
            Dicionário com métricas coletadas
        """
        try:
            # Obter logs de previsão
            logs_df = self.get_prediction_logs(days=30)  # Últimos 30 dias

            if logs_df.empty:
                logger.warning(f"Não há logs de previsão para coletar métricas")
                return {}

            # Calcular métricas
            metrics = self.calculate_metrics(
                logs_df['actual_value'].values,
                logs_df['predicted_value'].values
            )

            # Adicionar metadados
            metrics['ticker'] = self.symbol
            metrics['start_date'] = logs_df['date'].min().strftime('%Y-%m-%d')
            metrics['end_date'] = logs_df['date'].max().strftime('%Y-%m-%d')
            metrics['calculation_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Calcular tendência de erro (correlação entre data e erro)
            if len(logs_df) >= 5:  # Precisamos de algumas amostras para uma correlação significativa
                logs_df['date_numeric'] = (logs_df['date'] - logs_df['date'].min()).dt.days
                error_trend = np.corrcoef(logs_df['date_numeric'], logs_df['error'])[0, 1]
                metrics['error_trend'] = float(error_trend)

            # Salvar métricas
            with open(self.metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)

            logger.info(f"Métricas coletadas para {self.symbol}: MAE={metrics['mae']:.4f}, MAPE={metrics['mape']:.2f}%")
            return metrics

        except Exception as e:
            logger.error(f"Erro ao coletar métricas: {str(e)}")
            return {}

    def check_for_drift(self, threshold_mape=10.0):
        """
        Verifica se há drift no desempenho do modelo.

        Args:
            threshold_mape: Limiar de MAPE para considerar drift

        Returns:
            Dicionário com resultado da verificação
        """
        try:
            # Tentar carregar métricas existentes
            metrics = {}
            if os.path.exists(self.metrics_path):
                with open(self.metrics_path, 'r') as f:
                    try:
                        metrics = json.load(f)
                    except json.JSONDecodeError:
                        pass

            # Se não houver métricas, coletá-las
            if not metrics:
                metrics = self.collect_metrics()

            if not metrics:
                logger.warning(f"Não há métricas para verificar drift")
                return {'status': 'unknown', 'drift_detected': False}

            # Verificar se MAPE está acima do limiar
            mape = metrics.get('mape', 0)
            drift_detected = mape > threshold_mape

            # Verificar se há tendência significativa de erro
            error_trend = metrics.get('error_trend', 0)
            trend_drift = abs(error_trend) > 0.3  # Correlação moderada

            result = {
                'status': 'warning' if drift_detected or trend_drift else 'ok',
                'drift_detected': drift_detected or trend_drift,
                'mape': mape,
                'error_trend': error_trend,
                'threshold_mape': threshold_mape,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'message': ''
            }

            if drift_detected and trend_drift:
                result[
                    'message'] = f"Drift crítico detectado. MAPE={mape:.2f}% (>{threshold_mape}%), tendência de erro={error_trend:.2f}"
            elif drift_detected:
                result['message'] = f"Possível drift detectado. MAPE={mape:.2f}% (>{threshold_mape}%)"
            elif trend_drift:
                result['message'] = f"Possível drift detectado. Tendência de erro={error_trend:.2f}"
            else:
                result['message'] = f"Modelo estável. MAPE={mape:.2f}%, tendência de erro={error_trend:.2f}"

            # Salvar resultado
            drift_path = os.path.join(MONITORING_DIR, f"{self.symbol}_drift_check.json")
            with open(drift_path, 'w') as f:
                json.dump(result, f, indent=2)

            logger.info(f"Verificação de drift para {self.symbol}: {result['message']}")
            return result

        except Exception as e:
            logger.error(f"Erro ao verificar drift: {str(e)}")
            return {'status': 'error', 'message': str(e), 'drift_detected': False}

    def update_with_actual_data(self, actual_data):
        """
        Atualiza o monitoramento com dados reais.

        Args:
            actual_data: DataFrame com dados reais (colunas: 'date', 'close')

        Returns:
            Número de registros atualizados
        """
        try:
            if actual_data.empty:
                logger.warning("Dados vazios fornecidos para atualização")
                return 0

            # Certificar que as colunas necessárias existem
            required_cols = {'date', 'close'}
            actual_cols = set(actual_data.columns)

            if not required_cols.issubset(actual_cols):
                # Tentar com nomes alternativos
                alternatives = {'Date': 'date', 'Close': 'close'}
                for alt, std in alternatives.items():
                    if alt in actual_data.columns:
                        actual_data = actual_data.rename(columns={alt: std})

            # Verificar novamente
            if not {'date', 'close'}.issubset(set(actual_data.columns)):
                raise ValueError(
                    f"Colunas necessárias não encontradas. Necessárias: {required_cols}, Encontradas: {actual_cols}")

            # Garantir que date está no formato datetime
            if not pd.api.types.is_datetime64_dtype(actual_data['date']):
                actual_data['date'] = pd.to_datetime(actual_data['date'])

            # Carregar previsões anteriores
            predictions_path = os.path.join("reports", f"{self.symbol}_predictions.csv")
            if not os.path.exists(predictions_path):
                logger.warning(f"Arquivo de previsões não encontrado: {predictions_path}")
                return 0

            predictions = pd.read_csv(predictions_path)
            predictions['date'] = pd.to_datetime(predictions['date'])

            # Combinar previsões com valores reais
            merged = pd.merge(
                predictions,
                actual_data[['date', 'close']],
                on='date',
                how='inner'
            )

            if merged.empty:
                logger.warning("Nenhuma correspondência entre previsões e dados reais")
                return 0

            # Registrar cada par de previsão/valor real
            count = 0
            for _, row in merged.iterrows():
                self.log_prediction(
                    date=row['date'],
                    actual_value=row['close'],
                    predicted_value=row['predicted_price'],
                    prediction_date=row['date'] - timedelta(days=1)  # Assumindo que previsão foi feita no dia anterior
                )
                count += 1

            # Coletar métricas atualizadas
            self.collect_metrics()

            # Verificar drift
            self.check_for_drift()

            logger.info(f"Monitoramento atualizado com {count} novos registros")
            return count

        except Exception as e:
            logger.error(f"Erro ao atualizar monitoramento: {str(e)}")
            return 0


if __name__ == "__main__":
    # Exemplo de uso
    try:
        from src.data.collect_data import get_latest_stock_data

        # Definir símbolo
        symbol = "AAPL"  # Apple como exemplo

        # Criar monitor
        monitor = ModelMonitor(symbol)

        # Obter dados recentes
        actual_data = get_latest_stock_data(symbol, days=10)

        if not actual_data.empty:
            # Atualizar monitoramento
            count = monitor.update_with_actual_data(actual_data)
            print(f"Monitoramento atualizado com {count} novos registros")

            # Coletar métricas
            metrics = monitor.collect_metrics()
            if metrics:
                print(f"Métricas atuais:")
                print(f"- MAE: {metrics['mae']:.4f}")
                print(f"- RMSE: {metrics['rmse']:.4f}")
                print(f"- MAPE: {metrics['mape']:.2f}%")
                print(f"- R²: {metrics['r2']:.4f}")

            # Verificar drift
            drift_result = monitor.check_for_drift()
            print(f"Status do modelo: {drift_result['status']}")
            print(f"Mensagem: {drift_result['message']}")
        else:
            print("Nenhum dado recente encontrado para monitoramento")

    except Exception as e:
        logger.error(f"Erro: {str(e)}")