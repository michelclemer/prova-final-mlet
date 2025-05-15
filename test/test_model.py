"""
Testes para os módulos de modelos.
"""
import os
import pytest
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
import mlflow
from src.models.train import build_model, train_model
from src.models.evaluate_model import evaluate_model


class TestModelBuilding:
    """
    Testes para a construção de modelos.
    """

    def test_build_model(self):
        """
        Testa a construção do modelo LSTM.
        """
        # Definir forma dos dados de entrada
        input_shape = (60, 1)  # 60 timesteps, 1 feature

        # Executar a função
        model = build_model(input_shape)

        # Verificar o resultado
        assert model is not None
        assert isinstance(model, Sequential)

        # Verificar a arquitetura do modelo
        assert len(model.layers) == 6

        # Verificar a forma da saída
        assert model.output_shape == (None, 1)

        # Verificar se o modelo está compilado
        assert model.optimizer is not None
        assert model.loss is not None


class TestModelTraining:
    """
    Testes para o treinamento de modelos.
    """

    def test_train_model(self):
        """
        Testa o treinamento do modelo com dados sintéticos.
        """
        # Desativar o uso do MLflow para o teste
        mlflow.set_tracking_uri("")

        # Criar dados sintéticos
        window_size = 10
        n_samples = 100

        # X_train: (n_samples, window_size, 1)
        X_train = np.random.rand(n_samples, window_size, 1)
        # y_train: (n_samples, 1)
        y_train = np.random.rand(n_samples, 1)

        # X_test, y_test com o mesmo formato que os de treino, mas menos amostras
        X_test = np.random.rand(n_samples // 5, window_size, 1)
        y_test = np.random.rand(n_samples // 5, 1)

        # Diretório temporário para salvar o modelo
        temp_dir = "tests/temp_models"
        os.makedirs(temp_dir, exist_ok=True)
        model_path = f"{temp_dir}/test_model.h5"

        try:
            # Treinar o modelo com poucos epochs para o teste
            model = train_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                model_path=model_path,
                batch_size=32,
                epochs=2  # Usar poucas épocas para o teste ser rápido
            )

            # Verificar o resultado
            assert model is not None
            assert isinstance(model, Sequential)
            assert os.path.exists(model_path)

            # Carregar o modelo salvo e verificar se é válido
            loaded_model = tf.keras.models.load_model(model_path)
            assert loaded_model is not None

            # Fazer uma previsão para verificar se o modelo funciona
            test_input = np.random.rand(1, window_size, 1)
            prediction = loaded_model.predict(test_input)
            assert prediction.shape == (1, 1)

        finally:
            # Limpar arquivos temporários
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


@pytest.mark.skip(reason="Requer modelo treinado e dados de teste")
class TestModelEvaluation:
    """
    Testes para a avaliação de modelos.
    """

    def test_evaluate_model(self):
        """
        Testa a avaliação do modelo com dados sintéticos.
        Nota: Este teste é marcado para ser pulado, pois requer um modelo
        treinado e dados de teste reais para uma avaliação significativa.
        """
        # Desativar o uso do MLflow para o teste
        mlflow.set_tracking_uri("")

        # Criar diretórios temporários
        temp_dir = "tests/temp_eval"
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(f"{temp_dir}/models", exist_ok=True)
        os.makedirs(f"{temp_dir}/data/processed", exist_ok=True)
        os.makedirs(f"{temp_dir}/reports/figures", exist_ok=True)

        # Configurar caminhos
        model_path = f"{temp_dir}/models/test_model.h5"
        scaler_path = f"{temp_dir}/models/test_scaler.pkl"
        test_dates_path = f"{temp_dir}/data/processed/test_dates.csv"
        output_dir = f"{temp_dir}/reports/figures"

        try:
            # Aqui, você precisaria criar ou carregar:
            # 1. Um modelo treinado
            # 2. Um scaler para desnormalizar os dados
            # 3. Dados de teste (X_test, y_test)
            # 4. Datas de teste

            # Execute a função de avaliação
            metrics = evaluate_model(
                model_path=model_path,
                scaler_path=scaler_path,
                X_test=None,  # Substituir por dados reais
                y_test=None,  # Substituir por dados reais
                test_dates_path=test_dates_path,
                output_dir=output_dir
            )

            # Verificar as métricas retornadas
            assert metrics is not None
            assert 'mse' in metrics
            assert 'rmse' in metrics
            assert 'mae' in metrics
            assert 'r2' in metrics
            assert 'mape' in metrics

            # Verificar se os arquivos de saída foram gerados
            assert os.path.exists(f"{output_dir}/prediction_vs_actual.png")
            assert os.path.exists(f"{output_dir}/prediction_results.csv")

        finally:
            # Limpar arquivos temporários
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)