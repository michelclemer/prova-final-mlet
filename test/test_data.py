"""
Testes para os módulos de manipulação de dados.
"""
import os
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data.collect_data import collect_stock_data
from src.data.process import preprocess_data, create_sequences


class TestDataCollection:
    """
    Testes para a coleta de dados.
    """

    def test_collect_stock_data(self):
        """
        Testa a coleta de dados de ações.
        """
        # Definir um período mais recente para garantir que há dados
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        # Executar a função
        data = collect_stock_data(
            symbol="AAPL",
            start_date=start_date,
            end_date=end_date,
            save_path="tests/temp_data.csv"
        )

        # Verificar o resultado
        assert data is not None
        assert isinstance(data, pd.DataFrame)
        assert not data.empty
        assert 'Close' in data.columns
        assert 'Open' in data.columns
        assert 'High' in data.columns
        assert 'Low' in data.columns
        assert 'Volume' in data.columns
        assert 'Date' in data.columns

        # Limpar arquivos temporários
        if os.path.exists("tests/temp_data.csv"):
            os.remove("tests/temp_data.csv")

    def test_collect_stock_data_invalid_symbol(self):
        """
        Testa a coleta de dados com símbolo inválido.
        """
        with pytest.raises(Exception):
            collect_stock_data(
                symbol="INVALID_SYMBOL_12345",
                start_date="2022-01-01",
                end_date="2022-01-10",
                save_path="tests/temp_data.csv"
            )


class TestDataPreprocessing:
    """
    Testes para o pré-processamento de dados.
    """

    def test_create_sequences(self):
        """
        Testa a criação de sequências para o modelo LSTM.
        """
        # Criar dados de teste
        data = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]).reshape(-1, 1)
        window_size = 3

        # Executar a função
        X, y = create_sequences(data, window_size)

        # Verificar formato dos arrays
        assert X.shape == (len(data) - window_size, window_size, 1)
        assert y.shape == (len(data) - window_size, 1)

        # Verificar alguns valores específicos
        assert np.array_equal(X[0], np.array([0.1, 0.2, 0.3]).reshape(window_size, 1))
        assert np.array_equal(y[0], np.array([0.4]).reshape(1, 1))
        assert np.array_equal(X[1], np.array([0.2, 0.3, 0.4]).reshape(window_size, 1))
        assert np.array_equal(y[1], np.array([0.5]).reshape(1, 1))

    def test_preprocess_data(self):
        """
        Testa o pré-processamento completo dos dados.
        """
        # Criar dados sintéticos para teste
        dates = pd.date_range(start='2022-01-01', periods=100, freq='D')
        close_prices = np.sin(np.linspace(0, 10, 100)) * 50 + 100  # Valores sintéticos

        df = pd.DataFrame({
            'Date': dates.strftime('%Y-%m-%d'),
            'Open': close_prices * 0.99,
            'High': close_prices * 1.02,
            'Low': close_prices * 0.98,
            'Close': close_prices,
            'Volume': np.random.randint(1000000, 10000000, size=100)
        })

        # Salvar em um arquivo temporário
        temp_dir = "tests/temp"
        os.makedirs(temp_dir, exist_ok=True)

        input_path = f"{temp_dir}/test_stock_data.csv"
        output_path = f"{temp_dir}/test_processed"
        scaler_path = f"{temp_dir}/test_scaler.pkl"

        df.to_csv(input_path, index=False)

        # Executar a função
        try:
            result = preprocess_data(
                input_path=input_path,
                output_path=output_path,
                scaler_path=scaler_path,
                test_size=0.2,
                feature_window=10
            )

            # Verificar o resultado
            assert result is not None
            assert 'X_train' in result
            assert 'y_train' in result
            assert 'X_test' in result
            assert 'y_test' in result
            assert 'scaler' in result
            assert 'last_window' in result

            # Verificar arquivos gerados
            assert os.path.exists(f"{output_path}_X_train.npy")
            assert os.path.exists(f"{output_path}_y_train.npy")
            assert os.path.exists(f"{output_path}_X_test.npy")
            assert os.path.exists(f"{output_path}_y_test.npy")
            assert os.path.exists(scaler_path)

        finally:
            # Limpar arquivos temporários
            import shutil
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)