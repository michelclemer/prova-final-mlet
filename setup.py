from setuptools import setup, find_packages

setup(
    name="stock-prediction-mlops",
    version="0.1.0",
    description="Stock Price Prediction using Machine Learning and MLOps",
    author="Seu Nome",
    author_email="seu.email@exemplo.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.23.5",
        "scikit-learn>=1.2.2",
        "tensorflow>=2.12.0",
        "fastapi>=0.95.2",
        "uvicorn>=0.22.0",
        "mlflow>=2.3.1",
        "yfinance>=0.2.28",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)