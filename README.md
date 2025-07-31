# Stock Price Prediction Project

A comprehensive Python project for predicting stock price movements using machine learning techniques.

## Features

- **Data Collection**: Downloads daily historical stock data using yfinance
- **Technical Indicators**: Engineers various technical indicators (moving averages, RSI, MACD, etc.)
- **Model Training**: Supports both Random Forest and LSTM models
- **Time-based Splitting**: Proper train/test splits respecting temporal order
- **Performance Evaluation**: Comprehensive metrics including accuracy, precision, recall, and win rate
- **Backtesting**: Simulates trading with transaction costs and performance analysis

## Project Structure

```
Stonks/
├── data/                   # Data storage
├── models/                 # Trained model storage
├── src/
│   ├── data_collector.py  # Data collection utilities
│   ├── feature_engineering.py  # Technical indicators
│   ├── model_trainer.py   # Model training and evaluation
│   ├── backtester.py      # Backtesting simulation
│   └── utils.py           # Utility functions
├── notebooks/             # Jupyter notebooks for analysis
├── config.py              # Configuration settings
├── main.py                # Main execution script
└── requirements.txt       # Dependencies
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables (optional):
```bash
cp .env.example .env
# Edit .env with your Alpha Vantage API key if using
```

3. Run the main script:
```bash
python main.py
```

## Usage

### Basic Usage
```python
from src.data_collector import DataCollector
from src.feature_engineering import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.backtester import Backtester

# Collect data
collector = DataCollector()
data = collector.get_stock_data('AAPL', period='2y')

# Engineer features
engineer = FeatureEngineer()
features = engineer.create_features(data)

# Train model
trainer = ModelTrainer()
model, metrics = trainer.train_random_forest(features)

# Backtest
backtester = Backtester()
results = backtester.run_backtest(model, features)
```

## Configuration

Edit `config.py` to customize:
- Stock symbols to analyze
- Technical indicator parameters
- Model hyperparameters
- Backtesting parameters

## Models Supported

1. **Random Forest**: Fast, interpretable, good baseline
2. **LSTM**: Deep learning approach for time series
3. **Ensemble**: Combines multiple models

## Performance Metrics

- Accuracy
- Precision/Recall
- F1-Score
- Win Rate
- Sharpe Ratio
- Maximum Drawdown

## License

MIT License 