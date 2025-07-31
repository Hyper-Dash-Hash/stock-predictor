"""
Backtesting module for stock prediction models.
Simulates trading with transaction costs and calculates performance metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Backtester:
    """
    Backtests trading strategies based on model predictions.
    """
    
    def __init__(self, settings: Optional[Dict] = None):
        """
        Initialize the backtester.
        
        Args:
            settings: Backtesting settings
        """
        self.settings = settings or config.BACKTEST_SETTINGS
        self.metrics_settings = config.METRICS_SETTINGS
    
    def run_backtest(self, model, data: pd.DataFrame, symbol: Optional[str] = None) -> Dict:
        """
        Run backtest simulation for a trained model.
        
        Args:
            model: Trained model (Random Forest, LSTM, or ensemble)
            data: DataFrame with features and target
            symbol: Stock symbol (optional)
            
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Running backtest for {symbol or 'stock data'}")
        
        # Prepare data
        X, y = self._prepare_data(data)
        
        # Get predictions
        predictions = self._get_predictions(model, X)
        
        # Create trading signals
        signals = self._create_trading_signals(predictions, data)
        
        # Run simulation
        portfolio = self._simulate_trading(signals, data)
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(portfolio, data)
        
        # Store results
        results = {
            'portfolio': portfolio,
            'signals': signals,
            'metrics': metrics,
            'symbol': symbol
        }
        
        logger.info(f"Backtest complete. Final portfolio value: ${portfolio['portfolio_value'].iloc[-1]:,.2f}")
        
        return results
    
    def _prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for backtesting.
        
        Args:
            data: DataFrame with features and target
            
        Returns:
            Tuple of (features, target)
        """
        from src.feature_engineering import FeatureEngineer
        
        engineer = FeatureEngineer()
        X, y = engineer.prepare_data_for_training(data)
        
        return X, y
    
    def _get_predictions(self, model, X: pd.DataFrame) -> np.ndarray:
        """
        Get predictions from model.
        
        Args:
            model: Trained model
            X: Features DataFrame
            
        Returns:
            Array of predictions
        """
        if hasattr(model, 'predict_proba'):
            # Random Forest
            pred_proba = model.predict_proba(X)
            predictions = pred_proba[:, 1]  # Probability of positive class
        elif hasattr(model, 'predict'):
            # LSTM or other models
            pred_proba = model.predict(X)
            if len(pred_proba.shape) > 1:
                predictions = pred_proba[:, 0]
            else:
                predictions = pred_proba
        else:
            raise ValueError("Model must have predict or predict_proba method")
        
        return predictions
    
    def _create_trading_signals(self, predictions: np.ndarray, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create trading signals from model predictions.
        
        Args:
            predictions: Model predictions
            data: Original data
            
        Returns:
            DataFrame with trading signals
        """
        # Align predictions with data
        valid_data = data.dropna()
        predictions = predictions[:len(valid_data)]
        
        # Create signals DataFrame
        signals = pd.DataFrame(index=valid_data.index)
        signals['prediction'] = predictions
        signals['signal'] = 0  # 0: hold, 1: buy, -1: sell
        
        # Generate buy/sell signals based on prediction threshold
        threshold = 0.5
        signals.loc[signals['prediction'] > threshold, 'signal'] = 1
        signals.loc[signals['prediction'] < (1 - threshold), 'signal'] = -1
        
        # Add price data
        signals['close'] = valid_data['close']
        signals['returns'] = valid_data['close'].pct_change()
        
        return signals
    
    def _simulate_trading(self, signals: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate trading with transaction costs and position management.
        
        Args:
            signals: DataFrame with trading signals
            data: Original data
            
        Returns:
            DataFrame with portfolio performance
        """
        # Initialize portfolio
        initial_capital = self.settings['initial_capital']
        transaction_cost = self.settings['transaction_cost']
        position_size = self.settings['position_size']
        
        portfolio = pd.DataFrame(index=signals.index)
        portfolio['close'] = signals['close']
        portfolio['signal'] = signals['signal']
        portfolio['returns'] = signals['returns']
        
        # Initialize portfolio values
        portfolio['cash'] = initial_capital
        portfolio['position'] = 0
        portfolio['shares'] = 0
        portfolio['portfolio_value'] = initial_capital
        portfolio['trade_cost'] = 0
        
        current_cash = initial_capital
        current_shares = 0
        current_position = 0
        
        for i in range(1, len(portfolio)):
            signal = portfolio.iloc[i]['signal']
            price = portfolio.iloc[i]['close']
            returns = portfolio.iloc[i]['returns']
            
            # Update portfolio value from previous day
            if current_shares > 0:
                portfolio.iloc[i, portfolio.columns.get_loc('portfolio_value')] = (
                    current_cash + current_shares * price
                )
            else:
                portfolio.iloc[i, portfolio.columns.get_loc('portfolio_value')] = current_cash
            
            # Execute trades based on signals
            if signal == 1 and current_position <= 0:  # Buy signal
                # Calculate position size
                trade_value = current_cash * position_size
                shares_to_buy = trade_value / price
                trade_cost = trade_value * transaction_cost
                
                current_shares += shares_to_buy
                current_cash -= (trade_value + trade_cost)
                current_position = 1
                
                portfolio.iloc[i, portfolio.columns.get_loc('shares')] = current_shares
                portfolio.iloc[i, portfolio.columns.get_loc('cash')] = current_cash
                portfolio.iloc[i, portfolio.columns.get_loc('trade_cost')] = trade_cost
                
            elif signal == -1 and current_position >= 0:  # Sell signal
                if current_shares > 0:
                    trade_value = current_shares * price
                    trade_cost = trade_value * transaction_cost
                    
                    current_cash += (trade_value - trade_cost)
                    current_shares = 0
                    current_position = -1
                    
                    portfolio.iloc[i, portfolio.columns.get_loc('shares')] = current_shares
                    portfolio.iloc[i, portfolio.columns.get_loc('cash')] = current_cash
                    portfolio.iloc[i, portfolio.columns.get_loc('trade_cost')] = trade_cost
        
        # Calculate final portfolio metrics
        portfolio['total_value'] = portfolio['cash'] + portfolio['shares'] * portfolio['close']
        portfolio['daily_returns'] = portfolio['total_value'].pct_change()
        
        return portfolio
    
    def _calculate_performance_metrics(self, portfolio: pd.DataFrame, data: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            portfolio: Portfolio DataFrame
            data: Original data
            
        Returns:
            Dictionary with performance metrics
        """
        # Basic metrics
        total_return = (portfolio['total_value'].iloc[-1] / portfolio['total_value'].iloc[0]) - 1
        annual_return = total_return * (252 / len(portfolio))
        
        # Risk metrics
        daily_returns = portfolio['daily_returns'].dropna()
        volatility = daily_returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        risk_free_rate = self.metrics_settings['risk_free_rate']
        sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + daily_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        winning_days = (daily_returns > 0).sum()
        total_days = len(daily_returns)
        win_rate = winning_days / total_days if total_days > 0 else 0
        
        # Trading metrics
        total_trades = (portfolio['signal'] != 0).sum()
        total_cost = portfolio['trade_cost'].sum()
        
        # Compare with buy-and-hold strategy
        buy_hold_return = (data['close'].iloc[-1] / data['close'].iloc[0]) - 1
        buy_hold_annual = buy_hold_return * (252 / len(data))
        
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'total_transaction_cost': total_cost,
            'final_portfolio_value': portfolio['total_value'].iloc[-1],
            'buy_hold_return': buy_hold_return,
            'buy_hold_annual_return': buy_hold_annual,
            'excess_return': total_return - buy_hold_return
        }
        
        logger.info(f"Performance Metrics:")
        logger.info(f"Total Return: {total_return:.4f} ({total_return*100:.2f}%)")
        logger.info(f"Annual Return: {annual_return:.4f} ({annual_return*100:.2f}%)")
        logger.info(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        logger.info(f"Max Drawdown: {max_drawdown:.4f} ({max_drawdown*100:.2f}%)")
        logger.info(f"Win Rate: {win_rate:.4f} ({win_rate*100:.2f}%)")
        logger.info(f"Total Trades: {total_trades}")
        logger.info(f"Transaction Costs: ${total_cost:.2f}")
        
        return metrics
    
    def compare_strategies(self, models: Dict, data: pd.DataFrame, symbol: Optional[str] = None) -> Dict:
        """
        Compare multiple trading strategies.
        
        Args:
            models: Dictionary of trained models
            data: DataFrame with features and target
            symbol: Stock symbol (optional)
            
        Returns:
            Dictionary with comparison results
        """
        logger.info(f"Comparing strategies for {symbol or 'stock data'}")
        
        results = {}
        
        for model_name, model in models.items():
            try:
                result = self.run_backtest(model, data, symbol)
                results[model_name] = result['metrics']
                logger.info(f"{model_name} strategy completed")
            except Exception as e:
                logger.error(f"Error in {model_name} strategy: {str(e)}")
                continue
        
        # Create comparison summary
        comparison = pd.DataFrame(results).T
        
        logger.info(f"\nStrategy Comparison:")
        logger.info(comparison[['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']].round(4))
        
        return results
    
    def plot_backtest_results(self, results: Dict, save_path: Optional[str] = None):
        """
        Plot backtest results.
        
        Args:
            results: Backtest results dictionary
            save_path: Path to save plot (optional)
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            portfolio = results['portfolio']
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Portfolio value over time
            axes[0, 0].plot(portfolio.index, portfolio['total_value'])
            axes[0, 0].set_title('Portfolio Value Over Time')
            axes[0, 0].set_ylabel('Portfolio Value ($)')
            axes[0, 0].grid(True)
            
            # Daily returns distribution
            daily_returns = portfolio['daily_returns'].dropna()
            axes[0, 1].hist(daily_returns, bins=50, alpha=0.7)
            axes[0, 1].set_title('Daily Returns Distribution')
            axes[0, 1].set_xlabel('Daily Returns')
            axes[0, 1].set_ylabel('Frequency')
            
            # Cumulative returns
            cumulative_returns = (1 + daily_returns).cumprod()
            axes[1, 0].plot(cumulative_returns.index, cumulative_returns)
            axes[1, 0].set_title('Cumulative Returns')
            axes[1, 0].set_ylabel('Cumulative Returns')
            axes[1, 0].grid(True)
            
            # Drawdown
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            axes[1, 1].fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
            axes[1, 1].set_title('Drawdown')
            axes[1, 1].set_ylabel('Drawdown')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("matplotlib not available for plotting")


def main():
    """
    Example usage of Backtester.
    """
    from data_collector import DataCollector
    from feature_engineering import FeatureEngineer
    from model_trainer import ModelTrainer
    
    # Get sample data
    collector = DataCollector()
    data = collector.get_stock_data('AAPL', period='1y')
    
    # Create features
    engineer = FeatureEngineer()
    features = engineer.create_features(data, 'AAPL')
    
    # Train model
    trainer = ModelTrainer()
    model, metrics = trainer.train_random_forest(features, 'AAPL')
    
    # Run backtest
    backtester = Backtester()
    results = backtester.run_backtest(model, features, 'AAPL')
    
    print(f"Backtest Results:")
    print(f"Final Portfolio Value: ${results['metrics']['final_portfolio_value']:,.2f}")
    print(f"Total Return: {results['metrics']['total_return']:.4f}")
    print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.4f}")


if __name__ == "__main__":
    main() 