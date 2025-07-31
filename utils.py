"""
Utility functions for the stock prediction project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import logging
import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_stock_data(data: pd.DataFrame, symbol: str, save_path: Optional[str] = None):
    """
    Plot stock price data with volume.
    
    Args:
        data: Stock data DataFrame
        symbol: Stock symbol
        save_path: Path to save plot (optional)
    """
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Price chart
        ax1.plot(data.index, data['close'], label='Close Price', linewidth=2)
        ax1.set_title(f'{symbol} Stock Price')
        ax1.set_ylabel('Price ($)')
        ax1.grid(True)
        ax1.legend()
        
        # Volume chart
        ax2.bar(data.index, data['volume'], alpha=0.7, color='blue')
        ax2.set_title('Volume')
        ax2.set_ylabel('Volume')
        ax2.set_xlabel('Date')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Stock data plot saved to {save_path}")
        
        plt.show()
        
    except ImportError:
        logger.warning("matplotlib not available for plotting")


def plot_technical_indicators(data: pd.DataFrame, symbol: str, save_path: Optional[str] = None):
    """
    Plot technical indicators.
    
    Args:
        data: DataFrame with technical indicators
        symbol: Stock symbol
        save_path: Path to save plot (optional)
    """
    try:
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        # Price and moving averages
        axes[0].plot(data.index, data['close'], label='Close Price', linewidth=2)
        if 'sma_20' in data.columns:
            axes[0].plot(data.index, data['sma_20'], label='SMA 20', alpha=0.7)
        if 'sma_50' in data.columns:
            axes[0].plot(data.index, data['sma_50'], label='SMA 50', alpha=0.7)
        axes[0].set_title(f'{symbol} Price and Moving Averages')
        axes[0].set_ylabel('Price ($)')
        axes[0].grid(True)
        axes[0].legend()
        
        # RSI
        if 'rsi' in data.columns:
            axes[1].plot(data.index, data['rsi'], label='RSI', color='purple')
            axes[1].axhline(y=70, color='r', linestyle='--', alpha=0.7)
            axes[1].axhline(y=30, color='g', linestyle='--', alpha=0.7)
            axes[1].set_title('RSI')
            axes[1].set_ylabel('RSI')
        axes[1].grid(True)
        
        # MACD
        if 'macd' in data.columns and 'macd_signal' in data.columns:
            axes[2].plot(data.index, data['macd'], label='MACD', color='blue')
            axes[2].plot(data.index, data['macd_signal'], label='Signal', color='red')
            axes[2].bar(data.index, data['macd_histogram'], label='Histogram', alpha=0.3)
            axes[2].set_title('MACD')
            axes[2].set_ylabel('MACD')
        axes[2].grid(True)
        axes[2].legend()
        
        # Bollinger Bands
        if 'bb_upper' in data.columns and 'bb_lower' in data.columns:
            axes[3].plot(data.index, data['close'], label='Close Price', linewidth=2)
            axes[3].plot(data.index, data['bb_upper'], label='Upper Band', alpha=0.7)
            axes[3].plot(data.index, data['bb_lower'], label='Lower Band', alpha=0.7)
            axes[3].fill_between(data.index, data['bb_upper'], data['bb_lower'], alpha=0.1)
            axes[3].set_title('Bollinger Bands')
            axes[3].set_ylabel('Price ($)')
        axes[3].grid(True)
        axes[3].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Technical indicators plot saved to {save_path}")
        
        plt.show()
        
    except ImportError:
        logger.warning("matplotlib not available for plotting")


def plot_model_performance(metrics: Dict, model_name: str, save_path: Optional[str] = None):
    """
    Plot model performance metrics.
    
    Args:
        metrics: Dictionary of performance metrics
        model_name: Name of the model
        save_path: Path to save plot (optional)
    """
    try:
        # Create metrics for plotting
        metric_names = ['accuracy', 'precision', 'recall', 'f1_score', 'win_rate']
        metric_values = [metrics.get(metric, 0) for metric in metric_names]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(metric_names, metric_values, color=['blue', 'green', 'orange', 'red', 'purple'])
        ax.set_title(f'{model_name} Performance Metrics')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model performance plot saved to {save_path}")
        
        plt.show()
        
    except ImportError:
        logger.warning("matplotlib not available for plotting")


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 20, save_path: Optional[str] = None):
    """
    Plot feature importance.
    
    Args:
        importance_df: DataFrame with feature importance
        top_n: Number of top features to show
        save_path: Path to save plot (optional)
    """
    try:
        top_features = importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.barh(range(len(top_features)), top_features['importance'])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importance')
        
        # Add value labels
        for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
            ax.text(importance + 0.001, i, f'{importance:.3f}', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
        
    except ImportError:
        logger.warning("matplotlib not available for plotting")


def create_performance_report(metrics: Dict, model_name: str) -> str:
    """
    Create a formatted performance report.
    
    Args:
        metrics: Dictionary of performance metrics
        model_name: Name of the model
        
    Returns:
        Formatted report string
    """
    report = f"""
{'='*50}
{model_name.upper()} MODEL PERFORMANCE REPORT
{'='*50}

Classification Metrics:
- Accuracy: {metrics.get('accuracy', 0):.4f} ({metrics.get('accuracy', 0)*100:.2f}%)
- Precision: {metrics.get('precision', 0):.4f} ({metrics.get('precision', 0)*100:.2f}%)
- Recall: {metrics.get('recall', 0):.4f} ({metrics.get('recall', 0)*100:.2f}%)
- F1-Score: {metrics.get('f1_score', 0):.4f} ({metrics.get('f1_score', 0)*100:.2f}%)
- Win Rate: {metrics.get('win_rate', 0):.4f} ({metrics.get('win_rate', 0)*100:.2f}%)

Trading Metrics (if available):
- Total Return: {metrics.get('total_return', 0):.4f} ({metrics.get('total_return', 0)*100:.2f}%)
- Annual Return: {metrics.get('annual_return', 0):.4f} ({metrics.get('annual_return', 0)*100:.2f}%)
- Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}
- Max Drawdown: {metrics.get('max_drawdown', 0):.4f} ({metrics.get('max_drawdown', 0)*100:.2f}%)
- Total Trades: {metrics.get('total_trades', 0)}
- Transaction Costs: ${metrics.get('total_transaction_cost', 0):.2f}

{'='*50}
"""
    return report


def save_results_to_csv(results: Dict, filename: str):
    """
    Save results to CSV file.
    
    Args:
        results: Dictionary of results
        filename: Output filename
    """
    try:
        # Convert results to DataFrame
        if isinstance(results, dict):
            if 'metrics' in results:
                # Single result
                df = pd.DataFrame([results['metrics']])
            else:
                # Multiple results
                df = pd.DataFrame(results).T
        else:
            df = pd.DataFrame(results)
        
        filepath = f"{config.PATHS['data_dir']}/{filename}"
        df.to_csv(filepath)
        logger.info(f"Results saved to {filepath}")
        
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")


def load_results_from_csv(filename: str) -> pd.DataFrame:
    """
    Load results from CSV file.
    
    Args:
        filename: Input filename
        
    Returns:
        DataFrame with results
    """
    try:
        filepath = f"{config.PATHS['data_dir']}/{filename}"
        df = pd.read_csv(filepath, index_col=0)
        logger.info(f"Results loaded from {filepath}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading results: {str(e)}")
        return pd.DataFrame()


def calculate_rolling_metrics(data: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """
    Calculate rolling performance metrics.
    
    Args:
        data: DataFrame with returns
        window: Rolling window size
        
    Returns:
        DataFrame with rolling metrics
    """
    if 'daily_returns' not in data.columns:
        logger.warning("No 'daily_returns' column found")
        return pd.DataFrame()
    
    rolling_metrics = pd.DataFrame(index=data.index)
    
    # Rolling returns
    rolling_metrics['rolling_return'] = data['daily_returns'].rolling(window).mean() * 252
    
    # Rolling volatility
    rolling_metrics['rolling_volatility'] = data['daily_returns'].rolling(window).std() * np.sqrt(252)
    
    # Rolling Sharpe ratio
    risk_free_rate = config.METRICS_SETTINGS['risk_free_rate']
    rolling_metrics['rolling_sharpe'] = (
        (rolling_metrics['rolling_return'] - risk_free_rate) / rolling_metrics['rolling_volatility']
    )
    
    # Rolling maximum drawdown
    cumulative_returns = (1 + data['daily_returns']).cumprod()
    rolling_max = cumulative_returns.rolling(window).max()
    rolling_metrics['rolling_drawdown'] = (cumulative_returns - rolling_max) / rolling_max
    
    return rolling_metrics


def print_summary_statistics(data: pd.DataFrame, title: str = "Summary Statistics"):
    """
    Print summary statistics for a dataset.
    
    Args:
        data: DataFrame to analyze
        title: Title for the summary
    """
    print(f"\n{title}")
    print("=" * len(title))
    
    # Basic info
    print(f"Shape: {data.shape}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    print(f"Total days: {len(data)}")
    
    # Price statistics
    if 'close' in data.columns:
        print(f"\nPrice Statistics:")
        print(f"  Min: ${data['close'].min():.2f}")
        print(f"  Max: ${data['close'].max():.2f}")
        print(f"  Mean: ${data['close'].mean():.2f}")
        print(f"  Std: ${data['close'].std():.2f}")
    
    # Volume statistics
    if 'volume' in data.columns:
        print(f"\nVolume Statistics:")
        print(f"  Min: {data['volume'].min():,.0f}")
        print(f"  Max: {data['volume'].max():,.0f}")
        print(f"  Mean: {data['volume'].mean():,.0f}")
        print(f"  Std: {data['volume'].std():,.0f}")
    
    # Missing values
    missing_values = data.isnull().sum()
    if missing_values.sum() > 0:
        print(f"\nMissing Values:")
        for col, missing in missing_values[missing_values > 0].items():
            print(f"  {col}: {missing} ({missing/len(data)*100:.1f}%)")


def main():
    """
    Example usage of utility functions.
    """
    from data_collector import DataCollector
    from feature_engineering import FeatureEngineer
    
    # Get sample data
    collector = DataCollector()
    data = collector.get_stock_data('AAPL', period='6mo')
    
    # Print summary statistics
    print_summary_statistics(data, "AAPL Stock Data Summary")
    
    # Create features
    engineer = FeatureEngineer()
    features = engineer.create_features(data, 'AAPL')
    
    # Plot stock data
    plot_stock_data(data, 'AAPL')
    
    # Plot technical indicators
    plot_technical_indicators(features, 'AAPL')


if __name__ == "__main__":
    main() 