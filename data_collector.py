"""
Data collection module for stock price prediction.
Downloads historical stock data using yfinance and Alpha Vantage API.
"""

import pandas as pd
import yfinance as yf
import requests
import time
from datetime import datetime, timedelta
import logging
from typing import Optional, Dict, List
import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCollector:
    """
    Collects historical stock data from various sources.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the data collector.
        
        Args:
            api_key: Alpha Vantage API key (optional)
        """
        self.api_key = api_key or config.ALPHA_VANTAGE_API_KEY
        self.session = requests.Session()
    
    def get_stock_data(self, symbol: str, period: str = '2y', 
                      interval: str = '1d', source: str = 'yfinance') -> pd.DataFrame:
        """
        Get historical stock data for a given symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            source: Data source ('yfinance' or 'alpha_vantage')
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            if source.lower() == 'yfinance':
                return self._get_yfinance_data(symbol, period, interval)
            elif source.lower() == 'alpha_vantage':
                return self._get_alpha_vantage_data(symbol, period, interval)
            else:
                raise ValueError(f"Unsupported data source: {source}")
        except Exception as e:
            logger.error(f"Error collecting data for {symbol}: {str(e)}")
            raise
    
    def _get_yfinance_data(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """
        Get data from Yahoo Finance using yfinance.
        """
        logger.info(f"Downloading {symbol} data from Yahoo Finance...")
        
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        
        if data.empty:
            raise ValueError(f"No data found for symbol {symbol}")
        
        # Clean column names
        data.columns = [col.lower() for col in data.columns]
        
        # Add symbol column
        data['symbol'] = symbol
        
        logger.info(f"Downloaded {len(data)} records for {symbol}")
        return data
    
    def _get_alpha_vantage_data(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """
        Get data from Alpha Vantage API.
        """
        if not self.api_key:
            raise ValueError("Alpha Vantage API key is required")
        
        logger.info(f"Downloading {symbol} data from Alpha Vantage...")
        
        # Convert period to number of days
        period_days = self._period_to_days(period)
        
        # Alpha Vantage daily endpoint
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'apikey': self.api_key,
            'outputsize': 'full' if period_days > 100 else 'compact'
        }
        
        response = self.session.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if 'Error Message' in data:
            raise ValueError(f"Alpha Vantage error: {data['Error Message']}")
        
        if 'Note' in data:
            logger.warning(f"Alpha Vantage note: {data['Note']}")
            time.sleep(12)  # Rate limiting
        
        # Parse the data
        time_series = data.get('Time Series (Daily)', {})
        if not time_series:
            raise ValueError(f"No data found for symbol {symbol}")
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(time_series, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        # Rename columns
        column_mapping = {
            '1. open': 'open',
            '2. high': 'high',
            '3. low': 'low',
            '4. close': 'close',
            '5. volume': 'volume'
        }
        df = df.rename(columns=column_mapping)
        
        # Convert to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Filter by period
        if period_days:
            start_date = datetime.now() - timedelta(days=period_days)
            df = df[df.index >= start_date]
        
        # Add symbol column
        df['symbol'] = symbol
        
        logger.info(f"Downloaded {len(df)} records for {symbol}")
        return df
    
    def _period_to_days(self, period: str) -> int:
        """
        Convert period string to number of days.
        """
        period_map = {
            '1d': 1, '5d': 5, '1mo': 30, '3mo': 90, '6mo': 180,
            '1y': 365, '2y': 730, '5y': 1825, '10y': 3650
        }
        return period_map.get(period, 730)  # Default to 2 years
    
    def get_multiple_stocks(self, symbols: List[str], period: str = '2y', 
                           interval: str = '1d', source: str = 'yfinance') -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple stock symbols.
        
        Args:
            symbols: List of stock symbols
            period: Time period
            interval: Data interval
            source: Data source
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        data_dict = {}
        
        for symbol in symbols:
            try:
                data = self.get_stock_data(symbol, period, interval, source)
                data_dict[symbol] = data
                logger.info(f"Successfully collected data for {symbol}")
                
                # Rate limiting for Alpha Vantage
                if source.lower() == 'alpha_vantage':
                    time.sleep(12)
                    
            except Exception as e:
                logger.error(f"Failed to collect data for {symbol}: {str(e)}")
                continue
        
        return data_dict
    
    def save_data(self, data: pd.DataFrame, symbol: str, filename: Optional[str] = None) -> str:
        """
        Save stock data to CSV file.
        
        Args:
            data: Stock data DataFrame
            symbol: Stock symbol
            filename: Optional filename, if None will generate one
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_{timestamp}.csv"
        
        filepath = f"{config.PATHS['data_dir']}/{filename}"
        data.to_csv(filepath)
        logger.info(f"Data saved to {filepath}")
        return filepath
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load stock data from CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with stock data
        """
        data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        logger.info(f"Data loaded from {filepath}")
        return data


def main():
    """
    Example usage of DataCollector.
    """
    collector = DataCollector()
    
    # Get data for a single stock
    data = collector.get_stock_data('AAPL', period='1y')
    print(f"AAPL data shape: {data.shape}")
    print(data.head())
    
    # Get data for multiple stocks
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    data_dict = collector.get_multiple_stocks(symbols, period='6mo')
    
    for symbol, data in data_dict.items():
        print(f"{symbol}: {len(data)} records")


if __name__ == "__main__":
    main() 