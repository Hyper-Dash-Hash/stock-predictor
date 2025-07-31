"""
Feature engineering module for stock price prediction.
Creates technical indicators and features for machine learning models.
"""

import pandas as pd
import numpy as np
import ta
from typing import List, Dict, Optional, Tuple
import logging
import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Creates technical indicators and features for stock prediction models.
    """
    
    def __init__(self, settings: Optional[Dict] = None):
        """
        Initialize the feature engineer.
        
        Args:
            settings: Feature engineering settings
        """
        self.settings = settings or config.FEATURE_SETTINGS
        self.technical_settings = config.TECHNICAL_INDICATORS
    
    def create_features(self, data: pd.DataFrame, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Create all features for a stock dataset.
        
        Args:
            data: Stock data DataFrame with OHLCV columns
            symbol: Stock symbol (optional)
            
        Returns:
            DataFrame with original data and technical indicators
        """
        logger.info(f"Creating features for {symbol or 'stock data'}")
        
        # Make a copy to avoid modifying original data
        df = data.copy()
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Create technical indicators
        df = self._add_moving_averages(df)
        df = self._add_momentum_indicators(df)
        df = self._add_volatility_indicators(df)
        df = self._add_trend_indicators(df)
        df = self._add_volume_indicators(df)
        
        # Create price-based features
        df = self._add_price_features(df)
        
        # Create lag features
        df = self._add_lag_features(df)
        
        # Create rolling statistics
        df = self._add_rolling_features(df)
        
        # Create target variable
        df = self._create_target(df)
        
        # Remove rows with NaN values
        initial_rows = len(df)
        df = df.dropna()
        final_rows = len(df)
        
        logger.info(f"Feature engineering complete. Rows: {initial_rows} -> {final_rows}")
        logger.info(f"Total features: {len(df.columns)}")
        
        return df
    
    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add simple and exponential moving averages."""
        logger.info("Adding moving averages...")
        
        # Simple Moving Averages
        for period in self.technical_settings['sma_periods']:
            df[f'sma_{period}'] = ta.trend.sma_indicator(df['close'], window=period)
            df[f'price_sma_{period}_ratio'] = df['close'] / df[f'sma_{period}']
        
        # Exponential Moving Averages
        for period in self.technical_settings['ema_periods']:
            df[f'ema_{period}'] = ta.trend.ema_indicator(df['close'], window=period)
            df[f'price_ema_{period}_ratio'] = df['close'] / df[f'ema_{period}']
        
        return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators like RSI, Stochastic, Williams %R."""
        logger.info("Adding momentum indicators...")
        
        # RSI
        rsi_period = self.technical_settings['rsi_period']
        df['rsi'] = ta.momentum.rsi(df['close'], window=rsi_period)
        
        # Stochastic Oscillator
        stoch_k_period = self.technical_settings['stoch_k_period']
        stoch_d_period = self.technical_settings['stoch_d_period']
        df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'], 
                                         window=stoch_k_period)
        df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'],
                                                window=stoch_k_period, smooth_window=stoch_d_period)
        
        # Williams %R
        williams_period = self.technical_settings['williams_r_period']
        df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'],
                                                 lbp=williams_period)
        
        # Commodity Channel Index
        cci_period = self.technical_settings['cci_period']
        df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'], window=cci_period)
        
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators like Bollinger Bands, ATR."""
        logger.info("Adding volatility indicators...")
        
        # Bollinger Bands
        bb_period = self.technical_settings['bollinger_period']
        bb_std = self.technical_settings['bollinger_std']
        
        bb_indicator = ta.volatility.BollingerBands(df['close'], window=bb_period, 
                                                   window_dev=bb_std)
        df['bb_upper'] = bb_indicator.bollinger_hband()
        df['bb_lower'] = bb_indicator.bollinger_lband()
        df['bb_middle'] = bb_indicator.bollinger_mavg()
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Average True Range
        atr_period = self.technical_settings['atr_period']
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'],
                                                    window=atr_period)
        
        return df
    
    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicators like MACD, ADX."""
        logger.info("Adding trend indicators...")
        
        # MACD
        macd_fast = self.technical_settings['macd_fast']
        macd_slow = self.technical_settings['macd_slow']
        macd_signal = self.technical_settings['macd_signal']
        
        df['macd'] = ta.trend.macd(df['close'], window_fast=macd_fast, 
                                   window_slow=macd_slow)
        df['macd_signal'] = ta.trend.macd_signal(df['close'], window_fast=macd_fast,
                                                window_slow=macd_slow, window_sign=macd_signal)
        df['macd_histogram'] = ta.trend.macd_diff(df['close'], window_fast=macd_fast,
                                                 window_slow=macd_slow, window_sign=macd_signal)
        
        # ADX (Average Directional Index)
        adx_period = self.technical_settings['adx_period']
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=adx_period)
        
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators."""
        logger.info("Adding volume indicators...")
        
        # Volume SMA
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # On Balance Volume
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        
        # Volume Rate of Change
        df['volume_roc'] = df['volume'].pct_change(10)
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        logger.info("Adding price features...")
        
        # Price changes
        df['price_change'] = df['close'].pct_change()
        df['price_change_2d'] = df['close'].pct_change(2)
        df['price_change_5d'] = df['close'].pct_change(5)
        
        # High-Low ratio
        df['high_low_ratio'] = df['high'] / df['low']
        
        # Open-Close ratio
        df['open_close_ratio'] = df['open'] / df['close']
        
        # Price ranges
        df['daily_range'] = df['high'] - df['low']
        df['daily_range_pct'] = df['daily_range'] / df['close']
        
        # Gap up/down
        df['gap'] = df['open'] - df['close'].shift(1)
        df['gap_pct'] = df['gap'] / df['close'].shift(1)
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features."""
        logger.info("Adding lag features...")
        
        lag_periods = self.settings['feature_lag_periods']
        
        # Lag price features
        for lag in lag_periods:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            df[f'price_change_lag_{lag}'] = df['price_change'].shift(lag)
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistics."""
        logger.info("Adding rolling features...")
        
        windows = self.settings['rolling_windows']
        
        for window in windows:
            # Rolling statistics for price
            df[f'close_rolling_mean_{window}'] = df['close'].rolling(window).mean()
            df[f'close_rolling_std_{window}'] = df['close'].rolling(window).std()
            df[f'close_rolling_min_{window}'] = df['close'].rolling(window).min()
            df[f'close_rolling_max_{window}'] = df['close'].rolling(window).max()
            
            # Rolling statistics for volume
            df[f'volume_rolling_mean_{window}'] = df['volume'].rolling(window).mean()
            df[f'volume_rolling_std_{window}'] = df['volume'].rolling(window).std()
            
            # Rolling statistics for price change
            df[f'price_change_rolling_mean_{window}'] = df['price_change'].rolling(window).mean()
            df[f'price_change_rolling_std_{window}'] = df['price_change'].rolling(window).std()
        
        return df
    
    def _create_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create the target variable for prediction."""
        logger.info("Creating target variable...")
        
        horizon = self.settings['prediction_horizon']
        threshold = self.settings['price_change_threshold']
        
        # Future price change
        future_price_change = df['close'].shift(-horizon) / df['close'] - 1
        
        # Binary classification: 1 if price increases by threshold, 0 otherwise
        df[self.settings['target_column']] = (future_price_change > threshold).astype(int)
        
        # Multi-class target (optional)
        df['target_3class'] = pd.cut(future_price_change, 
                                    bins=[-np.inf, -threshold, threshold, np.inf],
                                    labels=[0, 1, 2])
        df['target_3class'] = df['target_3class'].fillna(1).astype(int)
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of feature columns (excluding target and metadata).
        
        Args:
            df: DataFrame with features
            
        Returns:
            List of feature column names
        """
        exclude_cols = [
            'symbol', 'target', 'target_3class',
            'open', 'high', 'low', 'close', 'volume'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols
    
    def prepare_data_for_training(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for model training.
        
        Args:
            df: DataFrame with features and target
            
        Returns:
            Tuple of (features, target)
        """
        feature_cols = self.get_feature_columns(df)
        target_col = self.settings['target_column']
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Remove rows where target is NaN (future data not available)
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        logger.info(f"Prepared data: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y


def main():
    """
    Example usage of FeatureEngineer.
    """
    from data_collector import DataCollector
    
    # Get sample data
    collector = DataCollector()
    data = collector.get_stock_data('AAPL', period='6mo')
    
    # Create features
    engineer = FeatureEngineer()
    features = engineer.create_features(data, 'AAPL')
    
    print(f"Original data shape: {data.shape}")
    print(f"Features data shape: {features.shape}")
    print(f"Feature columns: {len(engineer.get_feature_columns(features))}")
    
    # Prepare for training
    X, y = engineer.prepare_data_for_training(features)
    print(f"Training data: X={X.shape}, y={y.shape}")


if __name__ == "__main__":
    main() 