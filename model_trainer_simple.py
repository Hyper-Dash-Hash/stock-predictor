"""
Simplified model training module for stock price prediction.
Handles training Random Forest models with proper evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from typing import Dict, Tuple, Optional, List
import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleModelTrainer:
    """
    Trains and evaluates machine learning models for stock prediction.
    """
    
    def __init__(self, settings: Optional[Dict] = None):
        """
        Initialize the model trainer.
        
        Args:
            settings: Model training settings
        """
        self.settings = settings or config.MODEL_SETTINGS
        self.training_settings = config.TRAINING_SETTINGS
        self.scaler = StandardScaler()
        self.models = {}
        self.metrics = {}
    
    def train_random_forest(self, data: pd.DataFrame, symbol: Optional[str] = None) -> Tuple[RandomForestClassifier, Dict]:
        """
        Train a Random Forest model.
        
        Args:
            data: DataFrame with features and target
            symbol: Stock symbol (optional)
            
        Returns:
            Tuple of (trained_model, metrics)
        """
        logger.info(f"Training Random Forest model for {symbol or 'stock data'}")
        
        # Prepare data
        X, y = self._prepare_data(data)
        
        # Time-based split
        X_train, X_test, y_train, y_test = self._time_series_split(X, y)
        
        # Train model
        rf_params = self.settings['random_forest']
        model = RandomForestClassifier(**rf_params)
        model.fit(X_train, y_train)
        
        # Evaluate
        metrics = self._evaluate_model(model, X_test, y_test, 'random_forest')
        
        # Store model and metrics
        self.models['random_forest'] = model
        self.metrics['random_forest'] = metrics
        
        logger.info(f"Random Forest training complete. Accuracy: {metrics['accuracy']:.4f}")
        
        return model, metrics
    
    def _prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for training.
        
        Args:
            data: DataFrame with features and target
            
        Returns:
            Tuple of (features, target)
        """
        from src.feature_engineering import FeatureEngineer
        
        engineer = FeatureEngineer()
        X, y = engineer.prepare_data_for_training(data)
        
        return X, y
    
    def _time_series_split(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Perform time-based train/test split.
        
        Args:
            X: Features DataFrame
            y: Target Series
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        test_size = self.training_settings['test_size']
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        logger.info(f"Time series split: Train={len(X_train)}, Test={len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def _evaluate_model(self, model, X_test, y_test, model_type: str, y_pred: Optional[np.ndarray] = None) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            model_type: Type of model
            y_pred: Pre-computed predictions (optional)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if y_pred is None:
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Calculate win rate (percentage of correct predictions)
        win_rate = np.mean(y_pred == y_test)
        
        # Calculate additional metrics
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'win_rate': win_rate,
            'model_type': model_type
        }
        
        # Print detailed report
        logger.info(f"\n{model_type.upper()} Model Performance:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        logger.info(f"Win Rate: {win_rate:.4f}")
        
        return metrics
    
    def save_model(self, model, model_name: str, symbol: Optional[str] = None) -> str:
        """
        Save trained model to disk.
        
        Args:
            model: Trained model
            model_name: Name of the model
            symbol: Stock symbol (optional)
            
        Returns:
            Path to saved model
        """
        import os
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        symbol_suffix = f"_{symbol}" if symbol else ""
        filename = f"{model_name}{symbol_suffix}_{timestamp}.joblib"
        
        filepath = f"{config.PATHS['models_dir']}/{filename}"
        joblib.dump(model, filepath)
        
        logger.info(f"Model saved to {filepath}")
        return filepath
    
    def load_model(self, filepath: str) -> object:
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to model file
            
        Returns:
            Loaded model
        """
        model = joblib.load(filepath)
        
        logger.info(f"Model loaded from {filepath}")
        return model
    
    def get_feature_importance(self, model, feature_names: List[str]) -> pd.DataFrame:
        """
        Get feature importance for Random Forest model.
        
        Args:
            model: Trained Random Forest model
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance
        """
        if not hasattr(model, 'feature_importances_'):
            raise ValueError("Model does not have feature_importances_ attribute")
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df


def main():
    """
    Example usage of SimpleModelTrainer.
    """
    from src.data_collector import DataCollector
    from src.feature_engineering import FeatureEngineer
    
    # Get sample data
    collector = DataCollector()
    data = collector.get_stock_data('AAPL', period='1y')
    
    # Create features
    engineer = FeatureEngineer()
    features = engineer.create_features(data, 'AAPL')
    
    # Train model
    trainer = SimpleModelTrainer()
    model, metrics = trainer.train_random_forest(features, 'AAPL')
    
    print(f"Model accuracy: {metrics['accuracy']:.4f}")
    print(f"Win rate: {metrics['win_rate']:.4f}")


if __name__ == "__main__":
    main() 