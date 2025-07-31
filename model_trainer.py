"""
Model training module for stock price prediction.
Handles training Random Forest and LSTM models with proper evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import logging
from typing import Dict, Tuple, Optional, List
import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
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
    
    def train_lstm(self, data: pd.DataFrame, symbol: Optional[str] = None) -> Tuple[tf.keras.Model, Dict]:
        """
        Train an LSTM model.
        
        Args:
            data: DataFrame with features and target
            symbol: Stock symbol (optional)
            
        Returns:
            Tuple of (trained_model, metrics)
        """
        logger.info(f"Training LSTM model for {symbol or 'stock data'}")
        
        # Prepare data
        X, y = self._prepare_data(data)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences for LSTM
        sequence_length = self.settings['lstm']['sequence_length']
        X_sequences, y_sequences = self._create_sequences(X_scaled, y, sequence_length)
        
        # Time-based split
        split_idx = int(len(X_sequences) * (1 - self.training_settings['test_size']))
        X_train, X_test = X_sequences[:split_idx], X_sequences[split_idx:]
        y_train, y_test = y_sequences[:split_idx], y_sequences[split_idx:]
        
        # Build LSTM model
        model = self._build_lstm_model(X_train.shape[1], X_train.shape[2])
        
        # Train model
        lstm_params = self.settings['lstm']
        history = model.fit(
            X_train, y_train,
            epochs=lstm_params['epochs'],
            batch_size=lstm_params['batch_size'],
            validation_split=lstm_params['validation_split'],
            callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
            verbose=1
        )
        
        # Evaluate
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        metrics = self._evaluate_model(None, X_test, y_test, 'lstm', y_pred)
        
        # Store model and metrics
        self.models['lstm'] = model
        self.metrics['lstm'] = metrics
        
        logger.info(f"LSTM training complete. Accuracy: {metrics['accuracy']:.4f}")
        
        return model, metrics
    
    def train_ensemble(self, data: pd.DataFrame, symbol: Optional[str] = None) -> Tuple[Dict, Dict]:
        """
        Train an ensemble of models.
        
        Args:
            data: DataFrame with features and target
            symbol: Stock symbol (optional)
            
        Returns:
            Tuple of (ensemble_models, ensemble_metrics)
        """
        logger.info(f"Training ensemble model for {symbol or 'stock data'}")
        
        ensemble_models = {}
        ensemble_metrics = {}
        
        # Train individual models
        if 'random_forest' in self.settings['ensemble']['models']:
            rf_model, rf_metrics = self.train_random_forest(data, symbol)
            ensemble_models['random_forest'] = rf_model
            ensemble_metrics['random_forest'] = rf_metrics
        
        if 'lstm' in self.settings['ensemble']['models']:
            lstm_model, lstm_metrics = self.train_lstm(data, symbol)
            ensemble_models['lstm'] = lstm_model
            ensemble_metrics['lstm'] = lstm_metrics
        
        # Ensemble predictions
        ensemble_metrics['ensemble'] = self._evaluate_ensemble(ensemble_models, data)
        
        return ensemble_models, ensemble_metrics
    
    def _prepare_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for training.
        
        Args:
            data: DataFrame with features and target
            
        Returns:
            Tuple of (features, target)
        """
        from feature_engineering import FeatureEngineer
        
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
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            X: Scaled features
            y: Target values
            sequence_length: Number of time steps to look back
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(X)):
            X_sequences.append(X[i-sequence_length:i])
            y_sequences.append(y[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def _build_lstm_model(self, sequence_length: int, n_features: int) -> tf.keras.Model:
        """
        Build LSTM model architecture.
        
        Args:
            sequence_length: Number of time steps
            n_features: Number of features
            
        Returns:
            Compiled LSTM model
        """
        lstm_params = self.settings['lstm']
        
        model = Sequential([
            LSTM(units=lstm_params['units'], return_sequences=True, 
                 input_shape=(sequence_length, n_features)),
            Dropout(lstm_params['dropout']),
            LSTM(units=lstm_params['units']),
            Dropout(lstm_params['dropout']),
            Dense(units=lstm_params['units'] // 2, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
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
            if model_type == 'lstm':
                y_pred_proba = model.predict(X_test)
                y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            else:
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
    
    def _evaluate_ensemble(self, models: Dict, data: pd.DataFrame) -> Dict:
        """
        Evaluate ensemble model performance.
        
        Args:
            models: Dictionary of trained models
            data: Full dataset
            
        Returns:
            Ensemble evaluation metrics
        """
        X, y = self._prepare_data(data)
        X_train, X_test, y_train, y_test = self._time_series_split(X, y)
        
        # Get predictions from each model
        predictions = {}
        weights = self.settings['ensemble']['weights']
        
        for i, (model_name, model) in enumerate(models.items()):
            if model_name == 'lstm':
                # Scale features for LSTM
                X_test_scaled = self.scaler.transform(X_test)
                # Create sequences
                sequence_length = self.settings['lstm']['sequence_length']
                X_test_sequences, _ = self._create_sequences(X_test_scaled, y_test, sequence_length)
                pred_proba = model.predict(X_test_sequences)
                predictions[model_name] = (pred_proba > 0.5).astype(int).flatten()
            else:
                predictions[model_name] = model.predict(X_test)
        
        # Weighted ensemble prediction
        ensemble_pred = np.zeros(len(y_test))
        for i, (model_name, pred) in enumerate(predictions.items()):
            ensemble_pred += weights[i] * pred
        
        ensemble_pred = (ensemble_pred > 0.5).astype(int)
        
        # Calculate ensemble metrics
        metrics = self._evaluate_model(None, X_test, y_test, 'ensemble', ensemble_pred)
        
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
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        symbol_suffix = f"_{symbol}" if symbol else ""
        filename = f"{model_name}{symbol_suffix}_{timestamp}"
        
        if model_name == 'lstm':
            filepath = f"{config.PATHS['models_dir']}/{filename}.h5"
            model.save(filepath)
        else:
            filepath = f"{config.PATHS['models_dir']}/{filename}.joblib"
            joblib.dump(model, filepath)
        
        logger.info(f"Model saved to {filepath}")
        return filepath
    
    def load_model(self, filepath: str, model_type: str) -> object:
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to model file
            model_type: Type of model ('random_forest' or 'lstm')
            
        Returns:
            Loaded model
        """
        if model_type == 'lstm':
            model = tf.keras.models.load_model(filepath)
        else:
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
    Example usage of ModelTrainer.
    """
    from data_collector import DataCollector
    from feature_engineering import FeatureEngineer
    
    # Get sample data
    collector = DataCollector()
    data = collector.get_stock_data('AAPL', period='1y')
    
    # Create features
    engineer = FeatureEngineer()
    features = engineer.create_features(data, 'AAPL')
    
    # Train models
    trainer = ModelTrainer()
    
    # Train Random Forest
    rf_model, rf_metrics = trainer.train_random_forest(features, 'AAPL')
    
    # Train LSTM
    lstm_model, lstm_metrics = trainer.train_lstm(features, 'AAPL')
    
    # Train Ensemble
    ensemble_models, ensemble_metrics = trainer.train_ensemble(features, 'AAPL')
    
    print("\nModel Performance Summary:")
    for model_name, metrics in ensemble_metrics.items():
        print(f"{model_name}: Accuracy={metrics['accuracy']:.4f}, Win Rate={metrics['win_rate']:.4f}")


if __name__ == "__main__":
    main() 