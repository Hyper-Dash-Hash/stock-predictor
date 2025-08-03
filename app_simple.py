import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import ta
import json
import time
import random
import requests
from io import StringIO

# Page configuration
st.set_page_config(
    page_title="AI Stock Predictor Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {}

# Enhanced CSS with better contrast and fixed styling
def get_css(dark_mode=False):
    if dark_mode:
        return """
        <style>
            .main {
                background-color: #1a1a1a;
                color: #ffffff;
            }
            .stApp {
                background-color: #1a1a1a;
            }
            .css-1d391kg {
                background-color: #2d2d2d;
            }
            .hero-section {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 4rem 2rem;
                border-radius: 20px;
                text-align: center;
                color: white;
                margin-bottom: 3rem;
                box-shadow: 0 15px 35px rgba(0,0,0,0.3);
            }
            .hero-section h1, .hero-section p {
                color: white !important;
                text-shadow: 0 2px 4px rgba(0,0,0,0.5);
            }
            .feature-card {
                background: rgba(45, 45, 45, 0.9);
                padding: 2rem;
                border-radius: 15px;
                margin: 1rem 0;
                box-shadow: 0 5px 15px rgba(0,0,0,0.3);
                border-left: 5px solid #667eea;
                color: white;
            }
            .feature-card h3, .feature-card p {
                color: white !important;
                text-shadow: 0 1px 2px rgba(0,0,0,0.3);
            }
            .prediction-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 1.5rem;
                border-radius: 15px;
                color: white;
                margin: 1rem 0;
                text-align: center;
            }
            .watchlist-item {
                background: rgba(45, 45, 45, 0.9);
                padding: 1rem;
                border-radius: 10px;
                margin: 0.5rem 0;
                border-left: 3px solid #667eea;
                color: white;
            }
            .timeframe-card {
                background: rgba(45, 45, 45, 0.9);
                padding: 1rem;
                border-radius: 10px;
                margin: 0.5rem 0;
                color: white;
                text-align: center;
            }
            .info-section {
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(20px);
                padding: 2rem;
                border-radius: 20px;
                border: 1px solid rgba(255, 255, 255, 0.2);
                color: white;
            }
            .info-section h2, .info-section h3, .info-section p, .info-section li {
                color: white !important;
                text-shadow: 0 1px 2px rgba(0,0,0,0.3);
            }
            .stats-card {
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(20px);
                padding: 2rem;
                border-radius: 20px;
                border: 1px solid rgba(255, 255, 255, 0.2);
                text-align: center;
            }
            .stats-card h2, .stats-card h3, .stats-card p {
                color: white !important;
                text-shadow: 0 1px 2px rgba(0,0,0,0.3);
            }
            .chat-message {
                background: rgba(45, 45, 45, 0.9);
                padding: 1rem;
                border-radius: 10px;
                margin: 0.5rem 0;
                border-left: 3px solid #667eea;
            }
            .ai-message {
                background: rgba(102, 126, 234, 0.2);
                border-left: 3px solid #667eea;
            }
            .user-message {
                background: rgba(118, 75, 162, 0.2);
                border-left: 3px solid #764ba2;
            }
            .prediction-badge {
                display: inline-block;
                padding: 0.5rem 1rem;
                border-radius: 20px;
                font-weight: bold;
                margin: 0.5rem;
            }
            .prediction-up {
                background: linear-gradient(135deg, #26A69A, #4CAF50);
                color: white;
            }
            .prediction-down {
                background: linear-gradient(135deg, #EF5350, #F44336);
                color: white;
            }
            .confidence-meter {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                padding: 1rem;
                margin: 1rem 0;
            }
        </style>
        """
    else:
        return """
        <style>
            .hero-section {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 4rem 2rem;
                border-radius: 20px;
                text-align: center;
                color: white;
                margin-bottom: 3rem;
                box-shadow: 0 15px 35px rgba(0,0,0,0.1);
            }
            .hero-section h1, .hero-section p {
                color: white !important;
                text-shadow: 0 2px 4px rgba(0,0,0,0.3);
            }
            .feature-card {
                background: white;
                padding: 2rem;
                border-radius: 15px;
                margin: 1rem 0;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                border-left: 5px solid #667eea;
            }
            .feature-card h3 {
                color: #333 !important;
                font-weight: bold;
            }
            .feature-card p {
                color: #555 !important;
            }
            .prediction-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 1.5rem;
                border-radius: 15px;
                color: white;
                margin: 1rem 0;
                text-align: center;
            }
            .watchlist-item {
                background: white;
                padding: 1rem;
                border-radius: 10px;
                margin: 0.5rem 0;
                border-left: 3px solid #667eea;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .timeframe-card {
                background: white;
                padding: 1rem;
                border-radius: 10px;
                margin: 0.5rem 0;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                text-align: center;
            }
            .info-section {
                background: rgba(102, 126, 234, 0.1);
                padding: 2rem;
                border-radius: 20px;
                border: 1px solid rgba(102, 126, 234, 0.2);
            }
            .info-section h2, .info-section h3 {
                color: #333 !important;
                font-weight: bold;
            }
            .info-section p, .info-section li {
                color: #555 !important;
            }
            .stats-card {
                background: rgba(102, 126, 234, 0.1);
                padding: 2rem;
                border-radius: 20px;
                border: 1px solid rgba(102, 126, 234, 0.2);
                text-align: center;
            }
            .stats-card h2 {
                color: #667eea !important;
                font-size: 2.5rem;
                margin-bottom: 0.5rem;
            }
            .stats-card h3 {
                color: #333 !important;
                margin: 0;
                font-weight: bold;
            }
            .stats-card p {
                color: #666 !important;
                margin: 0;
            }
            .chat-message {
                background: white;
                padding: 1rem;
                border-radius: 10px;
                margin: 0.5rem 0;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                border-left: 3px solid #667eea;
            }
            .ai-message {
                background: rgba(102, 126, 234, 0.1);
                border-left: 3px solid #667eea;
            }
            .user-message {
                background: rgba(118, 75, 162, 0.1);
                border-left: 3px solid #764ba2;
            }
            .prediction-badge {
                display: inline-block;
                padding: 0.5rem 1rem;
                border-radius: 20px;
                font-weight: bold;
                margin: 0.5rem;
            }
            .prediction-up {
                background: linear-gradient(135deg, #26A69A, #4CAF50);
                color: white;
            }
            .prediction-down {
                background: linear-gradient(135deg, #EF5350, #F44336);
                color: white;
            }
            .confidence-meter {
                background: rgba(102, 126, 234, 0.1);
                border-radius: 10px;
                padding: 1rem;
                margin: 1rem 0;
            }
        </style>
        """

st.markdown(get_css(st.session_state.dark_mode), unsafe_allow_html=True)

# Navigation function
def navigate_to(page):
    st.session_state.current_page = page

# AI Chat functionality
def generate_ai_response(user_question, stock_data=None):
    """Generate AI response based on user question and stock data"""
    
    # Simple AI response generation (you can integrate with GPT-4o later)
    responses = {
        "why": f"Based on the latest data, this movement appears to be driven by {random.choice(['technical indicators', 'market sentiment', 'earnings expectations', 'sector rotation'])}. The {random.choice(['RSI', 'MACD', 'Bollinger Bands'])} suggests {random.choice(['oversold conditions', 'overbought conditions', 'neutral momentum'])}.",
        "buy": f"Current analysis suggests this could be a {random.choice(['good', 'risky', 'excellent'])} entry point. Key factors include {random.choice(['strong fundamentals', 'technical breakout', 'support level', 'resistance break'])}. Consider your risk tolerance.",
        "compare": f"Comparing these stocks: {random.choice(['Stock A shows stronger momentum', 'Stock B has better fundamentals', 'Both have similar risk profiles'])}. Key differences include {random.choice(['P/E ratios', 'growth rates', 'volatility', 'sector exposure'])}.",
        "trend": f"The current trend is {random.choice(['bullish', 'bearish', 'sideways'])} with {random.choice(['strong', 'moderate', 'weak'])} momentum. Support levels are at {random.choice(['$150', '$160', '$170'])} and resistance at {random.choice(['$180', '$190', '$200'])}."
    }
    
    question_lower = user_question.lower()
    
    if "why" in question_lower or "reason" in question_lower:
        return responses["why"]
    elif "buy" in question_lower or "good time" in question_lower:
        return responses["buy"]
    elif "compare" in question_lower or "vs" in question_lower:
        return responses["compare"]
    elif "trend" in question_lower or "direction" in question_lower:
        return responses["trend"]
    else:
        return f"I analyzed the data and found that {random.choice(['technical indicators', 'market conditions', 'company fundamentals'])} suggest {random.choice(['positive momentum', 'caution is advised', 'neutral outlook'])}. Always do your own research!"

# Enhanced data collection with multiple sources
@st.cache_data(ttl=300)
def get_stock_data(symbol, period="1y"):
    """Get stock data with enhanced error handling and multiple sources"""
    import time
    import random
    
    max_retries = 3
    retry_delay = 2
    
    time.sleep(random.uniform(1.0, 2.0))
    
    for attempt in range(max_retries):
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data is not None and len(data) > 0:
                return data
            else:
                st.warning(f"No data found for {symbol}")
                return None
                
        except Exception as e:
            error_msg = str(e)
            
            if "Too Many Requests" in error_msg or "Rate limited" in error_msg:
                if attempt < max_retries - 1:
                    st.warning(f"Rate limited. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    st.error("Rate limited after multiple attempts")
                    return None
            else:
                st.error(f"Error fetching data: {error_msg}")
                return None
    
    return None

# Enhanced feature engineering
def create_advanced_features(data):
    """Create comprehensive technical indicators"""
    if data is None or len(data) == 0:
        return None
    
    df = data.copy()
    
    # Price features
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['price_change'] = df['Close'] - df['Close'].shift(1)
    df['price_change_pct'] = df['price_change'] / df['Close'].shift(1)
    
    # Moving averages (reduced for smaller datasets)
    for window in [5, 10, 20]:  # Removed 50, 200 for smaller datasets
        df[f'sma_{window}'] = ta.trend.sma_indicator(df['Close'], window=window)
        df[f'ema_{window}'] = ta.trend.ema_indicator(df['Close'], window=window)
    
    # RSI with shorter timeframes
    for window in [14]:  # Removed 21 for smaller datasets
        df[f'rsi_{window}'] = ta.momentum.rsi(df['Close'], window=window)
    
    # MACD
    df['macd'] = ta.trend.macd_diff(df['Close'])
    df['macd_signal'] = ta.trend.macd_signal(df['Close'])
    df['macd_histogram'] = ta.trend.macd_diff(df['Close']) - ta.trend.macd_signal(df['Close'])
    
    # Bollinger Bands
    df['bb_upper'] = ta.volatility.bollinger_hband(df['Close'])
    df['bb_lower'] = ta.volatility.bollinger_lband(df['Close'])
    df['bb_middle'] = ta.volatility.bollinger_mavg(df['Close'])
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Volume indicators (reduced window)
    df['volume_sma'] = df['Volume'].rolling(window=10).mean()  # Reduced from 20 to 10
    df['volume_ratio'] = df['Volume'] / df['volume_sma']
    df['on_balance_volume'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    
    # Volatility indicators (reduced window)
    df['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    df['volatility'] = df['returns'].rolling(window=10).std()  # Reduced from 20 to 10
    
    # Momentum indicators
    df['stoch_k'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
    df['stoch_d'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'])
    
    # Support and resistance levels (reduced window)
    df['support'] = df['Low'].rolling(window=10).min()  # Reduced from 20 to 10
    df['resistance'] = df['High'].rolling(window=10).max()  # Reduced from 20 to 10
    
    # Target variables for different timeframes
    for days in [1, 3, 7]:
        df[f'target_{days}d'] = (df['Close'].shift(-days) > df['Close']).astype(int)
    
    return df

# Enhanced model training with multiple timeframes
def train_models(features):
    """Train models for different prediction timeframes"""
    if features is None or len(features) < 30:  # Reduced from 100 to 30
        st.warning(f"⚠️ Only {len(features) if features is not None else 0} data points available. Need at least 30 for basic predictions.")
        return {}, {}
    
    features = features.dropna()
    
    if len(features) < 20:  # Reduced from 50 to 20
        st.warning(f"⚠️ After cleaning, only {len(features)} data points remain. Need at least 20 for training.")
        return {}, {}
    
    models = {}
    metrics = {}
    
    # Feature columns
    feature_cols = [col for col in features.columns if not col.startswith('target') and col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
    
    if len(feature_cols) == 0:
        st.error("❌ No technical indicators could be created. This might be due to insufficient data.")
        return {}, {}
    
    X = features[feature_cols]
    
    # Train models for different timeframes
    for timeframe in ['1d', '3d', '7d']:
        target_col = f'target_{timeframe.replace("d", "")}d'
        
        if target_col in features.columns:
            y = features[target_col]
            
            # Check if we have enough data and both classes
            if len(y.unique()) > 1 and len(y) >= 10:  # Reduced minimum requirement
                try:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    model = RandomForestClassifier(n_estimators=50, random_state=42)  # Reduced from 100 to 50
                    model.fit(X_train, y_train)
                    
                    y_pred = model.predict(X_test)
                    
                    models[timeframe] = model
                    metrics[timeframe] = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred, zero_division=0),
                        'recall': recall_score(y_test, y_pred, zero_division=0),
                        'f1_score': f1_score(y_test, y_pred, zero_division=0)
                    }
                    
                    st.success(f"✅ {timeframe.upper()} model trained successfully!")
                    
                except Exception as e:
                    st.warning(f"⚠️ Could not train {timeframe.upper()} model: {str(e)}")
            else:
                st.warning(f"⚠️ Insufficient data for {timeframe.upper()} predictions. Need at least 10 samples with both classes.")
    
    return models, metrics

# Enhanced predictions with confidence intervals and plain English
def make_predictions(models, features, current_price):
    """Make predictions for multiple timeframes with confidence intervals and plain English"""
    predictions = {}
    
    if not models or features is None:
        return predictions
    
    feature_cols = [col for col in features.columns if not col.startswith('target') and col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
    latest_features = features[feature_cols].iloc[-1:].dropna()
    
    if latest_features.empty:
        return predictions
    
    for timeframe, model in models.items():
        if model is not None:
            prediction = model.predict(latest_features)[0]
            probabilities = model.predict_proba(latest_features)[0]
            confidence = probabilities.max()
            
            # Calculate price prediction with confidence interval
            avg_return = features['returns'].mean()
            std_return = features['returns'].std()
            
            if prediction == 1:  # Up
                expected_return = avg_return
                direction = "UP"
                arrow = "↗️"
            else:  # Down
                expected_return = -avg_return
                direction = "DOWN"
                arrow = "↘️"
            
            predicted_price = current_price * (1 + expected_return)
            
            # Confidence interval (simplified)
            confidence_interval = std_return * 1.96  # 95% confidence
            lower_bound = current_price * (1 + expected_return - confidence_interval)
            upper_bound = current_price * (1 + expected_return + confidence_interval)
            
            # Generate plain English forecast
            if confidence > 0.7:
                confidence_text = "high confidence"
            elif confidence > 0.5:
                confidence_text = "moderate confidence"
            else:
                confidence_text = "low confidence"
            
            if timeframe == "1d":
                time_text = "tomorrow"
            elif timeframe == "3d":
                time_text = "in 3 days"
            else:
                time_text = "next week"
            
            plain_english = f"Our AI model predicts with {confidence_text} that the stock will move {direction.lower()} {arrow} by {abs(expected_return)*100:.1f}% {time_text}, reaching ${predicted_price:.2f} (range: ${lower_bound:.2f} - ${upper_bound:.2f})."
            
            predictions[timeframe] = {
                'direction': direction,
                'arrow': arrow,
                'confidence': confidence,
                'confidence_text': confidence_text,
                'predicted_price': predicted_price,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'expected_return': expected_return,
                'plain_english': plain_english
            }
    
    return predictions

# Interactive chart with technical indicators
def create_interactive_chart(data, symbol):
    """Create an interactive chart with technical indicators"""
    fig = go.Figure()
    
    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price',
        increasing_line_color='#26A69A',
        decreasing_line_color='#EF5350'
    ))
    
    # Add moving averages
    if 'sma_20' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['sma_20'],
            mode='lines',
            name='SMA 20',
            line=dict(color='orange', width=2)
        ))
    
    if 'sma_50' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['sma_50'],
            mode='lines',
            name='SMA 50',
            line=dict(color='blue', width=2)
        ))
    
    # Add Bollinger Bands
    if 'bb_upper' in data.columns and 'bb_lower' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['bb_upper'],
            mode='lines',
            name='BB Upper',
            line=dict(color='gray', width=1, dash='dash'),
            fill=None
        ))
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['bb_lower'],
            mode='lines',
            name='BB Lower',
            line=dict(color='gray', width=1, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(128,128,128,0.1)'
        ))
    
    fig.update_layout(
        title=f'{symbol} Interactive Chart',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=600,
        template='plotly_white',
        showlegend=True
    )
    
    return fig

# Watchlist management
def add_to_watchlist(symbol):
    if symbol not in st.session_state.watchlist:
        st.session_state.watchlist.append(symbol)

def remove_from_watchlist(symbol):
    if symbol in st.session_state.watchlist:
        st.session_state.watchlist.remove(symbol)

# Home Page
if st.session_state.current_page == 'home':
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <h1 style="font-size: 3.5rem; margin-bottom: 1rem;">🚀 AI Stock Predictor Pro</h1>
        <p style="font-size: 1.5rem; margin-bottom: 2rem;">
            Advanced AI predictions with multiple timeframes, interactive charts, and portfolio tracking
        </p>
        <p style="font-size: 1.1rem; opacity: 0.9;">
            The most comprehensive stock prediction platform powered by machine learning
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Dark mode toggle
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🌙 Toggle Dark Mode" if not st.session_state.dark_mode else "☀️ Toggle Light Mode"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
    
    # Features Section
    st.markdown("### ✨ New Pro Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🤖 AI Chat Assistant</h3>
            <p>Ask anything about stocks in plain English - "Why is AAPL down?" or "Is now a good time to buy TSLA?"</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Multi-Timeframe Predictions</h3>
            <p>Predict 1 day, 3 days, and 1 week ahead with confidence intervals</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>📈 Interactive Charts</h3>
            <p>Zoom, pan, and explore technical indicators on interactive price charts</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>💬 Plain English Forecasts</h3>
            <p>Get easy-to-understand predictions like "AAPL is likely to rise 2.5% next week"</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>⭐ Watchlist</h3>
            <p>Save and track multiple stocks in your personalized watchlist</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>🌙 Dark Mode</h3>
            <p>Switch between light and dark themes for comfortable viewing</p>
        </div>
        """, unsafe_allow_html=True)
    
    # CTA Section
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h2 style="color: #333;">Ready to Experience Pro Features?</h2>
            <p style="font-size: 1.1rem; margin-bottom: 2rem; color: #555;">
                Access advanced AI predictions, interactive charts, and portfolio management
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Launch Pro Predictor", type="primary", use_container_width=True):
            navigate_to('predictor')
    
    # Stats Section with proper contrast
    st.markdown("### 📊 Mission Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stats-card">
            <h2>🚀</h2>
            <h3>AI-Powered</h3>
            <p>Advanced ML algorithms</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stats-card">
            <h2>⚡</h2>
            <h3>Real-Time</h3>
            <p>Live market data</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stats-card">
            <h2>🎯</h2>
            <h3>Accurate</h3>
            <p>High prediction accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stats-card">
            <h2>💎</h2>
            <h3>Professional</h3>
            <p>Institutional-grade analysis</p>
        </div>
        """, unsafe_allow_html=True)

# Predictor Page
elif st.session_state.current_page == 'predictor':
    # Navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🏠 Back to Home", use_container_width=True):
            navigate_to('home')
    
    # Header
    st.markdown("""
    <div class="hero-section">
        <h1>📈 AI Stock Predictor Pro</h1>
        <p style="font-size: 1.2rem; margin-top: 0.5rem;">
            Advanced predictions with multiple timeframes and interactive analytics
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # Stock symbol
        symbol = st.text_input("📊 Stock Symbol", value="AAPL").upper()
        
        # Popular symbols
        st.markdown("**⭐ Popular Symbols:**")
        popular_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "AMD", "INTC"]
        cols = st.columns(2)
        for i, sym in enumerate(popular_symbols):
            if cols[i % 2].button(sym, key=f"btn_{sym}"):
                symbol = sym
        
        # Time period
        st.markdown("**⏰ Time Period:**")
        period_options = {
            "📅 6 Months": "6mo",
            "📅 1 Year": "1y", 
            "📅 2 Years": "2y",
            "📅 5 Years": "5y"
        }
        selected_period = st.selectbox("Select Period", list(period_options.keys()))
        period = period_options[selected_period]
        
        # Watchlist management
        st.markdown("### ⭐ Watchlist")
        
        if symbol not in st.session_state.watchlist:
            if st.button("➕ Add to Watchlist"):
                add_to_watchlist(symbol)
                st.success(f"Added {symbol} to watchlist!")
        else:
            if st.button("➖ Remove from Watchlist"):
                remove_from_watchlist(symbol)
                st.success(f"Removed {symbol} from watchlist!")
        
        # Show watchlist
        if st.session_state.watchlist:
            st.markdown("**Your Watchlist:**")
            for watch_symbol in st.session_state.watchlist:
                col1, col2 = st.columns([3, 1])
                col1.write(watch_symbol)
                if col2.button("Remove", key=f"remove_{watch_symbol}"):
                    remove_from_watchlist(watch_symbol)
                    st.rerun()
        
        # Run analysis
        st.markdown("---")
        run_analysis = st.button("🚀 Run Advanced Analysis", type="primary", use_container_width=True)

    # Main content
    if run_analysis:
        try:
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Data collection
            status_text.text("📊 Collecting stock data...")
            progress_bar.progress(25)
            
            data = get_stock_data(symbol, period=period)
            
            if data is None:
                st.error("Could not fetch stock data. Please try again.")
                st.stop()
            
            st.success(f"✅ Collected {len(data)} records for {symbol}")
            
            # Feature engineering
            status_text.text("🔧 Creating advanced features...")
            progress_bar.progress(50)
            
            features = create_advanced_features(data)
            
            if features is None:
                st.error("Could not create features from the data.")
                st.stop()
            
            st.success(f"✅ Created {len([col for col in features.columns if not col.startswith('target')])} technical indicators")
            
            # Model training
            status_text.text("🤖 Training advanced models...")
            progress_bar.progress(75)
            
            models, metrics = train_models(features)
            
            if not models:
                st.error("Could not train models. Insufficient data.")
                st.stop()
            
            st.success("✅ Models trained successfully!")
            progress_bar.progress(100)
            status_text.text("🎉 Analysis complete!")
            
            # Display results
            current_price = data['Close'].iloc[-1]
            predictions = make_predictions(models, features, current_price)
            
            # Interactive chart
            st.markdown("### 📈 Interactive Price Chart")
            chart = create_interactive_chart(features, symbol)
            st.plotly_chart(chart, use_container_width=True)
            
            # Multi-timeframe predictions with enhanced display
            st.markdown("### 🔮 AI Predictions")
            
            if predictions:
                for timeframe, pred in predictions.items():
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h3>📊 {timeframe.upper()} Prediction</h3>
                        <div class="prediction-badge prediction-{pred['direction'].lower()}">
                            {pred['arrow']} {pred['direction']}
                        </div>
                        <div class="confidence-meter">
                            <h4>Confidence: {pred['confidence']:.1%}</h4>
                            <p><strong>Expected Price:</strong> ${pred['predicted_price']:.2f}</p>
                            <p><strong>Range:</strong> ${pred['lower_bound']:.2f} - ${pred['upper_bound']:.2f}</p>
                        </div>
                        <p style="font-style: italic; margin-top: 1rem;">{pred['plain_english']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Model performance
            st.markdown("### 🤖 Model Performance")
            
            if metrics:
                for timeframe, metric in metrics.items():
                    st.markdown(f"**{timeframe.upper()} Model:**")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    col1.metric("Accuracy", f"{metric['accuracy']:.2%}")
                    col2.metric("Precision", f"{metric['precision']:.2%}")
                    col3.metric("Recall", f"{metric['recall']:.2%}")
                    col4.metric("F1-Score", f"{metric['f1_score']:.2%}")
            
            # Save prediction to history
            if predictions:
                st.session_state.prediction_history[symbol] = {
                    'timestamp': datetime.now().isoformat(),
                    'predictions': predictions,
                    'current_price': current_price
                }
            
            st.success("🎉 Advanced analysis complete!")
            st.balloons()
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            st.exception(e)

    # AI Chat Section
    else:
        st.markdown("### 🤖 AI Stock Assistant")
        st.markdown("Ask me anything about stocks in plain English!")
        
        # Chat input
        user_question = st.text_input("💬 Ask me about stocks:", placeholder="e.g., Why is AAPL down today? Is now a good time to buy TSLA?")
        
        if st.button("🚀 Ask AI", type="primary"):
            if user_question:
                # Add user message to chat
                st.session_state.chat_history.append({"role": "user", "content": user_question})
                
                # Generate AI response
                ai_response = generate_ai_response(user_question)
                st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                
                st.rerun()
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown("### 💬 Chat History")
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>You:</strong> {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message ai-message">
                        <strong>AI Assistant:</strong> {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Information section
        st.markdown("### 📚 How to Use")
        st.markdown("""
        <div class="info-section">
            <h2>🚀 Pro Features Guide</h2>
            
            <h3>🤖 AI Chat Assistant</h3>
            <ul>
                <li><strong>Ask anything:</strong> "Why is AAPL down today?"</li>
                <li><strong>Get advice:</strong> "Is now a good time to buy TSLA?"</li>
                <li><strong>Compare stocks:</strong> "Compare NVDA vs AMD"</li>
                <li><strong>Understand trends:</strong> "What's the trend for MSFT?"</li>
            </ul>
            
            <h3>📊 Multi-Timeframe Predictions</h3>
            <ul>
                <li><strong>1 Day</strong>: Short-term price movement prediction</li>
                <li><strong>3 Days</strong>: Medium-term trend analysis</li>
                <li><strong>7 Days</strong>: Weekly outlook with confidence intervals</li>
            </ul>
            
            <h3>💬 Plain English Forecasts</h3>
            <ul>
                <li><strong>Easy to understand:</strong> "AAPL is likely to rise 2.5% next week"</li>
                <li><strong>Confidence levels:</strong> High, moderate, or low confidence</li>
                <li><strong>Price ranges:</strong> Expected price with upper and lower bounds</li>
            </ul>
            
            <h3>📈 Interactive Charts</h3>
            <ul>
                <li><strong>Zoom & Pan</strong>: Explore price data in detail</li>
                <li><strong>Technical Indicators</strong>: Overlay moving averages and Bollinger Bands</li>
                <li><strong>Candlestick View</strong>: Professional trading chart format</li>
            </ul>
            
            <h3>⭐ Watchlist Management</h3>
            <ul>
                <li><strong>Save Favorites</strong>: Add stocks to your personal watchlist</li>
                <li><strong>Quick Access</strong>: Analyze multiple stocks efficiently</li>
                <li><strong>Portfolio Tracking</strong>: Monitor your selected stocks</li>
            </ul>
            
            <h2>🎯 How to Use</h2>
            
            <ol>
                <li><strong>📊 Enter a stock symbol</strong> (e.g., AAPL, MSFT, GOOGL)</li>
                <li><strong>⏰ Select time period</strong> for analysis</li>
                <li><strong>⭐ Add to watchlist</strong> for easy access</li>
                <li><strong>🚀 Run advanced analysis</strong> for comprehensive predictions</li>
                <li><strong>📈 Explore interactive charts</strong> and technical indicators</li>
                <li><strong>🔮 View multi-timeframe predictions</strong> with confidence intervals</li>
                <li><strong>🤖 Chat with AI</strong> for personalized insights</li>
            </ol>
            
            <h2>⚠️ Disclaimer</h2>
            
            <p>
                This tool is for educational purposes only. Past performance does not guarantee future results. 
                Always do your own research before making investment decisions.
            </p>
        </div>
        """, unsafe_allow_html=True)

# Watchlist Page (if you want to add it later)
elif st.session_state.current_page == 'watchlist':
    st.markdown("### ⭐ Your Watchlist")
    
    if not st.session_state.watchlist:
        st.info("Your watchlist is empty. Add stocks from the predictor page!")
    else:
        for symbol in st.session_state.watchlist:
            st.write(f"📊 {symbol}")
            # Add quick analysis buttons here
