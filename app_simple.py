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

# Enhanced CSS with dark mode support
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
            .feature-card {
                background: #2d2d2d;
                padding: 2rem;
                border-radius: 15px;
                margin: 1rem 0;
                box-shadow: 0 5px 15px rgba(0,0,0,0.3);
                border-left: 5px solid #667eea;
                color: white;
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
                background: #2d2d2d;
                padding: 1rem;
                border-radius: 10px;
                margin: 0.5rem 0;
                border-left: 3px solid #667eea;
                color: white;
            }
            .timeframe-card {
                background: #2d2d2d;
                padding: 1rem;
                border-radius: 10px;
                margin: 0.5rem 0;
                color: white;
                text-align: center;
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
            .feature-card {
                background: white;
                padding: 2rem;
                border-radius: 15px;
                margin: 1rem 0;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                border-left: 5px solid #667eea;
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
        </style>
        """

st.markdown(get_css(st.session_state.dark_mode), unsafe_allow_html=True)

# Navigation function
def navigate_to(page):
    st.session_state.current_page = page

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
    
    # Moving averages
    for window in [5, 10, 20, 50, 200]:
        df[f'sma_{window}'] = ta.trend.sma_indicator(df['Close'], window=window)
        df[f'ema_{window}'] = ta.trend.ema_indicator(df['Close'], window=window)
    
    # RSI with multiple timeframes
    for window in [14, 21]:
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
    
    # Volume indicators
    df['volume_sma'] = df['Volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_sma']
    df['on_balance_volume'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    
    # Volatility indicators
    df['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    df['volatility'] = df['returns'].rolling(window=20).std()
    
    # Momentum indicators
    df['stoch_k'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
    df['stoch_d'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'])
    
    # Support and resistance levels
    df['support'] = df['Low'].rolling(window=20).min()
    df['resistance'] = df['High'].rolling(window=20).max()
    
    # Target variables for different timeframes
    for days in [1, 3, 7]:
        df[f'target_{days}d'] = (df['Close'].shift(-days) > df['Close']).astype(int)
    
    return df

# Enhanced model training with multiple timeframes
def train_models(features):
    """Train models for different prediction timeframes"""
    if features is None or len(features) < 100:
        return {}, {}
    
    features = features.dropna()
    
    if len(features) < 50:
        return {}, {}
    
    models = {}
    metrics = {}
    
    # Feature columns
    feature_cols = [col for col in features.columns if not col.startswith('target') and col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
    
    if len(feature_cols) == 0:
        return {}, {}
    
    X = features[feature_cols]
    
    # Train models for different timeframes
    for timeframe in ['1d', '3d', '7d']:
        target_col = f'target_{timeframe.replace("d", "")}d'
        
        if target_col in features.columns:
            y = features[target_col]
            
            if len(y.unique()) > 1:  # Only train if we have both classes
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                
                models[timeframe] = model
                metrics[timeframe] = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred, zero_division=0),
                    'f1_score': f1_score(y_test, y_pred, zero_division=0)
                }
    
    return models, metrics

# Enhanced predictions with confidence intervals
def make_predictions(models, features, current_price):
    """Make predictions for multiple timeframes with confidence intervals"""
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
            else:  # Down
                expected_return = -avg_return
            
            predicted_price = current_price * (1 + expected_return)
            
            # Confidence interval (simplified)
            confidence_interval = std_return * 1.96  # 95% confidence
            lower_bound = current_price * (1 + expected_return - confidence_interval)
            upper_bound = current_price * (1 + expected_return + confidence_interval)
            
            predictions[timeframe] = {
                'direction': 'UP' if prediction == 1 else 'DOWN',
                'confidence': confidence,
                'predicted_price': predicted_price,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'expected_return': expected_return
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
        
        st.markdown("""
        <div class="feature-card">
            <h3>⭐ Watchlist</h3>
            <p>Save and track multiple stocks in your personalized watchlist</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>🌙 Dark Mode</h3>
            <p>Switch between light and dark themes for comfortable viewing</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>🎯 Advanced Analytics</h3>
            <p>Comprehensive technical indicators and risk assessment</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Performance Tracking</h3>
            <p>Track prediction accuracy and model performance over time</p>
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
            
            # Multi-timeframe predictions
            st.markdown("### 🔮 Multi-Timeframe Predictions")
            
            if predictions:
                cols = st.columns(len(predictions))
                
                for i, (timeframe, pred) in enumerate(predictions.items()):
                    with cols[i]:
                        st.markdown(f"""
                        <div class="timeframe-card">
                            <h3>📊 {timeframe.upper()} Prediction</h3>
                            <h2>{pred['direction']}</h2>
                            <p>Confidence: {pred['confidence']:.1%}</p>
                            <p>Price: ${pred['predicted_price']:.2f}</p>
                            <p>Range: ${pred['lower_bound']:.2f} - ${pred['upper_bound']:.2f}</p>
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

    # Information section
    else:
        st.markdown("""
        ## 🚀 Pro Features Guide
        
        ### 📊 Multi-Timeframe Predictions
        - **1 Day**: Short-term price movement prediction
        - **3 Days**: Medium-term trend analysis
        - **7 Days**: Weekly outlook with confidence intervals
        
        ### 📈 Interactive Charts
        - **Zoom & Pan**: Explore price data in detail
        - **Technical Indicators**: Overlay moving averages and Bollinger Bands
        - **Candlestick View**: Professional trading chart format
        
        ### ⭐ Watchlist Management
        - **Save Favorites**: Add stocks to your personal watchlist
        - **Quick Access**: Analyze multiple stocks efficiently
        - **Portfolio Tracking**: Monitor your selected stocks
        
        ### 🌙 Dark Mode
        - **Toggle Theme**: Switch between light and dark modes
        - **Eye Comfort**: Reduce eye strain during extended use
        - **Professional Look**: Modern interface design
        
        ## 🎯 How to Use
        
        1. **📊 Enter a stock symbol** (e.g., AAPL, MSFT, GOOGL)
        2. **⏰ Select time period** for analysis
        3. **⭐ Add to watchlist** for easy access
        4. **🚀 Run advanced analysis** for comprehensive predictions
        5. **📈 Explore interactive charts** and technical indicators
        6. **🔮 View multi-timeframe predictions** with confidence intervals
        
        ## ⚠️ Disclaimer
        
        This tool is for educational purposes only. Past performance does not guarantee future results. 
        Always do your own research before making investment decisions.
        """)

# Watchlist Page (if you want to add it later)
elif st.session_state.current_page == 'watchlist':
    st.markdown("### ⭐ Your Watchlist")
    
    if not st.session_state.watchlist:
        st.info("Your watchlist is empty. Add stocks from the predictor page!")
    else:
        for symbol in st.session_state.watchlist:
            st.write(f"📊 {symbol}")
            # Add quick analysis buttons here
