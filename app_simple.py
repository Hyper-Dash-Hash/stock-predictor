import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import ta
import time
import random

# Page configuration
st.set_page_config(
    page_title="AI Stock Predictor Pro",
    page_icon="🚀",
    layout="wide"
)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'symbol_input' not in st.session_state:
    st.session_state.symbol_input = "AAPL"

# Fixed CSS with proper contrast
def get_css(dark_mode=False):
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
        .prediction-card h3, .prediction-card p {
            color: white !important;
            text-shadow: 0 1px 2px rgba(0,0,0,0.3);
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
        .prediction-badge {
            display: inline-block;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: bold;
            margin: 0.5rem;
            color: white;
        }
        .prediction-up {
            background: linear-gradient(135deg, #26A69A, #4CAF50);
            color: white;
        }
        .prediction-down {
            background: linear-gradient(135deg, #EF5350, #F44336);
            color: white;
        }
    </style>
    """

st.markdown(get_css(st.session_state.dark_mode), unsafe_allow_html=True)

# Navigation function
def navigate_to(page):
    st.session_state.current_page = page

# Data collection
@st.cache_data(ttl=300)
def get_stock_data(symbol, period="1y"):
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        return data if len(data) > 0 else None
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# Feature engineering
def create_advanced_features(data):
    if data is None or len(data) == 0:
        return None
    
    df = data.copy()
    
    # Basic features
    df['returns'] = df['Close'].pct_change()
    df['volume_sma'] = df['Volume'].rolling(window=10).mean()
    df['on_balance_volume'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    
    # Moving averages
    for window in [5, 10, 20]:
        df[f'sma_{window}'] = ta.trend.sma_indicator(df['Close'], window=window)
    
    # RSI
    df['rsi'] = ta.momentum.rsi(df['Close'], window=14)
    
    # MACD
    df['macd'] = ta.trend.macd_diff(df['Close'])
    df['macd_signal'] = ta.trend.macd_signal(df['Close'])
    
    # Bollinger Bands
    df['bb_upper'] = ta.volatility.bollinger_hband(df['Close'])
    df['bb_lower'] = ta.volatility.bollinger_lband(df['Close'])
    
    # Target variables for different timeframes
    for days in [1, 3, 7]:
        df[f'target_{days}d'] = (df['Close'].shift(-days) > df['Close']).astype(int)
    
    return df

# Model training
def train_models(features):
    if features is None or len(features) < 30:
        return {}, {}
    
    features = features.dropna()
    if len(features) < 20:
        return {}, {}
    
    models = {}
    metrics = {}
    
    feature_cols = [col for col in features.columns if not col.startswith('target') and col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
    X = features[feature_cols]
    
    for timeframe in ['1d', '3d', '7d']:
        target_col = f'target_{timeframe.replace("d", "")}d'
        
        if target_col in features.columns:
            y = features[target_col]
            
            if len(y.unique()) > 1 and len(y) >= 10:
                try:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model = RandomForestClassifier(n_estimators=50, random_state=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    models[timeframe] = model
                    metrics[timeframe] = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred, zero_division=0),
                        'recall': recall_score(y_test, y_pred, zero_division=0),
                        'f1_score': f1_score(y_test, y_pred, zero_division=0)
                    }
                except Exception as e:
                    st.warning(f"Could not train {timeframe.upper()} model: {str(e)}")
    
    return models, metrics

# Enhanced predictions with longer timeframes
def make_predictions(models, features, current_price, prediction_timeframe="7d"):
    predictions = {}
    
    if not models or features is None:
        return predictions
    
    feature_cols = [col for col in features.columns if not col.startswith('target') and col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
    latest_features = features[feature_cols].iloc[-1:].dropna()
    
    if latest_features.empty:
        return predictions
    
    # Convert prediction timeframe to days
    timeframe_days = int(prediction_timeframe.replace('d', ''))
    
    for timeframe, model in models.items():
        if model is not None:
            prediction = model.predict(latest_features)[0]
            probabilities = model.predict_proba(latest_features)[0]
            confidence = probabilities.max()
            
            # Calculate price prediction
            avg_return = features['returns'].mean()
            std_return = features['returns'].std()
            
            # Adjust for longer timeframes
            if timeframe_days > 7:
                compound_return = avg_return * (timeframe_days / 7)
                compound_volatility = std_return * np.sqrt(timeframe_days / 7)
            else:
                compound_return = avg_return
                compound_volatility = std_return
            
            if prediction == 1:
                expected_return = compound_return
                direction = "UP"
                arrow = "↗️"
            else:
                expected_return = -compound_return
                direction = "DOWN"
                arrow = "↘️"
            
            predicted_price = current_price * (1 + expected_return)
            confidence_interval = compound_volatility * 1.96
            lower_bound = current_price * (1 + expected_return - confidence_interval)
            upper_bound = current_price * (1 + expected_return + confidence_interval)
            
            # Generate plain English forecast
            if confidence > 0.7:
                confidence_text = "high confidence"
            elif confidence > 0.5:
                confidence_text = "moderate confidence"
            else:
                confidence_text = "low confidence"
            
            # Timeframe-specific descriptions
            timeframe_texts = {
                1: "tomorrow",
                3: "in 3 days",
                7: "next week",
                30: "next month",
                90: "in 3 months",
                180: "in 6 months",
                365: "in 1 year",
                730: "in 2 years",
                1825: "in 5 years"
            }
            time_text = timeframe_texts.get(timeframe_days, f"in {timeframe_days} days")
            
            percentage_change = abs(expected_return) * 100
            plain_english = f"Our AI model predicts with {confidence_text} that the stock will move {direction.lower()} {arrow} by {percentage_change:.1f}% {time_text}, reaching ${predicted_price:.2f} (range: ${lower_bound:.2f} - ${upper_bound:.2f})."
            
            predictions[timeframe] = {
                'direction': direction,
                'arrow': arrow,
                'confidence': confidence,
                'confidence_text': confidence_text,
                'predicted_price': predicted_price,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'expected_return': expected_return,
                'plain_english': plain_english,
                'timeframe_days': timeframe_days,
                'percentage_change': percentage_change
            }
    
    return predictions

# Interactive chart
def create_interactive_chart(data, symbol):
    fig = go.Figure()
    
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
    
    if 'sma_20' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['sma_20'],
            mode='lines',
            name='SMA 20',
            line=dict(color='orange', width=2)
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
    st.markdown("""
    <div class="hero-section">
        <h1 style="font-size: 3.5rem; margin-bottom: 1rem;">🚀 AI Stock Predictor Pro</h1>
        <p style="font-size: 1.5rem; margin-bottom: 2rem;">
            Advanced AI predictions with multiple timeframes and interactive charts
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
    st.markdown("### ✨ Pro Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>🤖 AI Chat Assistant</h3>
            <p>Ask anything about stocks in plain English</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>📊 Multi-Timeframe Predictions</h3>
            <p>Predict 1 day to 5 years ahead</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>💬 Plain English Forecasts</h3>
            <p>Easy-to-understand predictions</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h3>⭐ Watchlist</h3>
            <p>Save and track multiple stocks</p>
        </div>
        """, unsafe_allow_html=True)
    
    # CTA Section
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🚀 Launch Pro Predictor", type="primary", use_container_width=True):
            navigate_to('predictor')
    
    # Stats Section
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
        
        # Stock symbol with auto-replace functionality
        symbol = st.text_input("📊 Stock Symbol", value=st.session_state.symbol_input, key="symbol_input").upper()
        
        # Popular symbols with auto-replace functionality
        st.markdown("**⭐ Popular Symbols:**")
        popular_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "AMD", "INTC"]
        cols = st.columns(2)
        for i, sym in enumerate(popular_symbols):
            if cols[i % 2].button(sym, key=f"btn_{sym}"):
                # Update the symbol input using session state
                st.session_state.symbol_input = sym
                st.rerun()
        
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
        
        # Prediction timeframes
        st.markdown("**🔮 Prediction Timeframes:**")
        prediction_timeframes = {
            "📊 1 Day": "1d",
            "📊 3 Days": "3d", 
            "📊 1 Week": "7d",
            "📊 1 Month": "30d",
            "📊 3 Months": "90d",
            "📊 6 Months": "180d",
            "📊 1 Year": "365d",
            "📊 2 Years": "730d",
            "📊 5 Years": "1825d"
        }
        selected_prediction = st.selectbox("Select Prediction Timeframe", list(prediction_timeframes.keys()))
        prediction_timeframe = prediction_timeframes[selected_prediction]
        
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
            predictions = make_predictions(models, features, current_price, prediction_timeframe)
            
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
                        <p><strong>Confidence:</strong> {pred['confidence']:.1%}</p>
                        <p><strong>Expected Price:</strong> ${pred['predicted_price']:.2f}</p>
                        <p><strong>Range:</strong> ${pred['lower_bound']:.2f} - ${pred['upper_bound']:.2f}</p>
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
            
            st.success("🎉 Advanced analysis complete!")
            st.balloons()
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            st.exception(e)

    # Information section
    else:
        st.markdown("### 📚 How to Use")
        
        st.markdown("""
        **🚀 Pro Features Guide**
        
        **📊 Multi-Timeframe Predictions**
        - 1 Day to 5 Years: Choose from 9 different prediction timeframes
        - Confidence Intervals: See expected price ranges
        - Plain English: Easy-to-understand forecasts
        
        **📈 Interactive Charts**
        - Zoom & Pan: Explore price data in detail
        - Technical Indicators: Overlay moving averages
        - Candlestick View: Professional trading chart format
        
        **⭐ Watchlist Management**
        - Save Favorites: Add stocks to your personal watchlist
        - Quick Access: Analyze multiple stocks efficiently
        - Portfolio Tracking: Monitor your selected stocks
        
        **🎯 How to Use**
        
        1. **📊 Enter a stock symbol** (e.g., AAPL, MSFT, GOOGL)
        2. **⏰ Select time period** for analysis
        3. **🔮 Choose prediction timeframe** (1 day to 5 years)
        4. **⭐ Add to watchlist** for easy access
        5. **🚀 Run advanced analysis** for comprehensive predictions
        6. **📈 Explore interactive charts** and technical indicators
        7. **🔮 View predictions** with confidence intervals
        
        **⚠️ Disclaimer**
        
        This tool is for educational purposes only. Past performance does not guarantee future results. 
        Always do your own research before making investment decisions.
        """)
