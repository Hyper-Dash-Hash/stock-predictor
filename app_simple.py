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

# Page configuration
st.set_page_config(
    page_title="AI Stock Predictor",
    page_icon="��",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .success-box {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .info-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        text-align: center;
    }
    .recommendation-buy {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        text-align: center;
    }
    .recommendation-sell {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        text-align: center;
    }
    .recommendation-hold {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: #333;
        margin: 1rem 0;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Beautiful header
st.markdown("""
<div class="main-header">
    <h1>🚀 AI Stock Price Predictor</h1>
    <p style="font-size: 1.2rem; margin-top: 0.5rem;">
        Advanced machine learning predictions with real-time trading recommendations
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar with better styling
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    # Stock symbol input with emoji
    symbol = st.text_input("📊 Stock Symbol", value="AAPL").upper()
    
    # Popular symbols with better formatting
    st.markdown("**�� Popular Symbols:**")
    popular_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "AMD", "INTC"]
    cols = st.columns(2)
    for i, sym in enumerate(popular_symbols):
        if cols[i % 2].button(sym, key=f"btn_{sym}"):
            symbol = sym
    
    # Time period with icons
    st.markdown("**⏰ Time Period:**")
    period_options = {
        "�� 6 Months": "6mo",
        "📅 1 Year": "1y", 
        "📅 2 Years": "2y",
        "�� 5 Years": "5y"
    }
    selected_period = st.selectbox("Select Period", list(period_options.keys()))
    period = period_options[selected_period]
    
    # Model selection
    st.markdown("**🤖 Model Type:**")
    model_type = st.selectbox("Choose Model", ["Random Forest", "LSTM (Coming Soon)"])
    
    # Demo mode with better description
    demo_mode = st.checkbox("�� Demo Mode", help="Use sample data when API is rate limited")
    
    # API Status with styling
    if not demo_mode:
        st.markdown("""
        <div class="warning-box">
            ⚠️ <strong>API Status</strong>: Yahoo Finance may be rate limiting requests. Try Demo Mode if needed.
        </div>
        """, unsafe_allow_html=True)
    
    # Run button with gradient
    st.markdown("---")
    run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_stock_data(symbol, period="1y"):
    """Get stock data from Yahoo Finance with retry logic and fallback"""
    import time
    import random
    
    max_retries = 2  # Reduced retries to avoid long waits
    retry_delay = 3  # Increased delay
    
    # Add random delay to avoid rate limiting
    time.sleep(random.uniform(1.0, 3.0))
    
    for attempt in range(max_retries):
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data is not None and len(data) > 0:
                st.success(f"✅ Successfully fetched {len(data)} records for {symbol}")
                return data
            else:
                st.warning(f"No data found for {symbol}. Please check the symbol.")
                return None
                
        except Exception as e:
            error_msg = str(e)
            
            if "Too Many Requests" in error_msg or "Rate limited" in error_msg:
                if attempt < max_retries - 1:
                    st.warning(f"Rate limited. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    st.error(f"Rate limited after {max_retries} attempts.")
                    st.info("💡 **Global Rate Limiting Detected:**")
                    st.info("• Yahoo Finance is blocking all requests")
                    st.info("• This affects all users, not just you")
                    st.info("• Try again in 15-30 minutes")
                    return None
            else:
                st.error(f"Error fetching data for {symbol}: {error_msg}")
                st.info("�� **Try:** Different symbol or check spelling")
                return None
    
    return None

def get_sample_data():
    """Get sample data for demonstration when API fails"""
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Create sample data for demonstration
    dates = pd.date_range(start=datetime.now() - timedelta(days=365), end=datetime.now(), freq='D')
    np.random.seed(42)
    
    # Generate realistic stock data
    base_price = 150
    returns = np.random.normal(0.001, 0.02, len(dates))  # Daily returns
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    sample_data = pd.DataFrame({
        'Open': [p * (1 + np.random.normal(0, 0.01)) for p in prices],
        'High': [p * (1 + abs(np.random.normal(0, 0.02))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.02))) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    return sample_data

def create_features(data):
    """Create technical indicators"""
    if data is None or len(data) == 0:
        return None
    
    # Copy data to avoid modifying original
    df = data.copy()
    
    # Basic price features
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Moving averages
    df['sma_5'] = ta.trend.sma_indicator(df['Close'], window=5)
    df['sma_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['sma_50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['ema_12'] = ta.trend.ema_indicator(df['Close'], window=12)
    df['ema_26'] = ta.trend.ema_indicator(df['Close'], window=26)
    
    # RSI
    df['rsi_14'] = ta.momentum.rsi(df['Close'], window=14)
    
    # MACD
    df['macd'] = ta.trend.macd_diff(df['Close'])
    df['macd_signal'] = ta.trend.macd_signal(df['Close'])
    
    # Bollinger Bands
    df['bollinger_hband'] = ta.volatility.bollinger_hband(df['Close'])
    df['bollinger_lband'] = ta.volatility.bollinger_lband(df['Close'])
    df['bollinger_mavg'] = ta.volatility.bollinger_mavg(df['Close'])
    
    # Volume indicators
    df['on_balance_volume'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
    
    # Volatility
    df['atr'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'])
    
    # Target variable (next day price movement)
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    return df

def train_model(features):
    """Train Random Forest model"""
    if features is None or len(features) < 100:
        return None, {}
    
    # Remove rows with NaN values
    features = features.dropna()
    
    if len(features) < 50:
        return None, {}
    
    # Select feature columns (exclude target and date columns)
    feature_cols = [col for col in features.columns if col not in ['target', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    if len(feature_cols) == 0:
        return None, {}
    
    X = features[feature_cols]
    y = features['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0)
    }
    
    return model, metrics

# Main content area
if run_analysis:
    try:
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Data Collection
        status_text.text("📊 Collecting stock data...")
        progress_bar.progress(25)
        
        if demo_mode:
            st.info("🎯 Using Demo Mode with sample data...")
            data = get_sample_data()
            if data is not None:
                st.success("✅ Using sample data for demonstration")
            else:
                st.error("Failed to generate sample data")
                st.stop()
        else:
            data = get_stock_data(symbol, period=period)
            
            if data is None:
                st.error("Could not fetch stock data. Please check the symbol and try again.")
                
                # Offer demo mode
                if st.button("🎯 Try Demo Mode (Sample Data)"):
                    st.info("📊 Running with sample data for demonstration...")
                    data = get_sample_data()
                    if data is not None:
                        st.success("✅ Using sample data for demonstration")
                    else:
                        st.error("Failed to generate sample data")
                        st.stop()
                else:
                    st.stop()
            else:
                st.success(f"✅ Collected {len(data)} records for {symbol}")
                
                # Display stock data with better styling
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### �� Stock Data Summary")
                    
                    # Create a styled summary card
                    summary_data = {
                        "Symbol": symbol,
                        "Period": selected_period,
                        "Records": len(data),
                        "Date Range": f"{data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}"
                    }
                    
                    for key, value in summary_data.items():
                        st.markdown(f"**{key}:** {value}")
                    
                    # Price statistics with better formatting
                    st.markdown("### 💰 Price Statistics")
                    price_stats = {
                        "Min Price": f"${data['Close'].min():.2f}",
                        "Max Price": f"${data['Close'].max():.2f}",
                        "Mean Price": f"${data['Close'].mean():.2f}",
                        "Current Price": f"${data['Close'].iloc[-1]:.2f}"
                    }
                    
                    for stat, value in price_stats.items():
                        st.markdown(f"**{stat}:** {value}")
                
                with col2:
                    # Enhanced price chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['Close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color='#667eea', width=3),
                        fill='tonexty'
                    ))
                    fig.update_layout(
                        title=f"📈 {symbol} Stock Price",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        height=400,
                        template="plotly_white",
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Step 2: Feature Engineering
        status_text.text("🔧 Engineering features...")
        progress_bar.progress(50)
        
        features = create_features(data)
        
        if features is not None and len(features) > 0:
            feature_cols = [col for col in features.columns if col not in ['target', 'Open', 'High', 'Low', 'Close', 'Volume']]
            st.success(f"✅ Created {len(feature_cols)} technical indicators")
            
            # Check if we have enough data for training
            if len(features) < 50:
                st.warning(f"⚠️ Only {len(features)} data points available. Need at least 50 for reliable predictions.")
                st.info("💡 **Tip**: Try a longer time period (2-5 years) for better results.")
            
            # Display feature info with better styling
            st.markdown("### 🔧 Technical Indicators")
            st.write(f"**Total Features:** {len(feature_cols)}")
            
            # Show some key indicators in a nice table
            if len(features) > 0:
                key_indicators = ['rsi_14', 'macd', 'sma_20', 'bollinger_hband', 'bollinger_lband']
                available_indicators = [ind for ind in key_indicators if ind in features.columns]
                
                if available_indicators:
                    indicator_data = features[available_indicators].tail(10)
                    st.markdown("**Recent Technical Indicators:**")
                    st.dataframe(indicator_data, use_container_width=True)
        else:
            st.error("❌ Could not create features from the data.")
            st.info("💡 **Possible causes:**")
            st.info("• Insufficient data points")
            st.info("• Missing price/volume data")
            st.info("• Data format issues")
            st.stop()
        
        # Step 3: Model Training
        status_text.text("🤖 Training machine learning model...")
        progress_bar.progress(75)
        
        model, metrics = train_model(features)
        
        if model is not None and metrics:
            st.success("✅ Model training complete!")
            progress_bar.progress(100)
            status_text.text("🎉 Analysis complete!")
            
            # Display model performance with better styling
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🤖 Model Performance")
                
                # Create metric cards
                metrics_data = {
                    "Accuracy": f"{metrics['accuracy']:.2%}",
                    "Precision": f"{metrics['precision']:.2%}",
                    "Recall": f"{metrics['recall']:.2%}",
                    "F1-Score": f"{metrics['f1_score']:.2%}"
                }
                
                for metric, value in metrics_data.items():
                    st.metric(metric, value)
            
            with col2:
                st.markdown("### 📈 Model Info")
                model_info = {
                    "Model Type": "Random Forest",
                    "Features Used": "Technical Indicators",
                    "Target": "Next Day Price Movement",
                    "Training Data": "Historical Stock Data"
                }
                
                for key, value in model_info.items():
                    st.markdown(f"**{key}:** {value}")
            
            # Feature importance with better visualization
            if hasattr(model, 'feature_importances_'):
                feature_cols = [col for col in features.columns if col not in ['target', 'Open', 'High', 'Low', 'Close', 'Volume']]
                importance_df = pd.DataFrame({
                    'feature': feature_cols,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                if not importance_df.empty:
                    st.markdown("### 🎯 Top 5 Important Features")
                    top_features = importance_df.head(5)
                    
                    # Create a nice bar chart
                    fig = px.bar(top_features, x='importance', y='feature', 
                               orientation='h', color='importance',
                               color_continuous_scale='Viridis')
                    fig.update_layout(title="Feature Importance", height=300)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Make predictions with beautiful styling
            st.markdown("### 🔮 Price Predictions")
            
            if model is not None and features is not None:
                # Get the most recent data for prediction
                latest_features = features[feature_cols].iloc[-1:].dropna()
                
                if not latest_features.empty:
                    # Make prediction for next day
                    next_day_prediction = model.predict(latest_features)[0]
                    prediction_prob = model.predict_proba(latest_features)[0]
                    
                    current_price = data['Close'].iloc[-1]
                    
                    # Calculate predicted price change (simple approach)
                    avg_daily_return = features['returns'].mean()
                    predicted_return = avg_daily_return if next_day_prediction == 1 else -avg_daily_return
                    predicted_price = current_price * (1 + predicted_return)
                    
                    # Create prediction cards
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("""
                        <div class="prediction-card">
                            <h3>📊 Next Day Prediction</h3>
                            <h2>""" + ("📈 UP" if next_day_prediction == 1 else "�� DOWN") + """</h2>
                            <p>""" + f"{prediction_prob.max():.1%} confidence" + """</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>💰 Current Price</h3>
                            <h2>${current_price:.2f}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>�� Predicted Price</h3>
                            <h2>${predicted_price:.2f}</h2>
                            <p>{predicted_return:+.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Trading recommendation with beautiful styling
                    st.markdown("### 💡 Trading Recommendation")
                    
                    if next_day_prediction == 1 and prediction_prob.max() > 0.6:
                        st.markdown(f"""
                        <div class="recommendation-buy">
                            <h2>🟢 BUY</h2>
                            <h3>Strong upward momentum predicted</h3>
                            <p>• Confidence: {prediction_prob.max():.1%}</p>
                            <p>• Expected gain: {predicted_return:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif next_day_prediction == 0 and prediction_prob.max() > 0.6:
                        st.markdown(f"""
                        <div class="recommendation-sell">
                            <h2>🔴 SELL</h2>
                            <h3>Downward movement predicted</h3>
                            <p>• Confidence: {prediction_prob.max():.1%}</p>
                            <p>• Expected decline: {predicted_return:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="recommendation-hold">
                            <h2>🟡 HOLD</h2>
                            <h3>Uncertain market direction</h3>
                            <p>• Confidence: {prediction_prob.max():.1%}</p>
                            <p>• Consider waiting for clearer signals</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Risk warning with styling
                    st.markdown("""
                    <div class="warning-box">
                        ⚠️ <strong>Disclaimer</strong>: This is for educational purposes only. 
                        Past performance does not guarantee future results. 
                        Always do your own research and consider consulting a financial advisor.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show prediction confidence breakdown
                    st.markdown("### 📊 Prediction Confidence")
                    confidence_data = pd.DataFrame({
                        'Outcome': ['Price Up', 'Price Down'],
                        'Probability': [prediction_prob[1], prediction_prob[0]]
                    })
                    
                    fig = px.bar(confidence_data, x='Outcome', y='Probability', 
                               color='Probability', color_continuous_scale='RdYlGn')
                    fig.update_layout(title="Model Confidence in Prediction", height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.warning("⚠️ Insufficient recent data for prediction")
            else:
                st.error("Could not make predictions. Model training failed.")
        else:
            st.error("❌ Could not train model. Insufficient data or features.")
            st.info("�� **Common causes:**")
            st.info("• Less than 50 data points available")
            st.info("• Too many missing values in features")
            st.info("• All features are constant (no variation)")
            st.info("• Target variable has no variation")
            st.info("")
            st.info("**Try:**")
            st.info("• Use a longer time period (2-5 years)")
            st.info("• Try a different stock symbol")
            st.info("• Check if the stock has enough trading history")
        
        # Summary with celebration
        st.success("🎉 Analysis Complete!")
        st.balloons()
        
    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        st.exception(e)

# Information section with better styling
else:
    st.markdown("""
    ## 🚀 How to Use This App
    
    1. **📊 Enter a stock symbol** (e.g., AAPL, MSFT, GOOGL)
    2. **⏰ Select a time period** for analysis
    3. **🤖 Choose a model type** (Random Forest recommended)
    4. **🚀 Click "Run Analysis"** to start the prediction
    
    ## ✨ What This App Does
    
    - 📊 **Data Collection**: Downloads historical stock data from Yahoo Finance
    - 🔧 **Feature Engineering**: Creates technical indicators (RSI, MACD, etc.)
    - �� **Machine Learning**: Trains a Random Forest model to predict price movements
    - 🔮 **Price Predictions**: Forecasts next-day stock prices and price direction
    - 💡 **Trading Recommendations**: Provides BUY/SELL/HOLD advice based on predictions
    - 📈 **Performance Analysis**: Shows model accuracy and feature importance
    
    ## 📊 Technical Indicators Used
    
    - **�� Moving Averages**: SMA (5, 20, 50), EMA (12, 26)
    - **⚡ Momentum**: RSI (14), MACD
    - **�� Volatility**: Bollinger Bands, ATR
    - **📈 Volume**: On-Balance Volume
    - **💰 Price Action**: Returns, Log Returns
    
    ## 🎯 Understanding Predictions
    
    - **📈 UP**: Model predicts the stock price will increase tomorrow
    - **📉 DOWN**: Model predicts the stock price will decrease tomorrow
    - **🎯 Confidence**: How certain the model is about its prediction (0-100%)
    - **💡 Trading Recommendation**: BUY/SELL/HOLD advice based on prediction confidence
    - **⚠️ Risk Warning**: Always do your own research before making investment decisions
    
    ## ⚠️ Disclaimer
    
    This tool is for educational purposes only. Past performance does not guarantee future results. 
    Always do your own research before making investment decisions.
    
    ## 🔄 Rate Limiting
    
    If you encounter "Too Many Requests" errors:
    - **🎯 Use Demo Mode** (checkbox in sidebar) for immediate testing
    - ⏰ Wait 15-30 minutes before trying real data again
    - 🔄 Try a different stock symbol
    - 📅 Use shorter time periods (6 months instead of 5 years)
    - 🔄 The app will automatically retry up to 2 times
    
    **Note**: When Yahoo Finance is globally rate limiting, all users are affected.
    """)

# Footer with better styling
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">
    <h3>Built with ❤️ using Streamlit, Python, and Machine Learning</h3>
    <p>Advanced AI-powered stock prediction for smarter trading decisions</p>
</div>
""", unsafe_allow_html=True)
      
  
