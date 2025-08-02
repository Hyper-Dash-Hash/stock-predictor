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
    page_title="Stock Price Predictor",
    page_icon="��",
    layout="wide"
)

# Title and description
st.title("📈 Stock Price Predictor")
st.markdown("**Predict future stock prices and get trading recommendations using machine learning and technical analysis**")

# Sidebar for inputs
st.sidebar.header("Settings")

# Stock symbol input
symbol = st.sidebar.text_input("Stock Symbol", value="AAPL").upper()

# Popular stock symbols for quick access
popular_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX", "AMD", "INTC"]
st.sidebar.markdown("**Popular Symbols:** " + ", ".join(popular_symbols))

# Time period selection
period_options = {
    "6 Months": "6mo",
    "1 Year": "1y", 
    "2 Years": "2y",
    "5 Years": "5y"
}
selected_period = st.sidebar.selectbox("Time Period", list(period_options.keys()))
period = period_options[selected_period]

# Model selection
model_type = st.sidebar.selectbox("Model Type", ["Random Forest", "LSTM (Coming Soon)"])

# Demo mode option
demo_mode = st.sidebar.checkbox("🎯 Demo Mode (Use sample data)", help="Use sample data when API is rate limited")

# API Status Warning
if not demo_mode:
    st.warning("⚠️ **API Status**: Yahoo Finance may be rate limiting requests. If you encounter errors, try Demo Mode.")

# Run analysis button
run_analysis = st.sidebar.button("🚀 Run Analysis", type="primary")

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
        with st.spinner("Collecting stock data..."):
            # Step 1: Data Collection
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
                    
                    # Display stock data
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("�� Stock Data Summary")
                        st.write(f"**Symbol:** {symbol}")
                        st.write(f"**Period:** {selected_period}")
                        st.write(f"**Records:** {len(data)}")
                        st.write(f"**Date Range:** {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
                        
                        # Price statistics
                        price_stats = {
                            "Min Price": f"${data['Close'].min():.2f}",
                            "Max Price": f"${data['Close'].max():.2f}",
                            "Mean Price": f"${data['Close'].mean():.2f}",
                            "Current Price": f"${data['Close'].iloc[-1]:.2f}"
                        }
                        st.write("**Price Statistics:**")
                        for stat, value in price_stats.items():
                            st.write(f"• {stat}: {value}")
                    
                    with col2:
                        # Price chart
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data['Close'],
                            mode='lines',
                            name='Close Price',
                            line=dict(color='blue')
                        ))
                        fig.update_layout(
                            title=f"{symbol} Stock Price",
                            xaxis_title="Date",
                            yaxis_title="Price ($)",
                            height=300
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        with st.spinner("Engineering features..."):
            # Step 2: Feature Engineering
            features = create_features(data)
            
            if features is not None and len(features) > 0:
                feature_cols = [col for col in features.columns if col not in ['target', 'Open', 'High', 'Low', 'Close', 'Volume']]
                st.success(f"✅ Created {len(feature_cols)} technical indicators")
                
                # Check if we have enough data for training
                if len(features) < 50:
                    st.warning(f"⚠️ Only {len(features)} data points available. Need at least 50 for reliable predictions.")
                    st.info("💡 **Tip**: Try a longer time period (2-5 years) for better results.")
                
                # Display feature info
                st.subheader("🔧 Technical Indicators")
                st.write(f"**Total Features:** {len(feature_cols)}")
                
                # Show some key indicators
                if len(features) > 0:
                    key_indicators = ['rsi_14', 'macd', 'sma_20', 'bollinger_hband', 'bollinger_lband']
                    available_indicators = [ind for ind in key_indicators if ind in features.columns]
                    
                    if available_indicators:
                        indicator_data = features[available_indicators].tail(10)
                        st.write("**Recent Technical Indicators:**")
                        st.dataframe(indicator_data, use_container_width=True)
            else:
                st.error("❌ Could not create features from the data.")
                st.info("💡 **Possible causes:**")
                st.info("• Insufficient data points")
                st.info("• Missing price/volume data")
                st.info("• Data format issues")
                st.stop()
        
        with st.spinner("Training machine learning model..."):
            # Step 3: Model Training
            model, metrics = train_model(features)
            
            if model is not None and metrics:
                st.success("✅ Model training complete!")
                
                # Display model performance
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🤖 Model Performance")
                    st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
                    st.metric("Precision", f"{metrics['precision']:.2%}")
                    st.metric("Recall", f"{metrics['recall']:.2%}")
                    st.metric("F1-Score", f"{metrics['f1_score']:.2%}")
                
                with col2:
                    st.subheader("📈 Model Info")
                    st.write("**Model Type:** Random Forest")
                    st.write("**Features Used:** Technical Indicators")
                    st.write("**Target:** Next Day Price Movement")
                    st.write("**Training Data:** Historical Stock Data")
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    feature_cols = [col for col in features.columns if col not in ['target', 'Open', 'High', 'Low', 'Close', 'Volume']]
                    importance_df = pd.DataFrame({
                        'feature': feature_cols,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    if not importance_df.empty:
                        st.write("**Top 5 Important Features:**")
                        top_features = importance_df.head(5)
                        for _, row in top_features.iterrows():
                            st.write(f"• {row['feature']}: {row['importance']:.3f}")
                
                # Make predictions for the next few days
                st.subheader("🔮 Price Predictions")
                
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
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Next Day Prediction",
                                "📈 UP" if next_day_prediction == 1 else "📉 DOWN",
                                f"{prediction_prob.max():.1%} confidence"
                            )
                        
                        with col2:
                            st.metric(
                                "Current Price",
                                f"${current_price:.2f}"
                            )
                        
                        with col3:
                            st.metric(
                                "Predicted Price",
                                f"${predicted_price:.2f}",
                                f"{predicted_return:+.2%}"
                            )
                        
                        # Trading recommendation
                        st.subheader("💡 Trading Recommendation")
                        
                        if next_day_prediction == 1 and prediction_prob.max() > 0.6:
                            st.success("🟢 **BUY** - Strong upward momentum predicted")
                            st.write(f"• Confidence: {prediction_prob.max():.1%}")
                            st.write(f"• Expected gain: {predicted_return:.2%}")
                        elif next_day_prediction == 0 and prediction_prob.max() > 0.6:
                            st.error("🔴 **SELL** - Downward movement predicted")
                            st.write(f"• Confidence: {prediction_prob.max():.1%}")
                            st.write(f"• Expected decline: {predicted_return:.2%}")
                        else:
                            st.warning("🟡 **HOLD** - Uncertain market direction")
                            st.write(f"• Confidence: {prediction_prob.max():.1%}")
                            st.write("• Consider waiting for clearer signals")
                        
                        # Risk warning
                        st.info("⚠️ **Disclaimer**: This is for educational purposes only. Past performance does not guarantee future results. Always do your own research and consider consulting a financial advisor.")
                        
                        # Show prediction confidence breakdown
                        st.subheader("📊 Prediction Confidence")
                        confidence_data = pd.DataFrame({
                            'Outcome': ['Price Up', 'Price Down'],
                            'Probability': [prediction_prob[1], prediction_prob[0]]
                        })
                        
                        fig = px.bar(confidence_data, x='Outcome', y='Probability', 
                                   color='Probability', color_continuous_scale='RdYlGn')
                        fig.update_layout(title="Model Confidence in Prediction")
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
        
        # Summary
        st.success("🎉 Analysis Complete!")
        st.balloons()
        
    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        st.exception(e)

# Information section
else:
    st.markdown("""
    ## How to Use This App
    
    1. **Enter a stock symbol** (e.g., AAPL, MSFT, GOOGL)
    2. **Select a time period** for analysis
    3. **Choose a model type** (Random Forest recommended)
    4. **Click "Run Analysis"** to start the prediction
    
    ## What This App Does
    
    - 📊 **Data Collection**: Downloads historical stock data from Yahoo Finance
    - 🔧 **Feature Engineering**: Creates technical indicators (RSI, MACD, etc.)
    - �� **Machine Learning**: Trains a Random Forest model to predict price movements
    - 🔮 **Price Predictions**: Forecasts next-day stock prices and price direction
    - 💡 **Trading Recommendations**: Provides BUY/SELL/HOLD advice based on predictions
    - 📈 **Performance Analysis**: Shows model accuracy and feature importance
    
    ## Technical Indicators Used
    
    - **Moving Averages**: SMA (5, 20, 50), EMA (12, 26)
    - **Momentum**: RSI (14), MACD
    - **Volatility**: Bollinger Bands, ATR
    - **Volume**: On-Balance Volume
    - **Price Action**: Returns, Log Returns
    
    ## Understanding Predictions
    
    - **📈 UP**: Model predicts the stock price will increase tomorrow
    - **📉 DOWN**: Model predicts the stock price will decrease tomorrow
    - **Confidence**: How certain the model is about its prediction (0-100%)
    - **Trading Recommendation**: BUY/SELL/HOLD advice based on prediction confidence
    - **Risk Warning**: Always do your own research before making investment decisions
    
    ## Disclaimer
    
    This tool is for educational purposes only. Past performance does not guarantee future results. 
    Always do your own research before making investment decisions.
    
    ## Rate Limiting
    
    If you encounter "Too Many Requests" errors:
    - **Use Demo Mode** (checkbox in sidebar) for immediate testing
    - Wait 15-30 minutes before trying real data again
    - Try a different stock symbol
    - Use shorter time periods (6 months instead of 5 years)
    - The app will automatically retry up to 2 times
    
    **Note**: When Yahoo Finance is globally rate limiting, all users are affected.
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, Python, and Machine Learning")
      
  
