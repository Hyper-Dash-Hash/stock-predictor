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
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📈 Stock Price Predictor")
st.markdown("Predict stock price movements using machine learning and technical analysis")

# Sidebar for inputs
st.sidebar.header("Settings")

# Stock symbol input
symbol = st.sidebar.text_input("Stock Symbol", value="AAPL").upper()

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

# Run analysis button
run_analysis = st.sidebar.button("🚀 Run Analysis", type="primary")

def get_stock_data(symbol, period="1y"):
    """Get stock data from Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

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
    df['volume_sma'] = ta.volume.volume_sma(df['Close'], df['Volume'])
    
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
            data = get_stock_data(symbol, period=period)
            
            if data is None:
                st.error("Could not fetch stock data. Please check the symbol and try again.")
            else:
                st.success(f"✅ Collected {len(data)} records for {symbol}")
                
                # Display stock data
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Stock Data Summary")
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
            
            if features is not None:
                feature_cols = [col for col in features.columns if col not in ['target', 'Open', 'High', 'Low', 'Close', 'Volume']]
                st.success(f"✅ Created {len(feature_cols)} technical indicators")
                
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
                st.error("Could not create features from the data.")
        
        with st.spinner("Training machine learning model..."):
            # Step 3: Model Training
            model, metrics = train_model(features)
            
            if model is not None:
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
            else:
                st.error("Could not train model. Insufficient data or features.")
        
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
    - 🤖 **Machine Learning**: Trains a Random Forest model to predict price movements
    - 📈 **Performance Analysis**: Provides comprehensive trading metrics
    
    ## Technical Indicators Used
    
    - **Moving Averages**: SMA, EMA (5, 20, 50 periods)
    - **Momentum**: RSI, MACD
    - **Volatility**: Bollinger Bands, ATR
    - **Volume**: Volume SMA
    
    ## Disclaimer
    
    This tool is for educational purposes only. Past performance does not guarantee future results. 
    Always do your own research before making investment decisions.
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, Python, and Machine Learning") 