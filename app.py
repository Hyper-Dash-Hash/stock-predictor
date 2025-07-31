import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_collector import DataCollector
from src.feature_engineering import FeatureEngineer
from src.model_trainer_simple import SimpleModelTrainer
from src.backtester import Backtester
from src.utils import print_summary_statistics

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

# Main content area
if run_analysis:
    try:
        with st.spinner("Collecting stock data..."):
            # Step 1: Data Collection
            collector = DataCollector()
            data = collector.get_stock_data(symbol, period=period)
            
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
                    "Min Price": f"${data['close'].min():.2f}",
                    "Max Price": f"${data['close'].max():.2f}",
                    "Mean Price": f"${data['close'].mean():.2f}",
                    "Current Price": f"${data['close'].iloc[-1]:.2f}"
                }
                st.write("**Price Statistics:**")
                for stat, value in price_stats.items():
                    st.write(f"• {stat}: {value}")
            
            with col2:
                # Price chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['close'],
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
            engineer = FeatureEngineer()
            features = engineer.create_features(data, symbol)
            
            st.success(f"✅ Created {len(engineer.get_feature_columns(features))} technical indicators")
            
            # Display feature info
            st.subheader("🔧 Technical Indicators")
            feature_cols = engineer.get_feature_columns(features)
            st.write(f"**Total Features:** {len(feature_cols)}")
            
            # Show some key indicators
            if len(features) > 0:
                key_indicators = ['rsi_14', 'macd', 'sma_20', 'bollinger_hband', 'bollinger_lband']
                available_indicators = [ind for ind in key_indicators if ind in features.columns]
                
                if available_indicators:
                    indicator_data = features[available_indicators].tail(10)
                    st.write("**Recent Technical Indicators:**")
                    st.dataframe(indicator_data, use_container_width=True)
        
        with st.spinner("Training machine learning model..."):
            # Step 3: Model Training
            trainer = SimpleModelTrainer()
            model, metrics = trainer.train_random_forest(features, symbol)
            
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
                st.subheader("📈 Trading Metrics")
                st.metric("Win Rate", f"{metrics['win_rate']:.2%}")
                st.metric("Total Trades", metrics.get('total_trades', 'N/A'))
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    feature_cols = engineer.get_feature_columns(features)
                    importance_df = trainer.get_feature_importance(model, feature_cols)
                    
                    if not importance_df.empty:
                        st.write("**Top 5 Important Features:**")
                        top_features = importance_df.head(5)
                        for _, row in top_features.iterrows():
                            st.write(f"• {row['feature']}: {row['importance']:.3f}")
        
        with st.spinner("Running backtest simulation..."):
            # Step 4: Backtesting
            backtester = Backtester()
            results = backtester.run_backtest(model, features, symbol)
            
            st.success("✅ Backtest simulation complete!")
            
            # Display backtest results
            st.subheader("💰 Trading Simulation Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Final Portfolio Value", 
                    f"${results['metrics']['final_portfolio_value']:,.2f}",
                    f"{results['metrics']['total_return']:.2%}"
                )
            
            with col2:
                st.metric(
                    "Sharpe Ratio", 
                    f"{results['metrics']['sharpe_ratio']:.3f}"
                )
            
            with col3:
                st.metric(
                    "Max Drawdown", 
                    f"{results['metrics']['max_drawdown']:.2%}"
                )
            
            # Additional metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Trading Statistics:**")
                st.write(f"• Total Trades: {results['metrics']['total_trades']}")
                st.write(f"• Win Rate: {results['metrics']['win_rate']:.2%}")
                st.write(f"• Transaction Costs: ${results['metrics']['transaction_costs']:.2f}")
            
            with col2:
                st.write("**Risk Metrics:**")
                st.write(f"• Volatility: {results['metrics']['volatility']:.2%}")
                st.write(f"• Annual Return: {results['metrics']['annual_return']:.2%}")
                st.write(f"• Risk-Free Rate: {results['metrics']['risk_free_rate']:.2%}")
            
            # Portfolio performance chart
            if 'portfolio_data' in results:
                portfolio_df = results['portfolio_data']
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=portfolio_df.index,
                    y=portfolio_df['portfolio_value'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='green')
                ))
                fig.update_layout(
                    title="Portfolio Performance Over Time",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value ($)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
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
    - 🔧 **Feature Engineering**: Creates 80+ technical indicators (RSI, MACD, etc.)
    - 🤖 **Machine Learning**: Trains a Random Forest model to predict price movements
    - 💰 **Backtesting**: Simulates trading with realistic transaction costs
    - 📈 **Performance Analysis**: Provides comprehensive trading metrics
    
    ## Technical Indicators Used
    
    - **Moving Averages**: SMA, EMA (5, 10, 20, 50, 200 periods)
    - **Momentum**: RSI, Stochastic, Williams %R
    - **Trend**: MACD, ADX, CCI
    - **Volatility**: Bollinger Bands, ATR
    - **Volume**: OBV, Volume ROC
    
    ## Disclaimer
    
    This tool is for educational purposes only. Past performance does not guarantee future results. 
    Always do your own research before making investment decisions.
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, Python, and Machine Learning") 