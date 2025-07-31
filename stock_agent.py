import yfinance as yf
import pandas as pd
import sys
import matplotlib.pyplot as plt
from datetime import datetime
import os

def fetch_stock_data(ticker, period='6mo', interval='1d'):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    return df

def analyze_stock(df):
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    latest = df.iloc[-1]
    if latest['MA20'] > latest['MA50']:
        return 'BUY'
    elif latest['MA20'] < latest['MA50']:
        return 'SELL'
    else:
        return 'HOLD'

def plot_stock(df, ticker):
    plt.figure(figsize=(12,6))
    plt.plot(df.index, df['Close'], label='Close Price')
    plt.plot(df.index, df['MA20'], label='MA20')
    plt.plot(df.index, df['MA50'], label='MA50')
    plt.title(f'{ticker} Price and Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def log_recommendation(ticker, recommendation):
    log_file = 'recommendation_log.csv'
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    entry = f'{ticker},{now},{recommendation}\n'
    header = 'Ticker,Date,Recommendation\n'
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write(header)
    with open(log_file, 'a') as f:
        f.write(entry)

def main():
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
    else:
        ticker = input('Enter stock ticker (e.g., AAPL): ').upper()
    df = fetch_stock_data(ticker)
    if df.empty:
        print('No data found for ticker:', ticker)
        return
    recommendation = analyze_stock(df)
    print(f'Recommendation for {ticker}: {recommendation}')
    log_recommendation(ticker, recommendation)
    plot_stock(df, ticker)

if __name__ == '__main__':
    main() 