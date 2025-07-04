import streamlit as st
import pandas as pd
import requests
import yfinance as yf
import plotly.graph_objs as go
import plotly.express as px
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime, timedelta

# --- Config ---
st.set_page_config(page_title="Canadian Market & Forex Dashboard", layout="wide")
st.title("📈 Canadian Market & Forex Dashboard")

# --- API Keys ---
ALPHA_API_KEY = st.secrets["alpha_vantage"]["api_key"]
ts = TimeSeries(key=ALPHA_API_KEY, output_format='pandas')

# --- Stock Data ---
@st.cache_data(ttl=3600)
def get_stock_data_alpha(ticker):
    try:
        df, _ = ts.get_daily(symbol=ticker, outputsize='compact')
        df = df.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        })
        df = df.sort_index()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA50'] = df['Close'].rolling(50).mean()
        df['UpperBand'] = df['MA20'] + 2 * df['Close'].rolling(20).std()
        df['LowerBand'] = df['MA20'] - 2 * df['Close'].rolling(20).std()
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        rs = gain.rolling(14).mean() / loss.rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + rs))
        df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
        df['Signal'] = df['MACD'].ewm(span=9).mean()
        df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
        low14 = df['Low'].rolling(14).min()
        high14 = df['High'].rolling(14).max()
        df['%K'] = 100 * ((df['Close'] - low14) / (high14 - low14))
        df['%D'] = df['%K'].rolling(3).mean()
        return df.dropna()
    except:
        return None

# --- Forex Historical (via yfinance) ---
@st.cache_data(ttl=3600)
def get_forex_yf(from_cur, to_cur):
    symbol = f"{from_cur}{to_cur}=X"
    df = yf.download(symbol, period="3mo")
    if df.empty: return None
    return df[['Close']].rename(columns={'Close': f"{from_cur}/{to_cur}"})

# --- Interest Rate Differential ---
@st.cache_data(ttl=3600)
def get_bond_yield(currency):
    tickers = {"USD": "^TNX", "CAD": "^CCB10Y", "EUR": "^DE10Y", "GBP": "^GUKG10"}
    if currency not in tickers: return None
    hist = yf.Ticker(tickers[currency]).history(period="5d")
    return hist['Close'].dropna().iloc[-1] / (10 if currency == "USD" else 1)

def calc_ird(from_c, to_c):
    y1 = get_bond_yield(from_c)
    y2 = get_bond_yield(to_c)
    return round(y1 - y2, 2) if y1 and y2 else None

# --- Sidebar ---
st.sidebar.header("⚙️ Settings")
chart_type = st.sidebar.radio("Chart Type", ["Candlestick", "Line"], index=1)
with st.sidebar.expander("📊 Stock Metrics"):
    show_ma = st.checkbox("Show 20/50-Day MA", True)
    show_bollinger = st.checkbox("Show Bollinger Bands", True)
    show_volume = st.checkbox("Show Volume", False)

# --- TSX ETF Chart ---
st.header("TSX Composite Proxy ETF (XIC.TO)")
tsx_data = get_stock_data_alpha("XIC.TO")
if tsx_data is not None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tsx_data.index, y=tsx_data['Close'], name="Close"))
    if show_ma:
        fig.add_trace(go.Scatter(x=tsx_data.index, y=tsx_data['MA20'], name="MA20"))
        fig.add_trace(go.Scatter(x=tsx_data.index, y=tsx_data['MA50'], name="MA50"))
    if show_bollinger:
        fig.add_trace(go.Scatter(x=tsx_data.index, y=tsx_data['UpperBand'], name="Upper Band"))
        fig.add_trace(go.Scatter(x=tsx_data.index, y=tsx_data['LowerBand'], name="Lower Band"))
    st.plotly_chart(fig, use_container_width=True)
# --- Forex Section ---
st.header("💱 Forex Tracker")
col1, col2 = st.columns(2)
with col1:
    fx_from = st.text_input("From Currency", value="USD").strip().upper()
with col2:
    fx_to = st.text_input("To Currency", value="CAD").strip().upper()

if fx_from and fx_to and len(fx_from) == 3 and len(fx_to) == 3:
    fx_data = get_forex_yf(fx_from, fx_to)
    if fx_data is not None:
        st.metric(f"{fx_from}/{fx_to} Live Rate", value=f"{fx_data.iloc[-1,0]:.4f}")
        ird = calc_ird(fx_from, fx_to)
        if ird is not None:
            st.metric("Interest Rate Differential", f"{ird:.2f}%")
        with st.expander("📉 Forex History Chart"):
            fx_fig = go.Figure()
            fx_fig.add_trace(go.Scatter(x=fx_data.index, y=fx_data.iloc[:,0], name=f"{fx_from}/{fx_to}"))
            fx_fig.update_layout(title="90-Day Forex Rate", xaxis_title="Date", yaxis_title="Rate")
            st.plotly_chart(fx_fig, use_container_width=True)
    else:
        st.warning("No data for selected forex pair.")

# --- Portfolio Section ---
st.header("💼 Portfolio Analyzer")
tick_input = st.text_input("Enter tickers with weights (e.g., AAPL,10 GOOGL,5)").strip()
portfolio = {}
if tick_input:
    for entry in tick_input.split():
        try:
            ticker, wt = entry.split(',')
            portfolio[ticker.strip().upper()] = float(wt)
        except:
            st.warning("Invalid format. Use: TICKER,WEIGHT")

if portfolio:
    metrics = st.multiselect("Select Indicators", ["RSI", "MACD", "ATR", "%K/%D"], default=["RSI", "MACD"])
    for metric in metrics:
        fig = go.Figure()
        for i, (ticker, weight) in enumerate(portfolio.items()):
            df = get_stock_data_alpha(ticker)
            if df is None: continue
            color = px.colors.qualitative.Set1[i % 10]
            if metric == "RSI":
                fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name=f"{ticker} RSI", line=dict(color=color)))
            elif metric == "MACD":
                fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name=f"{ticker} MACD", line=dict(color=color)))
                fig.add_trace(go.Scatter(x=df.index, y=df["Signal"], name=f"{ticker} Signal", line=dict(color='gray', dash='dot')))
            elif metric == "ATR":
                fig.add_trace(go.Scatter(x=df.index, y=df["ATR"], name=f"{ticker} ATR", line=dict(color=color)))
            elif metric == "%K/%D":
                fig.add_trace(go.Scatter(x=df.index, y=df["%K"], name=f"{ticker} %K", line=dict(color=color)))
                fig.add_trace(go.Scatter(x=df.index, y=df["%D"], name=f"{ticker} %D", line=dict(color='black', dash='dot')))
        fig.update_layout(title=f"{metric} Chart", xaxis_title="Date", yaxis_title=metric)
        st.plotly_chart(fig, use_container_width=True)
