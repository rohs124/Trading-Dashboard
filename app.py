
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import plotly.graph_objs as go
from textblob import TextBlob
from datetime import datetime, timedelta

st.set_page_config(page_title="Trading Dashboard", layout="wide")
st.title("📈 Trading Dashboard with Sentiment, Portfolio Analytics & Forex")

# --- Sidebar Settings ---
st.sidebar.header("📊 Chart Settings")
show_ma = st.sidebar.checkbox("Show Moving Averages (20/50)", True)
show_rsi = st.sidebar.checkbox("Show RSI", True)
show_volume = st.sidebar.checkbox("Show Volume", True)

# --- Forex Section ---
st.header("💱 Forex Tracker")
fx_from = st.text_input("From Currency", "USD")
fx_to = st.text_input("To Currency", "CAD")
fx_symbol = f"{fx_from}{fx_to}=X"
forex_data = yf.download(fx_symbol, period="3mo", interval="1d")

if not forex_data.empty:
    st.subheader(f"Exchange Rate: {fx_from}/{fx_to}")
    st.line_chart(forex_data["Close"])
else:
    st.warning(f"Could not retrieve forex data for {fx_from}/{fx_to}")

# --- Ticker Sentiment Analysis ---
st.header("🧠 Ticker News Sentiment")
ticker = st.text_input("Enter Ticker Symbol for Sentiment (e.g., AAPL)", "AAPL")
if ticker:
    try:
        news = yf.Ticker(ticker).news[:5]
        for item in news:
            headline = item.get("title", "")
            sentiment = TextBlob(headline).sentiment.polarity
            tag = "Positive" if sentiment > 0.1 else "Negative" if sentiment < -0.1 else "Neutral"
            st.markdown(f"**{tag}** → {headline}")
    except Exception as e:
        st.warning(f"No news or failed sentiment check: {e}")

# --- Portfolio Section ---
st.header("💼 Portfolio Metrics")
portfolio_input = st.text_input("Enter tickers with weights (e.g., AAPL,50 MSFT,30)", "AAPL,50 MSFT,30")
portfolio = {}
try:
    for entry in portfolio_input.split():
        tkr, wt = entry.split(',')
        portfolio[tkr.strip().upper()] = float(wt)
except:
    st.error("Format error. Use: TICKER,WEIGHT")

metrics = st.multiselect("Select Portfolio Metrics", ["Close Price", "RSI", "Volume"], default=["Close Price", "RSI"])

if portfolio:
    price_fig = go.Figure()
    rsi_fig = go.Figure()
    cum_returns = pd.DataFrame()
    base = None

    for idx, (tkr, weight) in enumerate(portfolio.items()):
        data = yf.download(tkr, period="6mo", interval="1d")
        if data.empty:
            continue
        color = f"rgba({idx*40%255}, {idx*80%255}, {idx*120%255}, 0.9)"

        data["RSI"] = 100 - 100 / (1 + data["Close"].pct_change().rolling(14).mean() / data["Close"].pct_change().rolling(14).std())

        if "Close Price" in metrics:
            price_fig.add_trace(go.Scatter(x=data.index, y=data["Close"] * weight, name=f"{tkr} Close", line=dict(color=color)))
        if "RSI" in metrics:
            rsi_fig.add_trace(go.Scatter(x=data.index, y=data["RSI"], name=f"{tkr} RSI", line=dict(dash="dot", color=color)))
        if base is None:
            base = data["Close"].iloc[0]
        cum_returns[tkr] = data["Close"] / data["Close"].iloc[0] * weight

    if not cum_returns.empty:
        st.subheader("📈 Cumulative Returns")
        st.line_chart(cum_returns.sum(axis=1))

    if price_fig.data:
        st.subheader("📉 Price Metrics")
        st.plotly_chart(price_fig, use_container_width=True)
    if rsi_fig.data:
        st.subheader("📊 RSI Analysis")
        st.plotly_chart(rsi_fig, use_container_width=True)

# --- Placeholder for Intraday / Streaming Chart Integration ---
st.info("📡 Intraday streaming (e.g., Alpaca/Polygon.io) will be added in future enhancement.")
