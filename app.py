import streamlit as st
import pandas as pd
import requests
import plotly.graph_objs as go
import plotly.express as px
import yfinance as yf
from textblob import TextBlob
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime, timedelta

# --- Page Config ---
st.set_page_config(page_title="Canadian Market & Forex Dashboard", layout="wide")
st.title("📈 Canadian Market & Forex Dashboard")

# --- API Keys ---
ALPHA_API_KEY = st.secrets["alpha_vantage"]["api_key"]
EXCHANGE_API_KEY = st.secrets["exchange_rate_api"]["api_key"]
ts = TimeSeries(key=ALPHA_API_KEY, output_format='pandas')

# --- Caching & Data Functions ---
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
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['UpperBand'] = df['MA20'] + 2 * df['Close'].rolling(window=20).std()
        df['LowerBand'] = df['MA20'] - 2 * df['Close'].rolling(window=20).std()
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        return df.dropna()
    except Exception as e:
        st.error(f"Error loading {ticker}: {e}")
        return None

@st.cache_data(ttl=300)
def get_exchange_rate(from_currency, to_currency):
    url = f"https://v6.exchangerate-api.com/v6/{EXCHANGE_API_KEY}/pair/{from_currency}/{to_currency}"
    try:
        response = requests.get(url)
        data = response.json()
        if data["result"] == "success":
            return data["conversion_rate"]
        else:
            st.warning(f"Failed to get live rate for {from_currency}/{to_currency}: {data.get('error-type', 'Unknown error')}")
            return None
    except Exception as e:
        st.warning(f"Error fetching live exchange rate: {e}")
        return None

@st.cache_data(ttl=3600)
def get_forex_history(from_currency, to_currency):
    try:
        end = datetime.utcnow()
        start = end - timedelta(days=90)
        url = f"https://api.exchangerate.host/timeseries?start_date={start.date()}&end_date={end.date()}&base={from_currency}&symbols={to_currency}"
        response = requests.get(url)
        data = response.json()
        if data.get("rates"):
            df = pd.DataFrame(data["rates"]).T
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df.columns = [f"{from_currency}/{to_currency}"]
            return df
        else:
            st.warning(f"No historical data found for {from_currency}/{to_currency}")
            return None
    except Exception as e:
        st.warning(f"Error fetching forex history: {e}")
        return None

# --- Chart Renderer ---
def plot_chart(df, title, chart_type, show_ma, show_bollinger, show_volume, key_prefix):
    fig = go.Figure()

    if chart_type == 'Candlestick':
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'], name='Candlestick'))
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))

    if show_ma:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA20', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='MA50', line=dict(color='green')))

    if show_bollinger:
        fig.add_trace(go.Scatter(x=df.index, y=df['UpperBand'], name='Upper Band', line=dict(color='gray', dash='dot')))
        fig.add_trace(go.Scatter(x=df.index, y=df['LowerBand'], name='Lower Band', line=dict(color='gray', dash='dot')))

    if show_volume:
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='lightgray', yaxis='y2'))
        fig.update_layout(yaxis2=dict(overlaying='y', side='right', title='Volume'))

    fig.update_layout(title=title, height=400, xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_chart")

# --- Sidebar ---
st.sidebar.header("⚙️ Settings")
chart_type = st.sidebar.radio("Chart Type", ["Candlestick", "Line"], index=1)

# Separate metrics for stock charts
st.sidebar.markdown("📊 **Stock Chart Metrics**")
show_ma = st.sidebar.checkbox("Show 20/50-Day MA", True)
show_bollinger = st.sidebar.checkbox("Show Bollinger Bands", True)
show_volume = st.sidebar.checkbox("Show Volume", False)

# Separate metrics for forex chart
st.sidebar.markdown("📈 **Forex Chart Metrics**")
show_forex_ma = st.sidebar.checkbox("Show Forex 7/21-Day MA", True)

# --- TSX ETF Chart ---
st.header("TSX Composite Proxy ETF (XIC.TO)")
tsx_data = get_stock_data_alpha("XIC.TO")
if tsx_data is not None:
    plot_chart(tsx_data, "TSX Composite ETF (XIC.TO)", chart_type, show_ma, show_bollinger, show_volume, "tsx")

# --- Forex Section ---
st.header("💱 Forex Tracker")
st.subheader("USD/CAD Live Rate")
usd_cad = get_exchange_rate("USD", "CAD")
if usd_cad:
    st.metric(label="USD to CAD", value=f"{usd_cad:.4f}")
else:
    st.warning("Unable to fetch USD to CAD live rate.")

col1, col2 = st.columns(2)
with col1:
    fx_from = st.text_input("From Currency (e.g., EUR)", value="EUR").strip().upper()
with col2:
    fx_to = st.text_input("To Currency (e.g., USD)", value="USD").strip().upper()

if len(fx_from) != 3 or len(fx_to) != 3:
    st.error("Please enter valid 3-letter currency codes for 'From' and 'To' fields.")
else:
    rate = get_exchange_rate(fx_from, fx_to)
    if rate:
        st.metric(label=f"{fx_from} to {fx_to}", value=f"{rate:.4f}")
    else:
        st.warning(f"Unable to fetch live rate for {fx_from} to {fx_to}.")

    st.subheader("📉 Forex History (Last 90 Days)")
    forex_df = get_forex_history(fx_from, fx_to)
    if forex_df is not None and not forex_df.empty:
        fig_fx = go.Figure()
        fig_fx.add_trace(go.Scatter(
            x=forex_df.index,
            y=forex_df.iloc[:, 0],
            mode='lines',
            name=f"{fx_from}/{fx_to}"
        ))
        if show_forex_ma:
            forex_df['MA7'] = forex_df.iloc[:, 0].rolling(window=7).mean()
            forex_df['MA21'] = forex_df.iloc[:, 0].rolling(window=21).mean()
            fig_fx.add_trace(go.Scatter(x=forex_df.index, y=forex_df['MA7'], name='MA7', line=dict(color='blue', dash='dot')))
            fig_fx.add_trace(go.Scatter(x=forex_df.index, y=forex_df['MA21'], name='MA21', line=dict(color='green', dash='dot')))

        fig_fx.update_layout(
            title=f"{fx_from} to {fx_to} Exchange Rate (Last 90 Days)",
            height=400,
            xaxis_title="Date",
            yaxis_title="Exchange Rate"
        )
        st.plotly_chart(fig_fx, use_container_width=True)
    else:
        st.warning(f"No historical forex data available for {fx_from}/{fx_to}.")

# --- 🧠 Ticker News Sentiment Section ---
st.header("🧠 Ticker News Sentiment")
ticker_news_input = st.text_input("Enter ticker for news sentiment (e.g., AAPL)", value="AAPL").strip().upper()

def fetch_yahoo_news(ticker):
    try:
        yf_ticker = yf.Ticker(ticker)
        news_items = yf_ticker.news
        if not news_items:
            return []
        news_list = []
        for item in news_items[:5]:  # Top 5 news
            headline = item.get('title', '')
            if headline:
                polarity = TextBlob(headline).sentiment.polarity
                if polarity > 0.1:
                    sentiment = "Positive"
                elif polarity < -0.1:
                    sentiment = "Negative"
                else:
                    sentiment = "Neutral"
                news_list.append((headline, sentiment))
        return news_list
    except Exception as e:
        st.warning(f"Failed to fetch news for {ticker}: {e}")
        return []

news_data = fetch_yahoo_news(ticker_news_input)
if news_data:
    for headline, sentiment in news_data:
        if sentiment == "Positive":
            st.success(f"✅ {headline}  — *{sentiment}*")
        elif sentiment == "Negative":
            st.error(f"❌ {headline}  — *{sentiment}*")
        else:
            st.info(f"ℹ️ {headline}  — *{sentiment}*")
else:
    st.info("No news data available.")

# --- Portfolio Section ---
st.header("💼 Portfolio Metrics")
portfolio_input = st.text_input("Enter tickers with weights (e.g., SHOP.TO,10 ENB.TO,5 BNS.TO,20)")
portfolio = {}
if portfolio_input:
    try:
        for item in portfolio_input.split():
            ticker, weight = item.split(',')
            portfolio[ticker.strip().upper()] = float(weight)
    except:
        st.error("Invalid format. Use: TICKER,WEIGHT")

selected_metrics = st.multiselect(
    "Select portfolio metrics to display:",
    ["Close Price", "MA50", "UpperBand", "RSI", "Volume"],
    default=["Close Price", "RSI", "Volume"]
)

if portfolio:
    price_fig = go.Figure()
    rsi_fig = go.Figure()
    volume_fig = go.Figure()

    # For cumulative return calculation
    cum_returns = pd.DataFrame()

    for idx, (ticker, weight) in enumerate(portfolio.items()):
        data = get_stock_data_alpha(ticker)
        if data is None:
            continue
        color = px.colors.qualitative.Plotly[idx % 10]

        if "Close Price" in selected_metrics:
            price_fig.add_trace(go.Scatter(x=data.index, y=data["Close"] * weight, name=f"{ticker} Close", line=dict(color=color)))

        if "MA50" in selected_metrics:
            price_fig.add_trace(go.Scatter(x=data.index, y=data["MA50"] * weight, name=f"{ticker} MA50", line=dict(color=color, dash='dot')))

        if "UpperBand" in selected_metrics:
            price_fig.add_trace(go.Scatter(x=data.index, y=data["UpperBand"] * weight, name=f"{ticker} UpperBand", line=dict(color=color, dash='dash')))

        if "RSI" in selected_metrics:
            rsi_fig.add_trace(go.Scatter(x=data.index, y=data["RSI"], name=f"{ticker} RSI", line=dict(color=color, dash='dot')))

        if "Volume" in selected_metrics:
            volume_fig.add_trace(go.Scatter(x=data.index, y=data["Volume"], name=f"{ticker} Volume", line=dict(color=color)))

        # Prepare cumulative returns: normalize Close prices and weight them
        norm_close = data["Close"] / data["Close"].iloc[0]
        cum_returns[ticker] = norm_close * weight

    if any(metric in selected_metrics for metric in ["Close Price", "MA50", "UpperBand"]):
        st.subheader("📊 Portfolio Price Metrics")
        price_fig.update_layout(title="Portfolio Price", height=400, xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(price_fig, use_container_width=True)

    if "RSI" in selected_metrics:
        st.subheader("📉 RSI")
        rsi_fig.update_layout(title="Portfolio RSI", height=300, xaxis_title="Date", yaxis_title="RSI")
        st.plotly_chart(rsi_fig, use_container_width=True)

    if "Volume" in selected_metrics:
        st.subheader("📈 Volume")
        volume_fig.update_layout(title="Portfolio Volume", height=300, xaxis_title="Date", yaxis_title="Volume")
        st.plotly_chart(volume_fig, use_container_width=True)

    # --- Portfolio Cumulative Return Chart ---
    if not cum_returns.empty:
        cum_returns['Total'] = cum_returns.sum(axis=1)
        st.subheader("📈 Portfolio Cumulative Return")
        fig_cumret = go.Figure()
        fig_cumret.add_trace(go.Scatter(x=cum_returns.index, y=cum_returns['Total'], mode='lines', name='Cumulative Return'))
        fig_cumret.update_layout(title="Portfolio Cumulative Return (Weighted)", height=400, xaxis_title="Date", yaxis_title="Cumulative Return")
        st.plotly_chart(fig_cumret, use_container_width=True)

    if "RSI" in selected_metrics:
        st.subheader("🧠 AI RSI-Based Suggestions")
        for ticker in portfolio:
            data = get_stock_data_alpha(ticker)
            if data is not None and "RSI" in data.columns:
                latest_rsi = data["RSI"].iloc[-1]
                if latest_rsi < 30:
                    st.info(f"{ticker} is **oversold** (RSI={latest_rsi:.2f}) → Potential Buy")
                elif latest_rsi > 70:
                    st.warning(f"{ticker} is **overbought** (RSI={latest_rsi:.2f}) → Consider Reducing")
                else:
                    st.success(f"{ticker} RSI is neutral ({latest_rsi:.2f})")
