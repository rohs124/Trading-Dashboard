import streamlit as st
import pandas as pd
import requests
import plotly.graph_objs as go
import plotly.express as px
from alpha_vantage.timeseries import TimeSeries
from datetime import datetime, timedelta
import yfinance as yf
from textblob import TextBlob

# --- Page Config ---
st.set_page_config(page_title="Canadian Market & Forex Dashboard", layout="wide")
st.title("📈 Canadian Market & Forex Dashboard")

# --- API Keys ---
ALPHA_API_KEY = st.secrets["alpha_vantage"]["api_key"]
EXCHANGE_API_KEY = st.secrets["exchange_rate_api"]["api_key"]
ts = TimeSeries(key=ALPHA_API_KEY, output_format='pandas')

# --- Cached Data Functions ---
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
        df['DailyReturn'] = df['Close'].pct_change()
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
        return data.get("conversion_rate")
    except Exception as e:
        st.warning(f"Error fetching exchange rate: {e}")
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
            return None
    except Exception as e:
        return None

@st.cache_data(ttl=1800)
def get_news_sentiment_by_keyword(keyword):
    try:
        ticker_obj = yf.Ticker("AAPL")  # dummy to trigger the feed
        all_news = ticker_obj.news
        keyword_news = [n for n in all_news if keyword.lower() in n['title'].lower()]
        results = []
        for item in keyword_news[:5]:
            headline = item['title']
            polarity = TextBlob(headline).sentiment.polarity
            sentiment = "Positive" if polarity > 0.1 else "Negative" if polarity < -0.1 else "Neutral"
            results.append((headline, sentiment))
        return results
    except Exception as e:
        st.warning(f"Error fetching sentiment: {e}")
        return []

# --- Chart Function ---
def plot_chart(df, title, chart_type, show_ma, show_bollinger, show_volume, key_prefix):
    fig = go.Figure()

    if chart_type == 'Candlestick':
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'], name='Candlestick'))
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))

    if show_ma:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA20'))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='MA50'))

    if show_bollinger:
        fig.add_trace(go.Scatter(x=df.index, y=df['UpperBand'], name='UpperBand', line=dict(dash='dot')))
        fig.add_trace(go.Scatter(x=df.index, y=df['LowerBand'], name='LowerBand', line=dict(dash='dot')))

    if show_volume:
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', yaxis='y2'))
        fig.update_layout(yaxis2=dict(overlaying='y', side='right', title='Volume'))

    fig.update_layout(title=title, height=400)
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_chart")

# --- Sidebar Settings ---
st.sidebar.header("⚙️ Settings")
chart_type = st.sidebar.radio("Chart Type", ["Candlestick", "Line"], index=1)
stock_metrics = st.sidebar.multiselect("📊 Metrics (Stock Charts)", ["Show 20/50-Day MA", "Show Bollinger Bands", "Show Volume"], default=["Show 20/50-Day MA", "Show Bollinger Bands"])
portfolio_metrics = st.sidebar.multiselect("📈 Portfolio Metrics", ["Close Price", "MA50", "UpperBand", "RSI", "Volume", "Cumulative Return"], default=["Close Price", "RSI", "Cumulative Return"])

show_ma = "Show 20/50-Day MA" in stock_metrics
show_bollinger = "Show Bollinger Bands" in stock_metrics
show_volume = "Show Volume" in stock_metrics

# --- TSX ETF Chart ---
st.header("TSX Composite Proxy ETF (XIC.TO)")
tsx_data = get_stock_data_alpha("XIC.TO")
if tsx_data is not None:
    plot_chart(tsx_data, "TSX Composite ETF (XIC.TO)", chart_type, show_ma, show_bollinger, show_volume, "tsx")

# --- Forex Section ---
st.header("💱 Forex Tracker")
col1, col2 = st.columns(2)
with col1:
    fx_from = st.text_input("From Currency", value="USD").upper()
with col2:
    fx_to = st.text_input("To Currency", value="CAD").upper()

rate = get_exchange_rate(fx_from, fx_to)
if rate:
    st.metric(label=f"Exchange Rate: {fx_from}/{fx_to}", value=f"{rate:.4f}")

forex_df = get_forex_history(fx_from, fx_to)
if forex_df is not None and not forex_df.empty:
    fig_fx = go.Figure()
    fig_fx.add_trace(go.Scatter(x=forex_df.index, y=forex_df.iloc[:, 0], name=f"{fx_from}/{fx_to}"))
    fig_fx.update_layout(title="📉 Forex History (Last 90 Days)", height=400)
    st.plotly_chart(fig_fx, use_container_width=True)
else:
    st.warning(f"No historical forex data available for {fx_from}/{fx_to}.")

# --- News Sentiment Section ---
st.header("🧠 News Sentiment Analysis")
keyword = st.text_input("Enter keyword or company name for sentiment (e.g., Apple)").strip()
if keyword:
    news_items = get_news_sentiment_by_keyword(keyword)
    if news_items:
        for headline, sentiment in news_items:
            st.write(f"**{sentiment}** → {headline}")
        sentiments = [s for _, s in news_items]
        overall = "Positive" if sentiments.count("Positive") > 2 else "Negative" if sentiments.count("Negative") > 2 else "Neutral"
        st.subheader(f"📌 Overall Sentiment: {overall}")
    else:
        st.info("No recent news found for that keyword.")

# --- Portfolio Metrics Section ---
st.header("💼 Portfolio Metrics")
portfolio_input = st.text_input("Enter tickers with weights (e.g., AAPL,50 MSFT,50)").strip()
portfolio = {}
if portfolio_input:
    try:
        for item in portfolio_input.split():
            ticker, weight = item.split(',')
            portfolio[ticker.upper()] = float(weight)
    except:
        st.error("Invalid format. Use TICKER,WEIGHT")

if portfolio:
    price_fig, rsi_fig, volume_fig, return_fig = go.Figure(), go.Figure(), go.Figure(), go.Figure()
    for i, (ticker, weight) in enumerate(portfolio.items()):
        data = get_stock_data_alpha(ticker)
        if data is None: continue
        color = px.colors.qualitative.Plotly[i % 10]

        if "Close Price" in portfolio_metrics:
            price_fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name=f"{ticker}", line=dict(color=color)))
        if "MA50" in portfolio_metrics:
            price_fig.add_trace(go.Scatter(x=data.index, y=data["MA50"], name=f"{ticker} MA50", line=dict(color=color, dash='dot')))
        if "RSI" in portfolio_metrics:
            rsi_fig.add_trace(go.Scatter(x=data.index, y=data["RSI"], name=f"{ticker}", line=dict(color=color)))
        if "Volume" in portfolio_metrics:
            volume_fig.add_trace(go.Scatter(x=data.index, y=data["Volume"], name=f"{ticker}", line=dict(color=color)))
        if "Cumulative Return" in portfolio_metrics:
            cum_return = (1 + data['DailyReturn']).cumprod()
            return_fig.add_trace(go.Scatter(x=data.index, y=cum_return, name=f"{ticker} Return", line=dict(color=color)))

    if price_fig.data: 
        st.subheader("📉 Price Metrics")
        st.plotly_chart(price_fig, use_container_width=True)
    if rsi_fig.data:
        st.subheader("📊 RSI Analysis")
        st.plotly_chart(rsi_fig, use_container_width=True)
    if volume_fig.data:
        st.subheader("📈 Volume")
        st.plotly_chart(volume_fig, use_container_width=True)
    if return_fig.data:
        st.subheader("📈 Cumulative Returns")
        st.plotly_chart(return_fig, use_container_width=True)
