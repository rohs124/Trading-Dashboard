import streamlit as st
import pandas as pd
import requests
import plotly.graph_objs as go
from alpha_vantage.timeseries import TimeSeries

# --- API keys from Streamlit secrets ---
ALPHA_API_KEY = st.secrets.get("alpha_vantage", {}).get("api_key", "")
EXCHANGE_API_KEY = st.secrets.get("exchange_rate_api", {}).get("api_key", "")

ts = TimeSeries(key=ALPHA_API_KEY, output_format='pandas')

# --- Cached fetch of stock data ---
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
        st.error(f"Error loading {ticker} from Alpha Vantage: {e}")
        return None

# --- Cached fetch of exchange rates ---
@st.cache_data(ttl=300)
def get_exchange_rate(from_currency, to_currency):
    url = f"https://v6.exchangerate-api.com/v6/{EXCHANGE_API_KEY}/pair/{from_currency}/{to_currency}"
    try:
        response = requests.get(url)
        data = response.json()
        if data["result"] == "success":
            return data["conversion_rate"]
        else:
            st.error(f"Failed to get rate for {from_currency}/{to_currency}: {data.get('error-type')}")
            return None
    except Exception as e:
        st.error(f"Exception fetching rate for {from_currency}/{to_currency}: {e}")
        return None

# --- Stock analysis rules ---
def analyze_stock(df):
    if df is None or df.empty:
        return "No data to analyze."
    rsi_latest = df['RSI'].iloc[-1]
    price_latest = df['Close'].iloc[-1]
    price_week_ago = df['Close'].iloc[-6] if len(df) > 6 else df['Close'].iloc[0]
    pct_change_week = ((price_latest - price_week_ago) / price_week_ago) * 100

    analysis = []
    if rsi_latest < 30:
        analysis.append("RSI below 30 → Oversold, consider buying.")
    elif rsi_latest > 70:
        analysis.append("RSI above 70 → Overbought, consider selling.")
    else:
        analysis.append("RSI in normal range.")

    if pct_change_week > 5:
        analysis.append(f"Price increased {pct_change_week:.2f}% last week → Bullish trend.")
    elif pct_change_week < -5:
        analysis.append(f"Price decreased {pct_change_week:.2f}% last week → Bearish trend.")
    else:
        analysis.append("No significant price change last week.")

    return " ".join(analysis)

# --- Chart plotting function ---
def plot_chart(df, title, chart_type, show_ma, show_bollinger, show_rsi, show_volume, key_prefix):
    fig = go.Figure()
    if chart_type == 'Candlestick':
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'],
            name='Candlestick'
        ))
    else:
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))

    if show_ma:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], mode='lines', name='MA20', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], mode='lines', name='MA50', line=dict(color='green')))

    if show_bollinger:
        fig.add_trace(go.Scatter(x=df.index, y=df['UpperBand'], name='UpperBand', line=dict(dash='dot', color='gray')))
        fig.add_trace(go.Scatter(x=df.index, y=df['LowerBand'], name='LowerBand', line=dict(dash='dot', color='gray')))

    if show_volume and 'Volume' in df.columns:
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color='lightgray', yaxis='y2'))
        fig.update_layout(yaxis2=dict(overlaying='y', side='right', showgrid=False, title='Volume'))

    fig.update_layout(title=title, height=400, xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_main")

    if show_rsi:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'))
        fig_rsi.update_layout(title="RSI", height=200, yaxis_title='RSI')
        st.plotly_chart(fig_rsi, use_container_width=True, key=f"{key_prefix}_rsi")

# --- Sidebar for settings and portfolio ---
st.sidebar.header("⚙️ Settings")
chart_type = st.sidebar.radio("Chart Type", ["Candlestick", "Line"], index=0)

st.sidebar.markdown("📊 **Metrics**")
show_ma = st.sidebar.checkbox("Show 20/50-Day MA", True)
show_bollinger = st.sidebar.checkbox("Show Bollinger Bands", True)
show_rsi = st.sidebar.checkbox("Show RSI", False)
show_volume = st.sidebar.checkbox("Show Volume", False)

st.sidebar.header("📁 Portfolio Input")
portfolio_input = st.sidebar.text_area(
    "Enter your portfolio (Ticker:Weight comma-separated)",
    value="SHOP.TO:0.4, ENB.TO:0.3, BNS.TO:0.3"
)

portfolio = {}
try:
    for item in portfolio_input.split(","):
        ticker, weight = item.strip().split(":")
        portfolio[ticker.strip().upper()] = float(weight.strip())
except Exception:
    st.sidebar.error("Invalid portfolio input format. Use TICKER:WEIGHT, separated by commas.")

# --- Main App Title ---
st.title("📈 Canadian Market & Forex Dashboard")

# --- Portfolio Metrics & Analysis ---
st.header("Portfolio Metrics & Analysis")
metrics_data = []
for ticker, weight in portfolio.items():
    df = get_stock_data_alpha(ticker)
    if df is not None:
        latest_price = df['Close'].iloc[-1]
        analysis_text = analyze_stock(df)
        metrics_data.append((ticker, weight, latest_price, analysis_text))
    else:
        metrics_data.append((ticker, weight, None, "No data"))

for ticker, weight, price, analysis_text in metrics_data:
    st.subheader(f"{ticker} — Weight: {weight*100:.1f}%")
    if price:
        st.write(f"Latest Price: ${price:.2f}")
    else:
        st.write("Price data not available.")
    st.info(analysis_text)
    if price and ticker in portfolio:
        plot_chart(df, f"{ticker} Price Chart", chart_type, show_ma, show_bollinger, show_rsi, show_volume, f"portfolio_{ticker}")

# --- Forex Dashboard ---
st.header("💱 Forex Dashboard")
forex_pairs = [
    ("USD", "CAD"),
    ("EUR", "USD"),
    ("GBP", "USD"),
    ("USD", "JPY")
]

cols = st.columns(len(forex_pairs))
for i, (from_curr, to_curr) in enumerate(forex_pairs):
    with cols[i]:
        st.subheader(f"{from_curr}/{to_curr}")
        rate = get_exchange_rate(from_curr, to_curr)
        if rate is not None:
            st.metric(label="Exchange Rate", value=f"{rate:.4f}")

            # Simple line plot for recent rates (simulate with dummy data or extend for real history)
            # For demo, just show current rate as a single point plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0,1], y=[rate*0.98, rate], mode='lines+markers', name=f'{from_curr}/{to_curr}'))
            fig.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("Failed to fetch rate")

