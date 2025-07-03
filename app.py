import streamlit as st
import pandas as pd
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
import plotly.graph_objs as go
import requests

st.set_page_config(page_title="Canadian Market Dashboard + Portfolio & FX", layout="wide")
st.title("📈 Canadian Market & Tickers Dashboard with Portfolio Tracker & FX Rates")

# === Alpha Vantage Setup for Stock Data ===
ALPHA_API_KEY = "99VGI0IR5EUTR4CP"
ts = TimeSeries(key=ALPHA_API_KEY, output_format='pandas')

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

# === Sidebar Settings ===
st.sidebar.header("⚙️ Settings")
chart_type = st.sidebar.radio("Chart Type", ["Candlestick", "Line"], index=0)

st.sidebar.markdown("📊 **Metrics**")
show_ma = st.sidebar.checkbox("Show 20/50-Day MA", True)
show_bollinger = st.sidebar.checkbox("Show Bollinger Bands", True)
show_rsi = st.sidebar.checkbox("Show RSI", False)
show_volume = st.sidebar.checkbox("Show Volume", False)

# === TSX Composite Proxy ===
st.header("TSX Composite Proxy ETF (XIC.TO)")
tsx_data = get_stock_data_alpha("XIC.TO")
if tsx_data is not None:
    plot_chart(tsx_data, "TSX Composite Proxy ETF (XIC.TO)", chart_type, show_ma, show_bollinger, show_rsi, show_volume, "tsx")

# === Compare Any Two Tickers ===
st.header("📈 Compare Any Two Tickers")
col1, col2 = st.columns(2)
with col1:
    ticker1 = st.text_input("First Ticker", value="SHOP.TO")
with col2:
    ticker2 = st.text_input("Second Ticker", value="ENB.TO")

if ticker1 and ticker2:
    df1 = get_stock_data_alpha(ticker1)
    df2 = get_stock_data_alpha(ticker2)
    fig = go.Figure()
    if df1 is not None:
        fig.add_trace(go.Scatter(x=df1.index, y=df1['Close'], mode='lines', name=ticker1))
    if df2 is not None:
        fig.add_trace(go.Scatter(x=df2.index, y=df2['Close'], mode='lines', name=ticker2))
    fig.update_layout(title="Price Comparison", xaxis_title="Date", yaxis_title="Close Price", height=400)
    st.plotly_chart(fig, use_container_width=True)

# === Portfolio Tracker ===
st.header("💼 Portfolio Tracker")

portfolio_input = st.text_area(
    "Enter your portfolio holdings (one ticker and number of shares per line, comma separated). Example:\n"
    "SHOP.TO, 10\nENB.TO, 5\nBNS.TO, 20"
)

if portfolio_input:
    lines = portfolio_input.strip().split('\n')
    portfolio = []
    for line in lines:
        try:
            ticker, shares = line.split(',')
            ticker = ticker.strip().upper()
            shares = float(shares.strip())
            portfolio.append((ticker, shares))
        except Exception:
            st.error(f"Invalid input line: {line}")
            portfolio = []
            break

    if portfolio:
        data = []
        total_value = 0
        total_change = 0

        for ticker, shares in portfolio:
            df = get_stock_data_alpha(ticker)
            if df is not None and len(df) >= 2:
                latest_close = df['Close'].iloc[-1]
                prev_close = df['Close'].iloc[-2]
                value = latest_close * shares
                change = (latest_close - prev_close) * shares
                pct_change = (latest_close - prev_close) / prev_close * 100

                total_value += value
                total_change += change

                data.append({
                    "Ticker": ticker,
                    "Shares": shares,
                    "Latest Close": f"${latest_close:.2f}",
                    "Value": f"${value:.2f}",
                    "Change": f"${change:+.2f}",
                    "% Change": f"{pct_change:+.2f}%"
                })
            else:
                data.append({
                    "Ticker": ticker,
                    "Shares": shares,
                    "Latest Close": "N/A",
                    "Value": "N/A",
                    "Change": "N/A",
                    "% Change": "N/A"
                })

        df_portfolio = pd.DataFrame(data)
        st.dataframe(df_portfolio)

        st.markdown(f"### Total Portfolio Value: ${total_value:.2f}")
        st.markdown(f"### Total Daily Change: ${total_change:+.2f}")

# === Currency Exchange Rates ===
st.header("💱 Currency Exchange Rates (Base: CAD)")

@st.cache_data(ttl=300)
def get_fx_rates():
    try:
        # Free API from exchangerate-api.com (no key needed for basic usage)
        url = "https://open.er-api.com/v6/latest/CAD"
        response = requests.get(url)
        data = response.json()
        if data['result'] == 'success':
            rates = data['rates']
            return {
                "USD": rates.get("USD"),
                "EUR": rates.get("EUR"),
                "JPY": rates.get("JPY"),
            }
        else:
            st.error("Failed to fetch FX rates")
            return None
    except Exception as e:
        st.error(f"Error fetching FX rates: {e}")
        return None

fx_rates = get_fx_rates()
if fx_rates:
    df_fx = pd.DataFrame({
        "Currency": ["USD", "EUR", "JPY"],
        "Exchange Rate (CAD Base)": [fx_rates["USD"], fx_rates["EUR"], fx_rates["JPY"]]
    })
    st.table(df_fx)
