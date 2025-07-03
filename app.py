import streamlit as st
import pandas as pd
import requests
import plotly.graph_objs as go
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.foreignexchange import ForeignExchange

# --- Page config ---
st.set_page_config(page_title="Canadian Market & Forex Dashboard", layout="wide")
st.title("Canadian Market & Forex Dashboard")

# --- Load API keys securely ---
ALPHA_API_KEY = st.secrets["alpha_vantage"]["api_key"]
EXCHANGE_API_KEY = st.secrets["exchange_rate_api"]["api_key"]

ts = TimeSeries(key=ALPHA_API_KEY, output_format='pandas')
fx = ForeignExchange(key=ALPHA_API_KEY)

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

@st.cache_data(ttl=3600)
def get_fx_daily_data(from_currency, to_currency):
    try:
        data, _ = fx.get_currency_exchange_daily(from_symbol=from_currency, to_symbol=to_currency, outputsize='compact')
        df = pd.DataFrame.from_dict(data, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close'
        })
        df = df.sort_index()
        df = df.astype(float)
        return df
    except Exception as e:
        st.error(f"Error loading FX data for {from_currency}/{to_currency}: {e}")
        return None

# --- Sidebar Settings ---
st.sidebar.header("⚙️ Settings")
chart_type = st.sidebar.radio("Chart Type", ["Candlestick", "Line"], index=0)

st.sidebar.markdown("📊 **Portfolio Metrics to Display**")
available_metrics = ['Close', 'MA20', 'MA50', 'UpperBand', 'LowerBand', 'RSI', 'Volume']
selected_metrics = st.sidebar.multiselect(
    "Select metrics to plot for portfolio",
    options=available_metrics,
    default=['Close']
)

# --- Stock Ticker Selection ---
st.sidebar.header("🗂 Portfolio Settings")
portfolio_tickers_input = st.sidebar.text_area(
    "Enter your stock tickers separated by commas (e.g. SHOP.TO, ENB.TO, BNS.TO)",
    value="SHOP.TO, ENB.TO, BNS.TO"
)
portfolio_tickers = [t.strip().upper() for t in portfolio_tickers_input.split(",") if t.strip()]

# --- Plot function for stock charts ---
def plot_chart(df, title, chart_type, selected_metrics, key_prefix):
    fig = go.Figure()
    if "Close" in selected_metrics:
        if chart_type == 'Candlestick':
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'],
                name='Candlestick'
            ))
        else:
            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))

    line_styles = {
        "Close": "solid",
        "MA20": "dash",
        "MA50": "dot",
        "UpperBand": "dashdot",
        "LowerBand": "longdash",
        "RSI": "dash",
    }

    for metric in selected_metrics:
        if metric == "Close":
            continue
        if metric == "Volume":
            fig.add_trace(go.Bar(
                x=df.index,
                y=df["Volume"],
                name='Volume',
                marker_color='lightgray',
                yaxis='y2',
                opacity=0.3
            ))
        elif metric in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[metric],
                mode='lines',
                name=metric,
                line=dict(dash=line_styles.get(metric, 'solid'))
            ))

    layout = dict(
        title=title,
        xaxis_title='Date',
        height=450,
    )
    if "Volume" in selected_metrics:
        layout['yaxis2'] = dict(
            overlaying='y',
            side='right',
            title='Volume',
            showgrid=False,
            position=0.15
        )
    fig.update_layout(layout)
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_main")

@st.cache_data(ttl=3600)
def get_portfolio_data(tickers):
    data = {}
    for ticker in tickers:
        df = get_stock_data_alpha(ticker)
        if df is not None:
            data[ticker] = df
    return data

def plot_portfolio_metrics(portfolio_data, selected_metrics):
    fig = go.Figure()
    line_styles = {
        "Close": "solid",
        "MA20": "dash",
        "MA50": "dot",
        "UpperBand": "dashdot",
        "LowerBand": "longdash",
        "RSI": "dash",
    }
    volume_selected = "Volume" in selected_metrics

    for ticker, df in portfolio_data.items():
        for metric in selected_metrics:
            if metric == "Volume":
                continue
            if metric in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df[metric],
                    mode='lines',
                    name=f"{ticker} {metric}",
                    line=dict(dash=line_styles.get(metric, 'solid'))
                ))

    if volume_selected:
        for ticker, df in portfolio_data.items():
            if "Volume" in df.columns:
                fig.add_trace(go.Bar(
                    x=df.index,
                    y=df["Volume"],
                    name=f"{ticker} Volume",
                    yaxis="y2",
                    opacity=0.3
                ))

    layout = dict(
        title="Portfolio Metrics",
        xaxis_title="Date",
        height=500,
        legend_title="Stock and Metric"
    )
    if volume_selected:
        layout["yaxis2"] = dict(
            overlaying='y',
            side='right',
            title='Volume',
            showgrid=False,
            position=0.15
        )
    fig.update_layout(layout)
    st.plotly_chart(fig, use_container_width=True, key="portfolio_metrics")

# --- Forex Section ---
st.header("💱 Forex Dashboard")

# Fixed USD/CAD comparison metric
usd_cad_rate = get_exchange_rate("USD", "CAD")
if usd_cad_rate:
    st.metric("USD/CAD Exchange Rate", f"{usd_cad_rate:.4f}")
else:
    st.write("Failed to fetch USD/CAD rate")

# User selects forex pairs to compare
st.subheader("Compare Forex Pairs")
currency_options = ["USD", "CAD", "EUR", "GBP", "JPY", "AUD", "CHF", "NZD"]

user_forex_from = st.selectbox("From Currency", options=currency_options, index=0)
user_forex_to = st.selectbox("To Currency", options=currency_options, index=1)

if user_forex_from == user_forex_to:
    st.warning("Please select two different currencies.")
else:
    user_rate = get_exchange_rate(user_forex_from, user_forex_to)
    if user_rate:
        st.metric(f"{user_forex_from}/{user_forex_to} Exchange Rate", f"{user_rate:.4f}")
    else:
        st.write(f"Failed to fetch {user_forex_from}/{user_forex_to} rate")

    # Plot FX historical data using Alpha Vantage
    fx_df = get_fx_daily_data(user_forex_from, user_forex_to)
    if fx_df is not None:
        fig_fx = go.Figure()
        fig_fx.add_trace(go.Scatter(
            x=fx_df.index,
            y=fx_df['Close'],
            mode='lines',
            name=f"{user_forex_from}/{user_forex_to} Close"
        ))
        fig_fx.update_layout(
            title=f"Historical FX Rates: {user_forex_from}/{user_forex_to}",
            xaxis_title="Date",
            yaxis_title="Exchange Rate",
            height=400
        )
        st.plotly_chart(fig_fx, use_container_width=True, key="fx_historical")

# --- TSX Composite Proxy Section ---
st.header("TSX Composite Proxy ETF (XIC.TO)")
tsx_data = get_stock_data_alpha("XIC.TO")
if tsx_data is not None:
    plot_chart(tsx_data, "TSX Composite Proxy ETF (XIC.TO)", chart_type, selected_metrics, "tsx")
else:
    st.error("Failed to load TSX Composite data.")

# --- Portfolio Section ---
st.header("📊 Portfolio")
portfolio_data = get_portfolio_data(portfolio_tickers)
if not portfolio_data:
    st.warning("No portfolio data loaded. Check your tickers.")
else:
    plot_portfolio_metrics(portfolio_data, selected_metrics)

# --- Simple Rule-Based LLM-Like Analysis ---
st.header("🤖 Portfolio Analysis")
analysis_msgs = []
for ticker, df in portfolio_data.items():
    latest_rsi = df['RSI'].iloc[-1]
    weekly_change = (df['Close'].iloc[-1] - df['Close'].iloc[-6]) / df['Close'].iloc[-6] * 100
    msg = f"**{ticker}**: "
    if latest_rsi < 30:
        msg += "RSI below 30, stock might be oversold - consider buying. "
    elif latest_rsi > 70:
        msg += "RSI above 70, stock might be overbought - consider caution. "
    if weekly_change > 5:
        msg += "Price increased >5% in last week, trend is bullish. "
    elif weekly_change < -5:
        msg += "Price decreased >5% in last week, trend is bearish. "
    analysis_msgs.append(msg)
for msg in analysis_msgs:
    st.write(msg)
