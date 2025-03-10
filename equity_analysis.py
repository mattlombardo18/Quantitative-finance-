import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
import numpy as np

# =============================================================================
# Custom CSS
# =============================================================================
st.markdown("""
    <style>
        /* General Styling */
        h1, h2, h3 {
            color: #003366;
            font-weight: bold;
        }
        
        /* Improve widgets */
        div.stSlider, div.stSelectbox, div.stRadio {
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
        }

        /* Success message */
        .stAlert {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
            padding: 10px;
            border-radius: 5px;
        }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# Data Fetching Functions
# =============================================================================
@st.cache_data(ttl=3600)
def get_sp500_constituents():
    """Fetches the S&P 500 components from Wikipedia."""
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    df = tables[0][['Symbol', 'Security']]
    return df.set_index('Symbol')

@st.cache_data(ttl=3600)
def get_yahoo_constituents(url):
    """Fetches the components from Yahoo Finance using the provided URL."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        tables = pd.read_html(response.text)
        df = tables[0]
        return df.set_index(df.columns[0])
    except Exception as e:
        st.error(f"Error fetching Yahoo data: {e}")
        return pd.DataFrame()

# =============================================================================
# API Keys (Replace with your own keys)
# =============================================================================
ALPHA_VANTAGE_KEY = "UP4C9WN0SGZCS191"
NEWS_API_KEY = "206696ec95894e829f90e5192999ec96"

@st.cache_data(ttl=3600)
def get_stock_data(ticker, years, interval):
    """Fetches historical stock data for a given ticker."""
    period = f"{years}y"
    return yf.Ticker(ticker).history(period=period, interval=interval)


def get_alpha_vantage_data(ticker):
    """Récupère les données financières depuis Alpha Vantage."""
    url = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={ALPHA_VANTAGE_KEY}"
    try:
        response = requests.get(url, timeout=10)  # Timeout ajouté
        response.raise_for_status()
        data = response.json()
        return data if "Symbol" in data else None
    except requests.RequestException as e:
        st.error(f"Erreur lors de la récupération des données Alpha Vantage : {e}")
        return None

@st.cache_data(ttl=3600)
def get_news(ticker, company_name=""):
    """
    Fetch the latest financial news for a given ticker from NewsAPI.
    The filtering logic now uses the provided ticker and company name,
    so it works for any stock.
    """
    # Combine ticker and company name in the query for a more specific search
    if company_name:
        query = f"{ticker} {company_name}"
    else:
        query = f"{ticker} stock"
    
    url = (
        f"https://newsapi.org/v2/everything?q={query}"
        "&sortBy=publishedAt"
        f"&apiKey={NEWS_API_KEY}"
    )
    
    response = requests.get(url)
    data = response.json()
    articles = data.get("articles", [])
    
    # Generalized filtering: check if ticker or company name (if provided)
    # appear in the article title or description.
    filtered_articles = []
    ticker_lower = ticker.lower()
    company_name_lower = company_name.lower() if company_name else ""
    
    for article in articles:
        title = (article.get('title') or '').lower()
        description = (article.get('description') or '').lower()
        
        if (ticker_lower in title or ticker_lower in description or 
            (company_name_lower in title) or 
            (company_name_lower in description)):
            filtered_articles.append(article)
    
    return filtered_articles

@st.cache_data(ttl=3600)
def get_stock_returns(ticker, years, interval, benchmark="^GSPC"):
    """
    Calculates cumulative returns over the specified number of years and interval
    for both the ticker and its benchmark.
    """
    period = f"{years}y"
    stock = yf.Ticker(ticker).history(period=period, interval=interval)["Close"]
    bench = yf.Ticker(benchmark).history(period=period, interval=interval)["Close"]
    return (stock.pct_change(), bench.pct_change())

@st.cache_data(ttl=3600)
@st.cache_data(ttl=3600)
def get_ticker_info(ticker):
    """Fetches financial information for the ticker via yfinance, with error handling."""
    try:
        info = yf.Ticker(ticker).info
        if info is None or not info:
            st.error(f"Unable to retrieve info for ticker {ticker}.")
            return {}
        return info
    except Exception as e:
        st.error(f"Error fetching ticker info for {ticker}: {e}")
        return {}


# =============================================================================
# Technical Indicators Calculation
# =============================================================================
def compute_technical_indicators(data):
    """Calculates the 50-day SMA and 200-day SMA on the provided data."""
    data["SMA_50"] = data["Close"].rolling(window=50).mean()
    data["SMA_200"] = data["Close"].rolling(window=200).mean()
    return data

def compute_rsi(data, window=14):
    """Calculates the Relative Strength Index (RSI)."""
    delta = data["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))
    return data

def compute_macd(data, span_short=12, span_long=26, span_signal=9):
    """Calculates MACD and the signal line."""
    data["EMA_short"] = data["Close"].ewm(span=span_short, adjust=False).mean()
    data["EMA_long"] = data["Close"].ewm(span=span_long, adjust=False).mean()
    data["MACD"] = data["EMA_short"] - data["EMA_long"]
    data["Signal_Line"] = data["MACD"].ewm(span=span_signal, adjust=False).mean()
    return data

def compute_bollinger_bands(data, window=20):
    """Calculates Bollinger Bands."""
    data["SMA"] = data["Close"].rolling(window=window).mean()
    data["STD"] = data["Close"].rolling(window=window).std()
    data["Upper_Band"] = data["SMA"] + (2 * data["STD"])
    data["Lower_Band"] = data["SMA"] - (2 * data["STD"])
    return data

# =============================================================================
# Index Configuration and Ticker Selection
# =============================================================================
indices = {
    'S&P 500': get_sp500_constituents,
    'FTSE 100': lambda: get_yahoo_constituents('https://finance.yahoo.com/quote/%5EFTSE/components/'),
    'DAX 40': lambda: get_yahoo_constituents('https://finance.yahoo.com/quote/%5EGDAXI/components/'),
    'CAC 40': lambda: get_yahoo_constituents('https://finance.yahoo.com/quote/%5EFCHI/components/'),
    'Euro Stoxx 50': lambda: get_yahoo_constituents('https://finance.yahoo.com/quote/%5ESTOXX50E/components/'),
    'Other': None
}

benchmark_dict = {
    'S&P 500': '^GSPC',
    'FTSE 100': '^FTSE',
    'DAX 40': '^GDAXI',
    'CAC 40': '^FCHI',
    'Euro Stoxx 50': '^STOXX50E'
}

exchange_benchmark_dict = {
    "NasdaqGS": "^GSPC",
    "NASDAQ": "^GSPC",
    "NYQ": "^GSPC",
    "NYSE": "^GSPC",
    "LSE": "^FTSE",
    "FWB": "^GDAXI",
    "PAR": "^FCHI",
    "PA": "^FCHI",
    "Paris": "^FCHI",
    "EURONEXT": "^STOXX50E"
}

# =============================================================================
# Sidebar Settings
# =============================================================================
with st.sidebar:
    st.header("⚙️ Settings")
    selected_index = st.selectbox("Select an Index:", list(indices.keys()))
    
    if selected_index == "Other":
        user_ticker = st.text_input("Enter a custom ticker (e.g., TSLA, AAPL):").upper()
        if not user_ticker:
            st.warning("⚠️ Please enter a ticker symbol.")
            st.stop()
        selected_ticker = user_ticker
    else:
        tickers_df = indices[selected_index]()
        if not tickers_df.empty:
            selected_ticker = st.selectbox(
                "Select a Company:", tickers_df.index, 
                format_func=lambda x: f"{x} - {tickers_df.loc[x].values[0]}" if x in tickers_df.index else x
            )
        else:
            st.warning("⚠️ Unable to retrieve companies for this index.")
            st.stop()

    years = st.slider("📅 Analysis period (in years):", 1, 10, 5)
    interval = st.radio("📈 Time interval:", ["1d", "1mo", "1y"], index=1)

# =============================================================================
# Fetching Financial Data and Main Displays
# =============================================================================
with st.spinner("Loading data..."):
    stock_data = get_stock_data(selected_ticker, years, interval)
    ticker_info = get_ticker_info(selected_ticker)

if selected_index == "Other":
    if not ticker_info or ticker_info.get("regularMarketPrice") is None:
        st.error("Ticker not recognized or data unavailable.")
        st.stop()
    exchange = ticker_info.get("exchange", "")
    benchmark_symbol = exchange_benchmark_dict.get(exchange, "^GSPC")
else:
    benchmark_symbol = benchmark_dict.get(selected_index, '^GSPC')

st.title("📊 Stock Market Dashboard")

# =============================================================================
# Option to Download Data
# =============================================================================
st.download_button(
    label="Download CSV Data",
    data=stock_data.to_csv().encode('utf-8'),
    file_name=f"{selected_ticker}_data.csv",
    mime='text/csv'
)

# =============================================================================
# Stock Price Chart (Altair)
# =============================================================================
st.subheader("📊 Stock Price Evolution")
chart = alt.Chart(stock_data.reset_index()).mark_line().encode(
    x="Date:T",
    y="Close:Q",
    tooltip=["Date", "Close"]
).properties(
    title=f"Price Evolution of {selected_ticker}"
).interactive()
st.altair_chart(chart, use_container_width=True)

# =============================================================================
# Traded Volume Chart (Plotly)
# =============================================================================
st.subheader("📊 Traded Volume")
fig_volume = px.bar(
    stock_data, 
    x=stock_data.index, 
    y="Volume", 
    title=f"Traded Volume of {selected_ticker} over {years} years",
    labels={"x": "Date", "Volume": "Volume"}
)
st.plotly_chart(fig_volume, use_container_width=True)

# =============================================================================
# Financial Indicators (via yfinance)
# =============================================================================
st.subheader("📊 Financial Indicators")

def format_number(value):
    if isinstance(value, (int, float)):
        return f"{value:,.0f} USD".replace(",", " ")
    return "Data Unavailable"

def format_percentage(value):
    if isinstance(value, (int, float)):
        return f"{value:.2f} %"
    return "Data Unavailable"

def format_price(value):
    if isinstance(value, (int, float)):
        return f"{value:.2f} USD"
    return "Data Unavailable"

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"**🏢 Company Name:** {ticker_info.get('longName', 'Data Unavailable')}")
    st.markdown(f"**📌 Sector:** {ticker_info.get('sector', 'Data Unavailable')}")
    st.markdown(f"**💰 Market Capitalization:** {format_number(ticker_info.get('marketCap'))}")
    st.markdown(f"**💵 Current Price:** {format_price(ticker_info.get('regularMarketPrice'))}")
    st.markdown(f"**📊 P/E Ratio:** {ticker_info.get('trailingPE', 'Data Unavailable')}")
with col2:
    st.markdown(f"**🏦 EBITDA:** {format_number(ticker_info.get('ebitda'))}")
    st.markdown(f"**💰 Net Income:** {format_number(ticker_info.get('netIncomeToCommon'))}")
    st.markdown(f"**🔄 Cash Flow:** {format_number(ticker_info.get('operatingCashflow'))}")
    st.markdown(f"**🏦 Total Debt:** {format_number(ticker_info.get('totalDebt'))}")
    st.markdown(f"**📊 Dividend Yield:** {format_percentage(ticker_info.get('dividendYield'))}")

# =============================================================================
# Additional Data and Supplementary Displays
# =============================================================================
av_data = get_alpha_vantage_data(selected_ticker)
company_name = ticker_info.get('longName', '')
news_articles = get_news(selected_ticker, company_name)
returns, bench_returns = get_stock_returns(selected_ticker, years, interval, benchmark=benchmark_symbol)

# =============================================================================
# Advanced Technical Indicators
# =============================================================================
st.subheader("📈 Advanced Technical Indicators")
stock_data_ind = stock_data.copy()
stock_data_ind = compute_technical_indicators(stock_data_ind)
stock_data_ind = compute_rsi(stock_data_ind)
stock_data_ind = compute_macd(stock_data_ind)
stock_data_ind = compute_bollinger_bands(stock_data_ind)

# Chart with SMA, Bollinger Bands, and Price
fig_tech = go.Figure()
fig_tech.add_trace(go.Scatter(x=stock_data_ind.index, y=stock_data_ind["Close"], mode='lines', name="Price"))
fig_tech.add_trace(go.Scatter(x=stock_data_ind.index, y=stock_data_ind["SMA_50"], mode='lines', name="50-day SMA"))
fig_tech.add_trace(go.Scatter(x=stock_data_ind.index, y=stock_data_ind["SMA_200"], mode='lines', name="200-day SMA"))
fig_tech.add_trace(go.Scatter(x=stock_data_ind.index, y=stock_data_ind["Upper_Band"], mode='lines', name="Bollinger Upper"))
fig_tech.add_trace(go.Scatter(x=stock_data_ind.index, y=stock_data_ind["Lower_Band"], mode='lines', name="Bollinger Lower"))
st.plotly_chart(fig_tech, use_container_width=True)

# Display RSI
st.subheader("📈 RSI (Relative Strength Index)")
fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=stock_data_ind.index, y=stock_data_ind["RSI"], mode='lines', name="RSI"))
fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
st.plotly_chart(fig_rsi, use_container_width=True)

# Display MACD
st.subheader("📈 MACD (Moving Average Convergence Divergence)")
fig_macd = go.Figure()
fig_macd.add_trace(go.Scatter(x=stock_data_ind.index, y=stock_data_ind["MACD"], mode='lines', name="MACD"))
fig_macd.add_trace(go.Scatter(x=stock_data_ind.index, y=stock_data_ind["Signal_Line"], mode='lines', name="Signal"))
st.plotly_chart(fig_macd, use_container_width=True)

# =============================================================================
# Performance Comparison
# =============================================================================
if selected_index == "Other":
    comp_index_name = ticker_info.get("exchange", "Unknown Exchange")
else:
    comp_index_name = selected_index

st.subheader(f"📊 Performance Comparison with {comp_index_name}")
perf_df = pd.DataFrame({
    "Stock": returns.cumsum(),
    comp_index_name: bench_returns.cumsum()
})
st.line_chart(perf_df)

# =============================================================================
# Financial Metrics from Alpha Vantage
# =============================================================================
if av_data:
    st.subheader("💰 Financial Metrics (Alpha Vantage)")
    st.markdown(f"**📈 Beta:** {av_data.get('Beta', 'N/A')}")
    st.markdown(f"**💰 ROE:** {av_data.get('ReturnOnEquityTTM', 'N/A')}%")
    st.markdown(f"**💰 ROA:** {av_data.get('ReturnOnAssetsTTM', 'N/A')}%")
    st.markdown(f"**📊 EPS:** {av_data.get('EPS', 'N/A')}")

# =============================================================================
# Latest Financial News
# =============================================================================
st.subheader("📰 Latest Financial News")
if news_articles:
    for article in news_articles[:5]:
        st.markdown(f"[{article['title']}]({article['url']}) - {article['source']['name']}")
else:
    st.info("No news articles available.")

st.success("✅ Analysis completed successfully!")

# =============================================================================
# Footer
# =============================================================================
st.markdown("""
    ---
    📌 **Stock Market Dashboard** | Created by [Matthieu Lombardo](https://www.linkedin.com/in/matthieu-lombardo)
""", unsafe_allow_html=True)
