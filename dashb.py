import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import plotly.express as px

# Inject custom CSS for styling
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

# 📌 Retrieve index constituents from Wikipedia and Yahoo Finance
@st.cache_data
def get_sp500_constituents():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    df = tables[0][['Symbol', 'Security']]
    return df.set_index('Symbol')

@st.cache_data
def get_yahoo_constituents(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    tables = pd.read_html(response.text)
    df = tables[0]  # First table containing the constituents
    return df.set_index(df.columns[0])

# 🔥 Dictionary of indices
indices = {
    'S&P 500': get_sp500_constituents,
    'FTSE 100': lambda: get_yahoo_constituents('https://finance.yahoo.com/quote/%5EFTSE/components/'),
    'DAX 40': lambda: get_yahoo_constituents('https://finance.yahoo.com/quote/%5EGDAXI/components/'),
    'CAC 40': lambda: get_yahoo_constituents('https://finance.yahoo.com/quote/%5EFCHI/components/'),
    'Euro Stoxx 50': lambda: get_yahoo_constituents('https://finance.yahoo.com/quote/%5ESTOXX50E/components/'),
    'Other': None  # Custom ticker option
}

# 🌎 Sidebar - Settings Panel
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

    years = st.slider("📅 Select analysis period (in years):", 1, 10, 5)
    interval = st.radio("📈 Select time interval:", ["1d", "1mo", "1y"], index=1)

# 📊 Retrieve Financial Data
@st.cache_data
def get_stock_data(ticker, years, interval):
    period = f"{years}y"
    return yf.Ticker(ticker).history(period=period, interval=interval)

stock_data = get_stock_data(selected_ticker, years, interval)

# 🌟 Main Content
st.title("📊 Stock Market Dashboard - European Indices and S&P 500")

# 🔹 Price Chart (Using Plotly)
st.subheader("📊 Stock Price")
fig_price = px.line(stock_data, x=stock_data.index, y="Close", 
                    title=f"Price Evolution of {selected_ticker} over {years} years",
                    labels={"x": "Date", "Close": "Price (USD)"},
                    line_shape="linear")
st.plotly_chart(fig_price, use_container_width=True)

# 🔹 Volume Chart (Using Plotly)
st.subheader("📊 Traded Volume")
fig_volume = px.bar(stock_data, x=stock_data.index, y="Volume", 
                    title=f"Traded Volume of {selected_ticker} over {years} years",
                    labels={"x": "Date", "Volume": "Volume"},
                    color_discrete_sequence=["#FFA07A"])
st.plotly_chart(fig_volume, use_container_width=True)

# 🔹 Retrieve Financial Indicators
ticker_info = yf.Ticker(selected_ticker).info

st.subheader("📊 Financial Indicators")

def format_number(value):
    """ Format numbers with thousand separators, otherwise display 'Data Unavailable' """
    if isinstance(value, (int, float)):
        return f"{value:,.0f} USD".replace(",", " ")
    return "Data Unavailable"

def format_percentage(value):
    """ Format percentage """
    if isinstance(value, (int, float)):
        return f"{value:.2f} %"
    return "Data Unavailable"

def format_price(value):
    """ Format price """
    if isinstance(value, (int, float)):
        return f"{value:.2f} USD"
    return "Data Unavailable"

# 📊 Display indicators in columns
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

st.success("✅ Analysis successfully completed!")

# 📌 Footer
st.markdown("""
    ---
    📌 **Stock Market Dashboard** | Created by [Matthieu Lombardo](https://www.linkedin.com/in/matthieu-lombardo)
""", unsafe_allow_html=True)

