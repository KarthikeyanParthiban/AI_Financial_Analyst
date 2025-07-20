pip install nsepython streamlit google plotly ta

import streamlit as st
import pandas as pd
import requests
from io import StringIO
from datetime import datetime, timedelta
from nsepython import nsefetch
from google import genai
from google.genai import types
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta  # install with: pip install ta

# Configure Gemini API
client = genai.Client(api_key="AIzaSyAyCshk4-Pp5DY9XhdG7T2aGLn-WafWXZc")

# Constants
TABLE_NAMES = [
    "Quarterly Results", "Profit & Loss", "Compounded Sales Growth", "Compounded Profit Growth",
    "Stock Price CAGR", "Return on Equity", "Balance Sheet", "Cash Flows",
    "Ratios", "Shareholding Pattern (Quaterly)", "Shareholding Pattern (Yearly)"
]

# Streamlit setup
st.set_page_config(layout="wide")
st.sidebar.title("📈 Screener AI Assistant")

st.cache_data.clear()

# --- Compute Technical Indicators ---
def compute_indicators(df):
    df = df.copy()
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_hist"] = macd.macd_diff()
    
    # # Drop rows with NaNs in any of the indicator columns
    # df.dropna(subset=["RSI", "MACD", "MACD_signal", "MACD_hist"], inplace=True)
    
    return df
def plot_price_volume(df):
    st.caption(f"📅 Latest available data: `{df['Date'].max().strftime('%Y-%m-%d')}`")

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3]
    )

    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Close"],
        mode="lines",
        name="Close Price",
        line=dict(color="royalblue", width=2)
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=df["Date"],
        y=df["Volume"],
        name="Volume",
        marker_color="lightgray",
        opacity=0.6
    ), row=2, col=1)

    fig.update_layout(
        title="📈 Closing Price and Volume",
        height=600,
        template="plotly_dark",
        hovermode="x unified",
        margin=dict(t=40, b=40, l=60, r=30),
        font=dict(family="Segoe UI", size=13),
        showlegend=False
    )

    fig.update_yaxes(title_text="Close Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig


# Load NSE stock list
@st.cache_data(ttl=86400)
def load_nse_stock_list():
    url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "text/csv"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = StringIO(response.text)
        df = pd.read_csv(data)
        df = df[["SYMBOL", "NAME OF COMPANY"]].dropna()
        df["NAME OF COMPANY"] = df["NAME OF COMPANY"].str.strip()
        return df
    else:
        return pd.DataFrame(columns=["SYMBOL", "NAME OF COMPANY"])

# Clean Screener table
def clean_table(table):
    table = table.dropna(how="all")
    table.columns = [col.strip() for col in table.columns]
    if table.columns[0] == 'Unnamed: 0':
        table[table.columns[0]] = table[table.columns[0]].str.replace('+', '', regex=False).str.strip()
        table.rename(columns={table.columns[0]: 'Metric'}, inplace=True)
    return table

# Fetch Screener tables
@st.cache_data(ttl=3600)
def fetch_screener_tables(ticker):
    url = f"https://www.screener.in/company/{ticker}/consolidated/"
    tables = pd.read_html(url)
    return [clean_table(t) for t in tables]

# Chunk tables for AI input
def chunk_tables_for_ai(tables, max_rows_per_chunk=5):
    chunks = []
    for table in tables:
        for i in range(0, len(table), max_rows_per_chunk):
            chunk = table.iloc[i:i + max_rows_per_chunk]
            chunks.append(chunk.to_string(index=False))
    return "\n\n".join(chunks)

# AI analysis
def get_ai_analysis(table_text):
    prompt_template = """
You are a senior financial analyst with deep expertise in evaluating investment opportunities.

Your task is to analyze the provided financial document and give your expert assessment. Please do the following:

1. Provide a concise summary of the financial table and historical price data (max 100 words).
2. Highlight key trends: revenue growth, profit margins, leverage changes.
3. Based on this information, provide a BUY, SELL, or HOLD recommendation with justification.

Document:
\"\"\" {content} \"\"\" 

Output format:

Summary: concise summary

Key Trends:
- Revenue Growth: ...
- Profit Margins: ...
- Leverage: ...
- Price Trends: ...
- Key Support and resistances of past 3 months : ...

Recommendation: [BUY/SELL/HOLD] for long term and short term - [Justification]
"""
    prompt = prompt_template.format(content=table_text)

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=500,
                system_instruction=prompt,
                temperature=0.5
            )
        )
        return response.text
    except Exception as e:
        return f"❌ Gemini Error: {str(e)}"

# Fetch historical OHLCV from NSE
@st.cache_data(ttl=3600)
def fetch_ohlcv(symbol):
    to_date = datetime.today()
    from_date = to_date - timedelta(days=1*365)
    from_str = from_date.strftime("%d-%m-%Y")
    to_str = to_date.strftime("%d-%m-%Y")
    url = f"https://www.nseindia.com/api/historical/cm/equity?symbol={symbol}&series=[%22EQ%22]&from={from_str}&to={to_str}"
    data = nsefetch(url)
    df = pd.DataFrame(data['data'])

    df = df.rename(columns={
        'CH_TIMESTAMP': 'Date',
        'CH_OPENING_PRICE': 'Open',
        'CH_TRADE_HIGH_PRICE': 'High',
        'CH_TRADE_LOW_PRICE': 'Low',
        'CH_CLOSING_PRICE': 'Close',
        'CH_TOT_TRADED_QTY': 'Volume'
    })

    df['Date'] = pd.to_datetime(df['Date'])
    print(df.head)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Symbol'] = symbol
    return df[['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']]

# UI: Company selection
stock_df = load_nse_stock_list()
company_names = stock_df["NAME OF COMPANY"].tolist()
selected_company = st.sidebar.selectbox("🔍 Search by Company Name", sorted(company_names))

# Get ticker
ticker_row = stock_df[stock_df["NAME OF COMPANY"] == selected_company]
ticker = ticker_row["SYMBOL"].values[0] if not ticker_row.empty else None

if ticker:
    st.sidebar.markdown(f"*Ticker:* `{ticker}`")

if ticker:
    try:
        tables = fetch_screener_tables(ticker)
        df_price = fetch_ohlcv(ticker)
        df_p =  df_price.copy()
        table_labels = [TABLE_NAMES[i] if i < len(TABLE_NAMES) else f"Extra Table {i + 1}" for i in range(len(tables))]

        selected_labels = st.sidebar.multiselect("📋 Select tables to view & analyze", table_labels)
        selected_indices = [table_labels.index(lbl) for lbl in selected_labels if lbl in table_labels]
        selected_tables = [tables[i] for i in selected_indices]

        st.header("AI Financial Analyst")
        if 'ai_output' not in st.session_state:
            st.session_state.ai_output = None

        if st.sidebar.button("🔍 Run AI Analysis"):
            with st.spinner("Analyzing with Gemini..."):
                chunked_data = chunk_tables_for_ai(tables + [df_price], max_rows_per_chunk=7)
                analysis = get_ai_analysis(chunked_data)
                st.session_state.ai_output = analysis

        if st.session_state.ai_output:
            with st.expander("📄 AI Analysis Output", expanded=True):
                st.markdown(st.session_state.ai_output)

        import plotly.graph_objects as go

        st.header("📈 Price Chart")
        df_price = compute_indicators(df_price)
        fig = plot_price_volume(df_p)
        st.plotly_chart(fig, use_container_width=True)
        
        if selected_tables:
            st.markdown("---")
            st.header("📊 Selected Financial Tables")
            for label, df in zip(selected_labels, selected_tables):
                st.subheader(f"📌 {label}")
                st.dataframe(df, use_container_width=True)
                st.download_button(
                    label=f"⬇️ Download {label}.csv",
                    data=df.to_csv(index=False),
                    file_name=f"{label}.csv",
                    mime="text/csv"
                )

        st.markdown("---")
        

    except Exception as e:
        st.error(f"❌ Error: {e}")
