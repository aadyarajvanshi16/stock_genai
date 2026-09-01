"""
Fetches raw price history and company info from Yahoo Finance.
This is the "ground truth" data layer every agent reads from.
"""
import yfinance as yf
import pandas as pd
import streamlit as st


@st.cache_data(ttl=900, show_spinner=False)  # cache 15 min
def fetch_price_history(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """
    Returns OHLCV history for a ticker.
    period: 1mo, 3mo, 6mo, 1y, 2y, 5y, max
    interval: 1d, 1wk, 1mo
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    if df.empty:
        raise ValueError(f"No price data found for ticker '{ticker}'. Check the symbol.")
    df = df.reset_index()
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_company_info(ticker: str) -> dict:
    """Returns basic fundamentals/company info for the header cards."""
    stock = yf.Ticker(ticker)
    info = stock.info or {}
    return {
        "name": info.get("longName", ticker),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE"),
        "52w_high": info.get("fiftyTwoWeekHigh"),
        "52w_low": info.get("fiftyTwoWeekLow"),
        "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
        "previous_close": info.get("previousClose"),
        "currency": info.get("currency", "USD"),
        "summary": info.get("longBusinessSummary", ""),
        "website": info.get("website", ""),
    }


def format_market_cap(value):
    if not value:
        return "N/A"
    for unit, threshold in [("T", 1e12), ("B", 1e9), ("M", 1e6)]:
        if value >= threshold:
            return f"{value / threshold:.2f}{unit}"
    return str(value)
