"""
Fetches raw price history and company info from Yahoo Finance, with a
free, key-less fallback (Stooq) for when Yahoo rate-limits the app.

Yahoo Finance aggressively rate-limits requests that come from cloud provider
IPs (Streamlit Cloud, AWS, etc.), sometimes for extended periods regardless of
browser impersonation. We handle this in layers:
  1. Use a curl_cffi session that impersonates Chrome's TLS fingerprint.
  2. Retry with short exponential backoff on transient 429 errors.
  3. If Yahoo still fails, fall back to Stooq's free CSV endpoint for price
     history (no API key, much more lenient with cloud IPs).
"""
import time
import random
import io

import yfinance as yf
import pandas as pd
import requests
import streamlit as st

try:
    from curl_cffi import requests as curl_requests
    _HAS_CURL_CFFI = True
except ImportError:
    _HAS_CURL_CFFI = False

_PERIOD_TO_DAYS = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825}


@st.cache_resource(show_spinner=False)
def _get_session():
    """A browser-impersonating session, reused across all yfinance calls."""
    if _HAS_CURL_CFFI:
        return curl_requests.Session(impersonate="chrome")
    return None  # falls back to yfinance's default session


def _with_retry(fn, max_retries: int = 2):
    """Runs fn() with short exponential backoff on rate-limit / transient errors."""
    last_err = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if "429" in msg or "too many requests" in msg or "rate limit" in msg:
                time.sleep((2 ** attempt) + random.uniform(0, 1))
                continue
            raise
    raise last_err


def _fetch_from_stooq(ticker: str, period: str) -> pd.DataFrame:
    """
    Free fallback with no API key. Stooq expects e.g. 'aapl.us' for US tickers;
    tickers that already carry an exchange suffix (like 'TCS.NS' -> stooq
    doesn't support NSE) are passed through as-is on a best-effort basis.
    """
    symbol = ticker.lower()
    if "." not in symbol:
        symbol += ".us"

    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.text))
    if df.empty or "Date" not in df.columns:
        raise ValueError("stooq_empty")

    df["Date"] = pd.to_datetime(df["Date"])
    cutoff_days = _PERIOD_TO_DAYS.get(period, 180)
    df = df[df["Date"] >= (pd.Timestamp.now() - pd.Timedelta(days=cutoff_days))]
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].reset_index(drop=True)
    if df.empty:
        raise ValueError("stooq_empty")
    return df


@st.cache_data(ttl=900, show_spinner=False)  # cache 15 min
def fetch_price_history(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """
    Returns OHLCV history for a ticker. Tries Stooq FIRST — it's a free,
    key-less source that (unlike Yahoo) doesn't rate-limit Streamlit Cloud's
    shared IPs. Falls back to Yahoo Finance only for tickers Stooq doesn't
    cover (non-US exchanges) or if Stooq is briefly down.
    period: 1mo, 3mo, 6mo, 1y, 2y, 5y, max
    interval: 1d, 1wk, 1mo
    """
    try:
        return _fetch_from_stooq(ticker, period)
    except Exception:
        pass  # fall through to Yahoo below

    def _fetch_yahoo():
        stock = yf.Ticker(ticker, session=_get_session())
        return stock.history(period=period, interval=interval)

    try:
        df = _with_retry(_fetch_yahoo)
        if df.empty:
            raise ValueError("yahoo_empty")
        df = df.reset_index()
        return df
    except Exception:
        raise ValueError(
            f"Couldn't fetch price data for '{ticker}' from Stooq or Yahoo Finance. "
            f"Either the symbol is wrong, or both data sources are temporarily "
            f"unavailable — wait a minute and retry."
        )


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_company_info(ticker: str) -> dict:
    """Returns basic fundamentals/company info for the header cards."""
    def _fetch():
        stock = yf.Ticker(ticker, session=_get_session())
        return stock.info or {}

    try:
        info = _with_retry(_fetch)
    except Exception:
        info = {}  # non-fatal — header cards will just show N/A
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
