"""
Computes technical indicators from OHLCV data using plain pandas math
(no extra heavy dependency needed). Also produces a compact "summary dict"
that the Technical Analyst Agent reads to write its analysis — this keeps
the LLM prompt small and numeric instead of dumping a whole dataframe.
"""
import pandas as pd
import numpy as np


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["EMA_12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA_26"] = df["Close"].ewm(span=26, adjust=False).mean()
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "EMA_12" not in df or "EMA_26" not in df:
        df = add_moving_averages(df)
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    return df


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    df = df.copy()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))
    df["RSI_14"] = df["RSI_14"].fillna(50)  # neutral default for early rows
    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: int = 2) -> pd.DataFrame:
    df = df.copy()
    sma = df["Close"].rolling(window=window).mean()
    std = df["Close"].rolling(window=window).std()
    df["BB_Mid"] = sma
    df["BB_Upper"] = sma + num_std * std
    df["BB_Lower"] = sma - num_std * std
    return df


def add_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Daily_Return_%"] = df["Close"].pct_change() * 100
    return df


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = add_moving_averages(df)
    df = add_macd(df)
    df = add_rsi(df)
    df = add_bollinger_bands(df)
    df = add_daily_returns(df)
    return df


def get_technical_summary(df: pd.DataFrame) -> dict:
    """
    Condenses the indicator dataframe into the latest numeric snapshot +
    simple rule-based flags. This dict is what gets fed into the LLM prompt
    for the Technical Analyst Agent (numbers in, narrative out).
    """
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    volatility = df["Daily_Return_%"].std()
    trend = "uptrend" if latest["SMA_20"] > latest["SMA_50"] else "downtrend"

    rsi = latest["RSI_14"]
    if rsi >= 70:
        rsi_flag = "overbought"
    elif rsi <= 30:
        rsi_flag = "oversold"
    else:
        rsi_flag = "neutral"

    macd_signal = "bullish crossover" if latest["MACD"] > latest["MACD_Signal"] else "bearish crossover"

    price_vs_bb = "above upper band" if latest["Close"] > latest["BB_Upper"] else (
        "below lower band" if latest["Close"] < latest["BB_Lower"] else "within bands"
    )

    return {
        "latest_close": round(float(latest["Close"]), 2),
        "prev_close": round(float(prev["Close"]), 2),
        "change_pct": round(float((latest["Close"] - prev["Close"]) / prev["Close"] * 100), 2),
        "sma_20": round(float(latest["SMA_20"]), 2) if pd.notna(latest["SMA_20"]) else None,
        "sma_50": round(float(latest["SMA_50"]), 2) if pd.notna(latest["SMA_50"]) else None,
        "trend": trend,
        "rsi_14": round(float(rsi), 1),
        "rsi_flag": rsi_flag,
        "macd": round(float(latest["MACD"]), 3),
        "macd_signal_line": round(float(latest["MACD_Signal"]), 3),
        "macd_condition": macd_signal,
        "bollinger_position": price_vs_bb,
        "volatility_pct_std": round(float(volatility), 2) if pd.notna(volatility) else None,
        "52w_period_high": round(float(df["High"].max()), 2),
        "52w_period_low": round(float(df["Low"].min()), 2),
    }
