"""
Plotly chart builders. Kept separate from app.py so the main file stays readable.
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def candlestick_with_ma(df, ticker: str):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price"
    ))
    if "SMA_20" in df:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA_20"], name="SMA 20", line=dict(width=1.3)))
    if "SMA_50" in df:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA_50"], name="SMA 50", line=dict(width=1.3)))
    if "BB_Upper" in df:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_Upper"], name="BB Upper",
                                  line=dict(width=1, dash="dot"), opacity=0.5))
        fig.add_trace(go.Scatter(x=df["Date"], y=df["BB_Lower"], name="BB Lower",
                                  line=dict(width=1, dash="dot"), opacity=0.5,
                                  fill="tonexty", fillcolor="rgba(100,100,255,0.05)"))

    fig.update_layout(
        title=f"{ticker} — Price & Moving Averages",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def volume_chart(df, ticker: str):
    colors = ["#e74c3c" if row["Close"] < row["Open"] else "#2ecc71" for _, row in df.iterrows()]
    fig = go.Figure(go.Bar(x=df["Date"], y=df["Volume"], marker_color=colors, name="Volume"))
    fig.update_layout(title=f"{ticker} — Volume", template="plotly_dark", height=200,
                       margin=dict(t=40, b=20))
    return fig


def rsi_macd_chart(df, ticker: str):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                         subplot_titles=("RSI (14)", "MACD"))

    fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI_14"], name="RSI", line=dict(color="#f1c40f")), row=1, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=1, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=1, col=1)

    fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD"], name="MACD", line=dict(color="#3498db")), row=2, col=1)
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MACD_Signal"], name="Signal", line=dict(color="#e67e22")), row=2, col=1)
    fig.add_trace(go.Bar(x=df["Date"], y=df["MACD_Hist"], name="Histogram", marker_color="gray", opacity=0.4), row=2, col=1)

    fig.update_layout(template="plotly_dark", height=450, showlegend=True,
                       legend=dict(orientation="h", yanchor="bottom", y=1.05))
    return fig
