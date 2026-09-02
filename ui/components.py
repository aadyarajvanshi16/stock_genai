"""
Small reusable UI pieces: the RSI gauge and the Agent Activity Log renderer
(the creative bit that visualizes the agentic system actually "thinking").
"""
import streamlit as st
import plotly.graph_objects as go


def rsi_gauge(rsi_value: float):
    """Speedometer-style gauge for RSI, colored by zone."""
    if rsi_value >= 70:
        color = "#e74c3c"
    elif rsi_value <= 30:
        color = "#2ecc71"
    else:
        color = "#3498db"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rsi_value,
        title={"text": "RSI (Momentum)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 30], "color": "rgba(46,204,113,0.2)"},
                {"range": [30, 70], "color": "rgba(52,152,219,0.15)"},
                {"range": [70, 100], "color": "rgba(231,76,60,0.2)"},
            ],
        },
    ))
    fig.update_layout(height=250, margin=dict(t=40, b=10, l=20, r=20), template="plotly_dark")
    return fig


def render_agent_trace(trace: list[dict]):
    """Renders the step-by-step agent reasoning trace as a timeline."""
    agent_icons = {
        "Supervisor": "🕹️",
        "Technical Analyst": "📈",
        "News Analyst": "📰",
        "Summarizer": "🧠",
    }
    for step in trace:
        icon = agent_icons.get(step["agent"], "🤖")
        with st.container():
            st.markdown(f"**{icon} {step['agent']}** — {step['action']}")
            if step.get("detail"):
                st.caption(step["detail"])


def metric_cards(info: dict, format_market_cap_fn):
    cols = st.columns(4)
    price = info.get("current_price")
    prev = info.get("previous_close")
    delta = None
    if price and prev:
        delta = f"{((price - prev) / prev) * 100:.2f}%"

    cols[0].metric("Current Price", f"{info.get('currency', '')} {price}" if price else "N/A", delta)
    cols[1].metric("Market Cap", format_market_cap_fn(info.get("market_cap")))
    cols[2].metric("P/E Ratio", round(info["pe_ratio"], 2) if info.get("pe_ratio") else "N/A")
    cols[3].metric("52W Range",
                    f"{info.get('52w_low', 'N/A')} - {info.get('52w_high', 'N/A')}")
