"""
Stock GenAI Analyst — main Streamlit entry point.

Tabs:
  1. Overview & Charts   — raw price data, candlestick, RSI/MACD
  2. AI Analysis         — runs the 3-agent supervisor pipeline, shows the
                            Agent Activity Log so you can see the orchestration
  3. Chat with Stock      — RAG + routed chat, same pattern as the medical
                            reports project but with a routing supervisor
"""
import streamlit as st

import config
from data.stock_data import fetch_price_history, fetch_company_info, format_market_cap
from data.indicators import calculate_all_indicators, get_technical_summary
from data.news_fetcher import fetch_news, news_to_documents_text
from rag.vectorstore import build_news_vectorstore
from agents.supervisor import StockAnalysisSupervisor
from ui.charts import candlestick_with_ma, volume_chart, rsi_macd_chart
from ui.components import rsi_gauge, render_agent_trace, metric_cards

st.set_page_config(page_title="Stock GenAI Analyst", page_icon="📊", layout="wide")


# ---------- Sidebar ----------
with st.sidebar:
    st.title("📊 Stock GenAI Analyst")
    st.caption("Multi-agent stock research — technical + news + AI synthesis")

    api_key_input = st.text_input(
        "Google Gemini API Key",
        value=config.GOOGLE_API_KEY,
        type="password",
        help="Get a free key at aistudio.google.com/apikey. Not stored anywhere.",
    )

    ticker = st.text_input("Ticker symbol", value="AAPL", help="e.g. AAPL, TSLA, RELIANCE.NS, TCS.NS").upper().strip()
    period = st.selectbox("History period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=2)

    st.divider()
    generate_clicked = st.button("🚀 Generate AI Report", use_container_width=True, type="primary")
    st.caption("Runs Technical → News → Summarizer agents (uses API calls).")


if not ticker:
    st.info("Enter a ticker symbol in the sidebar to get started.")
    st.stop()


# ---------- Fetch data ----------
try:
    with st.spinner(f"Fetching price history for {ticker}..."):
        price_df = fetch_price_history(ticker, period=period)
        price_df = calculate_all_indicators(price_df)
        indicator_summary = get_technical_summary(price_df)
    with st.spinner("Fetching company info..."):
        company_info = fetch_company_info(ticker)
except ValueError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Something went wrong fetching data: {e}")
    st.stop()


# ---------- Header ----------
st.header(f"{company_info['name']} ({ticker})")
st.caption(f"{company_info['sector']} · {company_info['industry']}")
metric_cards(company_info, format_market_cap)

tab_overview, tab_ai, tab_chat = st.tabs(["📈 Overview & Charts", "🤖 AI Analysis (Agents)", "💬 Chat with Stock"])


# ---------- Tab 1: Overview & Charts ----------
with tab_overview:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(rsi_gauge(indicator_summary["rsi_14"]), use_container_width=True)
    with col2:
        st.markdown("#### Quick read")
        st.write(f"**Trend:** {indicator_summary['trend'].title()}")
        st.write(f"**RSI zone:** {indicator_summary['rsi_flag'].title()}")
        st.write(f"**MACD:** {indicator_summary['macd_condition'].title()}")
        st.write(f"**Bollinger position:** {indicator_summary['bollinger_position'].title()}")
        st.write(f"**Volatility (std dev of daily returns):** {indicator_summary['volatility_pct_std']}%")

    st.plotly_chart(candlestick_with_ma(price_df, ticker), use_container_width=True)
    st.plotly_chart(volume_chart(price_df, ticker), use_container_width=True)
    st.plotly_chart(rsi_macd_chart(price_df, ticker), use_container_width=True)

    with st.expander("Raw data"):
        st.dataframe(price_df.tail(30), use_container_width=True)


# ---------- Build news + vectorstore (needed by tabs 2 & 3) ----------
@st.cache_resource(show_spinner=False)
def _get_vectorstore(ticker_key: str, company_name: str):
    news_items = fetch_news(f"{ticker_key} {company_name}", max_items=config.MAX_NEWS_ITEMS)
    docs = news_to_documents_text(news_items)
    vs = build_news_vectorstore(docs, ticker_key)
    return vs, news_items


# ---------- Tab 2: AI Analysis ----------
with tab_ai:
    if not api_key_input:
        st.warning("Enter your Gemini API key in the sidebar to run the AI agents.")
    elif generate_clicked:
        with st.spinner("Indexing recent news for RAG..."):
            vectorstore, news_items = _get_vectorstore(ticker, company_info["name"])

        supervisor = StockAnalysisSupervisor(api_key=api_key_input)
        with st.spinner("Agents working: Technical Analyst → News Analyst → Summarizer..."):
            result = supervisor.generate_full_report(ticker, indicator_summary, vectorstore)

        st.session_state["last_report"] = result
        st.session_state["last_ticker"] = ticker

    if "last_report" in st.session_state and st.session_state.get("last_ticker") == ticker:
        result = st.session_state["last_report"]

        st.markdown("### 🧠 Final Investment Brief")
        st.markdown(result["final_brief"])
        st.info(config.DISCLAIMER)

        col1, col2 = st.columns(2)
        with col1:
            with st.expander("📈 Full Technical Analyst report"):
                st.write(result["technical_report"])
        with col2:
            with st.expander("📰 Full News Analyst report"):
                st.write(result["news_report"])

        with st.expander("🕹️ Agent Activity Log — see how the agents coordinated", expanded=False):
            render_agent_trace(result["trace"])
    elif not generate_clicked:
        st.info("Click **Generate AI Report** in the sidebar to run the multi-agent pipeline.")


# ---------- Tab 3: Chat ----------
with tab_chat:
    st.caption("Ask about price action, news, or both — a router agent decides which specialist(s) to call.")

    if not api_key_input:
        st.warning("Enter your Gemini API key in the sidebar to chat.")
    else:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                if msg.get("trace"):
                    with st.expander("🕹️ Which agent(s) answered this?"):
                        render_agent_trace(msg["trace"])

        user_question = st.chat_input(f"Ask something about {ticker}...")
        if user_question:
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.write(user_question)

            vectorstore, _ = _get_vectorstore(ticker, company_info["name"])
            supervisor = StockAnalysisSupervisor(api_key=api_key_input)

            with st.chat_message("assistant"):
                with st.spinner("Routing to the right agent(s)..."):
                    result = supervisor.route_chat_question(ticker, user_question, indicator_summary, vectorstore)
                st.write(result["answer"])
                with st.expander("🕹️ Which agent(s) answered this?"):
                    render_agent_trace(result["trace"])

            st.session_state.chat_history.append({
                "role": "assistant", "content": result["answer"], "trace": result["trace"]
            })
