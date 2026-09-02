# Stock GenAI Analyst 📊

A Streamlit stock research app powered by a **multi-agent GenAI system** — same core stack
as the "Chat with Medical Reports" project (HuggingFace embeddings + FAISS + LangChain LCEL +
Google Gemini), applied to stocks.

## What it does

- Search any ticker → get price history, candlestick chart, moving averages, RSI, MACD, Bollinger Bands
- Click **Generate AI Report** → three specialized agents run in sequence:
  - **📈 Technical Analyst Agent** — reads the indicator snapshot, writes a technical read
  - **📰 News Analyst Agent** — RAG over recent news (Google News RSS → chunked → embedded → FAISS), writes a sentiment read
  - **🧠 Summarizer Agent** — fuses both into one final investment brief (agreements, divergences, key risk)
- **Agent Activity Log** — expandable panel showing exactly which agent did what and why (the orchestration trace)
- **Chat tab** — ask a free-form question; a **Supervisor/router agent** classifies it (technical / news / both) and
  delegates to the right specialist(s) live, same as the report generation but reactive to your question

## Architecture

```
stock_genai_app/
├── app.py                    # Streamlit UI, tabs, chat
├── config.py                 # API keys / model config
├── data/
│   ├── stock_data.py         # yfinance price history + company info
│   ├── indicators.py         # RSI, MACD, SMA/EMA, Bollinger, volatility
│   └── news_fetcher.py       # Google News RSS (free, no API key)
├── rag/
│   ├── vectorstore.py        # FAISS + HuggingFace embeddings
│   └── retriever.py
├── agents/
│   ├── technical_agent.py    # Specialist: price/indicators only
│   ├── news_agent.py         # Specialist: RAG over news
│   ├── summarizer_agent.py   # Fusion: combines the two reports
│   └── supervisor.py         # Orchestrator + router (the "agentic" core)
└── ui/
    ├── charts.py              # Plotly candlestick/RSI/MACD charts
    └── components.py          # RSI gauge, agent trace log renderer
```

## Setup

```bash
cd stock_genai_app
pip install -r requirements.txt
cp .env.example .env
# then edit .env and paste your free Gemini key from https://aistudio.google.com/apikey
streamlit run app.py
```

You can also paste the API key directly into the sidebar at runtime instead of using `.env`.

## Notes

- News comes from Google News RSS — completely free, no API key, no rate limit headaches.
- Indian tickers work too — use the `.NS` suffix (e.g. `TCS.NS`, `RELIANCE.NS`).
- The vectorstore and price data are cached (`st.cache_resource` / `st.cache_data`) so repeated
  chat questions don't re-fetch or re-embed everything.
- This is an educational project — all AI output includes a disclaimer and gives no buy/sell advice.

## Ideas to extend further

- Add a **Risk Agent** that flags earnings dates, high volatility, or upcoming macro events
- Compare two tickers side-by-side with the same agent pipeline
- Export the final brief as a PDF (reuse your PDF skill from the medical reports project)
- Swap the router for LangGraph if you want a more formal state-machine supervisor pattern
