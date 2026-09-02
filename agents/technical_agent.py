"""
Technical Analyst Agent 📈
Reads the numeric indicator summary (RSI, MACD, moving averages, etc.)
and writes a plain-English technical read on the stock, using the same
LCEL pattern (prompt | llm | parser) as your medical reports project.
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import config

TECHNICAL_PROMPT = ChatPromptTemplate.from_template(
    """You are a Technical Analyst Agent inside a multi-agent stock research system.
You ONLY look at price action and indicators — you never speculate about news or fundamentals.

Ticker: {ticker}
Indicator snapshot (most recent trading day):
{indicator_summary}

Write a concise technical analysis (5-7 sentences) covering:
1. The current trend (using the moving averages) and today's move.
2. What RSI is signaling (overbought/oversold/neutral) and what that implies.
3. What the MACD crossover suggests about momentum.
4. Where price sits relative to the Bollinger Bands and what that implies about volatility.
5. One concrete thing a trader should watch next (a level or signal), phrased as an observation, not investment advice.

Be specific with the actual numbers given. Do not give buy/sell recommendations."""
)


class TechnicalAnalystAgent:
    """Specialized sub-agent: technical/price-action analysis only."""

    def __init__(self, api_key: str = None):
        self.llm = ChatGoogleGenerativeAI(
            model=config.LLM_MODEL_NAME,
            google_api_key=api_key or config.GOOGLE_API_KEY,
            temperature=0.3,
        )
        self.chain = TECHNICAL_PROMPT | self.llm | StrOutputParser()

    def analyze(self, ticker: str, indicator_summary: dict) -> str:
        summary_text = "\n".join(f"- {k}: {v}" for k, v in indicator_summary.items())
        return self.chain.invoke({"ticker": ticker, "indicator_summary": summary_text})
