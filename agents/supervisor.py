"""
Supervisor 🕹️
This is the "agentic" core of the app. It does two jobs:

1. generate_full_report(): runs Technical -> News -> Summarizer in sequence
   and logs every step it takes to a trace list (shown in the UI as the
   "Agent Activity Log" so you can literally demo the reasoning chain).

2. route_chat_question(): a lightweight router chain classifies an incoming
   chat question as needing TECHNICAL data, NEWS data, or BOTH, then
   delegates to the right specialist(s) — this is the "decides what to call"
   behavior that makes it agentic rather than a fixed pipeline.
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import config
from agents.technical_agent import TechnicalAnalystAgent
from agents.news_agent import NewsAnalystAgent
from agents.summarizer_agent import SummarizerAgent

ROUTER_PROMPT = ChatPromptTemplate.from_template(
    """Classify what kind of information is needed to answer this question about {ticker} stock.

Question: {question}

Reply with EXACTLY ONE word:
- TECHNICAL  (if it's about price, chart, indicators, trend, RSI, MACD, moving averages, support/resistance)
- NEWS       (if it's about news, events, sentiment, why the stock moved, announcements)
- BOTH       (if it genuinely needs both, e.g. "should I be worried about this stock right now")

Answer with just one word."""
)


class StockAnalysisSupervisor:
    """Coordinates the specialist agents. This is the orchestration layer."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or config.GOOGLE_API_KEY
        self.technical_agent = TechnicalAnalystAgent(api_key=self.api_key)
        self.news_agent = NewsAnalystAgent(api_key=self.api_key)
        self.summarizer_agent = SummarizerAgent(api_key=self.api_key)

        router_llm = ChatGoogleGenerativeAI(
            model=config.LLM_MODEL_NAME, google_api_key=self.api_key, temperature=0
        )
        self.router_chain = ROUTER_PROMPT | router_llm | StrOutputParser()

        self.trace = []  # list of {agent, action, detail} dicts for the UI

    def _log(self, agent: str, action: str, detail: str = ""):
        self.trace.append({"agent": agent, "action": action, "detail": detail})

    def generate_full_report(self, ticker: str, indicator_summary: dict, vectorstore) -> dict:
        """Runs the full pipeline: Technical -> News -> Summarizer."""
        self.trace = []

        self._log("Supervisor", "Dispatching to Technical Analyst Agent",
                   "Reason: full report requested, price/indicator data available.")
        technical_report = self.technical_agent.analyze(ticker, indicator_summary)
        self._log("Technical Analyst", "Completed analysis", f"{len(technical_report)} chars generated")

        self._log("Supervisor", "Dispatching to News Analyst Agent",
                   "Reason: full report requested, running RAG retrieval over news index.")
        news_report = self.news_agent.analyze_sentiment(ticker, vectorstore)
        self._log("News Analyst", "Completed RAG-based sentiment analysis", f"{len(news_report)} chars generated")

        self._log("Supervisor", "Dispatching to Summarizer Agent",
                   "Reason: fusing both specialist reports into final brief.")
        final_brief = self.summarizer_agent.synthesize(ticker, technical_report, news_report)
        self._log("Summarizer", "Completed synthesis", "Final investment brief ready")

        return {
            "technical_report": technical_report,
            "news_report": news_report,
            "final_brief": final_brief,
            "trace": self.trace,
        }

    def route_chat_question(self, ticker: str, question: str, indicator_summary: dict, vectorstore) -> dict:
        """
        Classifies the question and delegates to the right specialist(s).
        Returns the answer plus the trace of what was decided/called.
        """
        local_trace = []
        route = self.router_chain.invoke({"ticker": ticker, "question": question}).strip().upper()
        local_trace.append({"agent": "Supervisor", "action": f"Classified question as {route}", "detail": question})

        if "TECHNICAL" in route:
            summary_text = "\n".join(f"- {k}: {v}" for k, v in indicator_summary.items())
            prompt = f"Based on this technical data for {ticker}:\n{summary_text}\n\nAnswer concisely: {question}"
            answer = self.technical_agent.llm.invoke(prompt).content
            local_trace.append({"agent": "Technical Analyst", "action": "Answered from indicator data"})

        elif "NEWS" in route:
            answer = self.news_agent.answer_question(ticker, vectorstore, question)
            local_trace.append({"agent": "News Analyst", "action": "Answered via RAG retrieval over news"})

        else:  # BOTH
            summary_text = "\n".join(f"- {k}: {v}" for k, v in indicator_summary.items())
            tech_answer = self.technical_agent.llm.invoke(
                f"Technical data for {ticker}:\n{summary_text}\n\nBriefly answer: {question}"
            ).content
            local_trace.append({"agent": "Technical Analyst", "action": "Contributed technical angle"})

            news_answer = self.news_agent.answer_question(ticker, vectorstore, question)
            local_trace.append({"agent": "News Analyst", "action": "Contributed news angle via RAG"})

            answer = self.summarizer_agent.llm.invoke(
                f"Combine these two angles into one short answer to: '{question}'\n\n"
                f"Technical angle: {tech_answer}\n\nNews angle: {news_answer}"
            ).content
            local_trace.append({"agent": "Summarizer", "action": "Merged both angles into final answer"})

        return {"answer": answer, "trace": local_trace, "route": route}
