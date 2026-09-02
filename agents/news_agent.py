"""
News Analyst Agent 📰
This is the direct sibling of your medical-reports "chat with PDF" chain —
same RAG pattern, just retrieving news chunks from FAISS instead of
medical report chunks, then reasoning over them with Gemini.
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import config
from rag.retriever import retrieve_context

SENTIMENT_PROMPT = ChatPromptTemplate.from_template(
    """You are a News Analyst Agent inside a multi-agent stock research system.
You ONLY analyze recent news — you never look at price charts or indicators.

Ticker: {ticker}
Recent news excerpts:
{context}

Based ONLY on the excerpts above, write a concise news analysis (5-7 sentences) covering:
1. Overall sentiment (positive / negative / mixed / neutral) and why.
2. The 2-3 most significant recent events or headlines mentioned.
3. Any risks or catalysts investors are discussing in this coverage.

If the excerpts are too sparse or generic to say much, say so honestly instead of making things up."""
)

CHAT_PROMPT = ChatPromptTemplate.from_template(
    """You are a News Analyst Agent for {ticker}. Answer the user's question using ONLY
the news context below. If the context doesn't contain the answer, say you don't have
enough recent news coverage to answer that.

News context:
{context}

Question: {question}

Answer clearly and concisely."""
)


class NewsAnalystAgent:
    """Specialized sub-agent: RAG-based news sentiment + Q&A."""

    def __init__(self, api_key: str = None):
        self.llm = ChatGoogleGenerativeAI(
            model=config.LLM_MODEL_NAME,
            google_api_key=api_key or config.GOOGLE_API_KEY,
            temperature=0.3,
        )
        self.sentiment_chain = SENTIMENT_PROMPT | self.llm | StrOutputParser()
        self.chat_chain = CHAT_PROMPT | self.llm | StrOutputParser()

    def analyze_sentiment(self, ticker: str, vectorstore) -> str:
        if vectorstore is None:
            return "No recent news articles were found for this ticker, so a news sentiment read isn't available right now."
        context = retrieve_context(vectorstore, query=f"{ticker} stock recent news events sentiment", k=6)
        return self.sentiment_chain.invoke({"ticker": ticker, "context": context})

    def answer_question(self, ticker: str, vectorstore, question: str) -> str:
        if vectorstore is None:
            return "No news index is available for this ticker yet — try generating the report first."
        context = retrieve_context(vectorstore, query=question)
        return self.chat_chain.invoke({"ticker": ticker, "context": context, "question": question})
