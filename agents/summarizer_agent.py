"""
Summarizer Agent 🧠
The "fusion" agent — it doesn't look at raw data at all, only at the
other two agents' outputs, and synthesizes them into one coherent brief.
This mirrors how a real analyst desk works: specialists report in,
someone senior writes the final note.
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import config

SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    """You are the Lead Analyst Agent. Two specialist agents have reported their findings
on {ticker}. Synthesize their reports into ONE cohesive investment brief.

--- Technical Analyst Report ---
{technical_report}

--- News Analyst Report ---
{news_report}

Write a final brief with these exact sections, using markdown headers:
### Snapshot
One or two sentences combining the technical trend + news sentiment into a single read.

### Where They Agree
Points where the technical picture and news sentiment point the same direction.

### Where They Diverge
Any tension between what the chart shows and what the news suggests (this is often the most useful insight).

### Key Risk to Watch
The single most important thing that could change the picture.

Keep it tight — under 200 words total. Do not give buy/sell recommendations, only a synthesis of the two reports."""
)


class SummarizerAgent:
    """Fusion sub-agent: combines technical + news reports into one brief."""

    def __init__(self, api_key: str = None):
        self.llm = ChatGoogleGenerativeAI(
            model=config.LLM_MODEL_NAME,
            google_api_key=api_key or config.GOOGLE_API_KEY,
            temperature=0.4,
        )
        self.chain = SUMMARY_PROMPT | self.llm | StrOutputParser()

    def synthesize(self, ticker: str, technical_report: str, news_report: str) -> str:
        return self.chain.invoke({
            "ticker": ticker,
            "technical_report": technical_report,
            "news_report": news_report,
        })
