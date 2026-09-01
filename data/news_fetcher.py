"""
Pulls recent news headlines/snippets for a ticker using Google News RSS.
No API key needed (unlike NewsAPI), which keeps the app free to run —
same free-tooling philosophy as your medical reports project.
"""
import feedparser
import streamlit as st
from urllib.parse import quote_plus


@st.cache_data(ttl=1800, show_spinner=False)  # cache 30 min
def fetch_news(query: str, max_items: int = 12) -> list[dict]:
    """
    query: usually "TICKER company name stock"
    Returns list of {title, summary, link, published}
    """
    encoded_query = quote_plus(f"{query} stock")
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"

    feed = feedparser.parse(url)
    items = []
    for entry in feed.entries[:max_items]:
        items.append({
            "title": getattr(entry, "title", ""),
            "summary": getattr(entry, "summary", ""),
            "link": getattr(entry, "link", ""),
            "published": getattr(entry, "published", ""),
            "source": entry.get("source", {}).get("title", "") if hasattr(entry, "source") else "",
        })
    return items


def news_to_documents_text(news_items: list[dict]) -> list[str]:
    """
    Flattens each news item into a single text blob suitable for chunking
    + embedding in the vector store.
    """
    docs = []
    for item in news_items:
        text = f"Title: {item['title']}\nSource: {item.get('source', 'Unknown')}\nPublished: {item.get('published', '')}\nSummary: {item['summary']}"
        docs.append(text)
    return docs
