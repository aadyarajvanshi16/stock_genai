"""
Central config for the Stock GenAI app.
Loads Gemini API key from environment (.env) and defines model names
used across all agents, so there's a single place to change them.
"""
import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Same embedding model family used in the medical-reports RAG project
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Chat model used by every agent (kept cheap/fast; swap here if needed)
LLM_MODEL_NAME = "gemini-1.5-flash"

# RAG chunking config
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
RETRIEVER_K = 4

# How many news items to pull per ticker
MAX_NEWS_ITEMS = 12

DISCLAIMER = (
    "⚠️ This report is AI-generated for educational purposes only. "
    "It is NOT financial advice. Always do your own research before investing."
)
