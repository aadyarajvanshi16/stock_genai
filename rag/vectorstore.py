"""
Builds a FAISS vector store over a ticker's news articles using the same
HuggingFace sentence-transformer embeddings as your medical-reports project.
This is what lets the News Analyst Agent do retrieval instead of dumping
every headline into the prompt.
"""
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

import config


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    """Cached so the model loads from disk/HF hub only once per session."""
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)


def build_news_vectorstore(news_docs: list[str], ticker: str):
    """
    news_docs: list of flattened news text blobs (from news_fetcher.news_to_documents_text)
    Returns a FAISS vectorstore, or None if there's no news to index.
    """
    if not news_docs:
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    documents = [
        Document(page_content=doc, metadata={"ticker": ticker, "chunk_id": i})
        for i, doc in enumerate(news_docs)
    ]
    chunks = splitter.split_documents(documents)

    embeddings = get_embedding_model()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore
