"""
Thin wrapper around the vectorstore retriever so agents don't need to know
FAISS/LangChain internals directly.
"""
import config


def get_retriever(vectorstore, k: int = None):
    if vectorstore is None:
        return None
    return vectorstore.as_retriever(search_kwargs={"k": k or config.RETRIEVER_K})


def retrieve_context(vectorstore, query: str, k: int = None) -> str:
    """Returns retrieved chunks joined into one context string for a prompt."""
    retriever = get_retriever(vectorstore, k)
    if retriever is None:
        return ""
    docs = retriever.invoke(query)
    return "\n\n---\n\n".join(d.page_content for d in docs)
