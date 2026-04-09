from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from config import GEMINI_API_KEY, EMBEDDING_MODEL, TOP_K_RESULTS


def create_vector_store(text_chunks: list):
    """Convert text chunks into embeddings and store in FAISS vector DB."""

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GEMINI_API_KEY
    )

    vector_store = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )

    return vector_store


def get_retriever(vector_store):
    """Return a retriever that fetches top-K relevant chunks for a query."""

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K_RESULTS}
    )

    return retriever