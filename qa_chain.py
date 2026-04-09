from langchain_google_genai import ChatGoogleGenerativeAI
from config import GEMINI_API_KEY, LLM_MODEL


def build_qa_chain(retriever):
    """Build a QA chain with Gemini."""

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=0.2,
        convert_system_message_to_human=True
    )

    def qa_chain(question):
        docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"""
You are a helpful assistant that answers questions based strictly on the provided PDF context.
If the answer is not found in the context, say "I couldn't find this in the uploaded documents."

Context:
{context}

Question: {question}
Answer:"""
        answer = llm.invoke(prompt)
        return {"answer": answer.content, "source_documents": docs}

    return qa_chain


def ask_question(qa_chain, question: str) -> dict:
    """Send a question through the QA chain and return answer."""

    return qa_chain(question)