import streamlit as st
from pdf_processor import process_pdfs
from vector_store import create_vector_store, get_retriever
from qa_chain import build_qa_chain, ask_question

# ─── Page Config ───────────────────────────────────────────
st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="📄",
    layout="wide"
)

# ─── Title ─────────────────────────────────────────────────
st.title("📄 PDF Chat Assistant")
st.markdown("Upload one or more PDFs and ask questions about them.")

# ─── Session State Init ─────────────────────────────────────
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pdfs_processed" not in st.session_state:
    st.session_state.pdfs_processed = False

# ─── Sidebar: PDF Upload ────────────────────────────────────
with st.sidebar:
    st.header("📁 Upload PDFs")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    process_btn = st.button("⚡ Process PDFs", use_container_width=True)
    
    if process_btn:
        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
        else:
            with st.spinner("Processing PDFs..."):
                try:
                    # Step 1: Chunk PDFs
                    chunks = process_pdfs(uploaded_files)
                    st.success(f"✅ Extracted {len(chunks)} chunks")

                    # Step 2: Build vector store
                    with st.spinner("Creating embeddings..."):
                        vector_store = create_vector_store(chunks)
                        retriever = get_retriever(vector_store)
                    st.success("✅ Vector store ready")

                    # Step 3: Build QA chain
                    st.session_state.qa_chain = build_qa_chain(retriever)
                    st.session_state.pdfs_processed = True
                    st.session_state.chat_history = []
                    st.success("✅ Ready to chat!")

                except Exception as e:
                    st.error(f"Error: {str(e)}")

    if st.session_state.pdfs_processed:
        st.divider()
        if st.button("🗑️ Clear & Reset", use_container_width=True):
            st.session_state.qa_chain = None
            st.session_state.chat_history = []
            st.session_state.pdfs_processed = False
            st.rerun()

# ─── Main Chat Area ─────────────────────────────────────────
if not st.session_state.pdfs_processed:
    st.info("👈 Upload PDFs from the sidebar and click **Process PDFs** to begin.")
else:
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    user_question = st.chat_input("Ask a question about your PDFs...")

    if user_question:
        # Show user message
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })

        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = ask_question(st.session_state.qa_chain, user_question)
                answer = result["answer"]
                st.markdown(answer)

                # Show source chunks (expandable)
                if result["source_documents"]:
                    with st.expander("📚 Source Chunks Used"):
                        for i, doc in enumerate(result["source_documents"]):
                            st.markdown(f"**Chunk {i+1}:**")
                            st.caption(doc.page_content[:300] + "...")
                            st.divider()

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })