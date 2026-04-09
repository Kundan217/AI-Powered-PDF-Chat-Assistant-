from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


def load_pdf_text(pdf_files) -> str:
    """Extract raw text from one or multiple PDF files."""
    full_text = ""

    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                full_text += f"\n[Page {page_num + 1}]\n{text}"

    return full_text


def split_text_into_chunks(raw_text: str) -> list:
    """Split extracted text into overlapping chunks for better context retrieval."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_text(raw_text)
    return chunks


def process_pdfs(pdf_files) -> list:
    """Full pipeline: load PDFs → extract text → split into chunks."""

    if not pdf_files:
        return []

    raw_text = load_pdf_text(pdf_files)

    if not raw_text.strip():
        raise ValueError("No text could be extracted from the uploaded PDFs.")

    chunks = split_text_into_chunks(raw_text)
    return chunks