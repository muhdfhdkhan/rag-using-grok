import os
from pathlib import Path

import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq

# ----------------------- App configuration -----------------------
st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📘",
    layout="wide",
)

st.title("📘 HR Policy Assistant")
st.caption("Ask questions about the uploaded HR Policy document. Answers are grounded in retrieved document passages.")

DOCUMENT_PATH = Path(__file__).parent / "HR-Policy.pdf"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_MODEL = "openai/gpt-oss-20b"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 180
TOP_K = 4


def get_api_key() -> str:
    """Read the Groq API key from Streamlit Secrets or environment variables."""
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        key = ""
    return key or os.getenv("GROQ_API_KEY", "")


@st.cache_resource(show_spinner="Loading the local embedding model...")
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def extract_pdf_pages(pdf_path: Path):
    reader = PdfReader(str(pdf_path))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append({"page": page_number, "text": text})
    return pages


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split extracted text into overlapping chunks, retaining readable boundaries."""
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start + int(chunk_size * 0.65):
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


@st.cache_resource(show_spinner="Reading and indexing HR-Policy.pdf...")
def build_index(pdf_path_string: str, file_modified_time: float):
    pdf_path = Path(pdf_path_string)
    pages = extract_pdf_pages(pdf_path)
    chunks = []
    for page in pages:
        for chunk in split_text(page["text"]):
            chunks.append({"page": page["page"], "text": chunk})

    if not chunks:
        raise ValueError(
            "No selectable text was found in HR-Policy.pdf. If it is a scanned PDF, run OCR first."
        )

    model = load_embedding_model()
    embeddings = model.encode(
        [item["text"] for item in chunks],
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return chunks, np.asarray(embeddings, dtype=np.float32)


def retrieve(question: str, chunks, embeddings, model, top_k: int = TOP_K):
    query_embedding = model.encode([question], normalize_embeddings=True)[0]
