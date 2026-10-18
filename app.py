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
CHAT_MODEL = "llama-3.3-70b-versatile"
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
    # Embeddings are normalized, so dot product equals cosine similarity.
    scores = embeddings @ query_embedding
    best_indices = np.argsort(scores)[::-1][:top_k]
    return [
        {**chunks[int(i)], "score": float(scores[int(i)])}
        for i in best_indices
    ]


def answer_question(client, question: str, passages):
    context = "\n\n".join(
        f"[Page {p['page']} | relevance {p['score']:.3f}]\n{p['text']}"
        for p in passages
    )
    system_prompt = """You are an HR policy assistant.
Answer using only the supplied excerpts from the user's HR Policy PDF.
Do not invent policy details or fill gaps with general knowledge.
If the excerpts do not contain the answer, say: "I couldn't find that information in the provided HR Policy document."
Clearly distinguish exact policy wording from a cautious interpretation.
Keep the answer helpful and concise. Mention the relevant page number(s).
The document may contain old dates or policy statements; report what it says rather than claiming it is current law."""
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nRetrieved document excerpts:\n{context}",
            },
        ],
    )
    return response.choices[0].message.content or "No answer was returned."


api_key = get_api_key()
if not api_key:
    st.warning(
        "Add your Groq API key as `GROQ_API_KEY` in Streamlit Cloud → "
        "App settings → Secrets. Do not put the key directly in this Python file."
    )
    st.code('GROQ_API_KEY = "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"', language="toml")
    st.stop()

if not DOCUMENT_PATH.exists():
    st.error(
        "The document HR-Policy.pdf was not found. Upload it alongside app.py "
        "in the same repository/folder."
    )
    st.stop()

try:
    chunks, embeddings = build_index(
        str(DOCUMENT_PATH.resolve()), DOCUMENT_PATH.stat().st_mtime
    )
    embedding_model = load_embedding_model()
except Exception as exc:
    st.error(f"Could not read or index the PDF: {exc}")
    st.stop()

client = Groq(api_key=api_key)

with st.sidebar:
    st.header("Document")
    st.write("**HR-Policy.pdf**")
    st.write(f"Indexed text chunks: {len(chunks)}")
    st.write("**Chat model:**")
    st.code(CHAT_MODEL)
    st.caption(
        "The PDF is indexed on the server. Retrieved passages are sent to Groq "
        "to generate an answer."
    )
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Retrieved document passages"):
                for source in message["sources"]:
                    st.markdown(f"**Page {source['page']}** · relevance `{source['score']:.3f}`")
                    st.write(source["text"])

question = st.chat_input("Ask about leave, attendance, travel, dress code, or another HR policy...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the document and preparing an answer..."):
            try:
                passages = retrieve(question, chunks, embeddings, embedding_model)
                answer = answer_question(client, question, passages)
                st.markdown(answer)
                with st.expander("Retrieved document passages"):
                    for source in passages:
                        st.markdown(
                            f"**Page {source['page']}** · relevance `{source['score']:.3f}`"
                        )
                        st.write(source["text"])
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": passages,
                    }
                )
            except Exception as exc:
                st.error(f"Request failed: {exc}")
                st.caption(
                    "Check that your Groq key is valid, the model is available, "
                    "and your app has internet access."
                )
