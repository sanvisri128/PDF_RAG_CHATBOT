import os

import streamlit as st
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError

from rag_backend import PDFRAGBackend, RAGError


load_dotenv()

st.set_page_config(page_title="PDF RAG Chatbot", layout="wide")


def get_google_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if api_key:
        return api_key
    try:
        return str(st.secrets.get("GOOGLE_API_KEY", "") or "").strip()
    except StreamlitSecretNotFoundError:
        return ""


@st.cache_resource
def get_backend() -> PDFRAGBackend:
    return PDFRAGBackend()


st.title("PDF RAG Chatbot")
st.caption("Upload PDFs and ask questions grounded in their contents.")

try:
    backend = get_backend()
except Exception as exc:
    st.error(f"Could not initialize the RAG backend: {exc}")
    st.stop()

with st.sidebar:
    st.header("Documents")
    if not get_google_api_key():
        st.warning(
            "Gemini is not configured. Add GOOGLE_API_KEY to .env or Streamlit Cloud secrets."
        )
    st.caption("PDFs are limited to 200 MB each. Scanned PDFs require OCR before they can be searched.")
    uploads = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Files are validated and saved locally in the data/ directory.",
    )
    if st.button("Index uploaded PDFs", disabled=not uploads, use_container_width=True):
        uploaded_files = uploads or []
        progress = st.progress(0, text="Preparing uploads...")
        for index, upload in enumerate(uploaded_files):
            progress.progress(index / len(uploaded_files), text=f"Processing {upload.name}...")
            try:
                with st.spinner(f"Extracting and indexing {upload.name}..."):
                    result = backend.ingest_pdf(upload.name, upload.getvalue())
                    if result.already_indexed:
                        st.info(f"{result.filename} is already indexed.")
                    else:
                        st.success(f"Indexed {result.filename} ({result.chunk_count} chunks).")
            except RAGError as exc:
                st.error(f"{upload.name}: {exc}")
            except Exception as exc:
                st.error(f"Could not index {upload.name}: {exc}")
            finally:
                progress.progress((index + 1) / len(uploaded_files))
        progress.empty()

    try:
        indexed_sources = backend.list_sources()
    except Exception as exc:
        st.error(f"Could not load the indexed document list: {exc}")
        indexed_sources = []

    if indexed_sources:
        st.subheader("Indexed PDFs")
        st.caption(f"{len(indexed_sources)} document(s) in the knowledge base")
        for source in indexed_sources:
            st.write(f"- {source}")
    else:
        st.info("No PDFs have been indexed yet.")

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for source in message.get("sources", []):
            st.caption(f"Source: {source['filename']} - page {source['page']}")

question = st.chat_input("Ask a question about your indexed PDFs")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    api_key = get_google_api_key()
    with st.chat_message("assistant"):
        try:
            if not api_key:
                raise RAGError(
                    "A Google Gemini API key is required. Configure GOOGLE_API_KEY in .env "
                    "or Streamlit Cloud secrets."
                )
            with st.spinner("Searching your PDFs and generating an answer..."):
                answer, source_documents = backend.ask(question, api_key)
            st.markdown(answer)
            sources = [
                {
                    "filename": document.metadata["filename"],
                    "page": document.metadata["page"],
                }
                for document in source_documents
            ]
            for source in sources:
                st.caption(f"Source: {source['filename']} - page {source['page']}")
            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except RAGError as exc:
            message = str(exc)
            st.error(message)
            st.session_state.messages.append({"role": "assistant", "content": message})
        except Exception as exc:
            details = str(exc).replace(api_key, "[redacted]") if api_key else str(exc)
            message = f"An error occurred while answering your question: {details}"
            st.error(message)
            st.session_state.messages.append({"role": "assistant", "content": message})
