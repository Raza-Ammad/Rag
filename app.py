import os
import streamlit as st
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from PyPDF2 import PdfReader

INDEX_FILE = "vector_index.faiss"  # created by ingest.py
META_FILE = "metadata.csv"         # created by ingest.py
TOP_K = 5

CHUNK_SIZE = 400      # characters
CHUNK_OVERLAP = 50    # characters


def load_index_and_meta():
    """Load FAISS index and metadata from disk."""
    index = faiss.read_index(INDEX_FILE)
    meta = pd.read_csv(META_FILE)
    return index, meta


@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def extract_text_from_pdf(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    texts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        texts.append(page_text)
    return "\n".join(texts)


def retrieve(index, meta, model, query, top_k=TOP_K, source_filter=None):
    """Retrieve top-k chunks, optionally filtered by source filename."""
    query_emb = model.encode(query, convert_to_numpy=True).astype("float32")
    query_emb = np.expand_dims(query_emb, axis=0)

    distances, indices = index.search(query_emb, top_k)
    indices = indices[0]

    results = []
    for i in indices:
        row = meta.iloc[i]
        if source_filter is not None and row["source"] != source_filter:
            # skip chunks from other documents if a filter is applied
            continue
        results.append(
            {
                "source": row["source"],
                "chunk": row["chunk"],
            }
        )
    return results


def build_prompt(chunks, question, source_filter=None):
    if not chunks:
        context_str = "No relevant context found."
    else:
        context_texts = [c["chunk"] for c in chunks]
        context_str = "\n\n---\n\n".join(context_texts)

    doc_info = (
        f"Only use information from the document '{source_filter}'. "
        if source_filter is not None
        else "The context may contain multiple documents. Do not introduce any information that is not clearly present in the context. "
    )

    prompt = f"""You are a helpful assistant answering questions based only on the provided context.
{doc_info}
If the context does not contain the answer, say you don't know.
Do NOT add extra facts, names, IDs, module codes, or services that are not explicitly in the context.

Context:
{context_str}

Question: {question}

Answer:"""
    return prompt



def init_gemini_model():
    # Try Streamlit secrets first (for Streamlit Cloud), then environment variable (local)
    api_key = None
    try:
        # st.secrets is only available on Streamlit Cloud
        api_key = st.secrets.get("GOOGLE_API_KEY", None)
    except Exception:
        api_key = os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        # Small debug helper – shows what secret keys exist in the sidebar
        try:
            st.sidebar.write("Secrets keys:", list(st.secrets.keys()))
        except Exception:
            st.sidebar.write("Secrets not available in this environment")
        return None, "API Key not found. Please set GOOGLE_API_KEY in environment or Streamlit secrets."

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-flash-latest")
    return model, None


def call_gemini(model, prompt: str) -> str:
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error calling Gemini API: {e}"


def main():
    st.set_page_config(page_title="Saad RAG Demo", page_icon="💬")
    st.title("Saad RAG Demo – Gemini + FAISS (Local + PDF Upload + Chat)")

    st.write(
        "This app uses a local FAISS index for retrieval and the Gemini API "
        "for generating answers based on your documents. You can also upload PDFs "
        "to add them to the knowledge base. The chat below keeps your history "
        "for this session."
    )

    # Load index + metadata fresh on each run so PDF updates are reflected
    index, meta = load_index_and_meta()
    emb_model = get_embedding_model()
    gemini_model, gemini_error = init_gemini_model()

    # ---- PDF Upload Section ----
    st.subheader("Upload PDFs")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files to add to the knowledge base",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Process and index uploaded PDFs", key="upload_btn"):
        with st.spinner("Reading and indexing PDFs..."):
            new_rows = []
            new_embeddings = []

            for uploaded_file in uploaded_files:
                pdf_text = extract_text_from_pdf(uploaded_file)
                if not pdf_text.strip():
                    continue

                chunks = chunk_text(pdf_text)
                for chunk in chunks:
                    new_rows.append({"source": uploaded_file.name, "chunk": chunk})
                    emb = emb_model.encode(chunk, convert_to_numpy=True)
                    new_embeddings.append(emb)

            if new_embeddings:
                # Add new vectors to FAISS
                new_embeddings = np.vstack(new_embeddings).astype("float32")
                index.add(new_embeddings)

                # Build new metadata DataFrame and concatenate
                meta_new = pd.DataFrame(new_rows)
                meta = pd.concat([meta, meta_new], ignore_index=True)

                # Persist updated index + metadata to disk
                faiss.write_index(index, INDEX_FILE)
                meta.to_csv(META_FILE, index=False)

                st.success("Uploaded PDFs have been indexed and added to the knowledge base.")
            else:
                st.warning("No text could be extracted from the uploaded PDFs.")

    # ---- Document Filter ----
    st.subheader("Document filter (optional)")
    sources = sorted(meta["source"].unique().tolist())
    doc_options = ["All documents"] + sources
    selected = st.selectbox("Restrict answers to a specific document:", doc_options)
    source_filter = None if selected == "All documents" else selected

    # ---- Chat History Setup ----
    if "history" not in st.session_state:
        st.session_state["history"] = []  # list of {"role": "user"/"assistant", "content": str}

    st.subheader("Chat with your documents")

    if gemini_error:
        st.error(gemini_error)
        st.info(
            "Set GOOGLE_API_KEY as an environment variable locally, or add it to Streamlit secrets "
            "when deploying to Streamlit Cloud."
        )

    # Display previous messages
    for msg in st.session_state["history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input (new question)
    user_input = st.chat_input("Ask a question about your documents...")

    if user_input:
        # Add user message to history
        st.session_state["history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # RAG retrieval + Gemini answer
        with st.chat_message("assistant"):
            if gemini_model is None:
                answer = "Gemini model is not configured yet. Unable to generate an answer."
                st.warning(answer)
            else:
                with st.spinner("Retrieving relevant chunks and generating answer..."):
                    chunks = retrieve(index, meta, emb_model, user_input, TOP_K, source_filter=source_filter)
                    prompt = build_prompt(chunks, user_input, source_filter=source_filter)
                    answer = call_gemini(gemini_model, prompt)

                    # Show retrieved context in expandable sections
                    if chunks:
                        st.markdown("**Retrieved context:**")
                        for i, c in enumerate(chunks, start=1):
                            with st.expander(f"Chunk {i} – {c['source']}"):
                                st.write(c["chunk"])
                    else:
                        st.markdown("_No relevant context found for this question._")

                    st.markdown("**Answer:**")
                    st.write(answer)

            # Add assistant answer to history
            st.session_state["history"].append(
                {"role": "assistant", "content": answer}
            )


if __name__ == "__main__":
    main()