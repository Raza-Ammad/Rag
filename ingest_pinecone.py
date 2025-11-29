import os
import glob
import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from pinecone import Pinecone


CHUNK_SIZE = 400
CHUNK_OVERLAP = 50


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


def extract_text_from_pdf(path: str) -> str:
    with open(path, "rb") as f:
        reader = PdfReader(f)
        texts = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            texts.append(page_text)
    return "\n".join(texts)


def load_text_from_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def init_pinecone_index():
    api_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX_NAME")

    if not api_key or not index_name:
        raise RuntimeError("PINECONE_API_KEY and PINECONE_INDEX_NAME must be set.")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    return index


def main():
    docs_folder = "docs"
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = init_pinecone_index()

    # Collect .txt and .pdf files from docs/
    text_files = glob.glob(os.path.join(docs_folder, "*.txt"))
    pdf_files = glob.glob(os.path.join(docs_folder, "*.pdf"))

    vectors = []

    # Process text files
    for path in text_files:
        print(f"Processing TXT: {path}")
        text = load_text_from_file(path)
        if not text.strip():
            continue
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            emb = model.encode(chunk, convert_to_numpy=True).astype("float32")
            vector_id = f"{os.path.basename(path)}-chunk-{i}"
            vectors.append(
                {
                    "id": vector_id,
                    "values": emb.tolist(),
                    "metadata": {
                        "source": os.path.basename(path),
                        "chunk": chunk,
                    },
                }
            )

    # Process pdf files
    for path in pdf_files:
        print(f"Processing PDF: {path}")
        text = extract_text_from_pdf(path)
        if not text.strip():
            continue
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            emb = model.encode(chunk, convert_to_numpy=True).astype("float32")
            vector_id = f"{os.path.basename(path)}-chunk-{i}"
            vectors.append(
                {
                    "id": vector_id,
                    "values": emb.tolist(),
                    "metadata": {
                        "source": os.path.basename(path),
                        "chunk": chunk,
                    },
                }
            )

    if not vectors:
        print("No content found to index.")
        return

    print(f"Upserting {len(vectors)} vectors to Pinecone...")
    # Upsert in batches so we don't blow up payload size
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
        print(f"Upserted {i + len(batch)}/{len(vectors)}")

    print("Done. All base documents indexed into Pinecone.")


if __name__ == "__main__":
    main()