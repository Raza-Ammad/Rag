# ingest.py
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss

DOCS_DIR = "docs"
INDEX_FILE = "vector_index.faiss"
META_FILE = "metadata.csv"

CHUNK_SIZE = 400  # characters
CHUNK_OVERLAP = 50  # characters


def load_docs(docs_dir=DOCS_DIR):
    docs = []
    for path in glob.glob(os.path.join(docs_dir, "**", "*.txt"), recursive=True):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        docs.append({"path": path, "text": text})
    return docs


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]


def build_corpus(docs):
    rows = []
    for doc in docs:
        chunks = chunk_text(doc["text"])
        for chunk in chunks:
            rows.append({"source": doc["path"], "chunk": chunk})
    return pd.DataFrame(rows)


def main():
    print("Loading documents...")
    docs = load_docs()
    if not docs:
        print("No .txt files found in docs/ folder.")
        return

    print(f"Loaded {len(docs)} documents.")
    df = build_corpus(docs)
    print(f"Created {len(df)} text chunks.")

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Creating embeddings...")
    embeddings = []
    for text in tqdm(df["chunk"].tolist()):
        emb = model.encode(text, convert_to_numpy=True)
        embeddings.append(emb)

    embeddings = np.vstack(embeddings).astype("float32")

    print("Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    print(f"Saving index to {INDEX_FILE} and metadata to {META_FILE}...")
    faiss.write_index(index, INDEX_FILE)
    df.to_csv(META_FILE, index=False)

    print("Done! Ingestion complete.")


if __name__ == "__main__":
    main()