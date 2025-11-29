# rag.py
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

INDEX_FILE = "vector_index.faiss"
META_FILE = "metadata.csv"
TOP_K = 5


def load_index_and_meta():
    index = faiss.read_index(INDEX_FILE)
    meta = pd.read_csv(META_FILE)
    return index, meta


def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def retrieve(index, meta, model, query, top_k=TOP_K):
    query_emb = model.encode(query, convert_to_numpy=True).astype("float32")
    query_emb = np.expand_dims(query_emb, axis=0)

    distances, indices = index.search(query_emb, top_k)
    indices = indices[0]

    results = []
    for i in indices:
        row = meta.iloc[i]
        results.append(
            {
                "source": row["source"],
                "chunk": row["chunk"],
            }
        )
    return results


def build_prompt(chunks, question):
    context_texts = [c["chunk"] for c in chunks]
    context_str = "\n\n---\n\n".join(context_texts)

    prompt = f"""You are a helpful assistant answering questions based on the provided context.
Use only the information in the context. If the answer is not there, say you don't know.

Context:
{context_str}

Question: {question}

Answer:"""
    return prompt


def call_llm(prompt: str) -> str:
    # Placeholder for now
    return "LLM (Gemini) would answer here based on the retrieved context. (Placeholder)"


def main():
    print("Loading index and metadata...")
    index, meta = load_index_and_meta()

    print("Loading embedding model...")
    model = get_embedding_model()

    while True:
        question = input("\nAsk a question (or type 'exit'): ").strip()
        if question.lower() in ("exit", "quit"):
            break

        print("Retrieving relevant chunks...")
        chunks = retrieve(index, meta, model, question, TOP_K)

        print("\nTop retrieved context chunks:")
        for i, c in enumerate(chunks, start=1):
            print(f"\n[{i}] Source: {c['source']}")
            print(c["chunk"][:300], "..." if len(c["chunk"]) > 300 else "")

        prompt = build_prompt(chunks, question)
        answer = call_llm(prompt)

        print("\n--- Answer ---")
        print(answer)
        print("--------------")


if __name__ == "__main__":
    main()