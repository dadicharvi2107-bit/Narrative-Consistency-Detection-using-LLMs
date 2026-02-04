import os
import re
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer


# =========================
# Configuration
# =========================

DATA_DIR = "data"
BOOKS_DIR = os.path.join(DATA_DIR, "Books")
OUTPUT_DIR = "outputs"

MODEL_NAME = "all-MiniLM-L6-v2"
NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
LLM_MODEL = "meta/llama-3.1-8b-instruct"

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")


# =========================
# Utility Functions
# =========================

def load_books():
    books = {}
    for fname in os.listdir(BOOKS_DIR):
        path = os.path.join(BOOKS_DIR, fname)
        with open(path, encoding="utf-8") as f:
            books[fname.lower().replace(".txt", "")] = f.read()
    return books


def chunk_text(text, chunk_size=800, overlap=100):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start = max(0, end - overlap)

    return chunks


def split_claims(text):
    return [
        c.strip()
        for c in re.split(r"[.;]", text)
        if len(c.strip()) > 20
    ]


def shorten(text, max_words=200):
    return " ".join(text.split()[:max_words])


# =========================
# Retrieval
# =========================

def build_index(books, embedder):
    book_chunks = defaultdict(list)

    for book_name, text in books.items():
        chunks = chunk_text(text)
        embeddings = embedder.encode(
            chunks,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        for i, emb in enumerate(embeddings):
            book_chunks[book_name].append({
                "chunk_id": i,
                "text": chunks[i],
                "embedding": emb
            })

    return book_chunks


def retrieve_top_k(backstory, book_name, book_chunks, embedder, k=5):
    query_emb = embedder.encode(
        backstory,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scored = []
    for chunk in book_chunks[book_name]:
        score = np.dot(query_emb, chunk["embedding"])
        scored.append((score, chunk["text"]))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in scored[:k]]


# =========================
# LLM Judge
# =========================

def llm_decide(backstory, evidence_chunks):
    if NVIDIA_API_KEY is None:
        raise RuntimeError("NVIDIA_API_KEY not set")

    evidence = "\n\n".join(
        f"Evidence {i+1}:\n{e}"
        for i, e in enumerate(evidence_chunks)
    )

    prompt = f"""
You are a strict logical consistency judge.

BACKSTORY:
{backstory}

EVIDENCE:
{evidence}

Output 0 if the backstory contradicts the evidence.
Output 1 otherwise.

Answer with exactly one character.
"""

    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 1,
    }

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }

    r = requests.post(NIM_URL, headers=headers, json=payload, timeout=30)
    r.raise_for_status()

    text = r.json()["choices"][0]["message"]["content"]
    match = re.search(r"[01]", text)

    return int(match.group()) if match else 1


# =========================
# Main Pipeline
# =========================

def run_inference(df, book_chunks, embedder):
    results = []

    for _, row in df.iterrows():
        backstory = shorten(row["content"])
        book = row["book_name"].lower()

        claims = split_claims(backstory)
        prediction = 1

        for claim in claims:
            evidence = retrieve_top_k(
                claim, book, book_chunks, embedder, k=4
            )

            evidence = [shorten(e, 400) for e in evidence]

            try:
                pred = llm_decide(claim, evidence)
            except Exception:
                pred = 1

            if pred == 0:
                prediction = 0
                break

            time.sleep(0.8)

        results.append({
            "id": str(row["id"]),
            "prediction": prediction
        })

    return pd.DataFrame(results)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    embedder = SentenceTransformer(MODEL_NAME)
    books = load_books()
    book_chunks = build_index(books, embedder)

    train_preds = run_inference(train_df, book_chunks, embedder)
    train_preds.to_csv(os.path.join(OUTPUT_DIR, "train_predictions.csv"), index=False)

    test_preds = run_inference(test_df, book_chunks, embedder)
    test_preds.to_csv(os.path.join(OUTPUT_DIR, "results.csv"), index=False)


if __name__ == "__main__":
    main()
