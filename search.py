import numpy as np
from chunk import get_pages, chunk_all
from embed import embed

def build_index(chunks):
    texts = [c["text"] for c in chunks]
    vectors = embed(texts)
    return vectors  # shape: (2091, 384)

def search(query, chunks, vectors, top_k=5):
    query_vec = embed([query])[0]          # shape: (384,)
    scores = vectors @ query_vec           # dot product with every chunk = cosine similarity
    top_indices = np.argsort(scores)[::-1][:top_k]  # indices of top_k highest scores
    results = []
    for i in top_indices:
        results.append({
            "score": round(float(scores[i]), 4),
            "page": chunks[i]["page"],
            "text": chunks[i]["text"][:200]
        })
    return results

if __name__ == "__main__":
    print("Loading PDF and building index...")
    pages = get_pages("The Laws of Human Nature.pdf")
    chunks = chunk_all(pages)
    vectors = build_index(chunks)
    print(f"Index built: {vectors.shape}")

    query = "How do emotions affect decision making?"
    print(f"\nQuery: {query}\n")
    results = search(query, chunks, vectors)
    for r in results:
        print(f"Page {r['page']} (score {r['score']}): {r['text']}\n")