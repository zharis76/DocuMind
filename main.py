import os
from chunk import get_pages, chunk_all
from embed import embed
from search import build_index, search
from answer import answer
import numpy as np

# Load .env
with open(".env") as f:
    for line in f:
        if "=" in line and not line.startswith("#"):
            k, v = line.strip().split("=", 1)
            os.environ[k.strip()] = v.strip()

PDF = "The Laws of Human Nature.pdf"

print("Reading PDF...")
pages = get_pages(PDF)
chunks = chunk_all(pages)
print(f"{len(chunks)} chunks loaded")

print("Building index (this takes ~30 seconds)...")
vectors = build_index(chunks)
print("Index ready")

while True:
    question = input("\nAsk a question (or 'quit'): ").strip()
    if question.lower() == "quit":
        break
    results = search(question, chunks, vectors)
    ans = answer(question, results, os.environ["GROQ_API_KEY"])
    print(f"\n{ans}")
    print("\nSources:")
    for i, r in enumerate(results, 1):
        print(f"  [{i}] Page {r['page']} (score {r['score']}): {r['text'][:100]}...")