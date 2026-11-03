import os
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from chunk import get_pages, chunk_all
from embed import embed
from answer import answer

# Load .env
with open(".env") as f:
    for line in f:
        if "=" in line and not line.startswith("#"):
            k, v = line.strip().split("=", 1)
            os.environ[k.strip()] = v.strip()

app = FastAPI()

# In-memory store
chunks = []
vectors = None

class AskRequest(BaseModel):
    question: str

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    global chunks, vectors
    contents = await file.read()
    with open("uploaded.pdf", "wb") as f:
        f.write(contents)
    pages = get_pages("uploaded.pdf")
    chunks = chunk_all(pages)
    vectors = np.array(list(embed([c["text"] for c in chunks])))
    return {"chunks": len(chunks)}

@app.post("/api/ask")
async def ask(req: AskRequest):
    if vectors is None:
        return {"error": "Upload a PDF first"}
    query_vec = embed([req.question])[0]
    scores = vectors @ query_vec
    top = np.argsort(scores)[::-1][:5]
    results = [{"page": chunks[i]["page"], "text": chunks[i]["text"], "score": round(float(scores[i]), 4)} for i in top]
    ans = answer(req.question, results, os.environ["GROQ_API_KEY"])
    return {"answer": ans, "sources": results}

app.mount("/", StaticFiles(directory="static", html=True), name="static")