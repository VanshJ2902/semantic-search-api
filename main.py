import os
import glob
import time
import json
import requests
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Semantic Search with Reranking")

@app.get("/")
def root():
    return {"status": "ok"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi import Request
from fastapi.responses import JSONResponse

@app.options("/{rest_of_path:path}")
async def preflight_handler(rest_of_path: str, request: Request):
    return JSONResponse(content={"message": "OK"})

load_dotenv()

AIPIPE_API_KEY = os.getenv("AIPIPE_API_KEY")
AIPIPE_BASE_URL = "https://aipipe.org/openai/v1"


DOCS = []
DOC_EMBEDDINGS = None


class SearchRequest(BaseModel):
    query: str
    k: int = 6
    rerank: bool = True
    rerankK: int = 4


def load_documents():
    docs = []
    txt_files = sorted(glob.glob("data/*.txt"))

    for i, path in enumerate(txt_files):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()

        docs.append({
            "id": i,
            "path": path,
            "content": content
        })

    return docs


def embed_texts(texts):
    url = f"{AIPIPE_BASE_URL}/embeddings"
    headers = {
        "Authorization": f"Bearer {AIPIPE_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "text-embedding-3-small",
        "input": texts
    }

    resp = requests.post(url, headers=headers, json=payload)
    resp.raise_for_status()

    data = resp.json()["data"]
    vectors = [np.array(item["embedding"], dtype=np.float32) for item in data]
    return np.vstack(vectors)


def cosine_similarity_matrix(query_vec, doc_matrix):
    query_norm = np.linalg.norm(query_vec)
    doc_norms = np.linalg.norm(doc_matrix, axis=1)

    denom = query_norm * doc_norms
    denom[denom == 0] = 1e-9

    return np.dot(doc_matrix, query_vec) / denom


def build_index():
    global DOCS, DOC_EMBEDDINGS
    DOCS = load_documents()

    if len(DOCS) == 0:
        DOC_EMBEDDINGS = np.zeros((0, 1536), dtype=np.float32)
        return

    texts = [d["content"][:8000] for d in DOCS]
    DOC_EMBEDDINGS = embed_texts(texts)


def rerank_with_llm(query, docs):
    url = f"{AIPIPE_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {AIPIPE_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = (
        "You are a legal document relevance judge.\n"
        "Rate relevance between the QUERY and each DOCUMENT on a scale of 0 to 10.\n"
        "Return ONLY a valid JSON array like:\n"
        "[{\"id\":0,\"score\":7},{\"id\":1,\"score\":2}]\n\n"
        f"QUERY: {query}\n\n"
        "DOCUMENTS:\n"
    )

    for d in docs:
        prompt += f"\nID: {d['id']}\nTEXT:\n{d['content'][:2000]}\n"

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0
    }

    resp = requests.post(url, headers=headers, json=payload)
    resp.raise_for_status()

    text = resp.json()["choices"][0]["message"]["content"].strip()

    try:
        scores = json.loads(text)
    except:
        scores = [{"id": d["id"], "score": 0} for d in docs]

    score_map = {item["id"]: float(item["score"]) for item in scores}

    for d in docs:
        raw = score_map.get(d["id"], 0.0)
        raw = max(0.0, min(10.0, raw))
        d["score"] = raw / 10.0

    docs.sort(key=lambda x: x["score"], reverse=True)
    return docs


@app.on_event("startup")
def startup_event():
    build_index()


@app.post("/search")
def search(req: SearchRequest):
    start = time.time()

    if len(DOCS) == 0:
        return {
            "results": [],
            "reranked": req.rerank,
            "metrics": {
                "latency": int((time.time() - start) * 1000),
                "totalDocs": 0
            }
        }

    q_vec = embed_texts([req.query])[0]
    sims = cosine_similarity_matrix(q_vec, DOC_EMBEDDINGS)

    k = min(req.k, len(DOCS))
    top_idx = np.argsort(sims)[::-1][:k]

    candidates = []
    for idx in top_idx:
        candidates.append({
            "id": DOCS[idx]["id"],
            "score": float((sims[idx] + 1) / 2),
            "content": DOCS[idx]["content"],
            "metadata": {"source": DOCS[idx]["path"]}
        })

    if req.rerank:
        reranked_docs = rerank_with_llm(req.query, candidates)
        rerankK = min(req.rerankK, len(reranked_docs))
        results = reranked_docs[:rerankK]
    else:
        results = candidates

    latency = int((time.time() - start) * 1000)

    return {
        "results": results,
        "reranked": req.rerank,
        "metrics": {
            "latency": latency,
            "totalDocs": len(DOCS)
        }
    }
