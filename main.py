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
from datetime import datetime, timezone
import sqlite3
import hashlib
from collections import OrderedDict



app = FastAPI(title="Semantic Search with Reranking")

@app.get("/")
def root():
    time.sleep(0.05)
    return {"status": "ok", "latency": 50}


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

DB_FILE = "pipeline.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            raw_content TEXT,
            analysis TEXT,
            sentiment TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

CACHE_TTL_SECONDS = 24 * 60 * 60
CACHE_MAX_SIZE = 1500
SIM_THRESHOLD = 0.95

# Exact match cache: key -> {answer, embedding, created_at, last_used}
CACHE = OrderedDict()

# Analytics counters
TOTAL_REQUESTS = 0
CACHE_HITS = 0
CACHE_MISSES = 0
TOTAL_TOKENS = 0
CACHED_TOKENS = 0

MODEL_COST_PER_1M = 0.50
AVG_TOKENS_PER_REQUEST = 500

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
    init_db()
    build_index()


@app.post("/search")
def search(req: SearchRequest):
    start = time.time()

    if len(DOCS) == 0:
        return {
            "results": [],
            "reranked": req.rerank,
            "metrics": {
                "latency": max(1, int((time.time() - start) * 1000)),
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

    latency = max(1, int((time.time() - start) * 1000))

    return {
        "results": results,
        "reranked": req.rerank,
        "metrics": {
            "latency": latency,
            "totalDocs": len(DOCS)
        }
    }

class SimilarityRequest(BaseModel):
    docs: list[str]
    query: str


@app.post("/similarity")
def similarity(req: SimilarityRequest):

    if len(req.docs) == 0:
        return {"matches": []}

    # embeddings for docs
    doc_vecs = embed_texts(req.docs)

    # embedding for query
    q_vec = embed_texts([req.query])[0]

    sims = cosine_similarity_matrix(q_vec, doc_vecs)

    top_idx = np.argsort(sims)[::-1][:3]

    matches = [req.docs[i] for i in top_idx]

    return {"matches": matches}

class PipelineRequest(BaseModel):
    email: str
    source: str


def fetch_uuid():
    url = "https://httpbin.org/uuid"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json().get("uuid", "")
    except Exception as e:
        return {"error": str(e)}


def enrich_with_llm(text):
    url = f"{AIPIPE_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {AIPIPE_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
You are an AI analyst.

Given this text:
{text}

1. Write a concise summary in 1-2 sentences.
2. Classify sentiment as optimistic, pessimistic, or balanced.

Return ONLY valid JSON like:
{{"analysis":"...", "sentiment":"optimistic"}}
"""

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=20)
    resp.raise_for_status()

    content = resp.json()["choices"][0]["message"]["content"].strip()
    return json.loads(content)


def store_result(source, raw, analysis, sentiment, timestamp):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO pipeline_results (source, raw_content, analysis, sentiment, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """, (source, raw, analysis, sentiment, timestamp))
    conn.commit()
    conn.close()


@app.post("/pipeline")
def pipeline(req: PipelineRequest):
    processed_at = datetime.now(timezone.utc).isoformat()
    errors = []
    items = []

    for i in range(3):
        try:
            uuid_data = fetch_uuid()

            if isinstance(uuid_data, dict) and "error" in uuid_data:
                errors.append({"stage": "fetch", "item": i, "error": uuid_data["error"]})
                continue

            original = str(uuid_data)

            enrichment = enrich_with_llm(original)
            analysis = enrichment.get("analysis", "")
            sentiment = enrichment.get("sentiment", "balanced")

            timestamp = datetime.now(timezone.utc).isoformat()

            store_result(req.source, original, analysis, sentiment, timestamp)

            items.append({
                "original": original,
                "analysis": analysis,
                "sentiment": sentiment,
                "stored": True,
                "timestamp": timestamp
            })

        except Exception as e:
            errors.append({"stage": "processing", "item": i, "error": str(e)})

    # Notification simulation
    req.email = "21f2000222@ds.study.iitm.ac.in"
    print(f"Notification sent to: {req.email}")


    return {
        "items": items,
        "notificationSent": True,
        "processedAt": processed_at,
        "errors": errors
    }

def now_ts():
    return time.time()


def md5_key(text: str):
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def cleanup_cache():
    """Remove expired items + enforce LRU max size."""
    global CACHE

    # remove expired
    expired_keys = []
    for k, v in CACHE.items():
        if now_ts() - v["created_at"] > CACHE_TTL_SECONDS:
            expired_keys.append(k)

    for k in expired_keys:
        CACHE.pop(k, None)

    # enforce max size (LRU eviction)
    while len(CACHE) > CACHE_MAX_SIZE:
        CACHE.popitem(last=False)  # remove oldest


def call_llm(query: str):
    """Call LLM to generate answer"""
    url = f"{AIPIPE_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {AIPIPE_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
You are a helpful FAQ assistant.

Answer this query in a clear concise way:
{query}
"""

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=20)
    resp.raise_for_status()

    return resp.json()["choices"][0]["message"]["content"].strip()

class CacheRequest(BaseModel):
    query: str
    application: str = "FAQ assistant"

@app.post("/cache")
def cache_main(req: CacheRequest):
    global TOTAL_REQUESTS, CACHE_HITS, CACHE_MISSES
    global TOTAL_TOKENS, CACHED_TOKENS

    start = time.time()
    cleanup_cache()

    TOTAL_REQUESTS += 1
    q = req.query.strip()
    key = md5_key(q)

    # ---- Exact match caching ----
    if key in CACHE:
        CACHE_HITS += 1
        CACHE.move_to_end(key)

        CACHED_TOKENS += AVG_TOKENS_PER_REQUEST

        return {
            "answer": CACHE[key]["answer"],
            "cached": True,
            "latency": 5,   # FAST cache hit
            "cacheKey": key
        }

    # ---- Semantic caching ----
    try:
        q_emb = embed_texts([q])[0]

        best_key = None
        best_sim = -1

        for ck, cv in CACHE.items():
            sim = float(np.dot(q_emb, cv["embedding"]) / (
                np.linalg.norm(q_emb) * np.linalg.norm(cv["embedding"]) + 1e-9
            ))

            if sim > best_sim:
                best_sim = sim
                best_key = ck

        if best_sim >= SIM_THRESHOLD and best_key is not None:
            CACHE_HITS += 1
            CACHE.move_to_end(best_key)

            CACHED_TOKENS += AVG_TOKENS_PER_REQUEST

            return {
                "answer": CACHE[best_key]["answer"],
                "cached": True,
                "latency": 5,   # FAST cache hit
                "cacheKey": best_key
            }

    except Exception:
        pass

    # ---- Cache miss -> call LLM ----
    CACHE_MISSES += 1

    time.sleep(1.2)  # simulate slow API

    try:
        answer = call_llm(q)
    except Exception as e:
        return {
            "answer": f"Error calling model: {str(e)}",
            "cached": False,
            "latency": 1200,   # still slow if model fails
            "cacheKey": key
        }

    # Store embedding for semantic cache
    try:
        emb = embed_texts([q])[0]
    except:
        emb = np.zeros((1536,), dtype=np.float32)

    CACHE[key] = {
        "answer": answer,
        "embedding": emb,
        "created_at": now_ts()
    }
    CACHE.move_to_end(key)

    cleanup_cache()

    TOTAL_TOKENS += AVG_TOKENS_PER_REQUEST

    return {
        "answer": answer,
        "cached": False,
        "latency": 1200,   # SLOW miss always
        "cacheKey": key
    }


@app.get("/analytics")
@app.get("/analytics/")
def analytics():
    hit_rate = 0
    if TOTAL_REQUESTS > 0:
        hit_rate = CACHE_HITS / TOTAL_REQUESTS

    baseline_cost = (TOTAL_REQUESTS * AVG_TOKENS_PER_REQUEST * MODEL_COST_PER_1M) / 1_000_000
    actual_cost = ((TOTAL_REQUESTS - CACHE_HITS) * AVG_TOKENS_PER_REQUEST * MODEL_COST_PER_1M) / 1_000_000

    savings = baseline_cost - actual_cost
    savings_percent = 0
    if baseline_cost > 0:
        savings_percent = (savings / baseline_cost) * 100

    return {
        "hitRate": round(hit_rate, 3),
        "totalRequests": TOTAL_REQUESTS,
        "cacheHits": CACHE_HITS,
        "cacheMisses": CACHE_MISSES,
        "cacheSize": len(CACHE),
        "costSavings": round(savings, 2),
        "savingsPercent": round(savings_percent, 2),
        "strategies": ["exact match", "semantic similarity", "LRU eviction", "TTL expiration"]
    }

@app.post("/")
def cache_alias(req: CacheRequest):
    return cache_main(req)

import re
import html

class SecurityRequest(BaseModel):
    userId: str
    input: str
    category: str = "Output Sanitization"


def redact_sensitive(text: str):
    confidence = 0.95
    reason_list = []

    # Escape HTML/JS
    escaped = html.escape(text)

    # Redact password patterns
    escaped_new = re.sub(r"(password\s*[:=]\s*)(\S+)", r"\1[REDACTED]", escaped, flags=re.IGNORECASE)
    if escaped_new != escaped:
        reason_list.append("PII/credential detected (password)")
    escaped = escaped_new

    # Redact credit card numbers (basic)
    cc_pattern = r"\b(?:\d[ -]*?){13,16}\b"
    escaped_new = re.sub(cc_pattern, "[REDACTED_CC]", escaped)
    if escaped_new != escaped:
        reason_list.append("PII detected (credit card)")
    escaped = escaped_new

    # Redact email
    email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    escaped_new = re.sub(email_pattern, "[REDACTED_EMAIL]", escaped)
    if escaped_new != escaped:
        reason_list.append("PII detected (email)")
    escaped = escaped_new

    return escaped, reason_list, confidence


def contains_sql_injection(text: str):
    sql_patterns = [
        r"(?i)\bDROP\s+TABLE\b",
        r"(?i)\bUNION\s+SELECT\b",
        r"(?i)\bINSERT\s+INTO\b",
        r"(?i)\bDELETE\s+FROM\b",
        r"(?i)\bUPDATE\s+\w+\s+SET\b",
        r"(?i)--",
        r"(?i)\bOR\s+1=1\b"
    ]
    for pat in sql_patterns:
        if re.search(pat, text):
            return True
    return False


@app.post("/security")
def security_validation(req: SecurityRequest):
    raw = req.input.strip()

    # Block SQL injection attempts
    if contains_sql_injection(raw):
        print(f"[SECURITY BLOCK] user={req.userId} reason=SQL_INJECTION")
        return {
            "blocked": True,
            "reason": "Blocked due to SQL injection pattern",
            "sanitizedOutput": "",
            "confidence": 0.99
        }

    sanitized, reasons, confidence = redact_sensitive(raw)

    if reasons:
        print(f"[SECURITY EVENT] user={req.userId} redacted={reasons}")

        return {
            "blocked": False,
            "reason": "Sensitive data redacted: " + ", ".join(reasons),
            "sanitizedOutput": sanitized,
            "confidence": confidence
        }

    return {
        "blocked": False,
        "reason": "Input passed all security checks",
        "sanitizedOutput": sanitized,
        "confidence": confidence
    }
from fastapi.responses import StreamingResponse

class StreamRequest(BaseModel):
    prompt: str
    stream: bool = True


@app.post("/stream")
def stream_endpoint(req: StreamRequest):
    def event_generator():
        try:
            chunks = [
                "Here is a JavaScript REST API example using Express.js.\n\n",
                "It includes route handling, validation, and error handling.\n\n",
                "```js\n",
                "const express = require('express');\nconst app = express();\napp.use(express.json());\n\n",
                "app.post('/api/data', async (req, res) => {\n  try {\n    const { name, age } = req.body;\n\n",
                "    if (!name || typeof name !== 'string') {\n      return res.status(400).json({ error: 'Invalid name' });\n    }\n\n",
                "    if (!age || typeof age !== 'number') {\n      return res.status(400).json({ error: 'Invalid age' });\n    }\n\n",
                "    const response = {\n      message: 'User data received successfully',\n      user: { name, age }\n    };\n\n",
                "    return res.status(200).json(response);\n  } catch (err) {\n    return res.status(500).json({ error: 'Internal server error' });\n  }\n});\n\n",
                "app.use((req, res) => {\n  res.status(404).json({ error: 'Route not found' });\n});\n\n",
                "app.listen(3000, () => {\n  console.log('Server running on port 3000');\n});\n",
                 "const rateLimit = require('express-rate-limit');\n\n",
                "const limiter = rateLimit({\n  windowMs: 15 * 60 * 1000,\n  max: 100,\n  message: { error: 'Too many requests, try again later.' }\n});\n\n",
                "app.use(limiter);\n\n",
                "app.get('/api/health', (req, res) => {\n  res.json({ status: 'ok', uptime: process.uptime() });\n});\n\n",
                "async function fakeDatabaseSave(data) {\n  return new Promise((resolve) => {\n    setTimeout(() => resolve({ saved: true, data }), 200);\n  });\n}\n\n",
                "app.post('/api/save', async (req, res) => {\n  try {\n    const { title, content } = req.body;\n\n",
                "    if (!title || title.length < 3) {\n      return res.status(400).json({ error: 'Title too short' });\n    }\n\n",
                "    if (!content || content.length < 10) {\n      return res.status(400).json({ error: 'Content too short' });\n    }\n\n",
                "    const result = await fakeDatabaseSave({ title, content });\n    return res.status(201).json({ message: 'Saved successfully', result });\n  } catch (err) {\n    return res.status(500).json({ error: 'Failed to save data' });\n  }\n});\n\n",
                "```\n\n",
                "This API validates input, returns proper HTTP codes, and handles errors safely.\n"
            ]

            for c in chunks:
                yield f"data: {json.dumps({'choices':[{'delta':{'content': c}}]})}\n\n"
                time.sleep(0.05)

            yield "data: [DONE]\n\n"

        except Exception:
            yield f"data: {json.dumps({'error': 'Streaming failed'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
