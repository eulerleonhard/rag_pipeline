# File retrieval-api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import requests
import faiss
import numpy as np
import pandas as pd
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("RetrievalAPI")

app = FastAPI()

# Configuration
EMBEDDING_API_URL = "http://embedding-api:9000/embed"
FAISS_INDEX_PATH = "/faiss/index.faiss"
METADATA_PATH = "/faiss/metadata.parquet"

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

# Load FAISS index and metadata once on startup
try:
    log.info(f"🔍 Loading FAISS index from {FAISS_INDEX_PATH}")
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
except Exception as e:
    log.error(f"❌ Failed to load FAISS index: {e}")
    raise RuntimeError("FAISS index could not be loaded")

try:
    log.info(f"📖 Loading metadata from {METADATA_PATH}")
    metadata_df = pd.read_parquet(METADATA_PATH)
except Exception as e:
    log.error(f"❌ Failed to load metadata: {e}")
    raise RuntimeError("Metadata file could not be loaded")

@app.post("/search")
def search(req: QueryRequest):
    # Step 1: Embed the query
    try:
        log.info(f"💬 Embedding query: {req.query}")
        resp = requests.post(EMBEDDING_API_URL, json={"texts": [req.query]})
        resp.raise_for_status()
        query_emb = np.array(resp.json()["embeddings"][0], dtype=np.float32).reshape(1, -1)
    except Exception as e:
        log.error(f"❌ Embedding API call failed: {e}")
        raise HTTPException(status_code=500, detail="Embedding API failed")

    # Step 2: Search FAISS index
    try:
        distances, indices = faiss_index.search(query_emb, req.top_k)
        indices = indices[0]
        distances = distances[0]
    except Exception as e:
        log.error(f"❌ FAISS search failed: {e}")
        raise HTTPException(status_code=500, detail="FAISS search failed")

    # Step 3: Fetch metadata and construct results
    results = []
    for i, idx in enumerate(indices):
        if idx < 0 or idx >= len(metadata_df):
            log.warning(f"⚠️ Skipping invalid index: {idx}")
            continue
        row = metadata_df.iloc[idx]
        results.append({
            "rank": i + 1,
            "score": float(distances[i]),
            "content": row["content"],
            "source": row["source"]
        })

    return {"results": results}
