# File retrieval-api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import faiss
import numpy as np
import pandas as pd
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
    # Set index on chunk_int_id for fast lookup by ID
    metadata_df.set_index("chunk_int_id", inplace=True)
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
        distances, ids = faiss_index.search(query_emb, req.top_k)
        ids = ids[0]
        distances = distances[0]
    except Exception as e:
        log.error(f"❌ FAISS search failed: {e}")
        raise HTTPException(status_code=500, detail="FAISS search failed")

    # Step 3: Fetch metadata and construct results
    results = []
    for rank, faiss_id in enumerate(ids, start=1):
        if faiss_id < 0:
            log.warning(f"⚠️ Skipping invalid FAISS id: {faiss_id}")
            continue
        try:
            row = metadata_df.loc[faiss_id]
        except KeyError:
            log.warning(f"⚠️ Metadata not found for FAISS id: {faiss_id}")
            continue

        results.append({
            "rank": rank,
            "score": float(distances[rank - 1]),
            "content": row["content"],
            "source": row["source"]
        })

    return {"results": results}