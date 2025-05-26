from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import faiss
import numpy as np
import os
from typing import List, Tuple

app = FastAPI()

EMBEDDING_API_URL = "http://embedding-api:9000/embed"
FAISS_INDEX_PATH = "/faiss/index.faiss"
METADATA_PATH = "/faiss/metadata.parquet"

import pandas as pd

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

# Load FAISS index and metadata on startup
faiss_index = faiss.read_index(FAISS_INDEX_PATH)
metadata_df = pd.read_parquet(METADATA_PATH)

@app.post("/search")
def search(req: QueryRequest):
    # Embed query
    resp = requests.post(EMBEDDING_API_URL, json={"texts": [req.query]})
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail="Embedding API failed")
    query_emb = np.array(resp.json()["embeddings"][0], dtype=np.float32).reshape(1, -1)

    # Search FAISS
    distances, indices = faiss_index.search(query_emb, req.top_k)
    indices = indices[0]

    results = []
    for idx in indices:
        if idx == -1:
            continue
        print("Indices from FAISS:", indices)
        print("metadata_df length:", len(metadata_df))
        if idx < len(metadata_df):
            row = metadata_df.iloc[idx]
        else:
            print(f"Invalid index {idx} for metadata_df of length {len(metadata_df)}")
            continue  # or return an appropriate fallback
        results.append({
            "content": row["content"],
            "source": row["source"],
        })

    return {"results": results}