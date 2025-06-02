# rag_pipeline

This project implements a full Retrieval-Augmented Generation (RAG) pipeline using Spark, FAISS, FastAPI, and Docker Compose. It enables document ingestion, embedding, vector search, and large language model (LLM) answer generation.

---

## Architecture Overview

- **Data Ingestion:** PDFs are chunked and embedded using Spark and an embedding API.
- **Indexing:** Embeddings are indexed with FAISS for fast similarity search.
- **Retrieval API:** Exposes a search endpoint to find relevant chunks for a query.
- **LLM API:** Generates answers using a language model and retrieved context.
- **All components are containerized and orchestrated via Docker Compose.**

---

## Prerequisites

- [Docker](https://www.docker.com/products/docker-desktop)
- [Docker Compose](https://docs.docker.com/compose/)

---

## Folder Structure

```
rag_app1/
├── data/                  # Place your PDF files here
├── embedding-api/         # Embedding service (FastAPI + SentenceTransformers)
├── llm-api/               # LLM service (FastAPI + Transformers)
├── retrieval-api/         # Retrieval service (FastAPI + FAISS)
├── spark-jobs/            # Spark jobs for ingestion and indexing
├── compose.yaml           # Docker Compose orchestration
```

---

## Quickstart

### 1. Prepare Data

- Place your PDF files in the `data/` directory.

### 2. Build and Start Core Services

Start only the core infrastructure and embedding service first:

```sh
docker-compose up -d -scale --spark-worker=2
```

### 3. Ingest Documents and Build Index

**Step 1: Ingest and Embed Documents**

Run the ingestion pipeline inside the Spark container:

```sh
docker exec -it spark-master python /spark-jobs/ingestion_pipeline.py
```

This will:
- Read PDFs from HDFS
- Chunk and embed text
- Write embeddings to HDFS

**Step 2: Build FAISS Index**

```sh
docker exec -it spark-master python /spark-jobs/faiss_index_builder.py
```

This will:
- Read embeddings from HDFS
- Build a FAISS index and metadata
- Save them to a shared volume for the retrieval API

### 4. Start Retrieval and LLM APIs

After the index and metadata are built, start the retrieval and LLM API services:

```sh
docker-compose up -d retrieval-api llm-api
```

### 5. Query the RAG Pipeline

**End-to-end answer generation:**

Send a POST request to the LLM API.

- Linux:

```sh
curl -X POST http://localhost:9003/answer \
  -H "Content-Type: application/json" \
  -d '{"query": "YOUR QUESTION HERE", "top_k": 3}'
```

- Windows:

```sh
Invoke-RestMethod -Uri http://localhost:9003/answer `
  -Method Post `
  -Body (@{ query = "YOUR QUESTION HERE"; top_k = 3 } | ConvertTo-Json -Depth 10) `
  -ContentType "application/json"
```

You will receive an answer generated using retrieved context from your documents.

---

## API Endpoints

- **Embedding API:** `POST /embed` — Get embeddings for input texts.
- **Retrieval API:** `POST /search` — Retrieve top-k relevant document chunks for a query.
- **LLM API:**  
  - `POST /generate` — Generate answer from query and provided contexts.  
  - `POST /answer` — End-to-end: retrieve context and generate answer.

---

## Notes

- All services communicate over the internal Docker network.
- For large datasets, ensure your system has enough resources.
- If you add new data and rebuild the index, you must restart the retrieval-api and llm-api containers to load the latest index and metadata.
- You can extend the pipeline with a UI or additional data sources.

---

## Troubleshooting

- Check logs for each service using `docker compose logs <service-name>` or `docker-compose logs -f <service-name>`.
- Ensure your PDFs are readable and not encrypted.
- If you change code, rebuild the affected service with `docker compose build <service-name>`.

---

## License

VIETTEL