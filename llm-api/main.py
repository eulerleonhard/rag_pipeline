from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import logging
import requests
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import os

# Logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("LLMAPI")

app = FastAPI()

# Load FLAN-T5 model
MODEL_NAME = "google/flan-t5-large"
log.info(f"🧠 Loading model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

generator = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1  # CPU
)

RETRIEVAL_API_URL = "http://retrieval-api:9000/search"

# Pydantic models
class LLMQueryRequest(BaseModel):
    query: str
    retrieved_contexts: List[str]

class LLMQueryResponse(BaseModel):
    answer: str

class FullQueryRequest(BaseModel):
    query: str
    top_k: int = 3

# Build prompt for FLAN-T5
def build_prompt(query: str, contexts: List[str]) -> str:
    context_block = "\n".join(f"- {c}" for c in contexts)
    return (
        "Answer the question based on the context below.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\nAnswer:"
    )

# Local generation endpoint
@app.post("/generate", response_model=LLMQueryResponse)
def generate_response(req: LLMQueryRequest):
    try:
        prompt = build_prompt(req.query, req.retrieved_contexts)
        log.info("💬 Generating answer...")

        outputs = generator(
            prompt,
            max_new_tokens=256,
            do_sample=False,  # Deterministic for factuality
        )
        answer = outputs[0]["generated_text"].strip()

        return LLMQueryResponse(answer=answer)

    except Exception as e:
        log.error(f"❌ Generation failed: {e}")
        raise HTTPException(status_code=500, detail="LLM generation failed")

# End-to-end retrieval + generation
@app.post("/answer", response_model=LLMQueryResponse)
def answer(req: FullQueryRequest):
    try:
        log.info(f"🔗 Fetching context for: {req.query}")
        response = requests.post(RETRIEVAL_API_URL, json={"query": req.query, "top_k": req.top_k})
        response.raise_for_status()
        results = response.json()["results"]
        contexts = [r["content"] for r in results]
    except Exception as e:
        log.error(f"❌ Retrieval API call failed: {e}")
        raise HTTPException(status_code=500, detail="Retrieval API failed")

    return generate_response(LLMQueryRequest(query=req.query, retrieved_contexts=contexts))
