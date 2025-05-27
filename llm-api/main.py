from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import logging
from transformers import pipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("LLMAPI")

app = FastAPI()

log.info("Loading CPU-friendly model...")

# Use a smaller, CPU-friendly model like GPT-Neo 125M
generator = pipeline(
    "text-generation",
    model="EleutherAI/gpt-neo-125M",
    device=-1,  # CPU only
    pad_token_id=50256,  # GPT-2/Neo padding token
)

class LLMQueryRequest(BaseModel):
    query: str
    retrieved_contexts: List[str]

class LLMQueryResponse(BaseModel):
    answer: str

def build_prompt(query: str, contexts: List[str]) -> str:
    context_block = "\n\n".join(f"- {c}" for c in contexts)
    prompt = f"""You are an assistant answering questions based on the following documents:

{context_block}

Answer the following question based on the above content:
Q: {query}
A:"""
    return prompt

@app.post("/generate", response_model=LLMQueryResponse)
def generate_response(req: LLMQueryRequest):
    try:
        prompt = build_prompt(req.query, req.retrieved_contexts)
        log.info("🧠 Generating answer from CPU-friendly model")

        outputs = generator(
            prompt,
            max_length=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1,
        )
        # Extract generated text after prompt
        generated_text = outputs[0]["generated_text"]
        answer = generated_text[len(prompt):].strip()

        return LLMQueryResponse(answer=answer)

    except Exception as e:
        log.error(f"❌ LLM generation failed: {e}")
        raise HTTPException(status_code=500, detail="LLM generation failed")