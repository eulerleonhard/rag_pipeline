from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch
import uvicorn

app = FastAPI()

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Or any other HF model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

class EmbeddingRequest(BaseModel):
    texts: list[str]

def mean_pooling(token_embeddings, attention_mask):
    # Mean Pooling
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)

@app.post("/embed")
async def embed_texts(request: EmbeddingRequest):
    inputs = tokenizer(request.texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**inputs)
    embeddings = mean_pooling(model_output.last_hidden_state, inputs['attention_mask'])
    vectors = embeddings.numpy().tolist()
    return {"embeddings": vectors}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
