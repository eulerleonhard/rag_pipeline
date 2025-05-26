from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

class EmbeddingRequest(BaseModel):
    texts: list[str]

@app.post("/embed")
async def embed_texts(request: EmbeddingRequest):
    vectors = model.encode(request.texts, show_progress_bar=False, convert_to_numpy=True).tolist()
    return {"embeddings": vectors}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
