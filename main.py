"""
Server embedding chạy local, API giống OpenAI (POST /v1/embeddings).
Spring Boot gọi vào đây nên không cần internet/API key sau khi model đã tải về.

Chạy:  uvicorn main:app --host 127.0.0.1 --port 8000
Đổi model:  EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 uvicorn main:app ...
"""
import os
from typing import List, Optional, Union

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Tải model một lần khi khởi động (lần đầu sẽ tự tải ~90MB từ HuggingFace, các lần sau dùng cache).
model = SentenceTransformer(MODEL_NAME)
DIMENSIONS = model.get_sentence_embedding_dimension()

app = FastAPI(title="Local embedding server")


class EmbeddingRequest(BaseModel):
    model: Optional[str] = None
    input: Union[str, List[str]]
    dimensions: Optional[int] = None  # bỏ qua, chỉ để tương thích OpenAI


@app.post("/v1/embeddings")
def create_embeddings(req: EmbeddingRequest):
    texts = [req.input] if isinstance(req.input, str) else req.input
    # normalize_embeddings=True -> cosine similarity ổn định, khớp với toán tử <=> của pgvector
    vectors = model.encode(texts, normalize_embeddings=True)
    return {
        "object": "list",
        "model": MODEL_NAME,
        "data": [
            {"object": "embedding", "index": i, "embedding": vec.tolist()}
            for i, vec in enumerate(vectors)
        ],
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "dimensions": DIMENSIONS}
