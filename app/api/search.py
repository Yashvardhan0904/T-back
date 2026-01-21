from fastapi import APIRouter
from pydantic import BaseModel
from app.services.search.orchestrator import search_orchestrator
from app.services.embeddings.service import embedding_service

router = APIRouter()

class SearchRequest(BaseModel):
    query: str

class EmbeddingRequest(BaseModel):
    text: str

@router.post("/search/hybrid")
async def debug_search(request: SearchRequest):
    results = await search_orchestrator.hybrid_search(request.query)
    return {"query": request.query, "results_count": len(results), "results": results}

@router.post("/embeddings")
async def get_embeddings(request: EmbeddingRequest):
    embedding = await embedding_service.get_embedding(request.text)
    return {"text": request.text, "embedding": embedding}
