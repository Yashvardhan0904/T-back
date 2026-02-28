import httpx
import logging
from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        self.api_url = settings.EXT_EMBEDDING_API_URL
        self.headers = {"Authorization": f"Bearer {settings.HF_TOKEN}"} if settings.HF_TOKEN else {}

    async def get_embedding(self, text: str) -> list[float]:
        """
        Fetches embedding from Hugging Face Inference API.
        If HF_TOKEN is missing, this will fail in production.
        """
        if not settings.HF_TOKEN:
            logger.warning("⚠️ HF_TOKEN is missing. Returning dummy embedding.")
            return [0.0] * 384 # MiniLM dimension

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.api_url,
                    headers=self.headers,
                    json={"inputs": text, "options": {"wait_for_model": True}},
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Embedding API Error: {response.status_code} - {response.text}")
                    # Fallback to dummy for testing/resilience
                    logger.warning("Falling back to dummy embedding due to API error.")
                    return [0.01] * 384
        except Exception as e:
            logger.error(f"Failed to fetch embedding: {e}")
            return [0.01] * 384

embedding_service = EmbeddingService()
