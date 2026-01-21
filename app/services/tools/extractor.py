from groq import Groq
from app.core.config import get_settings
import json
import logging

settings = get_settings()
logger = logging.getLogger(__name__)

class ToolExtractionService:
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)

    async def extract_product_id(self, message: str) -> str:
        """
        Uses Groq to extract a product ID or Name from the message.
        In a real app, this would match against recent search results.
        """
        prompt = f"""
        Extract the product name or ID from this user request.
        Request: "{message}"

        Return JSON ONLY with:
        - product_info: (string or null)
        
        Example: "add the red hoodie to cart" -> {{"product_info": "red hoodie"}}
        """

        try:
            completion = self.client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            data = json.loads(completion.choices[0].message.content)
            return data.get("product_info")
        except Exception as e:
            logger.error(f"Tool Extraction Error: {e}")
            return None

tool_extractor = ToolExtractionService()
