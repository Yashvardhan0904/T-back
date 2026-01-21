from groq import Groq
from app.core.config import get_settings
import json
import logging

settings = get_settings()
logger = logging.getLogger(__name__)

class SearchStrategyService:
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)

    async def extract_filters(self, query: str) -> dict:
        """
        Uses Groq to extract structured MongoDB filters from a shopping query.
        Handles fuzzy price ranges.
        """
        prompt = f"""
        Extract search filters from this query for a shopping app.
        Query: "{query}"

        Return JSON ONLY with these keys:
        - category: (string)
        - color: (string)
        - fit: (string)
        - occasion: (string)
        - price_range: {{"min": number, "max": number}} (Handle "around", "cheap", "premium" intelligently)
        - tags: [list of keywords]

        Rules:
        1. If "around 50", set range 40-60.
        2. If "cheap", set max 500.
        3. If "premium", set min 2000.
        4. Use null for missing values.

        Example Output:
        {{"category": "hoodie", "color": "red", "price_range": {{"min": 40, "max": 60}}, "fit": "oversized"}}
        """

        try:
            completion = self.client.chat.completions.create(
                model=settings.GROQ_MODEL_SMALL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            filters = json.loads(completion.choices[0].message.content)
            logger.info(f"Extracted Filters: {filters}")
            return filters
        except Exception as e:
            logger.error(f"Filter Extraction Error: {e}")
            return {}

search_strategy = SearchStrategyService()
