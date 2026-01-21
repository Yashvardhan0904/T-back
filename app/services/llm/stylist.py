from groq import Groq
from app.core.config import get_settings
import logging

settings = get_settings()
logger = logging.getLogger(__name__)

class StylistService:
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)

    async def get_style_tip(self, context: str) -> str:
        """
        Generates a quick, fun fashion tip based on the user's context.
        """
        prompt = f"""
        You are 'Trendora', a spicy, high-end fashion stylist from Mumbai. 
        Give a short, punchy (max 10-12 words) style tip that sounds like it's coming from a fashion editor.
        Context: "{context}"
        
        Rules:
        1. Mix in some Hinglish (e.g., "Arre waah!", "Ekdum chic", "Global standard").
        2. Be confident, slightly arrogant but lovable.
        3. Mention specific silhouettes or styling tricks if the context allows.
        
        Examples: 
        - "Arre, pair that blazer with sneakers for that 'cool CEO' energy."
        - "Drape that stole like a queen—minimalism is for the boring!"
        - "Monochrome is the secret to looking like a total crore-pati."
        """
        try:
            completion = self.client.chat.completions.create(
                model=settings.GROQ_MODEL_SMALL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=50
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Stylist Tip Error: {e}")
            return "Style is about confidence—wear what makes you feel like the best version of yourself."

stylist_service = StylistService()
