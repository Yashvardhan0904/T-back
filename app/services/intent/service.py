from groq import Groq
from app.core.config import get_settings
import json
import logging

settings = get_settings()
logger = logging.getLogger(__name__)

class IntentService:
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)

    async def classify_intent(self, message: str) -> str:
        """
        Classifies user intent into labels: 
        GREETING, PRODUCT_SEARCH, PRODUCT_CATEGORY, PRODUCT_BROWSE, SMALL_TALK, OUT_OF_SCOPE,
        ADD_TO_CART, ADD_TO_WISHLIST, VIEW_CART, CLEAR_CART, GET_TRENDING
        """
        prompt = f"""
        Strictly classify the following user message into ONE of these labels:
        - GREETING: Friendly hello or hi.
        - PRODUCT_SEARCH: Looking for a specific item with vague or specific descriptors (e.g. "cool stuff", "black hoodie").
        - PRODUCT_CATEGORY: Asking for a high-level category (e.g. "show me kurtas", "do you have shoes?").
        - PRODUCT_BROWSE: Vague interest in trends or suggestions (e.g. "what's new?", "suggest something").
        - ADD_TO_CART: Adding a specific item to the cart.
        - ADD_TO_WISHLIST: Saving an item for later.
        - VIEW_CART: Asking to see what's in their cart (e.g., "show my cart", "what did I buy?").
        - CLEAR_CART: Asking to empty the cart.
        - GET_TRENDING: Asking for what's popular or new arrivals specifically.
        - IDENTITY_CORRECTION: User is correcting a mistake about their name, gender, or person.
        - DISMISSIVE: User is saying "nothing", "no", "shut up", or being uninterested.
        - SMALL_TALK: Personal or non-shopping chat.
        - OUT_OF_SCOPE: Something the assistant cannot do.

        User Message: "{message}"

        Output ONLY the label name in all caps. No explanation.
        """
        
        try:
            completion = self.client.chat.completions.create(
                model=settings.GROQ_MODEL_SMALL,  # Back to 8B for efficiency
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10
            )

            intent = completion.choices[0].message.content.strip()
            logger.info(f"Detected Intent: {intent}")
            return intent
        except Exception as e:
            logger.error(f"Intent Classification Error: {e}")
            return "PRODUCT_SEARCH" # Default fallback for safety

intent_service = IntentService()
