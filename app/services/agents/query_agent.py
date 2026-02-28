"""
Query Understanding Agent - Analyzes user input and extracts intent.

Output MUST include:
- intent_type (search | recommend | compare | browse | info)
- product_category
- filters (price, brand, features)
- ambiguity_level (low | medium | high)
"""

from typing import Dict, Any, List
from app.services.llm.service import llm_service
from app.core.config import get_settings
from app.core.categories import normalize_category, get_all_categories
from groq import Groq
import logging
import json

logger = logging.getLogger(__name__)
settings = get_settings()

# Keywords for short response detection
size_keywords = ["xs", "s", "m", "l", "xl", "xxl", "xxxl", "small", "medium", "large", "extra large"]
color_keywords = ["black", "white", "red", "blue", "green", "yellow", "pink", "purple", "orange", "brown", "grey", "gray", "navy", "teal", "maroon"]
vibe_keywords = ["minimal", "streetwear", "classic", "trendy", "ethnic", "luxury", "shadi", "traditional", "casual", "formal"]
party_keywords = ["clubbing", "club", "wedding", "house party", "party", "formal", "office", "college", "shadi", "reception", "trip", "beach", "vacation", "travel", "date", "dinner", "festival", "sangeet", "function", "haldi", "gym", "brunch", "concert", "outing", "trek", "farewell", "prom"]
gender_male_keywords = ["men", "mens", "men's", "male", "man", "guy", "boy", "boys", "gents", "bro"]
gender_female_keywords = ["women", "womens", "women's", "female", "woman", "girl", "girls", "lady", "ladies"]
gender_unisex_keywords = ["unisex", "gender neutral", "both", "anyone", "neutral"]


class QueryAgent:
    """
    Analyzes user input and extracts structured intent.
    Outputs JSON only. Does NOT recommend products.
    """
    
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)
    
    async def understand_query(self, user_input: str, conversation_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze user input and extract intent.
        
        Returns:
        {
            "intent_type": "search" | "recommend" | "compare" | "browse" | "info",
            "product_category": str,
            "filters": {
                "price": {"min": int, "max": int},
                "brand": str,
                "features": List[str]
            },
            "ambiguity_level": "low" | "medium" | "high"
        }
        """
        # ============================================================
        # CONTEXT EXTRACTION FROM CONVERSATION HISTORY (CRITICAL FIX)
        # ============================================================
        context_info = ""
        pending_category = None
        pending_filters = {}
        pending_gender = None
        pending_occasion = None
        
        if conversation_history and len(conversation_history) > 0:
            # Look at recent messages to extract context
            recent_messages = conversation_history[-6:]  # Last 3 exchanges
            
            context_parts = []
            for i, msg in enumerate(recent_messages):
                role = msg.get("role", "user")
                content = msg.get("content", "").lower()
                
                # Extract category from previous messages using centralized config
                detected = normalize_category(content)
                if detected:
                    pending_category = detected
                
                # Extract gender from previous user messages
                if role == "user":
                    if any(kw in content for kw in gender_male_keywords):
                        pending_gender = "male"
                    elif any(kw in content for kw in gender_female_keywords):
                        pending_gender = "female"
                    elif any(kw in content for kw in gender_unisex_keywords):
                        pending_gender = "unisex"
                
                # Extract occasion from previous user messages
                if role == "user":
                    for kw in party_keywords:
                        if kw in content:
                            pending_occasion = kw
                            break
                
                # Check if AI asked for gender
                if role == "assistant" and ("men's, women's" in content or "gender" in content):
                    context_parts.append("AI previously asked for gender")
                
                # Check if AI asked for size
                if role == "assistant" and ("size" in content or "what size" in content or "fit btw" in content):
                    context_parts.append("AI previously asked for size")
                
                # Check if AI asked for vibe
                if role == "assistant" and ("vibe" in content or "going for" in content):
                    context_parts.append("AI previously asked for style vibe")

                # Check if AI asked for design
                if role == "assistant" and ("loud" in content or "clean" in content or "prints" in content):
                    context_parts.append("AI previously asked for design preference")
                
                # Check if AI asked for color
                if role == "assistant" and ("color" in content or "what color" in content):
                    context_parts.append("AI previously asked for color preference")
                
                # Check if AI asked for occasion/buying type
                if role == "assistant" and ("occasion" in content or "party" in content or "wedding" in content):
                    context_parts.append("AI previously asked about occasion")
            
            if pending_category:
                context_parts.append(f"User was searching for: {pending_category}")
            if pending_gender:
                context_parts.append(f"User's gender preference: {pending_gender}")
            if pending_occasion:
                context_parts.append(f"User's occasion: {pending_occasion}")
            
            if context_parts:
                context_info = f"""
CONVERSATION CONTEXT (CRITICAL - USE THIS):
{chr(10).join(context_parts)}

If user's current message is short/ambiguous (like a size "medium", "large", "small" or color "blue", "white"):
- This is likely a RESPONSE to AI's question, NOT a new search
- Use pending_category: "{pending_category}" for the search
- intent_type should be "search" with the pending category
- ALWAYS carry forward gender "{pending_gender}" and occasion "{pending_occasion}" from context
"""
        
        # Check if current input is just a short response to AI's question
        user_lower = user_input.lower().strip()
        
        is_size_response = user_lower in size_keywords or any(user_lower == kw for kw in size_keywords)
        is_color_response = user_lower in color_keywords
        is_vibe_response = user_lower in vibe_keywords
        # Fuzzy match for party response
        is_party_response = any(kw in user_lower for kw in party_keywords) and len(user_lower.split()) <= 4
        # Gender detection — match full word or phrase
        detected_gender = None
        if any(kw in user_lower.split() or user_lower == kw for kw in gender_male_keywords):
            detected_gender = "male"
        elif any(kw in user_lower.split() or user_lower == kw for kw in gender_female_keywords):
            detected_gender = "female"
        elif any(kw in user_lower for kw in gender_unisex_keywords):
            detected_gender = "unisex"
        is_gender_response = detected_gender is not None
        
        # GENDER FAST-PATH: Immediately return with gender when user answers gender question
        if is_gender_response:
            logger.info(f"[QueryAgent] Fast-path: gender detected '{detected_gender}' from '{user_input}'")
            # Also check if there's an occasion or category in the same message
            occasion_in_msg = None
            for kw in party_keywords:
                if kw in user_lower and kw not in ["formal", "office"]:
                    occasion_in_msg = kw
                    break
            
            category_in_msg = normalize_category(user_lower)
            
            result = {
                "intent_type": "search",
                "product_category": category_in_msg or pending_category,
                "gender": detected_gender,
                "occasion": occasion_in_msg or pending_occasion,
                "buying_type": "occasion" if occasion_in_msg else None,
                "filters": {},
                "preferences": {},
                "ambiguity_level": "low",
                "context_restored": True
            }
            return result
        
        # SHORT RESPONSE FAST-PATH (size/color/vibe/party) — carry gender from history
        if (is_size_response or is_color_response or is_vibe_response or is_party_response) and pending_category:
            logger.info(f"[QueryAgent] Detected follow-up response '{user_input}' for pending category '{pending_category}'")
            
            result = {
                "intent_type": "search",
                "product_category": pending_category,
                "gender": pending_gender,  # Carry forward from history!
                "occasion": (user_input if is_party_response else pending_occasion),
                "buying_type": "occasion" if is_party_response else None,
                "filters": {
                    "size": user_input if is_size_response else None,
                    "color": user_input if is_color_response else None,
                    "vibe": user_input if is_vibe_response else None,
                    "sub_occasion": user_input if is_party_response else None
                },
                "sub_occasion": user_input if is_party_response else None,
                "preferences": {
                    "vibe": user_input if is_vibe_response else None
                },
                "ambiguity_level": "low",
                "context_restored": True
            }
            return result
        
        prompt = f"""
        You are a Query Understanding Agent for a conversational recommendation system.
        
        Analyze this user input: "{user_input}"
        
        {context_info}
        
        CRITICAL: First check intent type in this order:
        
        1. PREFERENCE UPDATE (user giving feedback):
           - "don't like", "dont like", "hate", "dislike" → preference_update
           - "I like X", "I prefer X", "I want X" (when X is a style/fit/attribute) → preference_update
           - "not X", "avoid X" → preference_update
        
        2. RECOMMEND (user wants suggestions):
           - "suggest me some", "suggest something", "recommend", "what do you recommend" → recommend
           - "show me options", "what's good", "what should I get" → recommend
        
        3. BROWSE (user wants to explore):
           - "what's trending", "what's new", "show me what you have" → browse
        
        Examples:
        - "suggest me some" → recommend
        - "I don't like oversized, I like casual fits" → preference_update
        - "what's trending" → browse
        
        Extract and return ONLY valid JSON with these fields:
        {{
            "intent_type": "rejection" | "search" | "recommend" | "compare" | "browse" | "info" | "greeting" | "small_talk" | "preference_update" | "outfit_completion" | "product_appreciation",
            "user_name": "string or null (extract from 'I am X' or 'my name is X')",
            "gender": ["male" | "female" | "unisex"] | null,
            "buying_type": "regular" | "occasion" | null,
            "product_category": "string or null (USE CONTEXT if current message is ambiguous)",
            "occasion": "string or null (e.g., 'college', 'party', 'wedding', 'office', 'gym')",
            "sub_occasion": "string or null (e.g., 'clubbing', 'house party', 'formal wedding', 'daily wear')",
            "filters": {{
                "price": {{"min": number or null, "max": number or null}},
                "brand": "string or null",
                "features": ["string"],
                "color": "string or null",
                "style": "string or null",
                "fit": "string or null",
                "size": "string or null"
            }},
            "preferences": {{
                "vibe": "string or null (e.g., 'minimal', 'streetwear')",
                "design_preference": "string or null (e.g., 'loud', 'clean')",
                "liked_fits": ["string"],
                "disliked_fits": ["string"],
                "preferred_fit": "string or null",
                "disliked_colors": ["string"],
                "disliked_styles": ["string"]
            }},
            "wardrobe_context": {{
                "user_has": ["string - items user already owns"],
                "needs": "string - what user is looking for", 
                "colors_mentioned": ["string - colors in user's existing items"]
            }},
            "product_appreciation": {{
                "is_appreciating": true/false,
                "product_reference": "string - which product (e.g., 'first one', 'the black one', 'XYZ jeans')",
                "liked_features": ["string - what they liked: 'color', 'style', 'fit', 'price', 'brand', 'design']",
                "sentiment": "string - 'love', 'like', 'interested', 'fire'"
            }},
            "ambiguity_level": "low" | "medium" | "high"
        }}
        
        Rules:
        1. intent_type (PRIORITY ORDER):
           - "rejection": User rejects the SHOWN results and wants ALTERNATIVES. This is about the LIST, not a general style preference.
             Examples: "don't like these", "show me different ones", "not feeling this", "anything besides these?", "none of these work", 
                       "too basic", "boring options", "no, other ones", "again showing same products", "show different items", 
                       "same ass products", "not what I wanted", "try again"
           - "greeting": Simple greetings like "hi", "hello", "hey", "what's up"
           - "product_appreciation": User expressing POSITIVE interest/liking for a SPECIFIC PRODUCT shown
             Examples: "I like this one", "the first one looks fire", "love the black jeans", "this is nice"
                      "oooh I like it", "yeah this is the one", "perfect", "I'll take this"
                      "the second option is great", "XYZ jeans look good"
            - "preference_update": User is giving general feedback/preferences (e.g., "don't like oversized", "prefer casual", "I like regular fits")
            - "small_talk": Casual conversation not about products
            - "search": User wants to find specific products
            - "recommend": User wants personalized recommendations (e.g., "suggest me some", "what do you recommend", "show me options")
            - "compare": User wants to compare products
            - "browse": User wants to explore categories (e.g., "what's trending", "show me what you have")
            - "info": User wants information about products
        
        2. REJECTION vs PREFERENCE_UPDATE (CRITICAL):
           - If user says "don't like THESE" or "show OTHER ones" → rejection
           - If user says "I don't like OVERSIZED" or "I prefer BLUE" → preference_update
           - Rejection is about the LIST shown; Preference Update is about STYLE/FIT.
        
        3. OUTFIT_COMPLETION detection (CRITICAL):
           - If user says "I have X" or "I'm wearing X" or "I got X" → extract to wardrobe_context.user_has
           - If user asks "what to wear with" or "suggest something above/below" → intent is outfit_completion
           - Extract colors from existing items for matching suggestions
        
        4. CONTEXT USAGE (CRITICAL FOR FOLLOW-UP MESSAGES):
           - If conversation context mentions a pending category (like "jeans"), USE THAT CATEGORY
           - If user just says a size like "medium" or "large", it's a response to size question
           - Set product_category to the pending category from context
        
        4. IMPORTANT: Differentiate Feedback (CRITICAL):
           - "don't like THESE" or "none of THESE work" or "show me DIFFERENT ones" → rejection (about current list)
           - "don't like regular jeans" or "prefer blue" → preference_update (about general style)
           - "don't like [ATTRIBUTE]" → preference_update
           - Only set disliked_fits, disliked_colors, or preferred_fit if a SPECIFIC attribute (baggy, slim, white, blue, etc.) is mentioned.
           - If user says a size like "medium" or "large" after being asked → info or search (response to size)
        
        6. Gender Extraction (CRITICAL - EXAMPLES):
           - Can be a list if multiple are mentioned (e.g., "mens and unisex" → ["male", "unisex"]).
           - "men", "mens", "male", "guy", "boy" → "male"
           - "women", "womens", "female", "girl", "lady" → "female"  
           - "unisex", "gender neutral", "for both" → "unisex"
           
           EXAMPLES:
           - "mens" → "gender": "male"
           - "mens obviously" → "gender": "male"
           - "looking for mens stuff" → "gender": "male"
           - "womens clothing" → "gender": "female"
           - "I'm a guy" → "gender": "male"
           
        7. Buying Type Extraction (CRITICAL):
           - If user mentions specific event (party, wedding, etc.) → occasion
           - If user wants "regular", "daily", "everyday", "normal" clothes → regular
           - Default to null if not clear.
           
        8. Sub-Occasion Guard (CRITICAL):
           - DO NOT guess sub_occasion (clubbing, house party) unless explicitly mentioned or strongly implied by keywords (e.g., "dance", "dinner", "party at home").
           - If user just says "party", leave sub_occasion as null to trigger discovery flow.
           
        8. Fit extraction:
           - "oversized" → oversized
           - "slim" or "tight" → slim
           - "regular" or "casual" or "normal" → regular
           - "loose" or "baggy" → loose
        
        9. Price extraction (CRITICAL):
           - "under 2000" or "below 2000" or "< 2000" → {{"max": 2000}}
           - "over 5000" or "above 5000" or "> 5000" → {{"min": 5000}}
           - "around 1000" or "~1000" → {{"min": 800, "max": 1200}}
           - "between 500 and 1000" → {{"min": 500, "max": 1000}}
           - Extract numbers from text (e.g., "2000 rupees" → 2000)
        
        10. Category extraction (IMPORTANT - use flexible matching):
           - "sneakers", "shoes", "footwear", "trainers", "loafers", "boots" → "sneakers"
           - "t shirt", "tshirt", "tee", "tops", "upper wear", "top wear" → "shirt"
           - "jersey", "jerseys", "sports jersey", "jer sey", "jersy", "jersys" → "jersey"
            
        8. Occasion Extraction:
            - "party" is too broad. Extract specific sub_occasion if mentioned (club, house part, wedding, etc).
            - "college" implies casual/oversized Gen-Z style.
           - "jeans", "denims", "denim", "pants" → "jeans"
           - "bottom wear", "bottoms", "trousers", "chinos" → "bottom"
           - "hoodie", "hoddie", "sweatshirt", "sweater", "pullover" → "hoodie"
           - "jacket", "coat", "blazer", "puffer", "bomber", "overcoat" → "jacket"
           - "socks", "sock" → "socks"
           - "kurta", "kurti", "ethnic wear", "traditional", "sherwani" → "kurta"
           - "polo", "polos" → "polo"
           - "shorts", "half pants" → "shorts"
           - "joggers", "trackpants", "sweatpants" → "joggers"
           - "cargo", "cargos" → "cargos"
           - "tracksuit", "activewear", "gymwear", "workout" → "activewear"
           - "cap", "hat", "beanie" → "cap"
           - Be flexible with synonyms and common misspellings (e.g., "jersys", "botttom")
           - IMPORTANT: For jerseys, return "jersey" not "shirt"
        
        8. ambiguity_level:
           - "low": Clear intent, specific requirements (has category AND price)
           - "medium": Some requirements, but needs clarification
           - "high": Vague or unclear intent
        
        9. Do NOT recommend products. Only extract intent.
        10. Return valid JSON only. No markdown, no explanation.
        11. HANDLING MULTIPLE INTENTS (CRITICAL):
           - If user says "I like/prefer X so show me Y" or "Already have X, want Y":
             → intent_type is "search" or "recommend" (whichever applies to Y)
             → product_category is Y
             → extract X into preferences or wardrobe_context
           Example: "i prefer puffer jacket so for layering i want a jer sey"
             → intent_type: "search"
             → product_category: "jersey"
             → product_appreciation: {{ "is_appreciating": true, "product_reference": "puffer jacket", "sentiment": "interested" }}
        
        """
        
        try:
            completion = self.client.chat.completions.create(
                model=settings.GROQ_MODEL_SMALL,  # Use smaller model for fast intent extraction
                messages=[
                    {
                        "role": "system",
                        "content": "You are a specialized JSON parser. Return ONLY valid JSON. No markdown, no explanation."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(completion.choices[0].message.content)
            
            # Validate and set defaults
            if "intent_type" not in result:
                result["intent_type"] = "search"
            if "ambiguity_level" not in result:
                result["ambiguity_level"] = "medium"
            if "filters" not in result:
                result["filters"] = {}
            if "product_category" not in result:
                result["product_category"] = None
            if "preferences" not in result:
                result["preferences"] = {}
            if "wardrobe_context" not in result:
                result["wardrobe_context"] = {"user_has": [], "needs": None, "colors_mentioned": []}
            
            return result
            
        except Exception as e:
            logger.error(f"Query understanding error: {e}")
            # Safe default
            return {
                "intent_type": "search",
                "product_category": None,
                "filters": {},
                "ambiguity_level": "medium"
            }


# Singleton instance
query_agent = QueryAgent()
