"""
Gen-Z Fashion Trend Agent - Expert in current Gen-Z fashion aesthetics.

Understands:
- Current Gen-Z fashion aesthetics
- Streetwear, Y2K, clean-boy, old-money, gorpcore, techwear
- Social media driven fashion cycles (Instagram, Pinterest, TikTok)
- Slang and style descriptors used by Gen-Z
"""

from typing import Dict, Any, List
from app.core.config import get_settings
from groq import Groq
import logging
import json

logger = logging.getLogger(__name__)
settings = get_settings()


class GenZTrendAgent:
    """
    Gen-Z fashion trend expert.
    Identifies relevant fashion aesthetics and provides styling logic.
    Does NOT suggest products or access databases.
    """
    
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)
    
    async def analyze_trend_context(
        self,
        user_query: str,
        user_memory: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Analyze user query and identify relevant Gen-Z fashion aesthetics.
        
        Returns:
        {
            "aesthetic": "clean boy / street casual",
            "fit_logic": "relaxed silhouettes, neutral tones, minimal logos",
            "color_palette": ["black", "cream", "olive"],
            "vibe_keywords": ["lowkey", "effortless", "daily wear"],
            "outfit_structure": {
                "top": "oversized tee or hoodie",
                "bottom": "relaxed cargos or jeans",
                "footwear": "white sneakers or chunky sneakers",
                "layer": "optional denim jacket or puffer"
            }
        }
        """
        # Get user's preferred aesthetics from memory
        preferred_aesthetics = []
        style_vibe = "casual"
        if user_memory:
            long_term = user_memory.get("long_term", {})
            preferred_aesthetics = long_term.get("fashion_aesthetics", [])
            style_vibe = long_term.get("style_vibe", "casual")
        
        prompt = f"""
        You are a Gen-Z fashion trend expert who understands current aesthetics and styling.
        
        User query: "{user_query}"
        User's known preferences: {json.dumps(preferred_aesthetics) if preferred_aesthetics else "none"}
        User's style vibe: {style_vibe}
        
        Analyze the query and return ONLY valid JSON with:
        {{
            "aesthetic": "string (e.g., 'clean boy', 'streetwear', 'Y2K', 'old money', 'gorpcore', 'techwear', 'minimalist')",
            "fit_logic": "string explaining the styling approach",
            "color_palette": ["color1", "color2", "color3"],
            "vibe_keywords": ["keyword1", "keyword2", "keyword3"],
            "outfit_structure": {{
                "top": "description of top piece",
                "bottom": "description of bottom piece",
                "footwear": "description of footwear",
                "layer": "optional layer or accessory"
            }},
            "trend_relevance": "high" | "medium" | "low"
        }}
        
        Rules:
        1. Identify the aesthetic based on query and user preferences
        2. Provide styling logic that explains WHY this aesthetic works
        3. Suggest color palettes that are currently trending
        4. Use Gen-Z vibe keywords (lowkey, fire, drip, clean, etc.)
        5. Structure outfit suggestions (top, bottom, footwear, layer)
        6. Assess trend relevance based on current Gen-Z fashion cycles
        
        Common Gen-Z Aesthetics:
        - "clean boy": minimal, neutral, oversized, effortless
        - "streetwear": bold logos, baggy fits, statement pieces
        - "Y2K": nostalgic 2000s, low-rise, butterfly motifs
        - "old money": preppy, classic, quality basics
        - "gorpcore": outdoor aesthetic, technical fabrics
        - "techwear": futuristic, functional, dark tones
        
        Return valid JSON only. No markdown, no explanation.
        """
        
        try:
            completion = self.client.chat.completions.create(
                model=settings.GROQ_MODEL_SMALL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a Gen-Z fashion trend expert. Return ONLY valid JSON. No markdown, no explanation."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(completion.choices[0].message.content)
            
            # Validate and set defaults
            if "aesthetic" not in result:
                result["aesthetic"] = "casual"
            if "fit_logic" not in result:
                result["fit_logic"] = "comfortable and stylish"
            if "color_palette" not in result:
                result["color_palette"] = ["black", "white", "gray"]
            if "vibe_keywords" not in result:
                result["vibe_keywords"] = ["casual", "comfortable"]
            if "outfit_structure" not in result:
                result["outfit_structure"] = {
                    "top": "t-shirt or hoodie",
                    "bottom": "jeans or cargos",
                    "footwear": "sneakers"
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Gen-Z trend analysis error: {e}")
            # Safe default
            return {
                "aesthetic": "casual",
                "fit_logic": "comfortable and stylish everyday wear",
                "color_palette": ["black", "white", "gray"],
                "vibe_keywords": ["casual", "comfortable"],
                "outfit_structure": {
                    "top": "t-shirt or hoodie",
                    "bottom": "jeans or cargos",
                    "footwear": "sneakers"
                },
                "trend_relevance": "medium"
            }
    
    async def suggest_trending_outfit(
        self,
        user_memory: Dict[str, Any],
        occasion: str = None,
        budget: Dict[str, int] = None,
        category_hint: str = None
    ) -> Dict[str, Any]:
        """
        Proactively suggest a trending outfit based on user DNA.
        Called when user wants recommendations without specific product in mind.
        
        Returns:
        {
            "suggested_aesthetic": "clean boy",
            "outfit_pieces": ["oversized white tee", "relaxed jeans", "white sneakers"],
            "search_terms": ["oversized t-shirt", "relaxed fit jeans", "white sneakers"],
            "trend_reason": "Clean boy aesthetic is huge on TikTok right now",
            "personalization": "Your minimalist vibe matches this perfectly",
            "color_palette": ["white", "cream", "black"],
            "confidence": 0.85
        }
        """
        # Extract user DNA
        long_term = user_memory.get("long_term", {})
        style_vibe = long_term.get("style_vibe", "casual")
        preferred_fit = long_term.get("preferred_fit", "regular")
        fashion_aesthetics = long_term.get("fashion_aesthetics", [])
        liked_colors = long_term.get("style_preferences", {}).get("likedColors", [])
        disliked_fits = long_term.get("disliked_fits", [])
        gender = long_term.get("behavioral_patterns", {}).get("gender", "neutral")
        
        # Build context
        occasion_context = f"Occasion: {occasion}" if occasion else "Occasion: everyday casual"
        budget_context = f"Budget: ₹{budget.get('min', 0)}-₹{budget.get('max', 10000)}" if budget else "Budget: flexible"
        category_context = f"Category focus: {category_hint}" if category_hint else ""
        
        prompt = f"""
        You are a Gen-Z fashion trend expert. Based on the user's style DNA, suggest a COMPLETE trending outfit.
        
        USER STYLE DNA:
        - Vibe: {style_vibe}
        - Preferred Fit: {preferred_fit}
        - Known Aesthetics: {json.dumps(fashion_aesthetics) if fashion_aesthetics else "none yet"}
        - Liked Colors: {json.dumps(liked_colors) if liked_colors else "unknown"}
        - Disliked Fits: {json.dumps(disliked_fits) if disliked_fits else "none"}
        - Gender: {gender}
        
        CONTEXT:
        {occasion_context}
        {budget_context}
        {category_context}
        
        CURRENT TRENDING AESTHETICS (2024-2025):
        1. "clean boy" - neutral tones, oversized fits, minimal branding, effortless
        2. "quiet luxury" - understated elegance, quality basics, no logos
        3. "streetwear revival" - baggy jeans, graphic tees, chunky sneakers
        4. "Y2K comeback" - low-rise, butterfly motifs, metallics
        5. "gorpcore" - outdoor aesthetic, technical fabrics, earth tones
        6. "old money" - preppy, classic, quality over quantity
        
        Return ONLY valid JSON:
        {{
            "suggested_aesthetic": "string (one of the trending aesthetics)",
            "outfit_pieces": ["piece1", "piece2", "piece3", "optional_accessory"],
            "search_terms": ["searchable term 1", "searchable term 2", "searchable term 3"],
            "trend_reason": "Why this aesthetic is trending right now (1 sentence)",
            "personalization": "Why this fits the user's DNA (1 sentence, reference their preferences)",
            "color_palette": ["color1", "color2", "color3"],
            "confidence": 0.0-1.0 (how well this matches the user)
        }}
        
        RULES:
        1. Match the user's existing vibe as closely as possible
        2. Avoid suggesting fits the user dislikes
        3. Use searchable terms that will find products in database (e.g., "oversized t-shirt" not "relaxed tee")
        4. Be specific about colors and styles
        5. Make personalization feel genuine, not generic
        """
        
        try:
            completion = self.client.chat.completions.create(
                model=settings.GROQ_MODEL_SMALL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a Gen-Z fashion trend expert. Return ONLY valid JSON."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.8,  # Slightly higher for creative suggestions
                response_format={"type": "json_object"}
            )
            
            result = json.loads(completion.choices[0].message.content)
            
            # Validate and set defaults
            if "suggested_aesthetic" not in result:
                result["suggested_aesthetic"] = "casual"
            if "outfit_pieces" not in result:
                result["outfit_pieces"] = ["t-shirt", "jeans", "sneakers"]
            if "search_terms" not in result:
                result["search_terms"] = result.get("outfit_pieces", [])
            if "trend_reason" not in result:
                result["trend_reason"] = "Current trending style"
            if "personalization" not in result:
                result["personalization"] = "This matches your style"
            if "color_palette" not in result:
                result["color_palette"] = ["black", "white", "gray"]
            if "confidence" not in result:
                result["confidence"] = 0.7
            
            logger.info(f"[GenZTrend] Suggested: {result['suggested_aesthetic']} with confidence {result['confidence']}")
            return result
            
        except Exception as e:
            logger.error(f"Trend suggestion error: {e}")
            # Safe default based on user vibe
            return {
                "suggested_aesthetic": style_vibe if style_vibe else "casual",
                "outfit_pieces": ["relaxed t-shirt", "comfortable jeans", "white sneakers"],
                "search_terms": ["t-shirt", "jeans", "sneakers"],
                "trend_reason": "Classic combinations never go out of style",
                "personalization": "These basics work with any wardrobe",
                "color_palette": ["black", "white", "gray"],
                "confidence": 0.5
            }


# Singleton instance
genz_trend_agent = GenZTrendAgent()
