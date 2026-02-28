# Silent Insight Extractor
# Passively extracts user preferences from chat without asking directly

from groq import Groq
from app.core.config import get_settings
from app.db.mongodb import get_database
from typing import Dict, Any, Optional
import json
import logging
import re

settings = get_settings()
logger = logging.getLogger(__name__)


class SilentExtractor:
    """
    Silently extracts user insights from chat messages.
    No direct questions - just passive learning from natural conversation.
    """
    
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)
    
    async def extract_silent_insights(self, message: str, history: list = None) -> Dict[str, Any]:
        """
        Analyze a user message for latent preferences.
        Returns only non-null extracted values.
        """
        context = ""
        if history:
            for h in history[-5:]:
                context += f"{h.get('role', 'user').capitalize()}: {h.get('content', '')}\n"
        
        prompt = f"""
        You are a silent preference extractor. Analyze the user's message for ANY latent information about:
        
        1. **gender**: male/female/non-binary (look for: "I'm a guy", "as a woman", pronouns)
        2. **heightCm**: height in centimeters (convert from feet/inches if needed)
        3. **skinTone**: 0.0-1.0 scale (0.0=very light, 1.0=very dark). Look for: "fair", "dusky", "dark", "wheatish"
        4. **undertone**: warm/cool/neutral (look for mentions of gold/silver jewelry, veins, etc.)
        5. **vibe**: streetwear/minimalist/traditional/ethnic/casual/formal/bohemian/preppy
        6. **riskTolerance**: 0.0-1.0 (0=very safe, 1=very bold). Look for: "simple", "bold", "experimental", "classic"
        7. **likedColors**: array of colors they like (look for: "I love blue", "blue is my favorite")
        8. **rejectedColors**: array of colors they avoid (look for: "not a fan of red", "hate pink")
        9. **fit**: slim/regular/oversized preference
        10. **bodyShape**: ectoSlim/mesoAthletic/endoRound (look for body type mentions)
        11. **size**: S/M/L/XL/XXL/XS (look for: "I'm a medium", "size M", "large fits me", "I wear L")
        12. **shoeSize**: numeric shoe size (look for: "size 9", "UK 10", "42 EU", "I wear 8")
        13. **likes**: array of {{item, reason}} - things user explicitly likes and why
        14. **dislikes**: array of {{item, reason}} - things user explicitly dislikes and why
        
        Recent Context:
        {context}
        
        Current Message: "{message}"
        
        RULES:
        - Only return fields that have CLEAR evidence in the message
        - Return EMPTY JSON {{}} if nothing can be extracted
        - Be conservative - don't guess
        - Output ONLY valid JSON, no explanation
        
        Example outputs:
        - "I'm a tall guy, about 6 feet" → {{"gender": "male", "heightCm": 183}}
        - "I'm dusky and love blue" → {{"skinTone": 0.6, "likedColors": ["blue"]}}
        - "Just want something simple today" → {{"riskTolerance": 0.2}}
        - "I'm a medium, size 9 shoes" → {{"size": "M", "shoeSize": 9}}
        - "I hate oversized stuff, looks baggy on me" → {{"dislikes": [{{"item": "oversized fits", "reason": "looks baggy"}}]}}
        """
        
        try:
            completion = self.client.chat.completions.create(
                model=settings.GROQ_MODEL_SMALL,  # Reverting to 8B for speed/sanity
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200
            )
            
            content = completion.choices[0].message.content.strip()
            
            # Extract JSON from response
            if "{" in content and "}" in content:
                json_str = content[content.find("{"):content.rfind("}")+1]
                extracted = json.loads(json_str)
                
                # Filter out empty values
                result = {k: v for k, v in extracted.items() if v is not None and v != "" and v != []}
                
                if result:
                    logger.info(f"[SilentExtractor] Extracted: {result}")
                
                return result
            
            return {}
            
        except Exception as e:
            logger.error(f"[SilentExtractor] Error: {e}")
            return {}
    
    async def save_to_memory(self, user_id: str, email: str, insights: Dict[str, Any]) -> bool:
        """
        Merge extracted insights into UserMemory document.
        """
        if not insights:
            return False
        
        try:
            db = get_database()
            
            # Build update document
            update = {"$set": {"lastUpdated": __import__("datetime").datetime.utcnow()}}
            
            # Map insights to UserMemory fields
            field_mapping = {
                "gender": "behavior.gender",
                "heightCm": "physical.heightCm",
                "skinTone": "physical.skinTone",
                "undertone": "physical.undertone",
                "vibe": "style.vibe",
                "riskTolerance": "psychology.riskTolerance",
                "fit": "style.fit",
                "bodyShape": "physical.bodyShape",
                "size": "physical.size",  # NEW: clothing size
                "shoeSize": "physical.shoeSize"  # NEW: shoe size
            }
            
            for key, field_path in field_mapping.items():
                if key in insights:
                    update["$set"][field_path] = insights[key]
            
            # Handle array fields specially
            if "likedColors" in insights:
                update.setdefault("$addToSet", {})["revealed.likedColors"] = {"$each": insights["likedColors"]}
            
            if "rejectedColors" in insights:
                update.setdefault("$addToSet", {})["revealed.rejectedColors"] = {"$each": insights["rejectedColors"]}
            
            # NEW: Handle likes and dislikes arrays
            if "likes" in insights and isinstance(insights["likes"], list):
                update.setdefault("$addToSet", {})["revealed.likes"] = {"$each": insights["likes"]}
                logger.info(f"[SilentExtractor] Saving likes: {insights['likes']}")
            
            if "dislikes" in insights and isinstance(insights["dislikes"], list):
                update.setdefault("$addToSet", {})["revealed.dislikes"] = {"$each": insights["dislikes"]}
                logger.info(f"[SilentExtractor] Saving dislikes: {insights['dislikes']}")
            
            # Upsert to UserMemory
            query = {"userEmail": email} if email else {"userId": user_id}
            await db.usermemories.update_one(query, update, upsert=True)
            
            logger.info(f"[SilentExtractor] Saved insights for {email or user_id}")
            return True
            
        except Exception as e:
            logger.error(f"[SilentExtractor] Save error: {e}")
            return False
    
    def build_dna_flash(self, memory: dict) -> str:
        """
        Build a concise KEY FLASH summary of user dimensions for LLM context.
        """
        if not memory:
            return "USER DNA: New user, no preferences known yet."
        
        parts = []
        
        # Gender
        gender = (memory.get("behavior") or {}).get("gender")
        if gender:
            parts.append(f"Gender={gender}")
        
        # Physical
        physical = memory.get("physical") or {}
        if physical.get("heightCm"):
            parts.append(f"Height={physical['heightCm']}cm")
        if physical.get("skinTone") is not None:
            skin_desc = "light" if physical["skinTone"] < 0.3 else "medium" if physical["skinTone"] < 0.6 else "dusky"
            parts.append(f"Skin={skin_desc}")
        if physical.get("undertone"):
            parts.append(f"Undertone={physical['undertone']}")
        
        # Style
        style = memory.get("style") or {}
        if style.get("vibe"):
            parts.append(f"Vibe={style['vibe']}")
        if style.get("fit"):
            parts.append(f"Fit={style['fit']}")
        
        # Psychology
        psych = memory.get("psychology") or {}
        if psych.get("riskTolerance") is not None:
            risk_desc = "safe" if psych["riskTolerance"] < 0.3 else "moderate" if psych["riskTolerance"] < 0.7 else "bold"
            parts.append(f"Risk={risk_desc}")
        
        # Revealed preferences
        revealed = memory.get("revealed") or {}
        if revealed.get("likedColors"):
            parts.append(f"Likes=[{','.join(revealed['likedColors'][:3])}]")
        if revealed.get("rejectedColors"):
            parts.append(f"Avoids=[{','.join(revealed['rejectedColors'][:3])}]")
        
        if not parts:
            return "USER DNA: New user, building profile..."
        
        return "USER DNA: " + ", ".join(parts)


# Singleton instance
silent_extractor = SilentExtractor()
