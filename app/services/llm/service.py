from groq import Groq
from app.core.config import get_settings
from app.services.llm.stylist import stylist_service
from typing import List, Dict, Any
import logging
import asyncio
import traceback
import json
from datetime import datetime

settings = get_settings()
logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)

    async def generate_response(self, user_query: str, products: List[Dict[str, Any]], intent: str, memory: dict = None, context: dict = None) -> str:
        """
        Converts structured product data into a friendly, professional response.
        Uses DNA Flash for instant user context.
        """
        user_name = (memory or {}).get("userName", "Friend")
        if not user_name or user_name.lower() in ["trendora", "trediora", "tredora", "none", "gorgeous"]:
            user_name = "Friend"
        
        # 0. Build DNA Flash (KEY FLASH summary of user dimensions)
        from app.services.fashion.silent_extractor import silent_extractor
        dna_flash = silent_extractor.build_dna_flash(memory)
        
        gender_val = (memory or {}).get("behavior", {}).get("gender", "unknown")

        
        # 1. Build Conversation Context (Last 15 messages for "entire session" feel)
        history_str = ""
        if context and context.get("conversationHistory"):
            recent = context["conversationHistory"][-15:]
            for h in recent:
                history_str += f"{h['role'].capitalize()}: {h['content']}\n"

        # 2. Determine Dynamic Content & Verbosity
        style_tip = await stylist_service.get_style_tip(user_query)
        
        # Verbosity Level
        query_len = len(user_query.split())
        if intent == "DISMISSIVE":
            verbosity = "ENGAGING: Still witty and descriptive, but don't push products. 2-3 sentences."
            mode = "REACTIVE: Acknowledge the vibe wittily, but stay supportive."
        elif products:
            verbosity = "STYLING_DOSSIER: Generous and detailed. 4-6 sentences."
            mode = "PREMIUM_CURATOR: Provide a deep-dive into why these pieces are iconic for them."
        else:
            verbosity = "CONVERSATIONAL: Warm and expressive. 3-4 sentences."
            mode = "FASHION_CONFIDANTE: Engage deeply with their mood and style journey."

        product_context = ""
        if products:
            for p in products:
                product_context += f"- {p.get('name', 'Item')} (₹{p.get('price', 0)}): {p.get('description', '')}\n"
        else:
            product_context = "No specific items found."

        prompt = f"""
        You are 'Trendora', a professional AI Style Consultant with a master's degree in fashion theory.
        You are direct, stylish, and extremely accurate.
        
        Identity: Trendora (Expert Stylist).
        Current User: {user_name}
        {dna_flash}
        
        Available Collection:
        {product_context}
        
        Stylist Directives:
        1. **Product Fidelity**: ONLY recommend items explicitly listed in the "Available Collection". NEVER hallucinate brands or item names.
        2. **Fashion Logic**: Briefly explain your tips using professional logic (e.g., body proportion, color harmony).
        3. **Lowkey Vibe**: Be helpful and sophisticated. Keep it to 3-4 sentences.
        4. **Desi Flair**: It's okay to use a little Hinglish (e.g., "Arre", "Perfect fits"), but keep it professional.
        5. **Strict No-Tech**: Your job is fashion only. If a user asks for tech, politely tell them your collection is focused on style.

        """
        
        # Verbosity is now handled by the 'Eloquence' rule in the prompt itself.

        for attempt in range(2):
            try:
                completion = self.client.chat.completions.create(
                    model=settings.GROQ_MODEL, # Back to 70B but with GROUNDED prompt
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7
                )
                return completion.choices[0].message.content
            except Exception as e:
                if "rate_limit_exceeded" in str(e).lower() and attempt == 0:
                    await asyncio.sleep(2)
                    continue
                
                error_trace = traceback.format_exc()
                logger.error(f"Response Generation Error: {e}\n{error_trace}")
                # Fallback to 8B with same persona
                try:
                    completion_small = self.client.chat.completions.create(
                        model=settings.GROQ_MODEL_SMALL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7
                    )
                    return completion_small.choices[0].message.content
                except Exception as fallback_err:
                    logger.error(f"Critical Fallback Failure: {fallback_err}")
                    return "Arre, thoda network ka chakkar hai! But your Trendora stylist is still here. Let's try that again, shall we?"

    async def generate_response_with_outfits(
        self,
        user_query: str,
        products: List[Dict[str, Any]],
        outfits: List[Dict[str, Any]],
        intent: str,
        memory: dict = None,
        context: dict = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Enhanced response generator that understands complete outfits.
        Uses AI scoring insights to explain WHY outfits work.
        """
        user_name = (memory or {}).get("userName", "Friend")
        if not user_name or user_name.lower() in ["trendora", "trediora", "tredora", "none", "gorgeous"]:
            user_name = "Friend"
        
        style_dna = (memory or {}).get("style", {})
        gender_val = (memory or {}).get("behavior", {}).get("gender", "unknown")
        
        # Build conversation history
        history_str = ""
        if context and context.get("conversationHistory"):
            recent = context["conversationHistory"][-10:]
            for h in recent:
                history_str += f"{h['role'].capitalize()}: {h['content']}\n"
        
        # Build outfit description for the SINGLE perfect look
        outfit_context = ""
        if outfits:
            outfit = outfits[0]
            items = outfit.get("items", [])
            score = outfit.get("score", 0)
            selection_type = outfit.get("selection_type", "maximum_harmony")
            insights = outfit.get("insights", [])
            total_price = outfit.get("total_price", 0)
            
            outfit_context = f"PERFECT OUTFIT SELECTION:\n"
            for item in items:
                outfit_context += f"- {item.get('category', '').title()}: {item.get('name', 'Item')} (Color: {item.get('color', 'N/A')}, Brand: {item.get('brand', 'N/A')})\n"
            
            outfit_context += f"\nHarmony Score: {score:.2f} ({selection_type})\n"
            outfit_context += f"Dynamic Insights: {', '.join(insights) if insights else 'Tailored specifically for your DNA'}\n"
            outfit_context += f"Total Value: ₹{total_price}\n"
        else:
            outfit_context = "No complete outfits found. Please suggest browsing single items."

        
        # Product fallback context
        product_context = ""
        if products and not outfits:
            for p in products[:5]:
                product_context += f"- {p.get('name', 'Item')} (₹{p.get('price', 0)})\n"
        
        # Metadata insights
        exploration_mode = (metadata or {}).get("exploration_mode", False)
        mind_change = (metadata or {}).get("mind_change_detected", False)
        single_outfit_mode = (metadata or {}).get("single_outfit_mode", True)
        
        # Build DNA Flash for instant context
        from app.services.fashion.silent_extractor import silent_extractor
        dna_flash = silent_extractor.build_dna_flash(memory)
        
        mode_hint = ""
        if exploration_mode:
            mode_hint = "User is in EXPLORATION mode - show variety and experimental options."
        if mind_change:
            mode_hint += " Preference shift detected - be open to new directions."
        
        prompt = f"""
        You are 'Trendora', a professional AI Style Advisor.
        Your task is to present ONE curated outfit to the user based on the following data.
        
        PERFECT OUTFIT:
        {outfit_context}
        
        Stylist Directives:
        1. **Product Accuracy**: ONLY mention items exactly as they appear in the "PERFECT OUTFIT" list.
        2. **Brevity**: Limit your response to 2 sentences. Present the items and briefly state why they match the user's request.
        3. **Professional Tone**: Be helpful and direct. No excessive enthusiasm or Hinglish.
        4. **Natural Follow-up**: End with a short style question.
        """
        
        for attempt in range(2):
            try:
                completion = self.client.chat.completions.create(
                    model=settings.GROQ_MODEL, # Back up to 70B
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.75
                )
                return completion.choices[0].message.content
            except Exception as e:
                if "rate_limit_exceeded" in str(e).lower() and attempt == 0:
                    await asyncio.sleep(2)
                    continue
                    
                error_trace = traceback.format_exc()
                logger.error(f"Outfit Response Generation Error: {e}\n{error_trace}")
                # Fallback to 8B with same persona
                try:
                    return await self.client.chat.completions.create(
                        model=settings.GROQ_MODEL_SMALL,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.75
                    ).choices[0].message.content
                except:
                    # Final fallback to basic response
                    return await self.generate_response(user_query, products, intent, memory, context)

    async def generate_thinking_steps(self, query: str, intent: str) -> List[str]:
        """
        Quickly generates 3 context-aware thinking steps using the 8B model.
        """
        prompt = f"""
        You are Trendora's AI assistant. Based on user query: "{query}" and intent: "{intent}",
        give 3 extremely short, punchy 'thinking steps' in Hinglish.
        
        Rules:
        1. Mix Hindi & English (e.g., "Scanning database...", "Arre, checking fits...").
        2. Max 3-4 words per step.
        3. Output ONLY a JSON array of 3 strings.
        """
        try:
            completion = self.client.chat.completions.create(
                model=settings.GROQ_MODEL_SMALL, # 8B is perfect for this
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=100
            )
            content = completion.choices[0].message.content.strip()
            # Try to find JSON array in the text if it's not pure JSON
            if "[" in content and "]" in content:
                content = content[content.find("["):content.rfind("]")+1]
            data = json.loads(content)
            steps = data.get("steps", data) if isinstance(data, dict) else data
            return steps[:3] if isinstance(steps, list) else ["Scanning products...", "Applying fashion theory...", "Finalizing picks..."]
        except Exception as e:
            logger.error(f"Thinking Steps Error: {e}")
            return ["Scanning Trendora database...", "Analyzing your vibe...", "Curating the best fits..."]

    async def extract_json(self, prompt: str) -> Dict[str, Any]:
        """
        Generic helper to extract structured JSON from prompt.
        """
        try:
            completion = self.client.chat.completions.create(
                model=settings.GROQ_MODEL_SMALL,
                messages=[{
                    "role": "system", 
                    "content": "You are a specialized JSON parser. Return ONLY valid JSON. No markdown, no explanation."
                }, {
                    "role": "user", 
                    "content": prompt
                }],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            import json
            return json.loads(completion.choices[0].message.content)
        except Exception as e:
            logger.error(f"JSON Extraction Error: {e}")
            return {}

    async def extract_preferences(self, query: str) -> Dict[str, Any]:
        """
        Extract latent style preferences from a query.
        """
        prompt = f"""
        Extract style preferences from this query: "{query}"
        Return JSON with:
        - gender: male/female/neutral
        - vibe: traditional/modern/casual/formal/experimental
        - fit: slim/regular/oversized
        - colors: list of colors mentioned
        """
        return await self.extract_json(prompt)

llm_service = LLMService()

