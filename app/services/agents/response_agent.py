"""
Response Agent - Generates final user-facing response.

Rules:
- Conversational tone with natural acknowledgments
- Clear reasoning
- Transparent recommendation logic
- Offer next actions (compare, filter, buy)

Never mention internal agents or system architecture.
Never expose raw JSON.
"""

from typing import List, Dict, Any, Optional
from app.services.llm.service import llm_service
from app.services.agents.acknowledgment_engine import acknowledgment_engine, AcknowledgmentType, SlangLevel
from app.core.config import get_settings
from groq import Groq
import logging

logger = logging.getLogger(__name__)
settings = get_settings()


class ResponseAgent:
    """
    Generates natural language responses for users.
    Never exposes internal architecture or raw JSON.
    """
    
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)
    
    async def generate_response(
        self,
        user_query: str,
        recommendations: List[Dict[str, Any]],
        user_memory: Dict[str, Any],
        user_intent: Dict[str, Any],
        conversation_history: List[Dict[str, Any]] = None,
        search_authority: Dict[str, Any] = None,  # Backend authority
        trend_suggestion: Dict[str, Any] = None,  # Proactive trend context
        outfit_context: Dict[str, Any] = None,  # Outfit completion context
        fashion_context: Dict[str, Any] = None  # NEW: Fashion intelligence context
    ) -> str:
        """
        Generate conversational response based on recommendations.
        
        Args:
            user_query: Original user input
            recommendations: Validated recommendations from VALIDATION_AGENT
            user_memory: User memory from MEMORY_AGENT
            user_intent: Intent from QUERY_AGENT
            conversation_history: Previous conversation turns
            search_authority: Backend authority object (MUST BE OBEYED)
            trend_suggestion: Proactive trend context for styling tips
            outfit_context: Context for outfit completion (user's wardrobe)
        
        Returns:
            Natural language response string
        """
        # Log search authority for debugging
        logger.info(f"[Response] Search authority: {search_authority}")
        if trend_suggestion:
            logger.info(f"[Response] Trend suggestion: {trend_suggestion.get('suggested_aesthetic')}")
        if outfit_context:
            logger.info(f"[Response] Outfit context: user has {outfit_context.get('user_has')}, suggesting {outfit_context.get('suggesting')}")
        if fashion_context:
            logger.info(f"[Response] Fashion context: occasion={fashion_context.get('occasion')}, formality={fashion_context.get('formality_level')}")
        
        # Build context for LLM
        user_name = user_memory.get("long_term", {}).get("user_name", "Friend")
        
        # ============================================================
        # SEARCH AUTHORITY ENFORCEMENT (CRITICAL - PREVENTS HALLUCINATION)
        # ============================================================
        authority = search_authority or {}
        search_status = authority.get("search_status", "UNKNOWN")
        products_found = authority.get("products_found", len(recommendations))
        llm_guardrail = authority.get("llm_guardrail", "ALLOW_FALLBACK")
        confidence = authority.get("confidence", 0.5)
        
        # Edge case guard: Backend says FOUND but recommendations empty (upstream bug)
        if search_status == "FOUND" and products_found > 0 and not recommendations:
            logger.error(
                f"[Response] INCONSISTENCY: Authority says FOUND ({products_found} products) "
                f"but recommendations list is empty. Possible upstream slicing bug."
            )
            # Fail safe - trust authority and inform user products exist
            return "I found some options for you! Give me a moment to load them. Try refreshing if they don't appear."
        
        # Build authority instruction for LLM
        if search_status == "FOUND" and products_found > 0:
            has_products = True  # Backend overrides LLM judgment
            authority_instruction = f"""
SEARCH AUTHORITY (NON-NEGOTIABLE):
- Backend confirms {products_found} products EXIST
- Confidence: {confidence:.2f}
- You MUST NOT say "no products found", "couldn't find", "nothing available", "no luck"
- The search was SUCCESSFUL - present the products enthusiastically
"""
        elif search_status == "EMPTY":
            has_products = False
            authority_instruction = """
SEARCH AUTHORITY:
- Backend confirms no products match this query
- You may offer alternatives (different category, style, color)
- Do NOT suggest budget expansion
"""
        elif authority.get("variety_search"):
            has_products = len(recommendations) > 0
            authority_instruction = """
SEARCH AUTHORITY:
- This is a VARIETY SEARCH (user rejected previous options)
- You MUST acknowledge that these are DIFFERENT from before
- Explain what's different (e.g., "tried a different fit", "different color palette", "more minimal")
- Match user's "rejection" energy by being helpful and persistent
"""
        else:
            has_products = len(recommendations) > 0
            authority_instruction = ""
        
        # ================================================================
        # FASHION INTELLIGENCE CONTEXT (NEW - PREVENTS INAPPROPRIATE SUGGESTIONS)
        # ================================================================
        fashion_intelligence = ""
        if fashion_context and has_products:
            occasion = fashion_context.get("occasion", "casual")
            cultural_context = fashion_context.get("cultural_context", "indian")
            formality_level = fashion_context.get("formality_level", "casual")
            emotional_state = fashion_context.get("emotional_state", "neutral")
            
            fashion_intelligence = f"""
FASHION INTELLIGENCE CONTEXT (CRITICAL - USE THIS FOR APPROPRIATE SUGGESTIONS):
- Occasion: {occasion} (formality: {formality_level})
- Cultural context: {cultural_context}
- User's emotional state: {emotional_state}

FASHION APPROPRIATENESS RULES:
1. For {occasion} occasions, these products have been pre-validated as appropriate
2. Cultural context is {cultural_context} - respect cultural norms
3. User seems {emotional_state} - adjust tone accordingly
4. These recommendations passed fashion validation (no inappropriate items like socks for parties)

Your response should acknowledge the occasion appropriateness:
"Perfect for {occasion}" or "These work great for {occasion} occasions"
"""
        
        # ================================================================
        # PROACTIVE STYLING TIPS (NEW - FASHION AWARENESS)
        # ================================================================
        trend_context = ""
        if trend_suggestion and has_products:
            aesthetic = trend_suggestion.get("suggested_aesthetic", "")
            outfit_pieces = trend_suggestion.get("outfit_pieces", [])
            trend_reason = trend_suggestion.get("trend_reason", "")
            personalization = trend_suggestion.get("personalization", "")
            color_palette = trend_suggestion.get("color_palette", [])
            
            trend_context = f"""
PROACTIVE STYLING TIP (ADD THIS NATURALLY):
- Current trending aesthetic: "{aesthetic}"
- Complete outfit suggestion: {', '.join(outfit_pieces)}
- Why it's trending: {trend_reason}
- Personalization for this user: {personalization}
- Suggested color palette: {', '.join(color_palette)}

After showing products, ADD a natural styling tip like:
"This would go fire with [other pieces] btw 👀" or
"Pro tip: pair this with [complementary items] for that {aesthetic} look"
"""
        
        # ================================================================
        # OUTFIT COMPLETION CONTEXT (Phase 2 - Wardrobe Awareness)
        # ================================================================
        outfit_instruction = ""
        if outfit_context and has_products:
            user_has = outfit_context.get("user_has", [])
            suggesting = outfit_context.get("suggesting", "outfit pieces")
            color_match = outfit_context.get("color_match", [])
            
            outfit_instruction = f"""
OUTFIT COMPLETION CONTEXT (CRITICAL - ACKNOWLEDGE USER'S WARDROBE):
User already has: {', '.join(user_has)}
You are suggesting: {suggesting}
Color palette to match: {', '.join(color_match) if color_match else 'neutral, white, black'}

YOUR RESPONSE MUST:
1. Acknowledge what they already have: "Since you've got the [items]..."
2. Suggest products that COMPLETE the outfit with color coordination
3. Explain WHY these pieces work with their existing items
4. Example: "With your white sneakers and blue denim, a white oversized tee + light jacket would complete that clean streetwear look 👌"
"""
        # ================================================================
        # CONVERSATIONAL ACKNOWLEDGMENT SYSTEM (NEW - NATURAL FLOW)
        # ================================================================
        conversational_acknowledgment = ""
        
        # Generate acknowledgment for user input to make conversation natural
        try:
            acknowledgment = acknowledgment_engine.create_conversational_acknowledgment(
                user_input=user_query,
                user_memory=user_memory,
                conversation_history=conversation_history,
                user_intent=user_intent
            )
            
            # Only use acknowledgment if it adds value (not for simple searches)
            ack_type, _ = acknowledgment_engine.detect_acknowledgment_type(
                user_query, conversation_history, user_intent
            )
            
            # Use acknowledgment for preference statements, discovery answers, etc.
            if ack_type in [AcknowledgmentType.PREFERENCE_STATED, AcknowledgmentType.DISCOVERY_ANSWER, 
                           AcknowledgmentType.APPRECIATION, AcknowledgmentType.CLARIFICATION]:
                conversational_acknowledgment = f"""
CONVERSATIONAL ACKNOWLEDGMENT (CRITICAL - USE THIS TO START YOUR RESPONSE):
Start your response with this acknowledgment: "{acknowledgment}"

This acknowledgment should feel natural and flow into your product presentation.
Example: "{acknowledgment} Here are some options that match exactly what you're looking for..."
"""
                logger.info(f"[Response] Using conversational acknowledgment: {acknowledgment}")
        
        except Exception as e:
            logger.error(f"[Response] Acknowledgment generation failed: {e}")
            conversational_acknowledgment = ""
        
        # Get user styling preferences for context
        style_vibe = user_memory.get("long_term", {}).get("style_vibe", "casual")
        design_pref = user_memory.get("long_term", {}).get("design_preference", "clean")
        
        recommendations_text = ""
        if recommendations:
            # Check if it's an outfit or single product
            if "items" in recommendations[0]:
                # Format as outfits - weave reasons into narrative
                for i, outfit in enumerate(recommendations[:3], 1):
                    items = outfit.get("items", [])
                    total_price = outfit.get("total_price", 0)
                    reasons = outfit.get("match_reasons", [])
                    recommendations_text += f"\nOption {i} (₹{total_price}): {', '.join([it.get('name') for it in items])}\n"
                    recommendations_text += f"   Vibe: {', '.join(reasons[:2])}\n"
            else:
                # Format single products with Match Reasons
                for i, rec in enumerate(recommendations[:5], 1):
                    product = rec.get("product", {})
                    reasons = rec.get("match_reasons", [])
                    recommendations_text += f"\n- {product.get('name')} (₹{product.get('price', 0)})\n"
                    recommendations_text += f"  Styling Logic: {', '.join(reasons)}\n"
        else:
            recommendations_text = "No products found."
        
        # Build conversation context
        history_text = ""
        if conversation_history:
            recent = conversation_history[-5:]
            for turn in recent:
                role = turn.get("role", "user")
                content = turn.get("content", "")[:100]
                history_text += f"{role.capitalize()}: {content}\n"
        
        # Build prompt
        intent_type = user_intent.get("intent_type", "search")
        filters = user_intent.get("filters", {})
        price_filter = filters.get("price", {})
        
        # Build price constraint info for the prompt
        price_constraint = ""
        if price_filter:
            if price_filter.get("max"):
                price_constraint = f" The user specifically requested products under ₹{price_filter['max']}. DO NOT suggest they increase their budget."
            elif price_filter.get("min"):
                price_constraint = f" The user specifically requested products above ₹{price_filter['min']}."
        
        if not has_products:
            # Get user's slang tolerance for Gen-Z style
            slang_tolerance = user_memory.get("long_term", {}).get("slang_tolerance", "medium")
            
            # Special prompt for no products found (Gen-Z style)
            prompt = f"""
{authority_instruction}
You are Trendora, a Gen-Z fashion stylist. You're casual, confident, and friendly.

User Query: "{user_query}"
{price_constraint}
Slang tolerance: {slang_tolerance}

SITUATION: No products were found matching the user's criteria.

YOUR RESPONSE MUST:
1. Acknowledge that you couldn't find matching products (be chill about it)
2. Be empathetic and helpful
3. Suggest alternatives WITHOUT suggesting budget expansion
4. Keep it to 2-3 sentences
5. Match Gen-Z energy (casual, not corporate)

TONE RULES (POLITE & RESPECTFUL):
- Use "Would you like..." instead of "You should..."
- Be warm and genuine, never pushy or salesy
- Acknowledge user preferences: "Since you mentioned..."
- Ask permission before suggesting alternatives: "Mind if I suggest something?"
- Never be condescending or dismissive

FORBIDDEN PHRASES (NEVER USE THESE):
- "expand your budget"
- "try higher price points"
- "broaden your search to include higher prices"
- "increase your budget"
- "consider spending more"
- Any suggestion to spend more money
- Corporate-sounding phrases like "I understand how frustrating that can be"

GOOD ALTERNATIVES TO SUGGEST (Gen-Z style):
- "Wanna try a different style or category?"
- "I can help you find similar vibes in other categories"
- "Let's explore other options that might hit different"
- Different categories, styles, or colors

TONE EXAMPLES:
- "Arre, couldn't find anything matching that. Wanna try a different category or style?"
- "No luck with that search. I can help you explore other options though."
- "Couldn't find products for that. Want to try something else?"

Generate a helpful, Gen-Z friendly response that respects the user's budget constraints.
"""
        else:
            # Check if recommendations are outfits or single products
            is_outfit = len(recommendations) > 0 and "items" in recommendations[0]
            
            # Get user's slang tolerance
            slang_tolerance = user_memory.get("long_term", {}).get("slang_tolerance", "medium")
            style_vibe = user_memory.get("long_term", {}).get("style_vibe", "casual")
            
            # Build prompt for the "Vibe Stylist"
            prompt = f"""
{authority_instruction}
{conversational_acknowledgment}
{fashion_intelligence}
{trend_context}
{outfit_instruction}

You are Trendora, a Gen-Z "Vibe Stylist." You're chill, trend-aware, and speak like a stylish friend, not a robot or salesman.

USER CONTEXT:
- Name: {user_name}
- Query: "{user_query}"
- Their Style DNA: {style_vibe} vibe, prefers {design_pref} designs.
- History: {history_text}

RECOMMENDATIONS DATA:
{recommendations_text}

YOUR TASK:
1. NARRATIVE ONLY: Do NOT use bullet points or numbered lists in your final output. Weave everything into a natural paragraph or two.
2. STYLING LOGIC: Weave the "Styling Logic" reasons (budget, vibe, fit, design) into your story. e.g., "Since you're into that clean {style_vibe} look, I found this..."
3. DISCOVERY AWARENESS: If the user just gave you their Vibe or Size (check history), acknowledge it enthusiastically (e.g., "Got it! Streetwear is a fire choice.")
4. NO CORPORATE TALK: Never say "I recommend," "Here are your options," or "I found products."
5. NEXT STEPS: End with a casual question to keep the vibe going (e.g., "Wanna see these in a different color?" or "Should we look for shoes to match?")

TONE EXAMPLES:
- "Yo {user_name}! Since you're vibing with {style_vibe}, I've got some clean options that hit different. This first white tee is perfect because it's that minimal solid look you like, and it fits right in your ₹2000 budget."
- "Got it! Streetwear is definitely the move. I found this sick oversized hoodie and cargos combo that gives off exactly that energy. Both pieces are super high quality and keep you looking fresh without breaking the bank."

Rules:
- 2-4 sentences max.
- CASUAL Gen-Z slang (fire, vibe, clean, mid, drip, lowkey, fr) only if slang_tolerance is high.
- MAX 1-2 emojis.
- NO BULLET POINTS.
"""
        
        try:
            # Use lower temperature for more deterministic responses when no products found
            temp = 0.3 if not has_products else 0.7
            
            completion = self.client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are Trendora, a professional AI Style Consultant. You MUST follow all instructions exactly, especially regarding budget constraints. NEVER suggest users spend more money than they specified."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temp
            )
            
            response = completion.choices[0].message.content.strip()
            
            # Post-process to remove forbidden suggestions
            forbidden_phrases = [
                "expand your budget",
                "higher price points",
                "broaden your search to include",
                "increase your budget",
                "spending more",
                "try higher prices"
            ]
            
            response_lower = response.lower()
            for phrase in forbidden_phrases:
                if phrase in response_lower:
                    logger.warning(f"Response contained forbidden phrase: {phrase}. Regenerating...")
                    # Regenerate with stricter prompt
                    return await self._generate_strict_fallback(user_query, price_filter, user_intent)
            
            return response
            
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            # Fallback response
            if recommendations:
                product = recommendations[0].get("product", {})
                return f"I found {product.get('name', 'some products')} that might interest you. Would you like to see more details?"
            else:
                # Better fallback that doesn't contradict user preferences
                price_info = ""
                if price_filter:
                    if price_filter.get("max"):
                        price_info = f" under ₹{price_filter['max']}"
                    elif price_filter.get("min"):
                        price_info = f" above ₹{price_filter['min']}"
                
                return f"I couldn't find products{price_info} matching your search. Would you like to try a different category or adjust your filters?"


    async def generate_greeting_response(
        self,
        user_query: str,
        user_memory: Dict[str, Any],
        conversation_history: List[Dict[str, Any]] = None
    ) -> str:
        """
        Generate Gen-Z friendly greeting response using acknowledgment engine.
        """
        try:
            # Use acknowledgment engine for natural greeting
            greeting = acknowledgment_engine.create_conversational_acknowledgment(
                user_input=user_query,
                user_memory=user_memory,
                conversation_history=conversation_history,
                user_intent={"intent_type": "greeting"}
            )
            
            return greeting
            
        except Exception as e:
            logger.error(f"Greeting generation error: {e}")
            
            # Fallback to original logic
            user_name = user_memory.get("long_term", {}).get("user_name", "Friend")
            slang_tolerance = user_memory.get("long_term", {}).get("slang_tolerance", "medium")
            
            # Check if this is a returning user
            is_returning = len(conversation_history) > 2 if conversation_history else False
            
            if is_returning:
                greeting_options = [
                    "Hey! What's up? Ready to find some fire fits?",
                    "Yo! Back for more style inspo?",
                    "Hey there! What are we styling today?",
                    "What's good! What can I help you find?"
                ]
            else:
                greeting_options = [
                    "Hey! I'm Trendora, your Gen-Z style assistant. What are you looking for today?",
                    "Yo! What's up? I'm here to help you find some fire fits. What's on your mind?",
                    "Hey there! Ready to find some trendy pieces? What can I help you with?",
                    "What's good! I'm Trendora. What are you in the mood for today?"
                ]
            
            # Adjust based on slang tolerance
            if slang_tolerance == "low":
                greeting_options = [
                    "Hello! I'm Trendora, your style assistant. How can I help you today?",
                    "Hi there! What are you looking for?",
                    "Hello! Ready to find some great pieces? What can I help with?"
                ]
            
            import random
            return random.choice(greeting_options)


# Singleton instance
response_agent = ResponseAgent()
