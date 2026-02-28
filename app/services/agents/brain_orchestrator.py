"""
Central LLM Brain (Orchestrator) - Production-grade AI Orchestrator

This is the MAIN SYSTEM PROMPT orchestrator that:
- Acts as the single decision-making brain
- Never directly accesses databases
- Delegates tasks to specialist agents
- Maintains user context, preferences, and memory
- Ensures responses are accurate, grounded, and explainable
"""

from typing import Dict, List, Any, Optional
from app.services.agents.memory_agent import memory_agent
from app.services.agents.query_agent import query_agent
from app.services.agents.search_agent import search_agent
from app.services.agents.recommendation_agent import recommendation_agent
from app.services.agents.validation_agent import validation_agent
from app.services.agents.response_agent import response_agent
from app.services.agents.genz_trend_agent import genz_trend_agent
from app.services.agents.outfit_stylist_agent import outfit_stylist
from app.services.fashion.fashion_ontology import fashion_ontology
from app.core.config import get_settings
from groq import Groq
import logging
import json

logger = logging.getLogger(__name__)
settings = get_settings()

# MASTER SYSTEM PROMPT
MASTER_SYSTEM_PROMPT = """You are a production-grade AI Orchestrator responsible for powering a multi-agent conversational recommendation system.

Your role:
- Act as the single decision-making brain.
- Never directly access databases.
- Delegate tasks to specialist agents.
- Maintain user context, preferences, and memory.
- Ensure responses are accurate, grounded, and explainable.

SYSTEM GOALS:
1. Understand user intent from conversational input.
2. Retrieve relevant user memory and preferences.
3. Decide whether database search is required.
4. Generate personalized product recommendations.
5. Respond conversationally and naturally.
6. Avoid hallucination — only recommend products returned by agents.

You have access to the following agents:
- MEMORY_AGENT: Manages user memory (preferences, history, behavioral signals)
- QUERY_AGENT: Analyzes user input and extracts intent
- SEARCH_AGENT: Retrieves products from database
- RECOMMENDATION_AGENT: Ranks and personalizes products
- VALIDATION_AGENT: Validates output for accuracy
- RESPONSE_AGENT: Generates natural language responses

You MUST:
- Route tasks explicitly.
- Use structured JSON for agent communication.
- Never fabricate product data.
- Use user memory to personalize recommendations.
- Ask clarifying questions ONLY when necessary.

User memory should evolve over time.
You must update memory when new preferences are discovered.

When you need to make decisions, think step by step:
1. What is the user asking for? (Use QUERY_AGENT)
2. What do we know about the user? (Use MEMORY_AGENT)
3. Do we need to search for products? (Use SEARCH_AGENT if needed)
4. How should we rank results? (Use RECOMMENDATION_AGENT)
5. Are the results valid? (Use VALIDATION_AGENT)
6. How should we respond? (Use RESPONSE_AGENT)

Always return your decisions in structured JSON format."""


class BrainOrchestrator:
    """
    Central LLM Brain that orchestrates all specialist agents.
    Uses the master system prompt to make decisions.
    """
    
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)
    
    async def process_request(
        self,
        user_input: str,
        user_id: str,
        email: Optional[str] = None,
        chat_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for processing user requests.
        
        Orchestrates the full pipeline:
        1. Retrieve user memory
        2. Understand query intent
        3. Search products (if needed)
        4. Rank and personalize
        5. Validate results
        6. Generate response
        7. Update memory
        
        Returns:
        {
            "response": str,
            "products": List[Dict],
            "metadata": Dict,
            "memory_update": Dict (optional)
        }
        """
        # - [x] Verify AI remembers all style traits after one mention and recommends smartly.

        ## Occasion-Specific Logic ("Party" & "College")
        # - [x] Update `QueryAgent` to extract sub-occasions (Wedding, Clubbing, House Party).
        # - [x] Implement "Occasion Detail Discovery" in `BrainOrchestrator`.
        # - [x] Refactor `RecommendationAgent` scoring (Boost oversized for college, bold for clubbing).
        # - [x] Verify "College" and "Party" logic via targeted tests.
        try:
            # Step 1: Retrieve user memory
            logger.info(f"[Brain] Retrieving memory for {email or user_id}")
            user_memory = await memory_agent.retrieve_memory(user_id, email, chat_id)
            
            # Step 1.5: Get conversation history EARLY for context-aware query understanding
            from app.services.memory.service import memory_service
            try:
                user_data = await memory_service.get_user_memory(user_id, email, chat_id)
                conversation_history = user_data.get("context", {}).get("conversationHistory", [])
            except Exception as history_error:
                logger.warning(f"[Brain] Failed to get conversation history: {history_error}")
                conversation_history = []

            # Step 1.6: Load shown products from history for global exclusion
            global_excluded_ids = []
            for msg in conversation_history:
                p_ids = msg.get("product_ids", [])
                if p_ids and isinstance(p_ids, list):
                    global_excluded_ids.extend(p_ids)
            global_excluded_ids = list(set(global_excluded_ids))
            logger.info(f"[Brain] Global session exclusion count: {len(global_excluded_ids)}")
            
            # Step 2: Understand query intent (WITH CONTEXT!)
            logger.info(f"[Brain] Understanding query: {user_input} (with {len(conversation_history)} history messages)")
            user_intent = await query_agent.understand_query(user_input, conversation_history)
            
            # Step 2.5: Handle greetings and small talk immediately (no search needed)
            intent_type = user_intent.get("intent_type", "search")
            
            # SMART OVERRIDE: Don't treat rich messages as just greetings
            # "hi i am going for wedding suggest me something" is NOT just a greeting!
            if intent_type in ["greeting", "small_talk"]:
                shopping_intent_words = [
                    "buy", "suggest", "recommend", "show", "find", "looking", "need",
                    "want", "search", "outfit", "wear", "dress", "confused", "help",
                    "wedding", "party", "trip", "beach", "office", "college", "date",
                    "occasion", "festival", "interview", "something", "clothes", "fashion",
                    "shopping", "shop", "browse", "explore", "check", "options"
                ]
                input_lower = user_input.lower()
                has_shopping_intent = any(word in input_lower for word in shopping_intent_words)
                
                if has_shopping_intent and len(user_input.split()) > 4:
                    # This is NOT just "hi" — user has real shopping intent
                    logger.info(f"[Brain] Overriding '{intent_type}' → 'recommend' (message has shopping intent)")
                    intent_type = "recommend"
                    user_intent["intent_type"] = "recommend"
                    # Re-extract useful info the LLM might have missed
                    if not user_intent.get("occasion"):
                        for occ in ["wedding", "party", "trip", "beach", "interview", "date", "festival"]:
                            if occ in input_lower:
                                user_intent["occasion"] = occ
                                user_intent["buying_type"] = "occasion"
                                break
            
            if intent_type in ["greeting", "small_talk"]:
                logger.info(f"[Brain] Detected {intent_type}, skipping product search")
                # Generate greeting response directly
                from app.services.memory.service import memory_service
                user_data = await memory_service.get_user_memory(user_id, email, chat_id)
                history = user_data.get("context", {}).get("conversationHistory", [])
                
                response_text = await response_agent.generate_greeting_response(
                    user_query=user_input,
                    user_memory=user_memory,
                    conversation_history=history
                )
                
                return {
                    "response": response_text,
                    "products": [],
                    "products_found": 0,
                    "metadata": {
                        "intent_type": intent_type,
                        "skipped_search": True
                    },
                    "memory_update": None
                }
            
            # Step 2.55: Handle product appreciation (user likes a specific product)
            if intent_type == "product_appreciation":
                logger.info(f"[Brain] Detected product appreciation: {user_input}")
                
                # Get appreciation context from intent
                appreciation = user_intent.get("product_appreciation", {})
                product_ref = appreciation.get("product_reference", "that item")
                liked_features = appreciation.get("liked_features", [])
                sentiment = appreciation.get("sentiment", "like")
                
                # Get conversation history to find what products were shown
                history = conversation_history
                
                # Extract category from recent products/conversation
                last_category = None
                for msg in reversed(history[-10:]):
                    content = msg.get("content", "").lower()
                    if "jeans" in content or "denim" in content:
                        last_category = "jeans"
                        break
                    elif "sneaker" in content or "shoe" in content:
                        last_category = "sneakers"
                        break
                    elif "t-shirt" in content or "tshirt" in content or "tee" in content:
                        last_category = "shirt"
                        break
                    elif "jacket" in content:
                        last_category = "jacket"
                        break
                    elif "hoodie" in content:
                        last_category = "hoodie"
                        break
                
                # Build appreciation response based on sentiment
                if sentiment in ["love", "fire"]:
                    acknowledgments = [
                        f"Great taste! That's a fire pick 🔥",
                        f"Yooo, you got an eye for style! That one's a vibe 👌",
                        f"Solid choice! That's gonna look clean on you"
                    ]
                elif sentiment == "like":
                    acknowledgments = [
                        f"Nice choice! I can see why you like it 👀",
                        f"Good pick! That one's lowkey fire",
                        f"I feel you, that's a solid option"
                    ]
                else:
                    acknowledgments = [
                        f"Got it! I can see you're vibing with that one",
                        f"Noted! That's a good pick"
                    ]
                
                import random
                ack_response = random.choice(acknowledgments)
                
                # Save liked features to memory
                memory_update_data = {}
                if liked_features:
                    # Update style preferences based on what they liked
                    memory_update_data["liked_features"] = liked_features
                    
                    # If they liked specific things, note them
                    for feature in liked_features:
                        if feature in ["color", "colour"]:
                            memory_update_data["prefers_color_variety"] = True
                        elif feature in ["fit", "style"]:
                            memory_update_data["style_conscious"] = True
                        elif feature == "price":
                            memory_update_data["value_conscious"] = True
                
                if last_category:
                    memory_update_data["recently_appreciated_category"] = last_category
                
                # Save to memory
                if memory_update_data:
                    await memory_agent.update_memory(
                        user_id=user_id,
                        email=email,
                        memory_type="LONG_TERM",
                        data=memory_update_data
                    )
                    logger.info(f"[Brain] Saved appreciation preferences: {memory_update_data}")
                
                # Search for similar products to suggest more
                similar_products = []
                if last_category:
                    try:
                        search_results = await search_agent.search_products(
                            query=f"similar to {product_ref}" if product_ref else last_category,
                            filters={},
                            category=last_category,
                            limit=4,
                            exclude_ids=global_excluded_ids
                        )
                        similar_products = search_results.get("products", [])[:3]
                    except Exception as search_err:
                        logger.warning(f"[Brain] Similar product search failed: {search_err}")
                
                # Build final response with similar suggestions
                if similar_products:
                    ack_response += f"\n\nSince you liked that, you might also vibe with these similar options 👇"
                
                return {
                    "response": ack_response,
                    "products": similar_products,
                    "products_found": len(similar_products),
                    "metadata": {
                        "intent_type": intent_type,
                        "appreciation_detected": True,
                        "liked_features": liked_features,
                        "preferences_saved": bool(memory_update_data)
                    },
                    "memory_update": {"update_memory": True, "data": memory_update_data} if memory_update_data else None
                }
            
            # Step 2.6: Handle preference updates (re-filter existing results)
            if intent_type == "preference_update":
                logger.info(f"[Brain] Detected preference update: {user_input}")
                
                # Extract preferences from intent
                preferences = user_intent.get("preferences", {})
                disliked_fits = preferences.get("disliked_fits", [])
                preferred_fit = preferences.get("preferred_fit")
                
                # Get recent conversation to find last search results
                from app.services.memory.service import memory_service
                user_data = await memory_service.get_user_memory(user_id, email, chat_id)
                history = user_data.get("context", {}).get("conversationHistory", [])
                
                # Try to get last category from conversation
                last_category = None
                for msg in reversed(history[-10:]):
                    if msg.get("role") == "user":
                        # Extract category from last user message
                        last_query = msg.get("content", "").lower()
                        from app.core.categories import normalize_category
                        last_category = normalize_category(last_query)
                        if last_category:
                            break
                
                # Re-search with updated preferences
                # Prepare filters from preferences
                disliked_colors = user_intent.get("preferences", {}).get("disliked_colors", [])
                disliked_styles = user_intent.get("preferences", {}).get("disliked_styles", [])
                
                search_filters = {
                    "fit": preferred_fit,
                    "preferences": user_intent.get("preferences", {})
                }
                
                # Search again with filters
                search_results = await search_agent.search_products(
                    query=user_input,
                    filters=search_filters,
                    category=last_category,
                    limit=20,
                    exclude_ids=global_excluded_ids
                )
                products = search_results.get("products", [])
                
                # Filter out disliked fits
                if disliked_fits:
                    products = [p for p in products if (p.get("attributes", {}).get("fit") or "").lower() not in [df.lower() for df in disliked_fits]]
                
                # Re-rank with updated preferences
                recommendations = []
                if products:
                    # Update user memory temporarily for ranking
                    if preferred_fit:
                        user_memory["long_term"]["preferred_fit"] = preferred_fit
                    if disliked_fits:
                        user_memory["long_term"]["disliked_fits"] = disliked_fits
                    
                    recommendations = recommendation_agent.rank_products(
                        products=products,
                        user_memory=user_memory,
                        intent=user_intent
                    )
                
                # Update memory permanently
                memory_update_data = {}
                
                # CRITICAL FIX: Handle gender from intent
                if user_intent.get("gender"):
                    memory_update_data["gender"] = user_intent.get("gender")
                
                if preferred_fit:
                    memory_update_data["preferred_fit"] = preferred_fit
                if disliked_fits:
                    memory_update_data["disliked_fits"] = disliked_fits
                
                # Add new preference tracking
                disliked_colors = user_intent.get("preferences", {}).get("disliked_colors", [])
                if disliked_colors:
                    memory_update_data["disliked_colors"] = disliked_colors
                
                disliked_styles = user_intent.get("preferences", {}).get("disliked_styles", [])
                if disliked_styles:
                    memory_update_data["disliked_styles"] = disliked_styles
                
                # ALWAYS update memory if we have any data (including gender)
                if memory_update_data:
                    await memory_agent.update_memory(
                        user_id=user_id,
                        email=email,
                        memory_type="LONG_TERM",
                        data=memory_update_data
                    )
                
                # Generate response
                response_text = await response_agent.generate_response(
                    user_query=user_input,
                    recommendations=recommendations[:5],
                    user_memory=user_memory,
                    user_intent=user_intent,
                    conversation_history=history
                )
                
                final_products = [rec.get("product") for rec in recommendations[:5]]
                
                return {
                    "response": response_text,
                    "products": final_products,
                    "products_found": len(final_products),
                    "metadata": {
                        "intent_type": intent_type,
                        "preference_updated": True
                    },
                    "memory_update": {"update_memory": True, "data": memory_update_data} if memory_update_data else None
                }
            
            # Step 2.65: Handle rejections (exclude previously shown items)
            if intent_type == "rejection":
                logger.info(f"[Brain] Detected rejection: {user_input}")
                
                # Get conversation context to find recently shown products
                history = conversation_history
                
                # Collect IDs of products from ALL recent assistant messages in this session
                # Combine global excluded IDs with those specifically from recent history if needed
                # (global_excluded_ids already contains all of them, but this is explicit)
                combined_excluded_ids = global_excluded_ids
                logger.info(f"[Brain] Rejection logic: excluding {len(combined_excluded_ids)} products.")
                
                # Re-search with exclusion AND preferences
                search_results = await search_agent.search_products(
                    query=user_input,
                    filters={"preferences": user_intent.get("preferences", {})},
                    category=last_category,
                    exclude_ids=combined_excluded_ids,
                    limit=20
                )
                products = search_results.get("products", [])
                
                # Rank with user memory
                recommendations = recommendation_agent.rank_products(
                    products=products,
                    user_memory=user_memory,
                    intent=user_intent
                )
                
                # Generate response showing variety
                # Highlight that we are showing DIFFERENT things
                response_text = await response_agent.generate_response(
                    user_query=user_input,
                    recommendations=recommendations[:5],
                    user_memory=user_memory,
                    user_intent=user_intent,
                    conversation_history=history,
                    search_authority={"search_status": "FOUND" if products else "EMPTY", "products_found": len(products), "variety_search": True}
                )
                
                final_products = [rec.get("product") for rec in recommendations[:5]]
                
                return {
                    "response": response_text,
                    "products": final_products,
                    "products_found": len(final_products),
                    "metadata": {
                        "intent_type": intent_type,
                        "rejection_handled": True,
                        "excluded_count": len(combined_excluded_ids)
                    },
                    "memory_update": None
                }
            
            # Step 3: Decide if search is needed
            needs_search = intent_type in ["search", "recommend", "browse", "compare", "outfit_completion"]
            
            products = []
            search_results = {}
            trend_suggestion = None  # For proactive recommendations
            outfit_context = None  # For outfit completion
            
            # ================================================================
            # OUTFIT COMPLETION LOGIC (NEW - Phase 2)
            # User mentions what they have, AI suggests completing pieces
            # ================================================================
            if intent_type == "outfit_completion":
                wardrobe = user_intent.get("wardrobe_context", {})
                user_has = wardrobe.get("user_has", [])
                colors_mentioned = wardrobe.get("colors_mentioned", [])
                needs = wardrobe.get("needs")
                
                logger.info(f"[Brain] Outfit completion - user has: {user_has}, colors: {colors_mentioned}, needs: {needs}")
                
                # Build smart search based on what user has
                # If they have bottom (jeans/pants) + shoes, they need top
                # If they have top, they need bottom
                has_bottom = any(item for item in user_has if any(b in item.lower() for b in ["jeans", "pants", "trousers", "shorts"]))
                has_top = any(item for item in user_has if any(t in item.lower() for t in ["shirt", "tee", "hoodie", "jacket", "top"]))
                has_shoes = any(item for item in user_has if any(s in item.lower() for s in ["sneaker", "shoe", "boots", "sandal"]))
                
                # Determine what to search for
                search_categories = []
                if has_bottom and has_shoes and not has_top:
                    search_categories = ["shirt", "t-shirt", "jacket", "hoodie"]
                    outfit_context = {
                        "user_has": user_has,
                        "suggesting": "top pieces",
                        "color_match": colors_mentioned
                    }
                elif has_top and not has_bottom:
                    search_categories = ["jeans", "pants", "trousers"]
                    outfit_context = {
                        "user_has": user_has,
                        "suggesting": "bottom pieces",
                        "color_match": colors_mentioned
                    }
                else:
                    # Default: suggest trending outfit pieces
                    search_categories = ["t-shirt", "jacket"]
                    outfit_context = {
                        "user_has": user_has,
                        "suggesting": "complete outfit",
                        "color_match": colors_mentioned
                    }
                
                # Build color-aware search query
                color_query = " ".join(colors_mentioned[:2]) if colors_mentioned else ""
                search_query = f"{color_query} {search_categories[0]}".strip()
                # CRITICAL: Set category from outfit completion
                category = search_categories[0]
                logger.info(f"[Brain] Outfit search: '{search_query}' for category '{category}'")
            
            # Step 4: Search products
            if needs_search:
                logger.info(f"[Brain] Searching products with filters: {user_intent.get('filters')}")
                
                # Initialize category variable
                category = user_intent.get("product_category")
                
                # ================================================================
                # FASHION INTELLIGENCE LAYER (NEW - PREVENTS DISASTERS)
                # Apply fashion ontology BEFORE search to prevent inappropriate suggestions
                # ================================================================
                
                # Extract fashion context using Outfit Stylist
                fashion_context = outfit_stylist.analyze_fashion_intent(
                    user_query=user_input,
                    user_memory=user_memory,
                    conversation_history=conversation_history
                )
                
                # Generate outfit strategy to guide search
                outfit_strategy = outfit_stylist.generate_outfit_strategy(
                    fashion_context=fashion_context,
                    user_query=user_input
                )
                
                logger.info(f"[Brain] Fashion context: occasion={fashion_context.get('occasion')}, "
                           f"cultural_context={fashion_context.get('cultural_context')}")
                logger.info(f"[Brain] Outfit strategy: {len(outfit_strategy.get('blocked_categories', []))} blocked categories")
                
                # Apply fashion guardrails to filters
                filters = user_intent.get("filters", {})
                
                # Add blocked categories from fashion ontology
                blocked_categories = outfit_strategy.get("blocked_categories", [])
                if blocked_categories:
                    if "exclude_categories" not in filters:
                        filters["exclude_categories"] = []
                    filters["exclude_categories"].extend(blocked_categories)
                    logger.info(f"[Brain] Fashion guardrail: Blocked {len(blocked_categories)} inappropriate categories")
                
                # Prioritize appropriate categories
                primary_categories = outfit_strategy.get("primary_categories", [])
                if primary_categories and not category:
                    category = primary_categories[0]  # Use most appropriate category
                    logger.info(f"[Brain] Fashion guidance: Using category '{category}' from outfit strategy")
                
                # For outfit completion, use the computed category, otherwise from intent
                if intent_type == "outfit_completion":
                    # Override category for outfit completion logic
                    pass  # category already set above in outfit completion logic
                
                # If category is provided, add it to filters for better search
                if category and not filters.get("category"):
                    filters["category"] = category
                
                # ================================================================
                # PROACTIVE FASHION RECOMMENDATION (NEW)
                # When user wants recommendations without specific product
                # ================================================================
                # Build a SMART search query from extracted intent, NOT raw user input
                # Raw: "hi i am yashvardhan looking for wedding fits" → USELESS for DB search
                # Smart: "wedding formal wear" → ACTUALLY WORKS
                if intent_type != "outfit_completion":
                    # Extract meaningful search terms from the intent
                    intent_search_terms = []
                    
                    # 1. Use LLM-extracted search query if available
                    llm_query = user_intent.get("search_query") or user_intent.get("extracted_query")
                    if llm_query and llm_query.lower() not in ["", "none", "null"]:
                        intent_search_terms.append(llm_query)
                    
                    # 2. Use product category if extracted
                    if category:
                        intent_search_terms.append(category)
                    
                    # 3. Use occasion from intent/accumulated context
                    extracted_occasion = user_intent.get("occasion")
                    if extracted_occasion:
                        intent_search_terms.append(extracted_occasion)
                    
                    # 4. Use filters like color, brand, style
                    if filters.get("color"):
                        intent_search_terms.append(filters["color"])
                    if filters.get("brand"):
                        intent_search_terms.append(filters["brand"])
                    if filters.get("style"):
                        intent_search_terms.append(filters["style"])
                    
                    # 5. Extract meaningful words from user input as fallback
                    if not intent_search_terms:
                        noise_words = {"hi", "hello", "hey", "i", "am", "the", "a", "an", "is", "it", 
                                      "its", "my", "me", "im", "i'm", "for", "to", "of", "and", "or",
                                      "in", "on", "at", "with", "this", "that", "what", "how", "can",
                                      "could", "would", "should", "you", "your", "please", "suggest",
                                      "recommend", "show", "find", "looking", "want", "need", "buy",
                                      "something", "some", "get", "give", "help", "confused", "today",
                                      "close", "friend", "going"}
                        meaningful_words = [w for w in user_input.lower().split() 
                                          if w not in noise_words and len(w) > 2]
                        if meaningful_words:
                            intent_search_terms = meaningful_words[:3]
                    
                    # Build the search query
                    if intent_search_terms:
                        search_query = " ".join(intent_search_terms)
                    else:
                        search_query = user_input  # absolute last fallback
                    
                    logger.info(f"[Brain] Smart search query: '{search_query}' (from intent, not raw input)")
                
                if intent_type in ["recommend", "browse"] and not category:
                    logger.info(f"[Brain] No specific category - using trend agent for proactive recommendation")
                    
                    # Get trend-based outfit suggestion
                    trend_suggestion = await genz_trend_agent.suggest_trending_outfit(
                        user_memory=user_memory,
                        occasion=user_intent.get("occasion"),
                        budget=filters.get("price"),
                        category_hint=None
                    )
                    logger.info(f"[Brain] Trend suggestion: {trend_suggestion.get('suggested_aesthetic')}")
                    
                    # Use the search terms from trend suggestion
                    search_terms = trend_suggestion.get("search_terms", ["t-shirt", "jeans", "sneakers"])
                    search_query = " ".join(search_terms[:2])  # Use first 2 terms for search
                
                # ================================================================
                # CONTEXT-AWARE DISCOVERY (Agentic AI)
                # Scans ALL conversation history to build full picture first,
                # then only asks what's ACTUALLY missing.
                # ================================================================
                
                # --- Step A: Accumulate context from ENTIRE conversation ---
                accumulated = {
                    "gender": None, "occasion": None, "sub_occasion": None,
                    "buying_type": None, "vibe": None, "style_hints": [],
                    "category": category,
                }
                
                occasion_keywords = [
                    "trip", "beach", "vacation", "travel", "party", "wedding", 
                    "interview", "office", "college", "clubbing", "date", "dinner",
                    "festival", "sangeet", "function", "haldi", "reception", "gym",
                    "brunch", "concert", "outing", "hangout", "picnic", "trek",
                    "ceremony", "convocation", "farewell", "prom"
                ]
                
                gender_male_kw = ["men", "mens", "men's", "male", "man", "guy", "boy", "boys", "gents", "bro"]
                gender_female_kw = ["women", "womens", "women's", "female", "woman", "girl", "girls", "lady", "ladies"]
                gender_unisex_kw = ["unisex", "gender neutral", "both", "neutral"]
                vibe_kw = ["minimal", "streetwear", "classic", "trendy", "ethnic", "luxury", "casual", "formal", "traditional"]
                style_hint_kw = {
                    "light": ["light", "lightweight", "airy", "breezy", "cool", "breathable", "summer"],
                    "warm": ["warm", "heavy", "cozy", "thick", "winter", "layered"],
                    "bold": ["bold", "loud", "flashy", "statement", "graphic", "printed"],
                    "clean": ["clean", "simple", "solid", "minimal", "plain", "subtle"],
                }
                
                for msg in conversation_history:
                    if msg.get("role") != "user":
                        continue
                    content = msg.get("content", "").lower()
                    words = content.split()
                    
                    # Gender
                    if any(kw in words for kw in gender_male_kw):
                        accumulated["gender"] = "male"
                    elif any(kw in words for kw in gender_female_kw):
                        accumulated["gender"] = "female"
                    elif any(kw in content for kw in gender_unisex_kw):
                        accumulated["gender"] = "unisex"
                    
                    # Occasion
                    for occ_kw in occasion_keywords:
                        if occ_kw in content:
                            accumulated["occasion"] = occ_kw
                            accumulated["buying_type"] = "occasion"
                            break
                    
                    # Vibe
                    for v_kw in vibe_kw:
                        if v_kw in words:
                            accumulated["vibe"] = v_kw
                            break
                    
                    # Style hints
                    for hint_name, hint_keywords in style_hint_kw.items():
                        if any(hk in content for hk in hint_keywords):
                            if hint_name not in accumulated["style_hints"]:
                                accumulated["style_hints"].append(hint_name)
                    
                    # Buying type
                    if any(kw in content for kw in ["regular", "daily", "everyday", "normal"]):
                        if not accumulated["buying_type"]:
                            accumulated["buying_type"] = "regular"
                
                logger.info(f"[Brain] Accumulated context: gender={accumulated['gender']}, "
                           f"occasion={accumulated['occasion']}, vibe={accumulated['vibe']}, "
                           f"buying_type={accumulated['buying_type']}, hints={accumulated['style_hints']}")
                
                # --- Step B: Merge (priority: intent > memory > accumulated) ---
                user_gender = (user_intent.get("gender") 
                              or user_memory.get("long_term", {}).get("gender") 
                              or accumulated["gender"])
                
                user_vibe = (user_intent.get("preferences", {}).get("vibe") 
                            or user_memory.get("long_term", {}).get("style_vibe") 
                            or accumulated["vibe"])
                
                user_buying_type = (user_intent.get("buying_type") 
                                   or user_memory.get("long_term", {}).get("buying_type")
                                   or user_memory.get("short_term", {}).get("buying_type")
                                   or accumulated["buying_type"])
                
                user_occasion = (user_intent.get("occasion") 
                                or user_memory.get("long_term", {}).get("occasion")
                                or accumulated["occasion"])
                
                if user_occasion and not user_buying_type:
                    user_buying_type = "occasion"
                
                logger.info(f"[Brain] Merged context: gender={user_gender}, occasion={user_occasion}, "
                           f"vibe={user_vibe}, buying_type={user_buying_type}")
                
                # --- Helper: save everything we know ---
                async def _save_all_known():
                    data = {}
                    if user_gender: data["gender"] = user_gender
                    if user_vibe: data["style_vibe"] = user_vibe
                    if user_buying_type: data["buying_type"] = user_buying_type
                    if user_occasion: data["occasion"] = user_occasion
                    if user_intent.get("sub_occasion"): data["sub_occasion"] = user_intent["sub_occasion"]
                    if user_intent.get("user_name"): data["user_name"] = user_intent["user_name"]
                    if accumulated["style_hints"]: data["style_hints"] = accumulated["style_hints"]
                    if data:
                        logger.info(f"[Brain] Persisting preferences: {list(data.keys())}")
                        await memory_agent.update_memory(
                            user_id=user_id, email=email,
                            memory_type="LONG_TERM", data=data
                        )
                    return data
                
                # --- Step C: Ask ONLY what's truly missing ---
                
                # 1. GENDER
                if not user_gender and intent_type in ["search", "recommend", "browse"]:
                    query_lower = user_input.lower()
                    has_specific_product = any(word in query_lower for word in [
                        "shirt", "tshirt", "t-shirt", "jeans", "pants", "hoodie", "jacket", 
                        "sneakers", "shoes", "dress", "top", "bottom", "kurta", "saree",
                        "jersey", "joggers", "shorts", "socks", "cap"
                    ])
                    if not has_specific_product:
                        logger.info(f"[Brain] Gender unknown from ALL sources - asking")
                        saved = await _save_all_known()
                        return {
                            "response": "Yo! Before we search, are we looking for Men's, Women's, or maybe something Gender-Neutral? 👕👗",
                            "products": [], "products_found": 0,
                            "metadata": {"question_type": "gender_discovery",
                                        "pending_search": {"query": search_query, "category": category, "filters": filters}},
                            "memory_update": {"update_memory": True, "data": saved} if saved else None
                        }
                    else:
                        logger.info(f"[Brain] Specific product mentioned, skipping gender")
                
                # 2. BUYING TYPE — skip if we already have occasion from conversation
                if not user_buying_type and not user_occasion and intent_type in ["search", "recommend"]:
                    logger.info(f"[Brain] Buying type unknown from ALL sources - asking")
                    saved = await _save_all_known()
                    return {
                        "response": "Are you shopping for a specific occasion (like a party/wedding/trip) or just looking for some fire regulars? 🔥",
                        "products": [], "products_found": 0,
                        "metadata": {"question_type": "buying_type_discovery",
                                    "pending_search": {"query": search_query, "category": category, "filters": filters}},
                        "memory_update": {"update_memory": True, "data": saved} if saved else None
                    }
                
                # 3. OCCASION DETAIL — only if buying for occasion but no specific occasion
                if user_buying_type == "occasion" and not user_occasion:
                    saved = await _save_all_known()
                    return {
                        "response": "Bet! What's the occasion? (Party, Wedding, Trip, Interview, or something else?) 🥂",
                        "products": [], "products_found": 0,
                        "metadata": {"question_type": "occasion_discovery_vague",
                                    "pending_search": {"query": search_query, "category": category, "filters": filters}},
                        "memory_update": {"update_memory": True, "data": saved} if saved else None
                    }
                
                # Sub-occasion only for generic "party"
                if user_occasion == "party" and not (user_intent.get("sub_occasion") or accumulated.get("sub_occasion")):
                    saved = await _save_all_known()
                    return {
                        "response": "Ayo! What kind of party? 👀 (Wedding Party, Clubbing, or a chill House Party?)",
                        "products": [], "products_found": 0,
                        "metadata": {"question_type": "occasion_discovery",
                                    "pending_search": {"query": search_query, "category": category, "filters": filters}},
                        "memory_update": {"update_memory": True, "data": saved} if saved else None
                    }
                
                # Save everything before search
                await _save_all_known()
                
                # Inject style hints and occasion into search query for better results
                if accumulated["style_hints"]:
                    hint_str = " ".join(accumulated["style_hints"])
                    if hint_str not in search_query:
                        search_query = f"{hint_str} {search_query}"
                        logger.info(f"[Brain] Enhanced search with style hints: '{search_query}'")
                
                if user_occasion and user_occasion not in search_query:
                    search_query = f"{user_occasion} {search_query}"
                    logger.info(f"[Brain] Enhanced search with occasion: '{search_query}'")
                
                # Apply Global Search Guardrails (Gender & Category)
                if "exclude_categories" not in filters:
                    filters["exclude_categories"] = []
                
                # Map high-level gender intent (male/female/unisex) to DB values (Men/Women/Unisex)
                gender_targets = []
                # Ensure user_gender is a list for processing
                gender_intents = user_gender if isinstance(user_gender, list) else [user_gender] if user_gender else []
                
                for gi in gender_intents:
                    if gi == "male":
                        gender_targets.extend(["Men", "Unisex"])
                    elif gi == "female":
                        gender_targets.extend(["Women", "Unisex"])
                    elif gi == "unisex":
                        gender_targets.extend(["Unisex", "Men", "Women"])
                
                # De-duplicate
                gender_targets = list(set(gender_targets))
                
                # Extract occasion for guardrails
                intent_occasion = (user_intent.get("occasion") or filters.get("occasion") or "").lower()
                
                # CRITICAL: If user explicitly wants 'Men' or is 'male', and didn't ask for 'Women',
                # strictly exclude Women products to avoid mismatched results like Sarees.
                is_exclusively_male = ("male" in gender_intents or "Men" in gender_targets) and "female" not in gender_intents
                if is_exclusively_male:
                    if "Women" in gender_targets:
                        gender_targets.remove("Women")
                    # Add strict category exclusions for men - UNIVERSAL LIST
                    womens_excl = [
                        "Saree", "Women's Earrings", "Heels", "Women's Footwear", 
                        "Jewelry", "Jewellery", "Necklace", "Bangles", "Clutch", 
                        "Handbag", "Dress", "Skirts", "Kurti", "Leggings", "Palazzo",
                        "Salwar", "Sharara", "Bras", "Panties", "Lingerie"
                    ]
                    for cat in womens_excl:
                        if cat not in filters["exclude_categories"]:
                            filters["exclude_categories"].append(cat)
                
                if gender_targets:
                    filters["gender"] = gender_targets
                
                # Category Guardrail: If intent is professional/formal, exclude activewear
                if user_occasion in ["professional", "office", "interview", "wedding"] or user_vibe in ["classic", "luxury"]:
                    if "Activewear" not in filters["exclude_categories"]:
                        filters["exclude_categories"].extend(["Activewear", "Sportswear"])
                    logger.info(f"[Brain] Applying Guardrail: Excluding Activewear for {user_occasion or user_vibe} context.")

                # Step 4: Search products
                try:
                    search_results = await search_agent.search_products(
                        query=search_query,
                        filters=filters,
                        category=category,
                        exclude_ids=global_excluded_ids,
                        limit=20
                    )
                    products = search_results.get("products", [])
                    logger.info(f"[Brain] Found {len(products)} products")
                    
                    # ================================================================
                    # INTELLIGENT EMPTY-SEARCH FALLBACK
                    # If 0 products found, retry with broader/related terms
                    # saree→ethnic→traditional, wedding→formal→kurta, etc.
                    # ================================================================
                    if not products:
                        logger.info(f"[Brain] 0 products from primary search '{search_query}' - trying fallbacks")
                        
                        from app.core.categories import CATEGORY_MAP, CATEGORY_KEYWORDS
                        
                        # Build fallback queries
                        fallback_queries = []
                        
                        # 1. Try parent/related categories from CATEGORY_MAP
                        if category and category.lower() in CATEGORY_MAP:
                            related_cats = CATEGORY_MAP[category.lower()]
                            for rel_cat in related_cats[:3]:
                                if rel_cat != category.lower():
                                    fallback_queries.append(rel_cat)
                        
                        # 2. Try occasion-based terms
                        if user_occasion:
                            occasion_to_products = {
                                "wedding": ["formal wear", "ethnic wear", "kurta", "sherwani", "blazer"],
                                "party": ["trendy outfit", "hoodie", "jeans", "sneakers"],
                                "date": ["casual wear", "shirt", "jeans", "sneakers"],
                                "interview": ["formal shirt", "trousers", "blazer"],
                                "office": ["formal wear", "shirt", "trousers"],
                                "college": ["casual wear", "tshirt", "jeans", "sneakers"],
                                "trip": ["casual outfit", "shorts", "tshirt"],
                                "beach": ["casual wear", "shorts", "sandals"],
                                "festival": ["ethnic wear", "kurta", "traditional"],
                                "gym": ["activewear", "tracksuit", "joggers"],
                            }
                            occ_products = occasion_to_products.get(user_occasion, [])
                            for op in occ_products:
                                if op not in fallback_queries:
                                    fallback_queries.append(op)
                        
                        # 3. Try just the occasion + gender as a broad search
                        if user_occasion:
                            fallback_queries.append(f"{user_occasion} outfit")
                        
                        # 4. Ultimate fallback: just search broadly
                        fallback_queries.append("trending")
                        
                        for fq in fallback_queries:
                            logger.info(f"[Brain] Fallback search: '{fq}'")
                            fallback_result = await search_agent.search_products(
                                query=fq,
                                filters=filters,
                                category=None,  # Remove category constraint
                                exclude_ids=global_excluded_ids,
                                limit=20
                            )
                            fallback_products = fallback_result.get("products", [])
                            if fallback_products:
                                products = fallback_products
                                search_results = fallback_result
                                logger.info(f"[Brain] Fallback '{fq}' found {len(products)} products!")
                                break
                        
                        if not products:
                            logger.info(f"[Brain] All fallbacks exhausted, trying gender-only broad search")
                            broad_filters = {}
                            if "gender" in filters:
                                broad_filters["gender"] = filters["gender"]
                            broad_result = await search_agent.search_products(
                                query="clothing",
                                filters=broad_filters,
                                category=None,
                                exclude_ids=global_excluded_ids,
                                limit=20
                            )
                            products = broad_result.get("products", [])
                            if products:
                                search_results = broad_result
                                logger.info(f"[Brain] Broad gender-only search found {len(products)} products")
                    
                    # ================================================================
                    # FASHION VALIDATION LAYER (NEW - FINAL SAFETY CHECK)
                    # Validate search results against fashion logic to prevent disasters
                    # ================================================================
                    if products and outfit_strategy:
                        validation_result = outfit_stylist.validate_search_results(
                            products=products,
                            search_strategy=outfit_strategy
                        )
                        
                        if not validation_result.get("validation_passed"):
                            logger.warning(f"[Brain] Fashion validation failed: {validation_result.get('issues')}")
                            # Use only valid products
                            products = validation_result.get("valid_products", [])
                            logger.info(f"[Brain] After fashion validation: {len(products)} valid products")
                        
                        # Log blocked products for debugging
                        blocked_products = validation_result.get("blocked_products", [])
                        if blocked_products:
                            logger.info(f"[Brain] Fashion guardrail blocked {len(blocked_products)} inappropriate products")
                            for blocked in blocked_products[:3]:  # Log first 3
                                product_name = blocked.get("product", {}).get("name", "Unknown")
                                reason = blocked.get("reason", "Inappropriate")
                                logger.info(f"[Brain] Blocked: {product_name} - {reason}")
                    
                except Exception as search_error:
                    logger.error(f"[Brain] Search failed: {search_error}")
                    # Try fallback search without filters
                    try:
                        # Fallback: Preserve guardrails (gender and excluded categories)
                        fallback_filters = {}
                        if "gender" in filters: fallback_filters["gender"] = filters["gender"]
                        if "exclude_categories" in filters: fallback_filters["exclude_categories"] = filters["exclude_categories"]
                        
                        search_results = await search_agent.search_products(
                            query=user_input if search_query == "trending" else search_query,
                            filters=fallback_filters,
                            category=None,
                            exclude_ids=global_excluded_ids,
                            limit=20
                        )
                        products = search_results.get("products", [])
                        logger.info(f"[Brain] Fallback search found {len(products)} products")
                    except Exception as fallback_error:
                        logger.error(f"[Brain] Fallback search also failed: {fallback_error}")
                        products = []
            
            # Step 4.5: Analyze Gen-Z fashion trends
            trend_context = None
            if products:
                logger.info(f"[Brain] Analyzing Gen-Z fashion trends")
                trend_context = await genz_trend_agent.analyze_trend_context(
                    user_input, user_memory
                )
                logger.info(f"[Brain] Trend aesthetic: {trend_context.get('aesthetic')}")
            
            # Step 5: Build outfits or rank products (if we have products)
            recommendations = []
            if products:
                logger.info(f"[Brain] Building outfits from {len(products)} products")
                # Try to build complete outfits first
                try:
                    recommendations = recommendation_agent.build_outfits(
                        products=products,
                        user_memory=user_memory,
                        intent=user_intent,
                        trend_context=trend_context
                    )
                    logger.info(f"[Brain] Built {len(recommendations)} outfits")
                except Exception as e:
                    logger.warning(f"[Brain] Outfit building failed, falling back to product ranking: {e}")
                    # Fallback to single product ranking
                    try:
                        recommendations = recommendation_agent.rank_products(
                            products=products,
                            user_memory=user_memory,
                            intent=user_intent
                        )
                    except Exception as rank_error:
                        logger.error(f"[Brain] Ranking also failed: {rank_error}")
                        # Last resort: just return products as-is with basic scores
                        recommendations = [{"product": p, "score": 50.0, "explanation": "Available option", "match_reasons": []} for p in products[:5]]
                
                # Step 6: Validate recommendations
                logger.info(f"[Brain] Validating recommendations")
                try:
                    validation_result = validation_agent.validate_recommendations(
                        recommendations=recommendations,
                        original_products=products,
                        user_intent=user_intent
                    )
                    
                    if validation_result.get("is_valid"):
                        recommendations = validation_result.get("validated_recommendations", [])
                    else:
                        issues = validation_result.get("issues", [])
                        logger.warning(f"[Brain] Validation issues: {issues}")
                        # Use validated recommendations anyway, but log issues
                        recommendations = validation_result.get("validated_recommendations", recommendations)
                except Exception as validation_error:
                    logger.error(f"[Brain] Validation failed: {validation_error}, using recommendations as-is")
                    # Continue with recommendations even if validation fails
            
            # Step 7: Generate response
            logger.info(f"[Brain] Generating response")
            # Get actual conversation history from memory service
            from app.services.memory.service import memory_service
            try:
                user_data = await memory_service.get_user_memory(user_id, email, chat_id)
                history = user_data.get("context", {}).get("conversationHistory", [])
            except Exception as history_error:
                logger.warning(f"[Brain] Failed to get history: {history_error}")
                history = []
            
            # Extract search authority for response generation
            search_authority = search_results.get("search_authority", {
                "search_status": "FOUND" if products else "EMPTY",
                "products_found": len(products),
                "confidence": min(0.95, 0.4 + 0.1 * len(products)) if products else 0.0,
                "llm_guardrail": "DO_NOT_SAY_NOT_FOUND" if products else "ALLOW_FALLBACK"
            })
            logger.info(f"[Brain] Authority propagated: {search_authority}")
            
            try:
                response_text = await response_agent.generate_response(
                    user_query=user_input,
                    recommendations=recommendations[:3] if recommendations and len(recommendations) > 0 and "items" in recommendations[0] else recommendations[:5],  # Top 3 outfits or top 5 products
                    user_memory=user_memory,
                    user_intent=user_intent,
                    conversation_history=history,
                    search_authority=search_authority,  # Pass authority downstream
                    trend_suggestion=trend_suggestion,  # Pass trend context for proactive tips
                    outfit_context=outfit_context,  # Pass outfit completion context
                    fashion_context=fashion_context if 'fashion_context' in locals() else None  # Pass fashion intelligence
                )
            except Exception as response_error:
                logger.error(f"[Brain] Response generation failed: {response_error}")
                # Fallback response based on whether we have products
                if recommendations:
                    product_names = [rec.get("product", {}).get("name", "item") for rec in recommendations[:3] if rec.get("product")]
                    response_text = f"Found some great options for you: {', '.join(product_names[:3])}. Want to see more details?"
                elif intent_type in ["recommend", "browse"]:
                    response_text = "I'd love to suggest some fire fits, but I'm having trouble right now. Try asking for something specific like 'sneakers' or 'hoodies' and I'll hook you up!"
                else:
                    response_text = "Arre, something glitched. Mind trying again with something more specific?"
            
            # Step 8: Determine memory update
            memory_update = await self._determine_memory_update(
                user_input=user_input,
                user_intent=user_intent,
                user_memory=user_memory
            )
            
            # Update memory if needed
            if memory_update.get("update_memory"):
                logger.info(f"[Brain] Updating memory: {memory_update}")
                await memory_agent.update_memory(
                    user_id=user_id,
                    email=email,
                    memory_type=memory_update.get("memory_type", "LONG_TERM"),
                    data=memory_update.get("data", {})
                )
            
            # Format final products list (extract from outfits or use products directly)
            final_products = []
            new_shown_ids = []
            
            if recommendations and "items" in recommendations[0]:
                # Extract all items from outfits (rare case currently)
                for outfit in recommendations[:3]:
                    for item in outfit.get("items", []):
                        final_products.append(item)
                        if item.get("id"):
                            new_shown_ids.append(item["id"])
            else:
                # Single products - Attach score and match reasons for transparency
                for rec in recommendations[:5]:
                    if rec.get("product"):
                        p = rec.get("product").copy()
                        p["_score"] = rec.get("score")
                        p["_match_reasons"] = rec.get("match_reasons")
                        final_products.append(p)
                        if p.get("id"):
                            new_shown_ids.append(p["id"])
            
            # Update session memory
            memory_update = await self._determine_memory_update(
                user_input=user_input,
                user_intent=user_intent,
                user_memory=user_memory
            )
            
            # Update memory if needed
            if memory_update.get("update_memory"):
                logger.info(f"[Brain] Updating memory: {memory_update}")
                await memory_agent.update_memory(
                    user_id=user_id,
                    email=email,
                    memory_type=memory_update.get("memory_type", "LONG_TERM"),
                    data=memory_update.get("data", {})
                )
            
            return {
                "response": response_text,
                "products": final_products,
                "products_found": len(final_products),
                "metadata": {
                    "intent_type": intent_type,
                    "ambiguity_level": user_intent.get("ambiguity_level"),
                    "memory_updated": memory_update.get("update_memory", False),
                    "recommendations_count": len(recommendations)
                },
                "memory_update": memory_update if memory_update.get("update_memory") else None
            }
            
        except Exception as e:
            logger.error(f"[Brain] Error processing request: {e}")
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"[Brain] Full error trace: {error_trace}")
            
            # CRASH DUMP to file for debugging
            try:
                with open("crash_dump.log", "a") as f:
                    import datetime
                    f.write(f"\n{'='*60}\n")
                    f.write(f"CRASH AT: {datetime.datetime.now()}\n")
                    f.write(f"USER INPUT: {user_input}\n")
                    f.write(f"ERROR: {e}\n")
                    f.write(f"TRACE:\n{error_trace}\n")
            except:
                pass
            
            # Graceful fallback - try to understand what user wants and provide helpful response
            try:
                # Try to at least understand the intent
                user_intent = await query_agent.understand_query(user_input)
                intent_type = user_intent.get("intent_type", "search")
                
                # Generate a helpful, Gen-Z friendly error response
                if intent_type in ["recommend", "browse"]:
                    # User wants suggestions - try to give something helpful
                    fallback_response = "Arre, something went wrong on my end. But hey, I can still help! Try asking for something specific like 'sneakers' or 'hoodies' and I'll hook you up with some fire options."
                elif intent_type == "search":
                    fallback_response = "Hmm, hit a snag there. Could you try being a bit more specific? Like 'white sneakers' or 'black hoodie'?"
                else:
                    fallback_response = "Yo, my bad - something glitched. Mind trying again? Maybe rephrase what you're looking for?"
                
            except Exception as fallback_error:
                logger.error(f"[Brain] Even fallback failed: {fallback_error}")
                # Ultimate fallback - friendly and helpful
                fallback_response = "Hey, something went wrong on my end. No worries though - try asking for something specific like 'sneakers' or 't-shirts' and I'll help you out!"
            
            return {
                "response": fallback_response,
                "products": [],
                "products_found": 0,
                "metadata": {
                    "intent_type": "error_fallback",
                    "error_handled": True
                },
                "memory_update": None
            }
    
    async def _determine_memory_update(
        self,
        user_input: str,
        user_intent: Dict[str, Any],
        user_memory: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Determine if memory should be updated based on user input.
        
        Uses LLM to extract preferences from user input.
        """
        # CRITICAL FIX: Check if user_intent already has extracted preferences
        # This is more reliable than keyword matching
        has_gender = user_intent.get("gender") is not None
        has_user_name = user_intent.get("user_name") is not None
        has_buying_type = user_intent.get("buying_type") is not None
        has_preferences = bool(user_intent.get("preferences", {}))
        
        # Check if user input contains preference signals
        preference_keywords = [
            "prefer", "like", "love", "want", "need", "budget", "price",
            "under", "over", "afford", "favorite", "favourite", "style",
            "don't like", "dont like", "hate", "dislike", "avoid", "not",
            "vibe", "minimal", "streetwear", "size", "size:", "my size",
            "mens", "womens", "male", "female", "men", "women", "guy", "girl"  # Gender keywords
        ]
        
        user_lower = user_input.lower()
        has_preference_signal = any(keyword in user_lower for keyword in preference_keywords)
        
        # Update memory if we have extracted preferences OR keyword signals
        should_update = has_gender or has_user_name or has_buying_type or has_preferences or has_preference_signal
        
        if not should_update:
            return {"update_memory": False}
        
        # Use LLM to extract structured preferences
        prompt = f"""
        Analyze this user input and extract preferences that should be stored in long-term memory.
        
        User input: "{user_input}"
        Intent Data (Gender/Type): {json.dumps({"gender": user_intent.get("gender"), "buying_type": user_intent.get("buying_type")})}
        
        Return JSON with:
        {{
            "update_memory": true,
            "memory_type": "LONG_TERM" | "SHORT_TERM",
            "data": {{
                "user_name": "string or null",
                "gender": "male" | "female" | "unisex" | null,
                "buying_type": "regular" | "occasion" | null,
                "preferred_price_range": "string or null",
                "interested_categories": ["string"],
                "preferred_brands": ["string"],
                "style_preferences": {{}}
            }}
        }}
        
        Rules:
        1. If user introduced themselves ("I am Yash", "name is Aria"), set user_name.
        2. CRITICAL - Gender mapping:
           - "mens", "men", "male", "guy", "boy" → "male"
           - "womens", "women", "female", "girl", "lady" → "female"  
           - "unisex", "gender neutral", "both" → "unisex"
           - Use the gender from Intent Data if available
        3. Set buying_type if mentioned (regular buying vs occasion-based).
        4. Extract price range, categories, and brands if mentioned.
        5. Return update_memory: false if no clear preferences or identity info found.
        6. ALWAYS set update_memory: true if Intent Data contains gender, user_name, or buying_type.
        """
        
        try:
            completion = self.client.chat.completions.create(
                model=settings.GROQ_MODEL_SMALL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a memory extraction agent. Return ONLY valid JSON. No markdown, no explanation."
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
            return result
            
        except Exception as e:
            logger.error(f"Memory update determination error: {e}")
            return {"update_memory": False}


# Singleton instance
brain_orchestrator = BrainOrchestrator()
