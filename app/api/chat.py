from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from app.services.intent.service import intent_service
from app.services.search.orchestrator import search_orchestrator
from app.services.llm.service import llm_service
from app.services.memory.service import memory_service
from app.services.tools.cart import cart_service
from app.services.tools.wishlist import wishlist_service
from app.services.tools.extractor import tool_extractor
from datetime import datetime
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================

class ChatRequest(BaseModel):
    message: str
    userId: str = "guest"
    userEmail: Optional[str] = None
    chatId: Optional[str] = None
    useIntelligentSearch: bool = True  # Toggle for new AI system
    useMultiAgentSystem: bool = True  # Toggle for new multi-agent orchestrator

class ChatResponse(BaseModel):
    response: str
    intent: str
    products: List[dict] = []
    products_found: int
    outfits: List[dict] = []  # New: multi-item outfit suggestions
    metadata: Dict[str, Any] = {}  # New: AI system metadata

class FeedbackRequest(BaseModel):
    userId: str
    userEmail: Optional[str] = None
    outfitId: Optional[str] = None
    productId: Optional[str] = None
    action: str  # "liked" | "rejected" | "skipped" | "purchased"
    features: Dict[str, Any] = {}  # colors, style, etc.
    responseTimeMs: Optional[int] = None

class FeedbackResponse(BaseModel):
    success: bool
    message: str

class ProfileRequest(BaseModel):
    userId: str
    userEmail: str
    physical: Dict[str, Any] = {}  # skinTone, height, bodyShape, etc.
    psychology: Dict[str, Any] = {}  # confidence, riskTolerance, etc.

# ============================================================
# HELPER FUNCTIONS
# ============================================================

async def _extract_and_save_insights(message: str, history: list, user_id: str, email: str):
    """Background task to silently extract and save user insights."""
    try:
        from app.services.fashion.silent_extractor import silent_extractor
        insights = await silent_extractor.extract_silent_insights(message, history)
        if insights:
            await silent_extractor.save_to_memory(user_id, email, insights)
    except Exception as e:
        logger.error(f"[SilentExtract] Background error: {e}")

# ============================================================
# CHAT ENDPOINT (Enhanced with Intelligence)
# ============================================================


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    try:
        # ============================================================
        # NEW MULTI-AGENT SYSTEM (Brain Orchestrator)
        # ============================================================
        if request.useMultiAgentSystem:
            try:
                from app.services.agents.brain_orchestrator import brain_orchestrator
                
                logger.info(f"[Chat] Using multi-agent system for user {request.userId}")
                
                # Process request through brain orchestrator
                result = await brain_orchestrator.process_request(
                    user_input=request.message,
                    user_id=request.userId,
                    email=request.userEmail,
                    chat_id=request.chatId
                )
                
                # Save interactions
                background_tasks.add_task(
                    memory_service.save_interaction,
                    request.userId,
                    request.userEmail,
                    "user",
                    request.message,
                    result.get("metadata", {}).get("intent_type", "search")
                )
                background_tasks.add_task(
                    memory_service.save_interaction,
                    request.userId,
                    request.userEmail,
                    "assistant",
                    result.get("response", ""),
                    result.get("metadata", {}).get("intent_type", "search"),
                    result.get("products", [])
                )
                
                # Nuclear serialization insurance
                import json
                safe_products = json.loads(json.dumps(result.get("products", []), default=str))
                
                return ChatResponse(
                    response=result.get("response", ""),
                    intent=result.get("metadata", {}).get("intent_type", "search"),
                    products=safe_products,
                    products_found=result.get("products_found", 0),
                    outfits=[],  # Multi-agent system focuses on products
                    metadata={
                        **result.get("metadata", {}),
                        "multi_agent_system": True,
                        "memory_updated": result.get("memory_update") is not None
                    }
                )
                
            except Exception as e:
                logger.error(f"Multi-agent system error, falling back: {e}")
                import traceback
                error_trace = traceback.format_exc()
                logger.error(f"Full error trace: {error_trace}")
                
                # Try to provide a graceful response before falling back
                try:
                    # Quick intent check for better error message
                    from app.services.agents.query_agent import query_agent
                    user_intent = await query_agent.understand_query(request.message)
                    intent_type = user_intent.get("intent_type", "search")
                    
                    if intent_type in ["recommend", "browse"]:
                        error_response = "Arre, something glitched. But I got you - try asking for something specific like 'sneakers' or 'hoodies' and I'll find you some fire options!"
                    else:
                        error_response = "Yo, hit a snag there. Mind trying again with something more specific? Like 'white sneakers' or 'black t-shirt'?"
                    
                    return ChatResponse(
                        response=error_response,
                        intent=intent_type,
                        products=[],
                        products_found=0,
                        outfits=[],
                        metadata={
                            "error_handled": True,
                            "fallback": True
                        }
                    )
                except:
                    # Ultimate fallback
                    pass
                
                # Fall through to legacy system
        
        # ============================================================
        # LEGACY SYSTEM (Original implementation)
        # ============================================================
        
        # 0. Load Memory & DNA
        user_data = await memory_service.get_user_memory(request.userId, request.userEmail, request.chatId)
        memory = user_data.get("memory")
        context = user_data.get("context")
        
        # 0.5 SILENT INSIGHT EXTRACTION - Passively learn from EVERY message
        from app.services.fashion.silent_extractor import silent_extractor
        background_tasks.add_task(
            _extract_and_save_insights,
            request.message,
            context.get("conversationHistory", []) if context else [],
            request.userId,
            request.userEmail
        )
        
        # Build DNA Flash for LLM context (moved to memory for efficiency)
        dna_flash = silent_extractor.build_dna_flash(memory)
        
        # 1. Intent Detection (using 70B for accuracy)
        intent_task = intent_service.classify_intent(request.message)
        # Simultaneously start generating thinking steps
        thinking_task = llm_service.generate_thinking_steps(request.message, "analyzing...")
        
        import asyncio
        try:
            intent, thinking_steps = await asyncio.gather(intent_task, thinking_task)
        except Exception as e:
            logger.error(f"Async tasks failed: {e}")
            intent = "SEARCH" # Safe default
            thinking_steps = ["Thinking...", "Analyzing style...", "Searching database..."]

        
        # 2. Routing Logic
        if intent == "IDENTITY_CORRECTION":
            await memory_service.extract_preferences(request.userId, request.userEmail, request.message)
            user_data = await memory_service.get_user_memory(request.userId, request.userEmail, request.chatId)
            memory = user_data.get("memory")
            
            response = await llm_service.generate_response(request.message, [], intent, memory, context)
            
            background_tasks.add_task(memory_service.save_interaction, request.userId, request.userEmail, "user", request.message, intent)
            background_tasks.add_task(memory_service.save_interaction, request.userId, request.userEmail, "assistant", response, intent)
            
            return ChatResponse(response=response, intent=intent, products_found=0, metadata={"steps": thinking_steps})

        if intent in ["GREETING", "SMALL_TALK", "DISMISSIVE"]:
            response = await llm_service.generate_response(request.message, [], intent, memory, context)
            
            background_tasks.add_task(memory_service.save_interaction, request.userId, request.userEmail, "user", request.message, intent)
            background_tasks.add_task(memory_service.save_interaction, request.userId, request.userEmail, "assistant", response, intent)
            background_tasks.add_task(memory_service.extract_preferences, request.userId, request.userEmail, request.message)
            
            return ChatResponse(response=response, intent=intent, products_found=0, metadata={"steps": thinking_steps})

        # Special intents: surprise me / explore
        if intent == "EXPLORE" or "surprise" in request.message.lower():
            from app.services.fashion.intelligent_orchestrator import intelligent_orchestrator
            await intelligent_orchestrator.trigger_exploration(request.userId, request.userEmail)
            # Continue to search with exploration mode

        if intent in ["ADD_TO_CART", "ADD_TO_WISHLIST"]:
            product_info = await tool_extractor.extract_product_id(request.message)
            
            if product_info:
                search_results = await search_orchestrator.hybrid_search(product_info, limit=1)
                if search_results:
                    product_id = str(search_results[0]['_id'])
                    
                    if intent == "ADD_TO_CART":
                        success = await cart_service.add_to_cart(request.userId, product_id)
                        action_msg = "Successfully added to your cart!" if success else "Failed to add to cart."
                        products = search_results
                    else:
                        success = await wishlist_service.add_to_wishlist(request.userId, product_id)
                        action_msg = "Successfully added to your wishlist!" if success else "Failed to add to wishlist."
                        products = search_results
                    
                    response = await llm_service.generate_response(f"{request.message} (Action result: {action_msg})", products, intent, memory, context)
                    
                    background_tasks.add_task(memory_service.save_interaction, request.userId, request.userEmail, "user", request.message, intent)
                    background_tasks.add_task(memory_service.save_interaction, request.userId, request.userEmail, "assistant", response, intent)
                    
                    return ChatResponse(
                        response=response, 
                        intent=intent, 
                        products=products, 
                        products_found=len(products),
                        metadata={"steps": thinking_steps}
                    )

            response = "I'm ready to add that for you, but I'm not sure which item you mean. Could you specify the product name?"
            return ChatResponse(response=response, intent=intent, products_found=0)

        if intent == "VIEW_CART":
            products = await cart_service.get_cart(request.userId)
            response = await llm_service.generate_response(request.message, products, intent, memory, context)
            return ChatResponse(
                response=response, 
                intent=intent, 
                products=products, 
                products_found=len(products),
                metadata={"steps": thinking_steps}
            )

        if intent == "CLEAR_CART":
            success = await cart_service.clear_cart(request.userId)
            msg = "Cart cleared!" if success else "Failed to clear cart."
            response = await llm_service.generate_response(f"{request.message} ({msg})", [], intent, memory, context)
            return ChatResponse(response=response, intent=intent, products_found=0, metadata={"steps": thinking_steps})

        if intent == "GET_TRENDING":
            products = await search_orchestrator.hybrid_search("trending", limit=5)
            response = await llm_service.generate_response(request.message, products, intent, memory, context)
            return ChatResponse(
                response=response, 
                intent=intent, 
                products=products, 
                products_found=len(products),
                metadata={"steps": thinking_steps}
            )

        # 3. INTELLIGENT SEARCH (New Fashion AI System)
        if request.useIntelligentSearch:
            try:
                from app.services.fashion.intelligent_orchestrator import intelligent_orchestrator
                
                result = await intelligent_orchestrator.intelligent_search(
                    query=request.message,
                    user_id=request.userId,
                    email=request.userEmail,
                    chat_id=request.chatId,
                    limit=5
                )
                
                products = result.get("products", [])
                outfits = result.get("outfits", [])
                metadata = result.get("metadata", {})
                
                # Generate response with outfit context
                final_text = await llm_service.generate_response_with_outfits(
                    request.message, products, outfits, intent, memory, context, metadata
                )
                
                background_tasks.add_task(memory_service.save_interaction, request.userId, request.userEmail, "user", request.message, intent)
                background_tasks.add_task(memory_service.save_interaction, request.userId, request.userEmail, "assistant", final_text, intent)
                background_tasks.add_task(memory_service.extract_preferences, request.userId, request.userEmail, request.message)
                
                # Nuclear serialization insurance: ensure everything is JSON serializable
                import json
                safe_products = json.loads(json.dumps(products, default=str))
                safe_outfits = json.loads(json.dumps(outfits, default=str))

                return ChatResponse(
                    response=final_text,
                    intent=intent,
                    products=safe_products,
                    products_found=len(safe_products),
                    outfits=safe_outfits,
                    metadata={
                        "dna_used": bool(dna_flash),
                        "intelligent": True,
                        "steps": thinking_steps
                    }
                )
                
            except Exception as e:
                logger.warning(f"Intelligent search failed, falling back to basic: {e}")
                # Fall through to basic search
        
        # 4. FALLBACK: Basic Search (Original behavior) - WITH GENDER FILTERING
        # Extract gender from user memory for fallback filtering
        try:
            user_data = await memory_service.get_user_memory(request.userId, request.userEmail, request.chatId)
            user_gender = user_data.get("memory", {}).get("long_term", {}).get("gender")
            
            # Apply basic gender filtering to fallback search
            mandatory_filters = {}
            if user_gender:
                mandatory_filters["gender"] = user_gender
                logger.info(f"[Fallback] Applying gender filter: {user_gender}")
            
            products = await search_orchestrator.hybrid_search(
                request.message, 
                mandatory_filters=mandatory_filters
            )
        except Exception as fallback_error:
            logger.error(f"[Fallback] Even fallback search failed: {fallback_error}")
            products = []
        
        final_text = await llm_service.generate_response(request.message, products, intent, memory, context)
        
        background_tasks.add_task(memory_service.save_interaction, request.userId, request.userEmail, "user", request.message, intent)
        background_tasks.add_task(memory_service.save_interaction, request.userId, request.userEmail, "assistant", final_text, intent)
        background_tasks.add_task(memory_service.extract_preferences, request.userId, request.userEmail, request.message)
        
        # Nuclear serialization insurance for fallback
        import json
        safe_products = json.loads(json.dumps(products, default=str))

        return ChatResponse(
            response=final_text,
            intent=intent,
            products=safe_products,
            products_found=len(safe_products),
            metadata={"steps": thinking_steps, "fallback": True}
        )
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Global Chat Error: {e}\n{error_trace}")
        # Only return a simple string for the error to guarantee serialization
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


# ============================================================
# FEEDBACK ENDPOINT (New - For Learning)
# ============================================================

@router.post("/feedback", response_model=FeedbackResponse)
async def feedback_endpoint(request: FeedbackRequest):
    """
    Record user feedback for adaptive learning.
    
    This updates:
    - Preference distributions (color, style probabilities)
    - Session state (mood, decision speed)
    - Exploration arm statistics (UCB)
    """
    try:
        from app.services.fashion.intelligent_orchestrator import intelligent_orchestrator
        
        outfit_id = request.outfitId or request.productId or "unknown"
        
        await intelligent_orchestrator.record_feedback(
            user_id=request.userId,
            email=request.userEmail,
            outfit_id=outfit_id,
            action=request.action,
            item_features=request.features,
            response_time_ms=request.responseTimeMs
        )
        
        return FeedbackResponse(
            success=True,
            message=f"Feedback recorded: {request.action}"
        )
        
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        return FeedbackResponse(success=False, message=str(e))


# ============================================================
# PROFILE ENDPOINT (New - For Style Profile)
# ============================================================

@router.post("/profile/style")
async def update_style_profile(request: ProfileRequest):
    """
    Update user's style profile (physical + psychological attributes).
    
    Called from style profile onboarding page.
    """
    try:
        from app.services.fashion.user_vector_service import user_vector_service
        
        results = {}
        
        if request.physical:
            success = await user_vector_service.save_physical_profile(
                request.userId, request.userEmail, request.physical
            )
            results["physical"] = success
        
        if request.psychology:
            success = await user_vector_service.save_psychology_profile(
                request.userId, request.userEmail, request.psychology
            )
            results["psychology"] = success
        
        return {"success": True, "results": results}
        
    except Exception as e:
        logger.error(f"Profile update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# EXPLORATION TRIGGER ENDPOINT
# ============================================================

@router.post("/explore")
async def trigger_exploration(userId: str, userEmail: Optional[str] = None):
    """
    Trigger exploration mode for user.
    Use when user says "surprise me" or wants something different.
    """
    try:
        from app.services.fashion.intelligent_orchestrator import intelligent_orchestrator
        
        await intelligent_orchestrator.trigger_exploration(userId, userEmail)
        
        return {"success": True, "message": "Exploration mode activated"}
        
    except Exception as e:
        logger.error(f"Exploration trigger error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# RESET PREFERENCES ENDPOINT
# ============================================================

@router.post("/preferences/reset")
async def reset_preferences(userId: str, userEmail: Optional[str] = None):
    """
    Reset session preferences.
    Use when user says "forget what I said" or wants fresh start.
    """
    try:
        from app.services.fashion.intelligent_orchestrator import intelligent_orchestrator
        
        await intelligent_orchestrator.reset_preferences(userId, userEmail)
        
        return {"success": True, "message": "Preferences reset for this session"}
        
    except Exception as e:
        logger.error(f"Preference reset error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

