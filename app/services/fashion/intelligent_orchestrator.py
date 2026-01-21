# Intelligent Search Orchestrator
# Replaces basic search with full fashion intelligence pipeline

from typing import Dict, List, Any, Optional
from app.models.user_vector import FullUserVector, Outfit
from app.services.fashion.user_vector_service import user_vector_service
from app.services.fashion.outfit_generator import outfit_generator
from app.services.fashion.scoring_engine import scoring_engine
from app.services.fashion.preference_engine import preference_engine
from app.services.fashion.exploration_engine import exploration_engine
from app.services.fashion.session_state import session_engine
from app.db.mongodb import get_database
import logging

logger = logging.getLogger(__name__)


class IntelligentOrchestrator:
    """
    Main intelligence pipeline that replaces basic search.
    
    Pipeline:
    1. Load user vector
    2. Apply time decay to preferences
    3. Detect mind changes
    4. Generate outfit candidates
    5. Score all candidates
    6. Apply exploration mixing
    7. Return ranked results with explanations
    """
    
    async def intelligent_search(
        self,
        query: str,
        user_id: str,
        email: str = None,
        chat_id: str = None,
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Main search endpoint with full intelligence.
        
        Returns:
            {
                "outfits": List[dict],  # Scored outfits with explanations
                "products": List[dict],  # Fallback flat product list
                "metadata": {
                    "exploration_mode": bool,
                    "mind_change_detected": bool,
                    "dominant_style": str
                }
            }
        """
        try:
            # 1. Load full user vector
            user_vector = await user_vector_service.get_full_vector(user_id, email)
            logger.info(f"Loaded user vector with {user_vector.get_dimension_count()} dims")
            
            # 2. Apply time decay to preferences
            await preference_engine.apply_time_decay(user_id, email)
            
            # 3. Detect mind changes
            mind_change = await preference_engine.detect_mind_change(user_id, email)
            mind_change_detected = mind_change.get("mind_change_detected", False)
            
            if mind_change_detected:
                logger.info(f"Mind change detected for {email or user_id}: distance={mind_change.get('distance'):.3f}")
            
            # 4. Get exploration recommendation
            exploration_rec = exploration_engine.get_exploration_recommendation(
                user_vector, mind_change_detected
            )
            
            # 5. Generate outfit candidates
            outfits = await outfit_generator.generate_outfits(query, user_vector, limit=50)
            
            if not outfits:
                # Fallback to single items
                logger.warning("No outfits generated, falling back to single items")
                items = await outfit_generator.generate_single_item_suggestions(query, user_vector, limit)
                return {
                    "outfits": [],
                    "products": [self._item_to_dict(item) for item in items],
                    "metadata": {
                        "exploration_mode": user_vector.session.exploration_mode,
                        "mind_change_detected": mind_change_detected,
                        "fallback": True
                    }
                }
            
            # 6. Score all outfits
            scored_outfits = []
            for outfit in outfits:
                try:
                    score_result = scoring_engine.compute_score(outfit, user_vector)
                    scored_outfits.append((outfit, score_result["total_score"]))
                except Exception as e:
                    logger.error(f"Error scoring outfit {outfit.outfit_id}: {e}")
                    continue
            
            if not scored_outfits:
                logger.warning("All outfits failed scoring")
                return {
                    "outfits": [],
                    "products": [],
                    "metadata": {"error": "Scoring failed"}
                }
            
            # 7. ENFORCE 1-OUTFIT RULE: Select ONLY the single best outfit
            # Sort by score descending and take the top one
            scored_outfits.sort(key=lambda x: x[1], reverse=True)
            best_outfit, best_score = scored_outfits[0]
            
            # 8. Build response with SINGLE outfit
            score_details = scoring_engine.compute_score(best_outfit, user_vector)
            
            outfit_dict = {
                "outfit_id": best_outfit.outfit_id,
                "items": [self._item_to_dict(item) for item in best_outfit.items],
                "total_price": best_outfit.total_price,
                "score": score_details["total_score"],
                "selection_type": "maximum_harmony",
                "insights": score_details.get("insights", []),
                "warnings": score_details.get("warnings", []),
                "component_scores": score_details.get("scores", {}),
                "harmony_reason": f"This outfit was selected from {len(outfits)} candidates for maximum harmony with your profile."
            }
            
            # Products are just the items from this one outfit
            unique_products = [self._item_to_dict(item) for item in best_outfit.items]
            
            return {
                "outfits": [outfit_dict],  # SINGLE outfit
                "products": unique_products,
                "metadata": {
                    "exploration_mode": user_vector.session.exploration_mode,
                    "mind_change_detected": mind_change_detected,
                    "exploration_recommendation": exploration_rec,
                    "total_candidates_scored": len(outfits),
                    "user_confidence": user_vector.psychology.confidence,
                    "user_risk_tolerance": user_vector.psychology.risk_tolerance,
                    "single_outfit_mode": True
                }
            }

            
        except Exception as e:
            logger.error(f"Intelligent search error: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to basic search
            return await self._fallback_search(query, limit)
    
    def _item_to_dict(self, item) -> Dict[str, Any]:
        """Convert OutfitItem to dict for API response."""
        return {
            "_id": item.item_id,
            "name": item.name,
            "category": item.category,
            "color": item.color_name,
            "price": item.price,
            "fit": item.fit,
            "formality": item.formality,
            "images": [item.image_url] if item.image_url else [],
            "brand": item.brand
        }
    
    async def _fallback_search(self, query: str, limit: int) -> Dict[str, Any]:
        """Basic search fallback if intelligence fails."""
        from app.services.search.orchestrator import search_orchestrator
        
        products = await search_orchestrator.hybrid_search(query, limit)
        
        return {
            "outfits": [],
            "products": products,
            "metadata": {
                "fallback": True,
                "exploration_mode": False,
                "mind_change_detected": False
            }
        }
    
    async def record_feedback(
        self,
        user_id: str,
        email: str,
        outfit_id: str,
        action: str,  # "liked" | "rejected" | "skipped" | "purchased"
        item_features: Dict[str, Any] = None,
        response_time_ms: int = None
    ) -> None:
        """
        Record user feedback for learning.
        
        Updates:
        - Preference distributions
        - Session state
        - Exploration arm statistics
        """
        try:
            # 1. Update preference engine
            if item_features:
                await preference_engine.update_preference(
                    user_id, email, item_features, action
                )
            
            # 2. Update session state
            signals = {
                "action": action,
                "response_time_ms": response_time_ms or 3000,
                "features": item_features
            }
            await session_engine.update_from_behavior(user_id, email, signals)
            
            # 3. Update exploration arms
            style = item_features.get("style") if item_features else "unknown"
            reward = {"liked": 1.0, "purchased": 1.0, "rejected": -1.0, "skipped": -0.3}.get(action, 0)
            await exploration_engine.update_arm(user_id, style, reward)
            
            # 4. Save interaction to DB
            db = get_database()
            await db.fashion_interactions.insert_one({
                "userId": user_id,
                "userEmail": email,
                "outfitId": outfit_id,
                "action": action,
                "features": item_features,
                "responseTimeMs": response_time_ms,
                "timestamp": __import__("datetime").datetime.utcnow()
            })
            
            logger.info(f"Recorded feedback: {action} for {outfit_id}")
            
        except Exception as e:
            logger.error(f"Error recording feedback: {e}")
    
    async def trigger_exploration(self, user_id: str, email: str = None) -> None:
        """
        Trigger exploration mode (for "surprise me" requests).
        """
        await session_engine.trigger_exploration_mode(user_id, email)
    
    async def reset_preferences(self, user_id: str, email: str = None) -> None:
        """
        Reset preferences (for "forget what I said" requests).
        """
        await session_engine.reset_session(user_id, email)


# Singleton instance
intelligent_orchestrator = IntelligentOrchestrator()
