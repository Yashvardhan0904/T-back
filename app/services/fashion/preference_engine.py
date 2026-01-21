# Preference Engine - Handles Spontaneous Mind Changes
# Implements probability distributions, time decay, and dual-mind blending

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import math
import numpy as np
from app.db.mongodb import get_database
from app.core.config import get_settings
import logging

logger = logging.getLogger(__name__)
settings = get_settings()


class PreferenceEngine:
    """
    Handles dynamic preference modeling with:
    1. Preference as probability distributions (not boolean)
    2. Time-decay for old preferences
    3. Dual-mind blending (stated vs revealed)
    4. Rejection tracking (stronger than likes)
    5. Mind-change detection
    """
    
    # Decay parameters
    DECAY_LAMBDA = 0.1  # How fast old preferences fade (per day)
    
    # Update learning rates
    LIKE_LEARNING_RATE = 0.15
    REJECT_LEARNING_RATE = 0.25  # Rejection is stronger signal
    SKIP_LEARNING_RATE = 0.05
    
    # Mind change detection threshold
    CONTRADICTION_THRESHOLD = 0.3
    
    # Blending parameters
    BETA_CONTRADICTION = 0.2  # How fast alpha drops with contradictions
    
    async def get_effective_preference(self, user_id: str, email: str = None) -> Dict[str, Any]:
        """
        Get effective preference vector blending stated and revealed.
        
        U_eff = α(t) * U_stated + (1-α(t)) * U_revealed
        Where α(t) = e^(-β * contradiction_count)
        
        Returns preference dict with effective probabilities.
        """
        db = get_database()
        
        # Build query
        query = {"$or": []}
        if email:
            query["$or"].append({"userEmail": email})
        if user_id and user_id != "guest":
            query["$or"].append({"userId": user_id})
        
        if not query["$or"]:
            return self._get_default_preferences()
        
        try:
            # Fetch user fashion data
            user_data = await db.user_fashion_profiles.find_one(query)
            
            if not user_data:
                return self._get_default_preferences()
            
            # Calculate alpha based on contradictions
            contradiction_count = user_data.get("contradiction_count", 0)
            alpha = math.exp(-self.BETA_CONTRADICTION * contradiction_count)
            
            # Get stated and revealed preferences
            stated = user_data.get("stated_preferences", {})
            revealed = user_data.get("revealed_preferences", {})
            
            # Blend color preferences
            color_probs = {}
            all_colors = set(stated.get("color_probs", {}).keys()) | set(revealed.get("color_probs", {}).keys())
            
            for color in all_colors:
                stated_prob = stated.get("color_probs", {}).get(color, 0.5)
                revealed_prob = revealed.get("color_probs", {}).get(color, 0.5)
                color_probs[color] = alpha * stated_prob + (1 - alpha) * revealed_prob
            
            # Blend style preferences
            style_probs = {}
            all_styles = set(stated.get("style_probs", {}).keys()) | set(revealed.get("style_probs", {}).keys())
            
            for style in all_styles:
                stated_prob = stated.get("style_probs", {}).get(style, 0.5)
                revealed_prob = revealed.get("style_probs", {}).get(style, 0.5)
                style_probs[style] = alpha * stated_prob + (1 - alpha) * revealed_prob
            
            return {
                "color_probs": color_probs,
                "style_probs": style_probs,
                "alpha": alpha,
                "contradiction_count": contradiction_count,
                "interaction_count": user_data.get("interaction_count", 0)
            }
            
        except Exception as e:
            logger.error(f"Error getting effective preference: {e}")
            return self._get_default_preferences()
    
    def _get_default_preferences(self) -> Dict[str, Any]:
        """Return default preference distribution."""
        return {
            "color_probs": {
                "black": 0.7, "white": 0.6, "grey": 0.6,
                "blue": 0.55, "navy": 0.55, "red": 0.5,
                "maroon": 0.5, "beige": 0.55, "green": 0.45
            },
            "style_probs": {
                "casual": 0.6, "streetwear": 0.5, "formal": 0.4,
                "minimalist": 0.5, "ethnic": 0.45
            },
            "alpha": 1.0,
            "contradiction_count": 0,
            "interaction_count": 0
        }
    
    async def update_preference(
        self,
        user_id: str,
        email: str,
        features: Dict[str, Any],
        action: str  # "liked" | "rejected" | "skipped"
    ) -> None:
        """
        Update preference distributions based on user interaction.
        
        Bayesian update:
        - Like: P_new(f) = P_old(f) + η(1 - P_old(f))
        - Reject: P_new(f) = P_old(f) * (1 - η)
        
        Where η_reject > η_like
        """
        db = get_database()
        
        if not email and (not user_id or user_id == "guest"):
            return
        
        # Determine learning rate
        if action == "liked":
            eta = self.LIKE_LEARNING_RATE
            update_positive = True
        elif action == "rejected":
            eta = self.REJECT_LEARNING_RATE
            update_positive = False
        else:  # skipped
            eta = self.SKIP_LEARNING_RATE
            update_positive = False
        
        query = {"$or": []}
        if email:
            query["$or"].append({"userEmail": email})
        if user_id and user_id != "guest":
            query["$or"].append({"userId": user_id})
        
        if not query["$or"]:
            return
        
        try:
            # Fetch current preferences
            user_data = await db.user_fashion_profiles.find_one(query)
            
            if not user_data:
                user_data = {
                    "userId": user_id,
                    "userEmail": email,
                    "stated_preferences": self._get_default_preferences(),
                    "revealed_preferences": self._get_default_preferences(),
                    "interaction_count": 0,
                    "contradiction_count": 0,
                    "last_decay_at": datetime.utcnow()
                }
            
            revealed = user_data.get("revealed_preferences", self._get_default_preferences())
            
            # Update color preferences
            colors = features.get("colors", [])
            for color in colors:
                color_lower = color.lower()
                current_prob = revealed.get("color_probs", {}).get(color_lower, 0.5)
                
                if update_positive:
                    # Bayesian increase
                    new_prob = current_prob + eta * (1 - current_prob)
                else:
                    # Bayesian decrease
                    new_prob = current_prob * (1 - eta)
                
                revealed.setdefault("color_probs", {})[color_lower] = max(0.05, min(0.95, new_prob))
            
            # Update style preferences
            style = features.get("style")
            if style:
                style_lower = style.lower()
                current_prob = revealed.get("style_probs", {}).get(style_lower, 0.5)
                
                if update_positive:
                    new_prob = current_prob + eta * (1 - current_prob)
                else:
                    new_prob = current_prob * (1 - eta)
                
                revealed.setdefault("style_probs", {})[style_lower] = max(0.05, min(0.95, new_prob))
            
            # Check for contradiction with stated preferences
            stated = user_data.get("stated_preferences", {})
            contradiction_detected = self._detect_contradiction(stated, features, action)
            
            # Update in database
            update_doc = {
                "$set": {
                    "revealed_preferences": revealed,
                    "lastUpdated": datetime.utcnow(),
                    "userId": user_id,
                    "userEmail": email
                },
                "$inc": {
                    "interaction_count": 1,
                }
            }
            
            if contradiction_detected:
                update_doc["$inc"]["contradiction_count"] = 1
            
            await db.user_fashion_profiles.update_one(query, update_doc, upsert=True)
            
            logger.info(f"Updated preferences for {email or user_id}: {action} on colors={colors}")
            
        except Exception as e:
            logger.error(f"Error updating preference: {e}")
    
    def _detect_contradiction(
        self,
        stated: Dict[str, Any],
        features: Dict[str, Any],
        action: str
    ) -> bool:
        """
        Detect if user action contradicts stated preferences.
        
        Example: User says "I love red" but rejects red outfits.
        """
        colors = features.get("colors", [])
        
        for color in colors:
            color_lower = color.lower()
            stated_prob = stated.get("color_probs", {}).get(color_lower, 0.5)
            
            # High stated preference + rejection = contradiction
            if stated_prob > 0.7 and action == "rejected":
                return True
            
            # Low stated preference + enthusiastic like = contradiction (positive)
            if stated_prob < 0.3 and action == "liked":
                return True
        
        return False
    
    async def apply_time_decay(self, user_id: str, email: str = None) -> None:
        """
        Apply exponential decay to old preferences.
        
        w(t) = w(t₀) * e^(-λ(t-t₀))
        
        Call this periodically (e.g., once per session).
        """
        db = get_database()
        
        query = {"$or": []}
        if email:
            query["$or"].append({"userEmail": email})
        if user_id and user_id != "guest":
            query["$or"].append({"userId": user_id})
        
        if not query["$or"]:
            return
        
        try:
            user_data = await db.user_fashion_profiles.find_one(query)
            
            if not user_data:
                return
            
            last_decay = user_data.get("last_decay_at", datetime.utcnow())
            now = datetime.utcnow()
            
            # Calculate days since last decay
            days_elapsed = (now - last_decay).total_seconds() / 86400
            
            if days_elapsed < 1:
                return  # Don't decay more than once per day
            
            # Calculate decay factor
            decay_factor = math.exp(-self.DECAY_LAMBDA * days_elapsed)
            
            # Apply decay to revealed preferences (pull toward 0.5 neutral)
            revealed = user_data.get("revealed_preferences", {})
            
            for category in ["color_probs", "style_probs"]:
                probs = revealed.get(category, {})
                for key in probs:
                    current = probs[key]
                    # Decay toward neutral (0.5)
                    probs[key] = 0.5 + (current - 0.5) * decay_factor
            
            # Also decay contradiction count
            current_contradictions = user_data.get("contradiction_count", 0)
            decayed_contradictions = int(current_contradictions * decay_factor)
            
            await db.user_fashion_profiles.update_one(
                query,
                {
                    "$set": {
                        "revealed_preferences": revealed,
                        "contradiction_count": decayed_contradictions,
                        "last_decay_at": now
                    }
                }
            )
            
            logger.info(f"Applied time decay for {email or user_id}, decay_factor={decay_factor:.3f}")
            
        except Exception as e:
            logger.error(f"Error applying time decay: {e}")
    
    async def detect_mind_change(self, user_id: str, email: str = None) -> Dict[str, Any]:
        """
        Detect if user has significantly changed their mind.
        
        D = ||U_stated - U_revealed||_2
        If D > τ → exploration mode recommended
        """
        db = get_database()
        
        query = {"$or": []}
        if email:
            query["$or"].append({"userEmail": email})
        if user_id and user_id != "guest":
            query["$or"].append({"userId": user_id})
        
        if not query["$or"]:
            return {"mind_change_detected": False, "distance": 0}
        
        try:
            user_data = await db.user_fashion_profiles.find_one(query)
            
            if not user_data:
                return {"mind_change_detected": False, "distance": 0}
            
            stated = user_data.get("stated_preferences", {})
            revealed = user_data.get("revealed_preferences", {})
            
            # Calculate L2 distance for colors
            stated_colors = stated.get("color_probs", {})
            revealed_colors = revealed.get("color_probs", {})
            
            distance = 0.0
            count = 0
            
            for color in set(stated_colors.keys()) & set(revealed_colors.keys()):
                diff = stated_colors[color] - revealed_colors[color]
                distance += diff ** 2
                count += 1
            
            if count > 0:
                distance = (distance / count) ** 0.5
            
            mind_change = distance > self.CONTRADICTION_THRESHOLD
            
            return {
                "mind_change_detected": mind_change,
                "distance": distance,
                "recommendation": "exploration" if mind_change else "normal"
            }
            
        except Exception as e:
            logger.error(f"Error detecting mind change: {e}")
            return {"mind_change_detected": False, "distance": 0}
    
    async def update_stated_preference(
        self,
        user_id: str,
        email: str,
        preference_type: str,  # "color" | "style" | "fit"
        value: str,
        probability: float = 0.8
    ) -> None:
        """
        Update stated preference when user explicitly says something.
        
        Example: User says "I love red" → update stated color preference for red.
        """
        db = get_database()
        
        query = {"$or": []}
        if email:
            query["$or"].append({"userEmail": email})
        if user_id and user_id != "guest":
            query["$or"].append({"userId": user_id})
        
        if not query["$or"]:
            return
        
        try:
            field = None
            if preference_type == "color":
                field = f"stated_preferences.color_probs.{value.lower()}"
            elif preference_type == "style":
                field = f"stated_preferences.style_probs.{value.lower()}"
            elif preference_type == "fit":
                field = f"stated_preferences.fit_probs.{value.lower()}"
            
            if field:
                await db.user_fashion_profiles.update_one(
                    query,
                    {
                        "$set": {
                            field: probability,
                            "lastUpdated": datetime.utcnow()
                        }
                    },
                    upsert=True
                )
                
                logger.info(f"Updated stated preference for {email or user_id}: {preference_type}={value} ({probability})")
                
        except Exception as e:
            logger.error(f"Error updating stated preference: {e}")


# Singleton instance
preference_engine = PreferenceEngine()
