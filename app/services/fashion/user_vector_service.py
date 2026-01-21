# User Vector Service - Manages full user vector operations
# Loads, saves, and maintains user vectors

from typing import Dict, Any, Optional
from datetime import datetime
from app.models.user_vector import (
    FullUserVector, PhysicalVector, GenderExpressionVector,
    PsychologyVector, ContextVector, PreferenceDistribution,
    RevealedBehavior, SessionState, EconomicVector,
    ClimateVector, GeoVector, OccasionVector, TimeVector,
    Undertone, BodyShape, OccasionType, Season
)
from app.db.mongodb import get_database
from app.services.fashion.session_state import session_engine
import logging

logger = logging.getLogger(__name__)


class UserVectorService:
    """
    Service for managing full user vectors.
    
    Responsibilities:
    - Load/save user vectors from database
    - Build vectors from available data
    - Handle missing data with defaults
    - Integrate session state
    """
    
    async def get_full_vector(
        self,
        user_id: str,
        email: str = None,
        with_session: bool = True
    ) -> FullUserVector:
        """
        Get complete user vector, building from available data.
        """
        db = get_database()
        
        # Build query
        query = {"$or": []}
        if email:
            query["$or"].append({"userEmail": email})
        if user_id and user_id != "guest":
            query["$or"].append({"userId": user_id})
        
        # Default vector
        user_vector = FullUserVector(user_id=user_id, user_email=email)
        
        if not query["$or"]:
            return user_vector
        
        try:
            # 1. Load UserMemory (basic profile from existing system)
            user_memory = await db.usermemories.find_one(query)
            if user_memory:
                user_vector = self._apply_user_memory(user_vector, user_memory)
            
            # 2. Load fashion profile (extended attributes)
            fashion_profile = await db.user_fashion_profiles.find_one(query)
            if fashion_profile:
                user_vector = self._apply_fashion_profile(user_vector, fashion_profile)
            
            # 3. Load session state
            if with_session:
                session = await session_engine.get_or_create_session(user_id, email)
                user_vector.session = session
            
            # 4. Build context from current time
            user_vector.context = self._build_current_context()
            
            logger.info(f"Built user vector for {email or user_id}: {user_vector.get_dimension_count()} dims")
            return user_vector
            
        except Exception as e:
            logger.error(f"Error getting user vector: {e}")
            return user_vector
    
    def _apply_user_memory(
        self,
        vector: FullUserVector,
        memory: Dict[str, Any]
    ) -> FullUserVector:
        """
        Apply data from existing UserMemory schema.
        """
        # Style preferences
        style = memory.get("style", {})
        if style.get("fit"):
            fit_map = {"slim": 0.3, "regular": 0.5, "oversized": 0.7}
            # This would affect psychology vector
        
        if style.get("colors"):
            colors = style["colors"]
            for color in colors:
                vector.stated_preferences.color_probs[color.lower()] = 0.75
        
        if style.get("vibe"):
            vibe = style["vibe"].lower()
            vector.stated_preferences.style_probs[vibe] = 0.8
        
        # Budget
        budget = memory.get("budget", {})
        if budget.get("max"):
            vector.economic.budget_max = float(budget["max"])
        if budget.get("avg"):
            vector.economic.budget_avg = float(budget["avg"])
        
        # Behavior
        behavior = memory.get("behavior", {})
        if behavior.get("gender"):
            gender = behavior["gender"].lower()
            if gender == "male":
                vector.gender_expression.masc_femme = 0.2
            elif gender == "female":
                vector.gender_expression.masc_femme = 0.8
        
        # 4. Revealed preferences (Likes/Rejects)
        revealed = memory.get("revealed", {})
        if revealed.get("likedColors"):
            vector.revealed_behavior.liked_colors.extend(revealed["likedColors"])
        if revealed.get("rejectedColors"):
            vector.revealed_behavior.rejected_colors.extend(revealed["rejectedColors"])
            
            # Also lower probs for rejected colors
            for color in revealed["rejectedColors"]:
                vector.stated_preferences.color_probs[color.lower()] = 0.05
        
        # 5. Behavior (Legacy avoids)
        behavior = memory.get("behavior", {})
        avoids = behavior.get("avoids", [])
        for avoid in avoids:
            avoid_lower = avoid.lower()
            if avoid_lower not in vector.revealed_behavior.rejected_colors:
                vector.revealed_behavior.rejected_colors.append(avoid_lower)
            vector.stated_preferences.color_probs[avoid_lower] = 0.05
        
        return vector

    
    def _apply_fashion_profile(
        self,
        vector: FullUserVector,
        profile: Dict[str, Any]
    ) -> FullUserVector:
        """
        Apply data from extended fashion profile.
        """
        # Physical attributes
        physical = profile.get("physical", {})
        if physical:
            if physical.get("skinTone") is not None:
                vector.physical.skin_tone = physical["skinTone"]
            if physical.get("undertone"):
                try:
                    vector.physical.undertone = Undertone(physical["undertone"])
                except:
                    pass
            if physical.get("heightCm"):
                height = physical["heightCm"]
                vector.physical.height_cm = height
                vector.physical.height_normalized = (height - 150) / 50  # Normalize
            if physical.get("bodyShape"):
                vector.physical.body_shape_embedding = physical["bodyShape"]
        
        # Psychology
        psychology = profile.get("psychology", {})
        if psychology:
            if psychology.get("confidence") is not None:
                vector.psychology.confidence = psychology["confidence"]
            if psychology.get("riskTolerance") is not None:
                vector.psychology.risk_tolerance = psychology["riskTolerance"]
            if psychology.get("comfortPriority") is not None:
                vector.psychology.comfort_priority = psychology["comfortPriority"]
        
        # Stated preferences
        stated = profile.get("stated_preferences", {})
        if stated.get("color_probs"):
            vector.stated_preferences.color_probs.update(stated["color_probs"])
        if stated.get("style_probs"):
            vector.stated_preferences.style_probs.update(stated["style_probs"])
        
        # Revealed behavior
        revealed = profile.get("revealed_preferences", {})
        if revealed:
            # Store for preference blending
            pass
        
        # Interaction history
        vector.revealed_behavior.interaction_count = profile.get("interaction_count", 0)
        vector.revealed_behavior.contradiction_count = profile.get("contradiction_count", 0)
        
        if profile.get("color_interactions"):
            vector.revealed_behavior.color_interactions = profile["color_interactions"]
        
        return vector
    
    def _build_current_context(self) -> ContextVector:
        """
        Build context vector from current time and defaults.
        In production, this would integrate with weather API, location, etc.
        """
        now = datetime.utcnow()
        
        # Determine season (Northern Hemisphere approximation)
        month = now.month
        if month in [12, 1, 2]:
            season = Season.WINTER
        elif month in [3, 4, 5]:
            season = Season.SPRING
        elif month in [6, 7, 8]:
            season = Season.SUMMER
        else:
            season = Season.AUTUMN
        
        return ContextVector(
            climate=ClimateVector(
                temperature=25.0,
                temperature_normalized=0.5,
                humidity=0.5,
                rain_probability=0.1
            ),
            geo=GeoVector(
                urbanicity=0.7
            ),
            occasion=OccasionVector(
                occasion_type=OccasionType.CASUAL,
                formality=0.3
            ),
            time=TimeVector(
                hour=now.hour,
                day_of_week=now.weekday(),
                season=season
            )
        )
    
    async def save_physical_profile(
        self,
        user_id: str,
        email: str,
        physical_data: Dict[str, Any]
    ) -> bool:
        """
        Save physical attributes from profile form.
        """
        db = get_database()
        
        query = {"$or": []}
        if email:
            query["$or"].append({"userEmail": email})
        if user_id and user_id != "guest":
            query["$or"].append({"userId": user_id})
        
        if not query["$or"]:
            return False
        
        try:
            await db.user_fashion_profiles.update_one(
                query,
                {
                    "$set": {
                        "physical": physical_data,
                        "userId": user_id,
                        "userEmail": email,
                        "lastUpdated": datetime.utcnow()
                    }
                },
                upsert=True
            )
            logger.info(f"Saved physical profile for {email or user_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving physical profile: {e}")
            return False
    
    async def save_psychology_profile(
        self,
        user_id: str,
        email: str,
        psychology_data: Dict[str, Any]
    ) -> bool:
        """
        Save psychology attributes (confidence, risk tolerance, etc.).
        """
        db = get_database()
        
        query = {"$or": []}
        if email:
            query["$or"].append({"userEmail": email})
        if user_id and user_id != "guest":
            query["$or"].append({"userId": user_id})
        
        if not query["$or"]:
            return False
        
        try:
            await db.user_fashion_profiles.update_one(
                query,
                {
                    "$set": {
                        "psychology": psychology_data,
                        "userId": user_id,
                        "userEmail": email,
                        "lastUpdated": datetime.utcnow()
                    }
                },
                upsert=True
            )
            logger.info(f"Saved psychology profile for {email or user_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving psychology profile: {e}")
            return False
    
    async def update_context(
        self,
        user_id: str,
        email: str,
        context_data: Dict[str, Any]
    ) -> bool:
        """
        Update context (occasion, weather, etc.).
        """
        # Context is typically session-level, stored in session
        db = get_database()
        
        query = {"$or": []}
        if email:
            query["$or"].append({"userEmail": email})
        if user_id and user_id != "guest":
            query["$or"].append({"userId": user_id})
        
        if not query["$or"]:
            return False
        
        try:
            await db.user_sessions.update_one(
                query,
                {
                    "$set": {
                        "context": context_data,
                        "lastUpdated": datetime.utcnow()
                    }
                },
                upsert=True
            )
            return True
        except Exception as e:
            logger.error(f"Error updating context: {e}")
            return False


# Singleton instance
user_vector_service = UserVectorService()
