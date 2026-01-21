# Session State Engine - Short-term mood modeling
# Handles volatile preferences that expire after session

from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from app.models.user_vector import SessionState, Mood
from app.db.mongodb import get_database
import logging

logger = logging.getLogger(__name__)


class SessionStateEngine:
    """
    Models short-term session state that overrides long-term preferences.
    
    Session state includes:
    - Mood (exploratory, tired, bold, confused)
    - Decision speed
    - Time pressure
    - Session-level preference boosts
    
    U_final = γ * U_session + (1-γ) * U_eff
    
    Where γ is determined by session intensity.
    """
    
    # Session expiry
    SESSION_TIMEOUT_MINUTES = 30
    
    # Mood detection thresholds
    FAST_DECISION_THRESHOLD_MS = 2000
    SLOW_DECISION_THRESHOLD_MS = 8000
    
    # Time of day mood modifiers
    LATE_NIGHT_HOURS = {23, 0, 1, 2, 3}
    MORNING_HOURS = {6, 7, 8, 9}
    
    async def get_or_create_session(self, user_id: str, email: str = None) -> SessionState:
        """
        Get existing session or create new one.
        """
        db = get_database()
        
        query = {"$or": []}
        if email:
            query["$or"].append({"userEmail": email})
        if user_id and user_id != "guest":
            query["$or"].append({"userId": user_id})
        
        if not query["$or"]:
            return SessionState()
        
        try:
            session_data = await db.user_sessions.find_one(query)
            
            if session_data:
                created_at = session_data.get("created_at", datetime.utcnow())
                
                # Check if session expired
                if datetime.utcnow() - created_at > timedelta(minutes=self.SESSION_TIMEOUT_MINUTES):
                    # Session expired, create new one
                    return await self._create_session(user_id, email)
                
                return SessionState(
                    session_id=str(session_data.get("_id", "")),
                    mood=Mood(session_data.get("mood", "focused")),
                    decision_speed=session_data.get("decision_speed", 0.5),
                    time_pressure=session_data.get("time_pressure", 0.3),
                    exploration_mode=session_data.get("exploration_mode", False),
                    session_color_boost=session_data.get("session_color_boost", {}),
                    session_style_boost=session_data.get("session_style_boost", {}),
                    avg_response_time_ms=session_data.get("avg_response_time_ms", 3000),
                    items_viewed=session_data.get("items_viewed", 0),
                    items_liked=session_data.get("items_liked", 0),
                    items_rejected=session_data.get("items_rejected", 0),
                    created_at=created_at
                )
            
            return await self._create_session(user_id, email)
            
        except Exception as e:
            logger.error(f"Error getting session: {e}")
            return SessionState()
    
    async def _create_session(self, user_id: str, email: str = None) -> SessionState:
        """Create a new session."""
        db = get_database()
        
        now = datetime.utcnow()
        current_hour = now.hour
        
        # Infer initial mood from time
        if current_hour in self.LATE_NIGHT_HOURS:
            initial_mood = Mood.EXPLORATORY  # Late night = more experimental
        elif current_hour in self.MORNING_HOURS:
            initial_mood = Mood.FOCUSED  # Morning = task-oriented
        else:
            initial_mood = Mood.FOCUSED
        
        session = SessionState(
            mood=initial_mood,
            created_at=now
        )
        
        query = {"$or": []}
        if email:
            query["$or"].append({"userEmail": email})
        if user_id and user_id != "guest":
            query["$or"].append({"userId": user_id})
        
        if not query["$or"]:
            return session

        try:
            await db.user_sessions.update_one(
                query,
                {
                    "$set": {
                        "userId": user_id,
                        "userEmail": email,
                        "mood": session.mood.value,
                        "decision_speed": session.decision_speed,
                        "time_pressure": session.time_pressure,
                        "exploration_mode": session.exploration_mode,
                        "session_color_boost": {},
                        "session_style_boost": {},
                        "avg_response_time_ms": 3000,
                        "items_viewed": 0,
                        "items_liked": 0,
                        "items_rejected": 0,
                        "created_at": now
                    }
                },
                upsert=True
            )
        except Exception as e:
            logger.error(f"Error creating session: {e}")
        
        return session
    
    async def update_from_behavior(
        self,
        user_id: str,
        email: str,
        signals: Dict[str, Any]
    ) -> SessionState:
        """
        Update session state based on behavioral signals.
        
        Signals:
        - response_time_ms: How long user took to respond
        - action: "viewed" | "liked" | "rejected" | "skipped"
        - scroll_speed: How fast user scrolled
        """
        db = get_database()
        
        query = {"$or": []}
        if email:
            query["$or"].append({"userEmail": email})
        if user_id and user_id != "guest":
            query["$or"].append({"userId": user_id})
        
        if not query["$or"]:
            return SessionState()
        
        try:
            session_data = await db.user_sessions.find_one(query)
            
            if not session_data:
                return await self._create_session(user_id, email)
            
            # Update response time (exponential moving average)
            old_avg = session_data.get("avg_response_time_ms", 3000)
            new_time = signals.get("response_time_ms", old_avg)
            new_avg = old_avg * 0.7 + new_time * 0.3
            
            # Infer decision speed
            if new_avg < self.FAST_DECISION_THRESHOLD_MS:
                decision_speed = 0.8
            elif new_avg > self.SLOW_DECISION_THRESHOLD_MS:
                decision_speed = 0.2
            else:
                decision_speed = 0.5
            
            # Update counts
            items_viewed = session_data.get("items_viewed", 0)
            items_liked = session_data.get("items_liked", 0)
            items_rejected = session_data.get("items_rejected", 0)
            
            action = signals.get("action")
            if action == "viewed":
                items_viewed += 1
            elif action == "liked":
                items_liked += 1
                items_viewed += 1
            elif action == "rejected":
                items_rejected += 1
                items_viewed += 1
            
            # Infer mood from behavior
            mood = Mood(session_data.get("mood", "focused"))
            
            # High rejection rate = confused or tired
            if items_viewed > 5:
                reject_rate = items_rejected / items_viewed
                if reject_rate > 0.7:
                    mood = Mood.CONFUSED
                elif reject_rate > 0.5:
                    mood = Mood.TIRED
                elif items_liked / max(items_viewed, 1) > 0.5:
                    mood = Mood.BOLD
            
            # Fast decisions + many views = exploratory
            if decision_speed > 0.7 and items_viewed > 10:
                mood = Mood.EXPLORATORY
            
            # Update exploration mode
            exploration_mode = mood in [Mood.EXPLORATORY, Mood.BOLD]
            
            # Session-level boosts from recent likes
            session_color_boost = session_data.get("session_color_boost", {})
            session_style_boost = session_data.get("session_style_boost", {})
            
            if action == "liked" and signals.get("features"):
                features = signals["features"]
                for color in features.get("colors", []):
                    session_color_boost[color.lower()] = session_color_boost.get(color.lower(), 0) + 0.1
                if features.get("style"):
                    session_style_boost[features["style"].lower()] = session_style_boost.get(features["style"].lower(), 0) + 0.1
            
            # Persist updates
            await db.user_sessions.update_one(
                query,
                {
                    "$set": {
                        "mood": mood.value,
                        "decision_speed": decision_speed,
                        "avg_response_time_ms": new_avg,
                        "items_viewed": items_viewed,
                        "items_liked": items_liked,
                        "items_rejected": items_rejected,
                        "exploration_mode": exploration_mode,
                        "session_color_boost": session_color_boost,
                        "session_style_boost": session_style_boost
                    }
                }
            )
            
            return SessionState(
                mood=mood,
                decision_speed=decision_speed,
                exploration_mode=exploration_mode,
                session_color_boost=session_color_boost,
                session_style_boost=session_style_boost,
                avg_response_time_ms=new_avg,
                items_viewed=items_viewed,
                items_liked=items_liked,
                items_rejected=items_rejected
            )
            
        except Exception as e:
            logger.error(f"Error updating session: {e}")
            return SessionState()
    
    def get_gamma(self, session: SessionState) -> float:
        """
        Get blending weight for session state.
        Higher gamma = more weight to session preferences.
        """
        return session.get_gamma()
    
    async def apply_session_override(
        self,
        base_prefs: Dict[str, float],
        session: SessionState,
        pref_type: str  # "color" | "style"
    ) -> Dict[str, float]:
        """
        Apply session-level preference overrides.
        
        prefs_final = (1-γ) * prefs_base + γ * prefs_session
        """
        gamma = self.get_gamma(session)
        
        if gamma < 0.1:
            return base_prefs
        
        session_boost = (
            session.session_color_boost if pref_type == "color"
            else session.session_style_boost
        )
        
        result = {}
        for key, value in base_prefs.items():
            boost = session_boost.get(key, 0)
            result[key] = (1 - gamma) * value + gamma * min(1.0, value + boost)
        
        return result
    
    async def trigger_exploration_mode(self, user_id: str, email: str = None) -> None:
        """
        Manually trigger exploration mode (e.g., user says "surprise me").
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
            await db.user_sessions.update_one(
                query,
                {
                    "$set": {
                        "exploration_mode": True,
                        "mood": Mood.EXPLORATORY.value
                    }
                }
            )
            logger.info(f"Exploration mode triggered for {email or user_id}")
        except Exception as e:
            logger.error(f"Error triggering exploration: {e}")
    
    async def reset_session(self, user_id: str, email: str = None) -> SessionState:
        """
        Reset session (for "forget what I said earlier" requests).
        """
        return await self._create_session(user_id, email)


# Singleton instance
session_engine = SessionStateEngine()
