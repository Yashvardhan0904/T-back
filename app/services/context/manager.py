"""
Context Manager - Manages three-tier memory architecture and context persistence

This is the central component that coordinates:
1. Working Memory: Current conversation context within LLM context window
2. Session Memory: Temporary storage for current session preferences and state  
3. Long-term Memory: Persistent user profile and historical preferences

Key responsibilities:
- Load and save user context across all memory tiers
- Propagate context updates between agents via event-driven architecture
- Handle preference conflicts and hierarchy (explicit > inferred)
- Maintain conversation continuity and context persistence
"""

from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import logging
import json
from dataclasses import asdict

from app.models.context import (
    UserContext, UserProfile, SessionState, ConversationHistory,
    Message, StylePreference, ColorPreferences, BrandPreference,
    PriceRange, Interaction, PreferenceSource, Filter, FilterType
)
from app.db.mongodb import get_database
from app.services.memory.service import memory_service
from app.services.context.preference_resolver import preference_resolver, PreferenceConflict, ConflictResolutionStrategy
from bson import ObjectId

logger = logging.getLogger(__name__)


class ContextManager:
    """
    Central context management system with three-tier memory architecture.
    Ensures all agents receive consistent, up-to-date context through event-driven updates.
    """
    
    def __init__(self):
        self.context_cache: Dict[str, UserContext] = {}
        self.event_listeners: List[Callable] = []
        self.db = None
    
    async def _get_db(self):
        """Get database connection"""
        if not self.db:
            self.db = get_database()
        return self.db
    
    def register_event_listener(self, callback: Callable):
        """Register callback for context update events"""
        self.event_listeners.append(callback)
    
    async def _notify_context_update(self, user_id: str, context: UserContext):
        """Notify all registered listeners of context updates"""
        for callback in self.event_listeners:
            try:
                await callback(user_id, context)
            except Exception as e:
                logger.error(f"Error in context update callback: {e}")
    
    async def load_user_context(
        self,
        user_id: str,
        email: Optional[str] = None,
        chat_id: Optional[str] = None
    ) -> UserContext:
        """
        Load complete user context from all three memory tiers.
        
        Returns:
            UserContext with working memory, session memory, and user profile
        """
        cache_key = f"{user_id}:{email}:{chat_id}"
        
        # Check cache first (for performance)
        if cache_key in self.context_cache:
            cached_context = self.context_cache[cache_key]
            # Return cached if less than 5 minutes old
            if (datetime.utcnow() - cached_context.last_updated).seconds < 300:
                return cached_context
        
        logger.info(f"[ContextManager] Loading context for user_id={user_id}, email={email}, chat_id={chat_id}")
        
        # Load from existing memory service
        user_data = await memory_service.get_user_memory(user_id, email, chat_id)
        memory = user_data.get("memory", {})
        context_data = user_data.get("context", {})
        
        # 1. Build User Profile (Long-term Memory)
        user_profile = await self._build_user_profile(user_id, email, memory)
        
        # 2. Build Session State (Session Memory)
        session_state = await self._build_session_state(user_id, email, chat_id, context_data)
        
        # 3. Build Conversation History (Working Memory)
        conversation_history = self._build_conversation_history(context_data.get("conversationHistory", []))
        
        # 4. Create complete context
        user_context = UserContext(
            user_id=user_id,
            working_memory=conversation_history,
            session_memory=session_state,
            user_profile=user_profile,
            last_updated=datetime.utcnow()
        )
        
        # Cache the context
        self.context_cache[cache_key] = user_context
        
        logger.info(f"[ContextManager] Loaded context with {len(conversation_history.messages)} messages, "
                   f"discovery_state={session_state.discovery_state}")
        
        return user_context
    
    async def _build_user_profile(self, user_id: str, email: Optional[str], memory: Dict) -> UserProfile:
        """Build user profile from memory data"""
        if not memory:
            return UserProfile(user_id=user_id)
        
        # Extract preferences
        preferences = memory.get("preferences", {})
        style = memory.get("style", {})
        behavior = memory.get("behavior", {})
        physical = memory.get("physical", {})
        
        # Build style preferences
        style_preferences = []
        if style.get("fit"):
            style_preferences.append(StylePreference(
                style=style["fit"],
                confidence=0.8,
                source=PreferenceSource.EXPLICIT
            ))
        
        # Build color preferences
        color_preferences = ColorPreferences(
            liked_colors=style.get("liked_colors", []),
            disliked_colors=style.get("disliked_colors", [])
        )
        
        # Build brand preferences
        brand_preferences = []
        for brand in preferences.get("brands", []):
            brand_preferences.append(BrandPreference(
                brand=brand,
                preference_type="like",
                confidence=0.7,
                source=PreferenceSource.BEHAVIORAL
            ))
        
        # Build price range
        price_range = None
        if preferences.get("price_range"):
            pr = preferences["price_range"]
            price_range = PriceRange(
                min_price=pr.get("min"),
                max_price=pr.get("max")
            )
        
        return UserProfile(
            user_id=user_id,
            gender_preference=physical.get("gender"),
            style_preferences=style_preferences,
            color_preferences=color_preferences,
            brand_preferences=brand_preferences,
            price_range=price_range,
            user_name=memory.get("userName"),
            style_vibe=style.get("vibe", "casual"),
            design_preference=style.get("design_preference", "clean"),
            slang_tolerance=behavior.get("slang_tolerance", "medium"),
            preferred_fit=style.get("fit", "regular"),
            disliked_fits=style.get("disliked_fits", [])
        )
    
    async def _build_session_state(
        self,
        user_id: str,
        email: Optional[str],
        chat_id: Optional[str],
        context_data: Dict
    ) -> SessionState:
        """Build session state from context data"""
        session_id = chat_id or f"session_{user_id}_{int(datetime.utcnow().timestamp())}"
        
        # Load session-specific data from database
        db = await self._get_db()
        session_data = {}
        
        if chat_id:
            try:
                # Try to load existing session state
                session_doc = await db.session_states.find_one({"session_id": chat_id})
                if session_doc:
                    session_data = session_doc
            except Exception as e:
                logger.error(f"Error loading session state: {e}")
        
        # Determine discovery state based on user profile and session
        discovery_state = self._determine_discovery_state(context_data, session_data)
        
        return SessionState(
            session_id=session_id,
            current_intent=context_data.get("current_intent"),
            discovered_preferences=session_data.get("discovered_preferences", {}),
            search_context=session_data.get("search_context", {}),
            buying_type=session_data.get("buying_type"),
            occasion=session_data.get("occasion"),
            sub_occasion=session_data.get("sub_occasion"),
            discovery_state=discovery_state,
            excluded_product_ids=session_data.get("excluded_product_ids", [])
        )
    
    def _determine_discovery_state(self, context_data: Dict, session_data: Dict) -> str:
        """Determine current discovery state based on available information"""
        # Check what we already know
        has_gender = bool(session_data.get("discovered_preferences", {}).get("gender"))
        has_vibe = bool(session_data.get("discovered_preferences", {}).get("vibe"))
        has_size = bool(session_data.get("discovered_preferences", {}).get("size"))
        
        if not has_gender:
            return "gender"
        elif not has_vibe:
            return "vibe"
        elif not has_size:
            return "size"
        else:
            return "complete"
    
    def _build_conversation_history(self, history_data: List[Dict]) -> ConversationHistory:
        """Build conversation history from raw data"""
        messages = []
        for msg_data in history_data:
            message = Message(
                role=msg_data.get("role", "user"),
                content=msg_data.get("content", ""),
                timestamp=msg_data.get("timestamp", datetime.utcnow()),
                metadata=msg_data.get("metadata", {}),
                product_ids=msg_data.get("product_ids", [])
            )
            messages.append(message)
        
        return ConversationHistory(
            messages=messages,
            context_summary=self._generate_context_summary(messages)
        )
    
    def _generate_context_summary(self, messages: List[Message]) -> str:
        """Generate context summary from messages"""
        if not messages:
            return "No previous conversation."
        
        recent = messages[-5:]  # Last 5 messages
        summary_parts = []
        
        for msg in recent:
            role = msg.role
            content = msg.content[:50]  # First 50 chars
            summary_parts.append(f"{role}: {content}...")
        
        return " | ".join(summary_parts)
    
    async def update_session_preference(
        self,
        user_id: str,
        key: str,
        value: Any,
        email: Optional[str] = None,
        chat_id: Optional[str] = None
    ) -> None:
        """
        Update session preference and propagate to all agents.
        
        Args:
            user_id: User identifier
            key: Preference key (e.g., "gender", "vibe", "size")
            value: Preference value
            email: User email (optional)
            chat_id: Chat session ID (optional)
        """
        # Load current context
        context = await self.load_user_context(user_id, email, chat_id)
        
        # Update session preference
        context.update_session_preference(key, value)
        
        # Persist to database
        await self._persist_session_state(context.session_memory)
        
        # Update cache
        cache_key = f"{user_id}:{email}:{chat_id}"
        self.context_cache[cache_key] = context
        
        # Notify all agents of context update
        await self._notify_context_update(user_id, context)
        
        logger.info(f"[ContextManager] Updated session preference {key}={value} for {user_id}")
    
    async def persist_preference(
        self,
        user_id: str,
        preference_type: str,
        preference_data: Dict[str, Any],
        email: Optional[str] = None
    ) -> None:
        """
        Persist preference to long-term memory.
        
        Args:
            user_id: User identifier
            preference_type: Type of preference ("style", "color", "brand", etc.)
            preference_data: Preference data to persist
            email: User email (optional)
        """
        # Use existing memory agent for persistence
        from app.services.agents.memory_agent import memory_agent
        
        await memory_agent.update_memory(
            user_id=user_id,
            email=email,
            memory_type="LONG_TERM",
            data=preference_data
        )
        
        # Invalidate cache to force reload
        cache_keys_to_remove = [key for key in self.context_cache.keys() if key.startswith(f"{user_id}:")]
        for key in cache_keys_to_remove:
            del self.context_cache[key]
        
        logger.info(f"[ContextManager] Persisted {preference_type} preference for {user_id}")
    
    async def add_message_to_working_memory(
        self,
        user_id: str,
        message: Message,
        email: Optional[str] = None,
        chat_id: Optional[str] = None
    ) -> None:
        """Add message to working memory and persist"""
        context = await self.load_user_context(user_id, email, chat_id)
        context.working_memory.add_message(message)
        
        # Persist message
        await memory_service.save_interaction(
            user_id=user_id,
            email=email,
            role=message.role,
            content=message.content,
            products=[{"id": pid} for pid in message.product_ids]
        )
        
        # Update cache
        cache_key = f"{user_id}:{email}:{chat_id}"
        self.context_cache[cache_key] = context
        
        logger.info(f"[ContextManager] Added message to working memory for {user_id}")
    
    async def get_working_memory(
        self,
        user_id: str,
        email: Optional[str] = None,
        chat_id: Optional[str] = None
    ) -> ConversationHistory:
        """Get working memory (conversation history)"""
        context = await self.load_user_context(user_id, email, chat_id)
        return context.working_memory
    
    async def _persist_session_state(self, session_state: SessionState) -> None:
        """Persist session state to database"""
        try:
            db = await self._get_db()
            await db.session_states.update_one(
                {"session_id": session_state.session_id},
                {"$set": {
                    "discovered_preferences": session_state.discovered_preferences,
                    "search_context": session_state.search_context,
                    "buying_type": session_state.buying_type,
                    "occasion": session_state.occasion,
                    "sub_occasion": session_state.sub_occasion,
                    "discovery_state": session_state.discovery_state,
                    "excluded_product_ids": session_state.excluded_product_ids,
                    "updated_at": datetime.utcnow()
                }},
                upsert=True
            )
        except Exception as e:
            logger.error(f"Error persisting session state: {e}")
    
    async def handle_preference_conflict(
        self,
        user_id: str,
        conflicting_preferences: List[Dict[str, Any]],
        email: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Handle conflicting preferences by prioritizing explicit over inferred,
        and most recent over older statements.
        
        Returns:
            Resolved preference with conflict resolution metadata
        """
        if not conflicting_preferences:
            return {}
        
        # Use preference resolver to handle conflicts
        conflicts = preference_resolver.resolve_preference_conflicts(
            conflicting_preferences,
            strategy=ConflictResolutionStrategy.MOST_RECENT
        )
        
        # Get the first (and likely only) conflict
        if conflicts:
            conflict_key = list(conflicts.keys())[0]
            conflict = conflicts[conflict_key]
            
            # Check if user confirmation is needed
            if preference_resolver.should_ask_user_for_confirmation(conflict):
                conflict.requires_user_confirmation = True
                logger.info(f"[ContextManager] Conflict requires user confirmation for {user_id}: {conflict_key}")
            
            resolved_preference = {
                "key": conflict.preference_key,
                "value": conflict.resolved_value,
                "confidence": conflict.confidence,
                "resolution_strategy": conflict.resolution_strategy.value,
                "requires_confirmation": conflict.requires_user_confirmation,
                "conflict_message": preference_resolver.generate_conflict_resolution_message(conflict)
            }
            
            logger.info(f"[ContextManager] Resolved preference conflict for {user_id}: {resolved_preference}")
            return resolved_preference
        
        # Fallback to simple resolution
        sorted_prefs = sorted(
            conflicting_preferences,
            key=lambda p: (
                preference_resolver._get_source_priority(p.get("source", "inferred")),
                p.get("timestamp", datetime.min)
            ),
            reverse=True
        )
        
        resolved = sorted_prefs[0]
        logger.info(f"[ContextManager] Fallback preference resolution for {user_id}: {resolved}")
        return resolved
    
    def clear_cache(self, user_id: Optional[str] = None):
        """Clear context cache for user or all users"""
        if user_id:
            cache_keys_to_remove = [key for key in self.context_cache.keys() if key.startswith(f"{user_id}:")]
            for key in cache_keys_to_remove:
                del self.context_cache[key]
        else:
            self.context_cache.clear()
        
        logger.info(f"[ContextManager] Cleared cache for {user_id or 'all users'}")
    
    async def get_merged_preferences(
        self,
        user_id: str,
        email: Optional[str] = None,
        chat_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get merged preferences from all sources with proper hierarchy.
        
        Priority: conversation > session > profile
        
        Returns:
            Merged preferences dictionary
        """
        context = await self.load_user_context(user_id, email, chat_id)
        
        # Extract preferences from different sources
        profile_prefs = {}
        if context.user_profile.gender_preference:
            profile_prefs["gender"] = context.user_profile.gender_preference
        if context.user_profile.style_vibe:
            profile_prefs["vibe"] = context.user_profile.style_vibe
        if context.user_profile.preferred_fit:
            profile_prefs["fit"] = context.user_profile.preferred_fit
        
        session_prefs = context.session_memory.discovered_preferences
        
        # Extract conversation preferences from recent messages
        conversation_prefs = []
        for message in context.working_memory.get_recent_messages(5):
            # This would be enhanced to extract preferences from message content
            # For now, we'll use metadata if available
            if "preferences" in message.metadata:
                for pref_key, pref_value in message.metadata["preferences"].items():
                    conversation_prefs.append({
                        "type": pref_key,
                        "value": pref_value,
                        "source": "explicit",
                        "timestamp": message.timestamp
                    })
        
        # Use preference resolver to merge
        merged = preference_resolver.merge_preferences(
            session_preferences=session_prefs,
            profile_preferences=profile_prefs,
            conversation_preferences=conversation_prefs
        )
        
        logger.info(f"[ContextManager] Merged preferences for {user_id}: {list(merged.keys())}")
        return merged


# Singleton instance
context_manager = ContextManager()