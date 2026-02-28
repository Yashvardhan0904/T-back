"""
Enhanced Context Management Models

This module defines the three-tier memory architecture:
1. Working Memory: Current conversation context within LLM context window
2. Session Memory: Temporary storage for current session preferences and state
3. Long-term Memory: Persistent user profile and historical preferences
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum


class PreferenceSource(Enum):
    """Source of a preference"""
    EXPLICIT = "explicit"  # User directly stated
    INFERRED = "inferred"  # System inferred from behavior
    BEHAVIORAL = "behavioral"  # Learned from interactions


class FilterType(Enum):
    """Type of search filter"""
    MANDATORY = "mandatory"  # Must be applied (e.g., gender)
    PREFERENCE = "preference"  # Should be applied if possible
    SOFT = "soft"  # Nice to have


@dataclass
class StylePreference:
    """User style preference with confidence scoring"""
    style: str  # "oversized", "fitted", "casual", etc.
    confidence: float  # 0.0 to 1.0
    source: PreferenceSource
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ColorPreferences:
    """User color preferences"""
    liked_colors: List[str] = field(default_factory=list)
    disliked_colors: List[str] = field(default_factory=list)
    neutral_colors: List[str] = field(default_factory=list)


@dataclass
class BrandPreference:
    """User brand preference"""
    brand: str
    preference_type: str  # "like", "dislike", "neutral"
    confidence: float
    source: PreferenceSource


@dataclass
class PriceRange:
    """User price range preference"""
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    currency: str = "INR"


@dataclass
class Interaction:
    """User interaction record"""
    interaction_type: str  # "click", "purchase", "like", "reject"
    item_id: str
    item_features: Dict[str, Any]
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserProfile:
    """Long-term user profile with persistent preferences"""
    user_id: str
    gender_preference: Optional[str] = None  # "male", "female", "unisex"
    size_preferences: Dict[str, str] = field(default_factory=dict)  # category -> size
    style_preferences: List[StylePreference] = field(default_factory=list)
    color_preferences: ColorPreferences = field(default_factory=ColorPreferences)
    brand_preferences: List[BrandPreference] = field(default_factory=list)
    price_range: Optional[PriceRange] = None
    interaction_history: List[Interaction] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Additional profile fields
    user_name: Optional[str] = None
    style_vibe: str = "casual"  # "minimal", "streetwear", "classic", etc.
    design_preference: str = "clean"  # "loud", "clean", "printed"
    slang_tolerance: str = "medium"  # "low", "medium", "high"
    preferred_fit: str = "regular"  # "oversized", "slim", "regular"
    disliked_fits: List[str] = field(default_factory=list)


@dataclass
class Message:
    """Conversation message"""
    role: str  # "user", "assistant"
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    product_ids: List[str] = field(default_factory=list)


@dataclass
class ConversationHistory:
    """Working memory - current conversation context"""
    messages: List[Message] = field(default_factory=list)
    extracted_preferences: List[Dict[str, Any]] = field(default_factory=list)
    context_summary: str = ""
    
    def add_message(self, message: Message):
        """Add message to history"""
        self.messages.append(message)
        # Keep only last 20 messages for working memory
        if len(self.messages) > 20:
            self.messages = self.messages[-20:]
    
    def get_recent_messages(self, count: int = 5) -> List[Message]:
        """Get recent messages"""
        return self.messages[-count:] if self.messages else []


@dataclass
class SessionState:
    """Session memory - temporary preferences and state"""
    session_id: str
    current_intent: Optional[str] = None
    pending_questions: List[str] = field(default_factory=list)
    discovered_preferences: Dict[str, Any] = field(default_factory=dict)
    search_context: Dict[str, Any] = field(default_factory=dict)
    buying_type: Optional[str] = None  # "regular", "occasion"
    occasion: Optional[str] = None  # "party", "wedding", "office"
    sub_occasion: Optional[str] = None  # "clubbing", "house party"
    discovery_state: str = "initial"  # "initial", "gender", "vibe", "size", "complete"
    excluded_product_ids: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class UserContext:
    """Complete user context combining all memory tiers"""
    user_id: str
    working_memory: ConversationHistory
    session_memory: SessionState
    user_profile: UserProfile
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def update_session_preference(self, key: str, value: Any):
        """Update session preference"""
        self.session_memory.discovered_preferences[key] = value
        self.session_memory.updated_at = datetime.utcnow()
        self.last_updated = datetime.utcnow()
    
    def get_gender_preference(self) -> Optional[str]:
        """Get gender preference from session or profile"""
        # Check session first (most recent)
        session_gender = self.session_memory.discovered_preferences.get("gender")
        if session_gender:
            return session_gender
        # Fall back to profile
        return self.user_profile.gender_preference
    
    def get_style_preferences(self) -> List[str]:
        """Get all style preferences"""
        styles = []
        # Add from profile
        for pref in self.user_profile.style_preferences:
            styles.append(pref.style)
        # Add from session
        session_style = self.session_memory.discovered_preferences.get("style")
        if session_style and session_style not in styles:
            styles.append(session_style)
        return styles


@dataclass
class Filter:
    """Search filter"""
    field: str
    value: Any
    filter_type: FilterType
    confidence: float = 1.0


@dataclass
class SearchQuery:
    """Enhanced search query with context"""
    text: str
    mandatory_filters: List[Filter] = field(default_factory=list)
    preference_filters: List[Filter] = field(default_factory=list)
    context: Optional[UserContext] = None
    exclude_ids: List[str] = field(default_factory=list)
    limit: int = 20


@dataclass
class SearchResults:
    """Search results with metadata"""
    products: List[Dict[str, Any]] = field(default_factory=list)
    total_count: int = 0
    applied_filters: List[Filter] = field(default_factory=list)
    explanation: Optional[str] = None  # Why certain results were excluded
    search_authority: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationContext:
    """Context for response generation"""
    user_query: str
    user_context: UserContext
    search_results: Optional[SearchResults] = None
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    intent: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)