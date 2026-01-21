# User Vector Schema for Advanced Fashion AI
# This module defines the complete mathematical representation of a user

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
import numpy as np
from datetime import datetime

# ============================================================
# ENUMS & CONSTANTS
# ============================================================

class Undertone(str, Enum):
    WARM = "warm"
    COOL = "cool"
    NEUTRAL = "neutral"

class BodyShape(str, Enum):
    ECTOMORPH = "ectomorph"  # Lean, long limbs
    MESOMORPH = "mesomorph"  # Athletic, muscular
    ENDOMORPH = "endomorph"  # Soft, wider
    MIXED = "mixed"

class OccasionType(str, Enum):
    CASUAL = "casual"
    WORK = "work"
    DATE = "date"
    WEDDING = "wedding"
    STREET = "street"
    FORMAL = "formal"
    PARTY = "party"
    FESTIVAL = "festival"

class Season(str, Enum):
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"
    WINTER = "winter"

class Mood(str, Enum):
    EXPLORATORY = "exploratory"
    TIRED = "tired"
    BOLD = "bold"
    CONFUSED = "confused"
    FOCUSED = "focused"

# ============================================================
# PHYSICAL VECTOR (Immutable / Slow-changing)
# ============================================================

class PhysicalVector(BaseModel):
    """Physical attributes that affect how clothes look on the body."""
    
    # Skin & Face
    skin_tone: float = Field(0.5, ge=0, le=1, description="Fitzpatrick scale normalized 0-1")
    undertone: Undertone = Field(Undertone.NEUTRAL, description="Warm/Cool/Neutral")
    skin_contrast: float = Field(0.5, ge=0, le=1, description="Hair-skin contrast level")
    face_shape_embedding: List[float] = Field(default_factory=lambda: [0.0]*4, description="4-dim face shape embedding")
    
    # Body Geometry
    height_cm: float = Field(170.0, ge=100, le=250, description="Height in cm")
    height_normalized: float = Field(0.5, ge=0, le=1, description="Height normalized 0-1")
    weight_index: float = Field(0.5, ge=0, le=1, description="BMI-like index normalized")
    body_shape_embedding: List[float] = Field(
        default_factory=lambda: [0.33, 0.33, 0.34],
        description="Ecto/Meso/Endo mix (sums to 1)"
    )
    shoulder_waist_ratio: float = Field(1.0, ge=0.5, le=2.0, description="Shoulder to waist ratio")
    hip_ratio: float = Field(0.8, ge=0.3, le=1.5, description="Hip ratio")
    leg_torso_ratio: float = Field(1.0, ge=0.5, le=1.5, description="Leg to torso ratio")
    posture_score: float = Field(0.7, ge=0, le=1, description="Posture quality")
    
    def to_vector(self) -> np.ndarray:
        """Convert to numpy array for math operations (~25 dimensions)"""
        return np.array([
            self.skin_tone,
            *self._one_hot_undertone(),
            self.skin_contrast,
            *self.face_shape_embedding,
            self.height_normalized,
            self.weight_index,
            *self.body_shape_embedding,
            self.shoulder_waist_ratio,
            self.hip_ratio,
            self.leg_torso_ratio,
            self.posture_score
        ])
    
    def _one_hot_undertone(self) -> List[float]:
        return [
            1.0 if self.undertone == Undertone.WARM else 0.0,
            1.0 if self.undertone == Undertone.COOL else 0.0,
            1.0 if self.undertone == Undertone.NEUTRAL else 0.0
        ]

# ============================================================
# GENDER EXPRESSION VECTOR
# ============================================================

class GenderExpressionVector(BaseModel):
    """Gender expression is continuous, not binary."""
    
    masc_femme: float = Field(0.5, ge=0, le=1, description="0=masculine, 1=feminine")
    androgyny_preference: float = Field(0.5, ge=0, le=1, description="Preference for androgynous styles")
    silhouette_preference: float = Field(0.5, ge=0, le=1, description="Structured(0) to Flowing(1)")
    
    def to_vector(self) -> np.ndarray:
        return np.array([self.masc_femme, self.androgyny_preference, self.silhouette_preference])

# ============================================================
# PSYCHOLOGICAL VECTOR
# ============================================================

class PsychologyVector(BaseModel):
    """Psychological factors that affect what user will actually wear."""
    
    # Confidence & Risk
    confidence: float = Field(0.5, ge=0, le=1, description="Self-assurance level")
    risk_tolerance: float = Field(0.5, ge=0, le=1, description="Fashion risk tolerance")
    comfort_priority: float = Field(0.5, ge=0, le=1, description="Comfort vs style priority")
    self_expression: float = Field(0.5, ge=0, le=1, description="Self-expression index")
    attention_seeking: float = Field(0.5, ge=0, le=1, description="Desire to stand out")
    
    # Personality (OCEAN Model)
    openness: float = Field(0.5, ge=0, le=1)
    conscientiousness: float = Field(0.5, ge=0, le=1)
    extraversion: float = Field(0.5, ge=0, le=1)
    agreeableness: float = Field(0.5, ge=0, le=1)
    neuroticism: float = Field(0.5, ge=0, le=1)
    
    # Style Preferences
    minimalism_score: float = Field(0.5, ge=0, le=1, description="Minimal vs Maximal")
    trend_following: float = Field(0.5, ge=0, le=1, description="Trendy vs Classic")
    uniqueness_drive: float = Field(0.5, ge=0, le=1, description="Desire to be unique")
    
    def to_vector(self) -> np.ndarray:
        return np.array([
            self.confidence, self.risk_tolerance, self.comfort_priority,
            self.self_expression, self.attention_seeking,
            self.openness, self.conscientiousness, self.extraversion,
            self.agreeableness, self.neuroticism,
            self.minimalism_score, self.trend_following, self.uniqueness_drive
        ])

# ============================================================
# CONTEXT VECTOR (Dynamic / Real-time)
# ============================================================

class ClimateVector(BaseModel):
    """Weather and environmental context."""
    
    temperature: float = Field(25.0, ge=-30, le=50, description="Temperature in Celsius")
    temperature_normalized: float = Field(0.5, ge=0, le=1)
    humidity: float = Field(0.5, ge=0, le=1)
    rain_probability: float = Field(0.0, ge=0, le=1)
    uv_index: float = Field(0.3, ge=0, le=1, description="UV normalized")
    wind_factor: float = Field(0.2, ge=0, le=1)
    
    def to_vector(self) -> np.ndarray:
        return np.array([
            self.temperature_normalized, self.humidity,
            self.rain_probability, self.uv_index, self.wind_factor
        ])

class GeoVector(BaseModel):
    """Geographic context."""
    
    region_embedding: List[float] = Field(default_factory=lambda: [0.0]*8, description="8-dim region embedding")
    urbanicity: float = Field(0.7, ge=0, le=1, description="0=rural, 1=urban")
    cultural_index: float = Field(0.5, ge=0, le=1, description="Traditional vs Modern")
    
    def to_vector(self) -> np.ndarray:
        return np.array([*self.region_embedding, self.urbanicity, self.cultural_index])

class OccasionVector(BaseModel):
    """Event and social context."""
    
    occasion_type: OccasionType = Field(OccasionType.CASUAL)
    formality: float = Field(0.3, ge=0, le=1)
    duration_hours: float = Field(4.0, ge=0, le=24)
    social_power_distance: float = Field(0.5, ge=0, le=1, description="Hierarchy level of event")
    cultural_norm_strictness: float = Field(0.5, ge=0, le=1)
    
    def to_vector(self) -> np.ndarray:
        occasion_one_hot = [0.0] * 8
        occasion_map = {
            OccasionType.CASUAL: 0, OccasionType.WORK: 1, OccasionType.DATE: 2,
            OccasionType.WEDDING: 3, OccasionType.STREET: 4, OccasionType.FORMAL: 5,
            OccasionType.PARTY: 6, OccasionType.FESTIVAL: 7
        }
        occasion_one_hot[occasion_map.get(self.occasion_type, 0)] = 1.0
        return np.array([
            *occasion_one_hot, self.formality, self.duration_hours / 24,
            self.social_power_distance, self.cultural_norm_strictness
        ])

class TimeVector(BaseModel):
    """Temporal context with cyclical encoding."""
    
    hour: int = Field(12, ge=0, le=23)
    day_of_week: int = Field(0, ge=0, le=6)
    season: Season = Field(Season.SUMMER)
    
    def to_vector(self) -> np.ndarray:
        import math
        # Cyclical encoding for time
        hour_sin = math.sin(2 * math.pi * self.hour / 24)
        hour_cos = math.cos(2 * math.pi * self.hour / 24)
        day_sin = math.sin(2 * math.pi * self.day_of_week / 7)
        day_cos = math.cos(2 * math.pi * self.day_of_week / 7)
        
        season_one_hot = [0.0] * 4
        season_map = {Season.SPRING: 0, Season.SUMMER: 1, Season.AUTUMN: 2, Season.WINTER: 3}
        season_one_hot[season_map.get(self.season, 1)] = 1.0
        
        return np.array([hour_sin, hour_cos, day_sin, day_cos, *season_one_hot])

class ContextVector(BaseModel):
    """Complete contextual state."""
    
    climate: ClimateVector = Field(default_factory=ClimateVector)
    geo: GeoVector = Field(default_factory=GeoVector)
    occasion: OccasionVector = Field(default_factory=OccasionVector)
    time: TimeVector = Field(default_factory=TimeVector)
    
    def to_vector(self) -> np.ndarray:
        return np.concatenate([
            self.climate.to_vector(),
            self.geo.to_vector(),
            self.occasion.to_vector(),
            self.time.to_vector()
        ])

# ============================================================
# PREFERENCE DISTRIBUTION (NOT STATIC VALUES)
# ============================================================

class PreferenceDistribution(BaseModel):
    """
    Preferences as probability distributions, not boolean values.
    This allows smooth handling of mind changes.
    """
    
    # Color preferences as probabilities
    color_probs: Dict[str, float] = Field(
        default_factory=lambda: {
            "black": 0.7, "white": 0.6, "red": 0.5, "blue": 0.5,
            "green": 0.4, "yellow": 0.3, "pink": 0.4, "maroon": 0.5,
            "beige": 0.5, "grey": 0.6, "navy": 0.5, "brown": 0.4
        },
        description="P(likes | color)"
    )
    
    # Style preferences as probabilities
    style_probs: Dict[str, float] = Field(
        default_factory=lambda: {
            "streetwear": 0.5, "formal": 0.4, "casual": 0.6, "minimalist": 0.5,
            "maximalist": 0.3, "ethnic": 0.4, "western": 0.5, "fusion": 0.4
        },
        description="P(likes | style)"
    )
    
    # Fit preferences
    fit_probs: Dict[str, float] = Field(
        default_factory=lambda: {
            "slim": 0.5, "regular": 0.6, "oversized": 0.4, "tailored": 0.5
        }
    )
    
    # Decay tracking
    last_decay_at: datetime = Field(default_factory=datetime.utcnow)
    
    def get_preference_vector(self) -> np.ndarray:
        """Convert to fixed-size vector for optimization."""
        # Fixed order for consistency
        color_order = ["black", "white", "red", "blue", "green", "yellow", "pink", "maroon", "beige", "grey", "navy", "brown"]
        style_order = ["streetwear", "formal", "casual", "minimalist", "maximalist", "ethnic", "western", "fusion"]
        fit_order = ["slim", "regular", "oversized", "tailored"]
        
        color_vec = [self.color_probs.get(c, 0.5) for c in color_order]
        style_vec = [self.style_probs.get(s, 0.5) for s in style_order]
        fit_vec = [self.fit_probs.get(f, 0.5) for f in fit_order]
        
        return np.array(color_vec + style_vec + fit_vec)

# ============================================================
# REVEALED BEHAVIOR (What user actually does)
# ============================================================

class RevealedBehavior(BaseModel):
    """Tracks actual user behavior, not stated preferences."""
    
    # Outfit embeddings
    liked_outfit_embeddings: List[List[float]] = Field(
        default_factory=list,
        description="128-dim embeddings of liked outfits"
    )
    rejected_outfit_embeddings: List[List[float]] = Field(
        default_factory=list,
        description="128-dim embeddings of rejected outfits"
    )
    
    # Color interaction history
    color_interactions: Dict[str, Dict[str, int]] = Field(
        default_factory=lambda: {},
        description="{'red': {'liked': 5, 'rejected': 2, 'skipped': 10}}"
    )
    
    # Fit issues
    fit_issues: Dict[str, bool] = Field(
        default_factory=lambda: {},
        description="{'tight_shoulders': True, 'long_sleeves': False}"
    )
    
    # Counts
    interaction_count: int = Field(0)
    contradiction_count: int = Field(0, description="Times stated != revealed")
    
    def get_stated_revealed_distance(self, stated: PreferenceDistribution) -> float:
        """Calculate distance between stated and revealed preferences."""
        if self.interaction_count < 5:
            return 0.0  # Not enough data
        
        # Build revealed probs from interactions
        revealed_color_probs = {}
        for color, stats in self.color_interactions.items():
            total = stats.get("liked", 0) + stats.get("rejected", 0) + stats.get("skipped", 0)
            if total > 0:
                revealed_color_probs[color] = stats.get("liked", 0) / total
        
        # Calculate L2 distance for overlapping colors
        distance = 0.0
        count = 0
        for color in revealed_color_probs:
            if color in stated.color_probs:
                diff = stated.color_probs[color] - revealed_color_probs[color]
                distance += diff ** 2
                count += 1
        
        return (distance / max(count, 1)) ** 0.5

# ============================================================
# SESSION STATE (Short-term, volatile)
# ============================================================

class SessionState(BaseModel):
    """
    Short-term session-level state that expires after session.
    Used to detect current mood and override long-term preferences.
    """
    
    session_id: str = Field("")
    mood: Mood = Field(Mood.FOCUSED)
    decision_speed: float = Field(0.5, ge=0, le=1, description="Fast(1) vs Slow(0)")
    time_pressure: float = Field(0.3, ge=0, le=1)
    exploration_mode: bool = Field(False, description="User in exploration mode")
    
    # Session-level preference overrides
    session_color_boost: Dict[str, float] = Field(default_factory=dict)
    session_style_boost: Dict[str, float] = Field(default_factory=dict)
    
    # Behavioral signals
    avg_response_time_ms: float = Field(3000.0)
    items_viewed: int = Field(0)
    items_liked: int = Field(0)
    items_rejected: int = Field(0)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    def get_gamma(self) -> float:
        """
        Blending weight for session state.
        Higher gamma = more weight to session preferences.
        """
        # High gamma for exploration mode or time pressure
        if self.exploration_mode:
            return 0.7
        if self.time_pressure > 0.7:
            return 0.6
        if self.mood == Mood.EXPLORATORY:
            return 0.6
        return 0.3

# ============================================================
# ECONOMIC & PRACTICAL CONSTRAINTS
# ============================================================

class EconomicVector(BaseModel):
    """Budget and practical constraints."""
    
    budget_min: float = Field(0.0, ge=0)
    budget_max: float = Field(5000.0, ge=0)
    budget_avg: float = Field(1500.0, ge=0)
    
    brand_affinity_embedding: List[float] = Field(
        default_factory=lambda: [0.5]*8,
        description="8-dim brand preference embedding"
    )
    
    sustainability_preference: float = Field(0.5, ge=0, le=1)
    quality_over_quantity: float = Field(0.5, ge=0, le=1)
    
    def to_vector(self) -> np.ndarray:
        return np.array([
            self.budget_avg / 10000,  # Normalize to 0-1 range
            (self.budget_max - self.budget_min) / 10000,  # Budget flexibility
            *self.brand_affinity_embedding,
            self.sustainability_preference,
            self.quality_over_quantity
        ])

# ============================================================
# FULL USER VECTOR (THE COMPLETE REPRESENTATION)
# ============================================================

class FullUserVector(BaseModel):
    """
    Complete mathematical representation of a user.
    This is approximately 350+ dimensions when converted to numpy.
    """
    
    user_id: str
    user_email: Optional[str] = None
    
    # Core vectors
    physical: PhysicalVector = Field(default_factory=PhysicalVector)
    gender_expression: GenderExpressionVector = Field(default_factory=GenderExpressionVector)
    psychology: PsychologyVector = Field(default_factory=PsychologyVector)
    context: ContextVector = Field(default_factory=ContextVector)
    
    # Preference system
    stated_preferences: PreferenceDistribution = Field(default_factory=PreferenceDistribution)
    revealed_behavior: RevealedBehavior = Field(default_factory=RevealedBehavior)
    
    # Session state (volatile)
    session: SessionState = Field(default_factory=SessionState)
    
    # Economic constraints
    economic: EconomicVector = Field(default_factory=EconomicVector)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    def to_full_vector(self) -> np.ndarray:
        """
        Convert entire user state to a single numpy vector.
        Useful for similarity calculations and model input.
        """
        return np.concatenate([
            self.physical.to_vector(),
            self.gender_expression.to_vector(),
            self.psychology.to_vector(),
            self.context.to_vector(),
            self.stated_preferences.get_preference_vector(),
            self.economic.to_vector()
        ])
    
    def get_effective_preferences(self) -> np.ndarray:
        """
        U_eff = α(t) * U_stated + (1-α(t)) * U_revealed
        Where α = e^(-β * contradiction_count)
        """
        import math
        
        # Calculate alpha based on contradiction count
        beta = 0.2  # Sensitivity
        contradiction_count = self.revealed_behavior.contradiction_count
        alpha = math.exp(-beta * contradiction_count)
        
        stated_vec = self.stated_preferences.get_preference_vector()
        
        # If not enough revealed data, use stated only
        if self.revealed_behavior.interaction_count < 5:
            return stated_vec
        
        # Build revealed vector from interactions
        # For simplicity, blend at preference level
        return stated_vec * alpha + stated_vec * (1 - alpha)  # Placeholder for revealed
    
    def get_final_preferences(self) -> np.ndarray:
        """
        U_final = γ * U_session + (1-γ) * U_eff
        Apply session-level overrides.
        """
        effective = self.get_effective_preferences()
        gamma = self.session.get_gamma()
        
        # If no strong session signal, use effective
        if gamma < 0.2:
            return effective
        
        # Apply session boosts (simplified)
        return effective  # Full implementation would blend session preferences
    
    def get_dimension_count(self) -> int:
        """Get total dimensions of the user vector."""
        return len(self.to_full_vector())


# ============================================================
# OUTFIT REPRESENTATION
# ============================================================

class OutfitItem(BaseModel):
    """Single item in an outfit."""
    
    item_id: str
    category: str  # top, bottom, layer, shoes, accessories
    name: str
    
    # Visual properties
    color_vector: List[float] = Field(default_factory=lambda: [0.0]*3, description="RGB normalized")
    color_name: str = Field("")
    pattern: str = Field("solid")
    
    # Style properties
    style_embedding: List[float] = Field(default_factory=lambda: [0.0]*64, description="64-dim style embedding")
    fit: str = Field("regular")
    formality: float = Field(0.5, ge=0, le=1)
    
    # Physical properties
    fabric: str = Field("")
    weight: str = Field("medium")  # light, medium, heavy
    
    # Metadata
    price: float = Field(0.0)
    brand: str = Field("")
    season: List[str] = Field(default_factory=list)
    occasion: List[str] = Field(default_factory=list)
    
    # Image
    image_url: str = Field("")

class Outfit(BaseModel):
    """Complete outfit = set of items."""
    
    outfit_id: str
    items: List[OutfitItem]
    
    # Computed properties (cached)
    dominant_color: str = Field("")
    style_embedding: List[float] = Field(default_factory=lambda: [0.0]*128, description="Combined style embedding")
    total_price: float = Field(0.0)
    formality_score: float = Field(0.5)
    
    # Scores (filled by scoring engine)
    scores: Dict[str, float] = Field(default_factory=dict)
    total_score: float = Field(0.0)
    
    def get_by_category(self, category: str) -> Optional[OutfitItem]:
        """Get item by category."""
        for item in self.items:
            if item.category == category:
                return item
        return None
    
    def to_vector(self) -> np.ndarray:
        """Convert outfit to feature vector for scoring."""
        # Average style embeddings of all items
        if not self.items:
            return np.zeros(64)
        
        embeddings = [np.array(item.style_embedding) for item in self.items]
        return np.mean(embeddings, axis=0)
