# Fashion Knowledge Graph
# Symbolic AI rules for color theory, body proportion, and style balance

from typing import Dict, List, Tuple, Any, Optional
from enum import Enum
import colorsys
import math

# ============================================================
# COLOR THEORY ENGINE
# ============================================================

class ColorHarmony(Enum):
    COMPLEMENTARY = "complementary"  # Opposite on wheel
    ANALOGOUS = "analogous"  # Adjacent colors
    TRIADIC = "triadic"  # Three equidistant
    SPLIT_COMPLEMENTARY = "split_complementary"
    MONOCHROMATIC = "monochromatic"
    NEUTRAL = "neutral"

# Standard fashion color palette with HSL values
COLOR_PALETTE = {
    "black": (0, 0, 0.05),
    "white": (0, 0, 0.95),
    "grey": (0, 0, 0.5),
    "red": (0, 0.8, 0.5),
    "maroon": (0, 0.65, 0.35),
    "orange": (30, 0.85, 0.55),
    "yellow": (60, 0.9, 0.55),
    "olive": (60, 0.4, 0.35),
    "green": (120, 0.6, 0.4),
    "teal": (180, 0.5, 0.4),
    "blue": (210, 0.7, 0.5),
    "navy": (230, 0.65, 0.25),
    "purple": (270, 0.6, 0.45),
    "pink": (330, 0.6, 0.7),
    "beige": (35, 0.3, 0.75),
    "brown": (30, 0.55, 0.35),
    "cream": (40, 0.25, 0.9),
    "tan": (35, 0.35, 0.6),
    "gold": (45, 0.7, 0.5),
    "silver": (0, 0, 0.75),
}

# Neutral colors that go with everything
NEUTRAL_COLORS = {"black", "white", "grey", "beige", "cream", "tan", "navy", "brown"}


class ColorTheoryEngine:
    """Mathematical color theory for fashion."""
    
    @staticmethod
    def get_hsl(color_name: str) -> Tuple[float, float, float]:
        """Get HSL values for a color."""
        return COLOR_PALETTE.get(color_name.lower(), (0, 0, 0.5))
    
    @staticmethod
    def hue_distance(h1: float, h2: float) -> float:
        """Calculate angular distance between two hues (0-180)."""
        diff = abs(h1 - h2)
        return min(diff, 360 - diff)
    
    def get_harmony_type(self, colors: List[str]) -> ColorHarmony:
        """Determine the type of color harmony in a palette."""
        if len(colors) < 2:
            return ColorHarmony.MONOCHROMATIC
        
        # Check if all neutral
        if all(c.lower() in NEUTRAL_COLORS for c in colors):
            return ColorHarmony.NEUTRAL
        
        # Get hues of non-neutral colors
        hues = []
        for c in colors:
            h, s, l = self.get_hsl(c)
            if s > 0.1:  # Has saturation
                hues.append(h)
        
        if len(hues) < 2:
            return ColorHarmony.NEUTRAL
        
        # Calculate hue distances
        h1, h2 = hues[0], hues[1]
        dist = self.hue_distance(h1, h2)
        
        if dist < 30:
            return ColorHarmony.ANALOGOUS
        elif 150 < dist < 210:
            return ColorHarmony.COMPLEMENTARY
        elif 110 < dist < 130:
            return ColorHarmony.TRIADIC
        else:
            return ColorHarmony.SPLIT_COMPLEMENTARY
    
    def calculate_harmony_score(self, colors: List[str]) -> float:
        """
        Score color harmony (0-1).
        Higher = more harmonious.
        """
        if len(colors) < 2:
            return 1.0
        
        harmony = self.get_harmony_type(colors)
        
        # Base scores for different harmonies
        harmony_scores = {
            ColorHarmony.NEUTRAL: 0.9,
            ColorHarmony.ANALOGOUS: 0.85,
            ColorHarmony.COMPLEMENTARY: 0.75,
            ColorHarmony.MONOCHROMATIC: 0.8,
            ColorHarmony.TRIADIC: 0.7,
            ColorHarmony.SPLIT_COMPLEMENTARY: 0.65,
        }
        
        base = harmony_scores.get(harmony, 0.6)
        
        # Penalty for too many saturated colors
        saturated_count = sum(1 for c in colors if self.get_hsl(c)[1] > 0.5)
        if saturated_count > 2:
            base -= 0.1 * (saturated_count - 2)
        
        return max(0.1, min(1.0, base))
    
    def calculate_skin_contrast(self, colors: List[str], skin_tone: float, undertone: str) -> float:
        """
        Score how well colors contrast with skin tone.
        skin_tone: 0 (very light) to 1 (very dark)
        """
        if not colors:
            return 0.5
        
        # Ensure skin_tone is a float
        skin_tone = skin_tone if skin_tone is not None else 0.5
        
        scores = []
        for color in colors:
            h, s, l = self.get_hsl(color)
            
            # Lightness contrast
            lightness_diff = abs(l - skin_tone)
            
            # Dark skin + bright colors = good contrast
            if skin_tone > 0.6:
                if l > 0.6:  # Bright colors
                    contrast_score = 0.8 + lightness_diff * 0.2
                elif l < 0.3:  # Very dark colors
                    contrast_score = 0.5  # Low contrast, but can work
                else:
                    contrast_score = 0.6 + lightness_diff * 0.3
            
            # Light skin
            elif skin_tone < 0.4:
                if l < 0.4:  # Dark colors
                    contrast_score = 0.8 + lightness_diff * 0.2
                elif l > 0.8:  # Very bright (white)
                    contrast_score = 0.5  # Can wash out
                else:
                    contrast_score = 0.6 + lightness_diff * 0.3
            
            # Medium skin - most versatile
            else:
                contrast_score = 0.7 + lightness_diff * 0.2
            
            # Undertone adjustments
            if undertone == "warm":
                if color.lower() in ["gold", "orange", "red", "brown", "tan", "beige"]:
                    contrast_score += 0.1
                elif color.lower() in ["silver", "blue", "grey"]:
                    contrast_score -= 0.05
            elif undertone == "cool":
                if color.lower() in ["silver", "blue", "pink", "purple"]:
                    contrast_score += 0.1
                elif color.lower() in ["gold", "orange", "brown"]:
                    contrast_score -= 0.05
            
            scores.append(min(1.0, contrast_score))
        
        return sum(scores) / len(scores)


# ============================================================
# BODY PROPORTION RULES
# ============================================================

class BodyProportionEngine:
    """Rules for body type and proportion optimization."""
    
    # Body shape optimization rules
    BODY_SHAPE_RULES = {
        # Ectomorph (lean, long limbs)
        "ectomorph": {
            "recommended_fits": ["regular", "slim", "tailored"],
            "avoid_fits": ["oversized"],  # Can look drowned
            "silhouette_tips": [
                "Layering adds dimension",
                "Horizontal patterns add width",
                "Structured shoulders work well"
            ],
            "proportion_boost": {
                "slim_pants": 0.15,
                "layered_top": 0.2,
                "structured_jacket": 0.15,
            }
        },
        
        # Mesomorph (athletic, muscular)
        "mesomorph": {
            "recommended_fits": ["regular", "tailored", "slim"],
            "avoid_fits": ["very_tight"],  # Too revealing
            "silhouette_tips": [
                "V-neck emphasizes shoulders",
                "Tailored fits showcase physique",
                "Monochromatic elongates"
            ],
            "proportion_boost": {
                "tailored_jacket": 0.2,
                "v_neck": 0.15,
                "fitted_shirt": 0.15,
            }
        },
        
        # Endomorph (soft, wider)
        "endomorph": {
            "recommended_fits": ["regular", "tailored", "structured"],
            "avoid_fits": ["very_tight", "very_loose"],
            "silhouette_tips": [
                "Vertical lines elongate",
                "Dark colors are slimming",
                "Structured pieces add definition"
            ],
            "proportion_boost": {
                "dark_outer_layer": 0.2,
                "vertical_pattern": 0.15,
                "structured_blazer": 0.2,
            }
        }
    }
    
    # Height-specific rules
    HEIGHT_RULES = {
        "short": {  # < 165cm normalized
            "avoid": ["long_coat", "wide_pants", "horizontal_stripes"],
            "recommend": ["high_waist", "cropped_jacket", "vertical_stripes"],
            "penalties": {
                "long_coat": -0.4,
                "wide_pants": -0.2,
                "oversized_everything": -0.3,
            },
            "boosts": {
                "high_waist_pants": 0.25,
                "monochromatic": 0.2,
                "slim_fit": 0.15,
            }
        },
        "average": {  # 165-180cm
            "avoid": [],
            "recommend": ["balanced_proportions"],
            "penalties": {},
            "boosts": {"balanced_outfit": 0.1}
        },
        "tall": {  # > 180cm
            "avoid": [],
            "recommend": ["layering", "wide_pants", "horizontal_patterns"],
            "boosts": {
                "layered_look": 0.15,
                "wide_pants": 0.1,
                "oversized_top": 0.1,
            },
            "penalties": {}
        }
    }
    
    def get_body_type(self, body_shape_embedding: List[float]) -> str:
        """Determine dominant body type from embedding."""
        if len(body_shape_embedding) < 3:
            return "mesomorph"
        
        ecto, meso, endo = body_shape_embedding[:3]
        if ecto > meso and ecto > endo:
            return "ectomorph"
        elif endo > meso and endo > ecto:
            return "endomorph"
        else:
            return "mesomorph"
    
    def get_height_category(self, height_normalized: float) -> str:
        """Categorize height."""
        if height_normalized < 0.35:
            return "short"
        elif height_normalized > 0.65:
            return "tall"
        else:
            return "average"
    
    def calculate_proportion_score(
        self,
        outfit_features: Dict[str, Any],
        body_shape_embedding: List[float],
        height_normalized: float,
        shoulder_waist_ratio: float = 1.0
    ) -> float:
        """
        Calculate how well an outfit works for body proportions.
        Returns score 0-1.
        """
        base_score = 0.6
        
        # Ensure numerical values are not None
        body_shape_embedding = body_shape_embedding or [0.33, 0.33, 0.34]
        height_normalized = height_normalized if height_normalized is not None else 0.5
        shoulder_waist_ratio = shoulder_waist_ratio if shoulder_waist_ratio is not None else 1.0

        body_type = self.get_body_type(body_shape_embedding)
        height_cat = self.get_height_category(height_normalized)
        
        rules = self.BODY_SHAPE_RULES.get(body_type, {})
        height_rules = self.HEIGHT_RULES.get(height_cat, {})
        
        # Check fit compatibility
        outfit_fit = outfit_features.get("fit", "regular")
        if outfit_fit in rules.get("recommended_fits", []):
            base_score += 0.15
        elif outfit_fit in rules.get("avoid_fits", []):
            base_score -= 0.2
        
        # Apply body shape boosts/penalties
        for feature, boost in rules.get("proportion_boost", {}).items():
            if outfit_features.get(feature):
                base_score += boost
        
        # Apply height penalties
        for feature, penalty in height_rules.get("penalties", {}).items():
            if outfit_features.get(feature):
                base_score += penalty  # penalties are negative
        
        # Apply height boosts
        for feature, boost in height_rules.get("boosts", {}).items():
            if outfit_features.get(feature):
                base_score += boost
        
        # Shoulder ratio adjustments
        if shoulder_waist_ratio > 1.2:  # Broad shoulders
            if outfit_features.get("v_neck") or outfit_features.get("structured_jacket"):
                base_score += 0.1
        elif shoulder_waist_ratio < 0.9:  # Narrow shoulders
            if outfit_features.get("padded_shoulders") or outfit_features.get("structured_jacket"):
                base_score += 0.15
        
        return max(0.1, min(1.0, base_score))


# ============================================================
# STYLE BALANCE RULES
# ============================================================

class StyleBalanceEngine:
    """Rules for balancing outfit elements."""
    
    # Style clash rules
    CLASH_RULES = [
        {
            "condition": lambda o: o.get("bold_upper") and o.get("bold_lower"),
            "penalty": 0.3,
            "reason": "Two statement pieces compete for attention"
        },
        {
            "condition": lambda o: o.get("multiple_patterns") and not o.get("pattern_mixing_skill"),
            "penalty": 0.25,
            "reason": "Pattern mixing requires expertise"
        },
        {
            "condition": lambda o: o.get("formal_top") and o.get("casual_bottom"),
            "penalty": 0.2,
            "reason": "Formality mismatch"
        },
        {
            "condition": lambda o: o.get("statement_shoes") and o.get("statement_accessories"),
            "penalty": 0.15,
            "reason": "Too many focal points"
        },
    ]
    
    # Balance boost rules
    BALANCE_RULES = [
        {
            "condition": lambda o: o.get("bold_upper") and o.get("neutral_lower"),
            "boost": 0.2,
            "reason": "Balanced attention with statement piece"
        },
        {
            "condition": lambda o: o.get("neutral_outfit") and o.get("statement_accessory"),
            "boost": 0.15,
            "reason": "Accessory elevates neutral base"
        },
        {
            "condition": lambda o: o.get("monochromatic"),
            "boost": 0.1,
            "reason": "Cohesive color story"
        },
        {
            "condition": lambda o: o.get("third_piece"),
            "boost": 0.1,
            "reason": "Third piece adds sophistication"
        },
    ]
    
    def calculate_balance_score(self, outfit_features: Dict[str, Any]) -> Tuple[float, List[str]]:
        """
        Calculate style balance score and reasons.
        Returns (score, list of reasons).
        """
        score = 0.6
        reasons = []
        
        # Apply clash penalties
        for rule in self.CLASH_RULES:
            if rule["condition"](outfit_features):
                score -= rule["penalty"]
                reasons.append(f"⚠️ {rule['reason']}")
        
        # Apply balance boosts
        for rule in self.BALANCE_RULES:
            if rule["condition"](outfit_features):
                score += rule["boost"]
                reasons.append(f"✓ {rule['reason']}")
        
        return max(0.1, min(1.0, score)), reasons


# ============================================================
# CONFIDENCE & OCCASION RULES
# ============================================================

class ConfidenceOccasionEngine:
    """Rules for matching outfits to confidence and occasion."""
    
    # Confidence requirements for different outfit types
    CONFIDENCE_REQUIREMENTS = {
        "bold_color": 0.6,
        "statement_print": 0.65,
        "unconventional_silhouette": 0.7,
        "revealing_fit": 0.7,
        "mixed_patterns": 0.75,
        "avant_garde": 0.8,
    }
    
    # Occasion formality requirements
    OCCASION_FORMALITY = {
        "casual": (0.0, 0.4),
        "work": (0.4, 0.7),
        "date": (0.3, 0.7),
        "wedding": (0.6, 0.9),
        "street": (0.1, 0.5),
        "formal": (0.7, 1.0),
        "party": (0.4, 0.8),
        "festival": (0.2, 0.6),
    }
    
    def calculate_confidence_fit(
        self,
        outfit_features: Dict[str, Any],
        user_confidence: float,
        user_risk_tolerance: float
    ) -> float:
        """Calculate how well outfit matches user confidence."""
        score = 0.7  # Base
        
        # Check confidence requirements
        for feature, required_conf in self.CONFIDENCE_REQUIREMENTS.items():
            if outfit_features.get(feature):
                if user_confidence >= required_conf:
                    score += 0.1  # User ready for this
                else:
                    gap = required_conf - user_confidence
                    score -= gap * 0.5  # Penalty proportional to gap
        
        # Risk tolerance affects willingness to try new things
        if outfit_features.get("experimental"):
            if user_risk_tolerance > 0.6:
                score += 0.1
            else:
                score -= 0.1
        
        return max(0.1, min(1.0, score))
    
    def calculate_occasion_fit(
        self,
        outfit_formality: float,
        occasion_type: str
    ) -> float:
        """Calculate how well outfit formality matches occasion."""
        if occasion_type not in self.OCCASION_FORMALITY:
            return 0.6
        
        min_form, max_form = self.OCCASION_FORMALITY[occasion_type]
        
        if min_form <= outfit_formality <= max_form:
            # Perfect match - center of range is best
            center = (min_form + max_form) / 2
            distance = abs(outfit_formality - center)
            range_size = (max_form - min_form) / 2
            return 0.9 - (distance / range_size) * 0.2
        
        # Outside range
        if outfit_formality < min_form:
            return max(0.3, 0.6 - (min_form - outfit_formality))
        else:
            return max(0.3, 0.6 - (outfit_formality - max_form))


# ============================================================
# TREND ALIGNMENT
# ============================================================

class TrendEngine:
    """Track and score trend alignment."""
    
    # Current trends (would be updated dynamically in production)
    CURRENT_TRENDS = {
        "2024_winter": {
            "colors": ["burgundy", "forest_green", "camel", "grey"],
            "silhouettes": ["oversized", "layered", "structured"],
            "patterns": ["plaid", "houndstooth", "solid"],
            "styles": ["quiet_luxury", "old_money", "minimalist"],
        }
    }
    
    def calculate_trend_score(
        self,
        outfit_features: Dict[str, Any],
        user_trend_following: float
    ) -> float:
        """
        Calculate trend alignment.
        Users with low trend_following shouldn't be penalized for not following trends.
        """
        if user_trend_following < 0.3:
            # User doesn't care about trends
            return 0.6  # Neutral score
        
        current = self.CURRENT_TRENDS.get("2024_winter", {})
        
        matches = 0
        total_checks = 0
        
        # Check color trends
        outfit_colors = outfit_features.get("colors", [])
        for color in outfit_colors:
            total_checks += 1
            if color in current.get("colors", []):
                matches += 1
        
        # Check silhouette
        outfit_silhouette = outfit_features.get("silhouette")
        if outfit_silhouette:
            total_checks += 1
            if outfit_silhouette in current.get("silhouettes", []):
                matches += 1
        
        # Check style
        outfit_style = outfit_features.get("style")
        if outfit_style:
            total_checks += 1
            if outfit_style in current.get("styles", []):
                matches += 1
        
        if total_checks == 0:
            return 0.6
        
        match_ratio = matches / total_checks
        
        # Scale by user's trend following preference
        return 0.4 + (match_ratio * 0.5 * user_trend_following)


# ============================================================
# UNIFIED KNOWLEDGE GRAPH
# ============================================================

class FashionKnowledgeGraph:
    """
    Unified fashion knowledge engine combining all rule systems.
    """
    
    def __init__(self):
        self.color_engine = ColorTheoryEngine()
        self.proportion_engine = BodyProportionEngine()
        self.balance_engine = StyleBalanceEngine()
        self.confidence_engine = ConfidenceOccasionEngine()
        self.trend_engine = TrendEngine()
    
    def evaluate_outfit(
        self,
        outfit_features: Dict[str, Any],
        user_physical: Dict[str, Any],
        user_psychology: Dict[str, Any],
        user_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Comprehensive outfit evaluation using all knowledge systems.
        
        Returns:
            {
                "scores": {
                    "color_harmony": float,
                    "skin_contrast": float,
                    "body_proportion": float,
                    "style_balance": float,
                    "confidence_fit": float,
                    "occasion_fit": float,
                    "trend_alignment": float,
                },
                "total_score": float,
                "insights": List[str],
                "warnings": List[str]
            }
        """
        scores = {}
        insights = []
        warnings = []
        
        # 1. Color Harmony
        colors = outfit_features.get("colors", [])
        scores["color_harmony"] = self.color_engine.calculate_harmony_score(colors)
        
        # 2. Skin Contrast
        skin_tone = user_physical.get("skin_tone", 0.5)
        undertone = user_physical.get("undertone", "neutral")
        scores["skin_contrast"] = self.color_engine.calculate_skin_contrast(
            colors, skin_tone, undertone
        )
        
        # 3. Body Proportion
        body_shape = user_physical.get("body_shape_embedding", [0.33, 0.33, 0.34])
        height_norm = user_physical.get("height_normalized", 0.5)
        shoulder_ratio = user_physical.get("shoulder_waist_ratio", 1.0)
        scores["body_proportion"] = self.proportion_engine.calculate_proportion_score(
            outfit_features, body_shape, height_norm, shoulder_ratio
        )
        
        # 4. Style Balance
        balance_score, balance_reasons = self.balance_engine.calculate_balance_score(outfit_features)
        scores["style_balance"] = balance_score
        for reason in balance_reasons:
            if reason.startswith("⚠️"):
                warnings.append(reason)
            else:
                insights.append(reason)
        
        # 5. Confidence Fit
        confidence = user_psychology.get("confidence", 0.5)
        risk_tolerance = user_psychology.get("risk_tolerance", 0.5)
        scores["confidence_fit"] = self.confidence_engine.calculate_confidence_fit(
            outfit_features, confidence, risk_tolerance
        )
        
        # 6. Occasion Fit
        outfit_formality = outfit_features.get("formality", 0.5)
        occasion = user_context.get("occasion_type", "casual")
        scores["occasion_fit"] = self.confidence_engine.calculate_occasion_fit(
            outfit_formality, occasion
        )
        
        # 7. Trend Alignment
        trend_following = user_psychology.get("trend_following", 0.5)
        scores["trend_alignment"] = self.trend_engine.calculate_trend_score(
            outfit_features, trend_following
        )
        
        # Calculate weighted total
        weights = {
            "color_harmony": 0.20,
            "skin_contrast": 0.18,
            "body_proportion": 0.18,
            "style_balance": 0.14,
            "confidence_fit": 0.12,
            "occasion_fit": 0.10,
            "trend_alignment": 0.08,
        }
        
        total = sum(scores[k] * weights[k] for k in weights)
        
        # Generate DYNAMIC insights based on actual scores and features
        insight_generators = [
            # Color insights
            (scores["color_harmony"] > 0.85, "Perfect color palette synergy"),
            (scores["color_harmony"] > 0.7, "Cohesive color story"),
            (0.5 < scores["color_harmony"] <= 0.7, "Interesting color contrast"),
            
            # Skin tone insights  
            (scores["skin_contrast"] > 0.85, f"Colors that make your {undertone} undertone glow"),
            (scores["skin_contrast"] > 0.7, "Colors complement your skin beautifully"),
            
            # Body proportion insights
            (scores["body_proportion"] > 0.85, "Perfectly balanced proportions"),
            (scores["body_proportion"] > 0.7, "Flattering silhouette for your frame"),
            
            # Style insights
            (outfit_features.get("neutral_outfit"), "Timeless neutral palette"),
            (outfit_features.get("bold_upper") and outfit_features.get("neutral_lower"), "Balanced attention with statement piece"),
            (outfit_features.get("has_third_piece"), "Layered depth adds sophistication"),
            (outfit_features.get("monochromatic"), "Sleek monochromatic look"),
            
            # Confidence insights
            (scores["confidence_fit"] > 0.8 and risk_tolerance > 0.6, "Perfectly bold for your style DNA"),
            (scores["confidence_fit"] > 0.8 and risk_tolerance <= 0.4, "Refined and understated elegance"),
            
            # Occasion fit
            (scores["occasion_fit"] > 0.85, f"Ideal for your {occasion} plans"),
            (scores["occasion_fit"] > 0.7, "Versatile for multiple settings"),
        ]
        
        # Add insights where condition is True
        for condition, insight in insight_generators:
            if condition and insight not in insights:
                insights.append(insight)
        
        # Limit to top 4 most relevant insights
        insights = insights[:4]
        
        return {
            "scores": scores,
            "total_score": total,
            "insights": insights,
            "warnings": warnings
        }


# Singleton instance
fashion_knowledge = FashionKnowledgeGraph()
