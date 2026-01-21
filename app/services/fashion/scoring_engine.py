# Scoring Engine - The Mathematical Brain of Fashion AI
# This is the core optimization engine that scores outfits

from typing import Dict, List, Any, Tuple
import numpy as np
from app.models.user_vector import FullUserVector, Outfit, OutfitItem
from app.services.fashion.knowledge_graph import fashion_knowledge
import logging

logger = logging.getLogger(__name__)


class ScoringEngine:
    """
    Mathematical optimization engine that scores outfits.
    
    Score = Σ(wₖ * Cₖ) - Penalties
    
    Components:
    - C_color: Color harmony score
    - C_skin: Skin contrast score
    - C_body: Body proportion score
    - C_balance: Style balance score
    - C_conf: Confidence fit score
    - C_ctx: Context/occasion fit score
    - C_trend: Trend alignment score
    - C_hist: Personal history fit score
    """
    
    # Configurable weights (can be learned over time)
    DEFAULT_WEIGHTS = {
        "color_harmony": 0.20,
        "skin_contrast": 0.18,
        "body_proportion": 0.18,
        "style_balance": 0.14,
        "confidence_fit": 0.12,
        "context_fit": 0.10,
        "trend_fit": 0.08,
    }
    
    # Penalty multipliers
    CLASH_PENALTY_WEIGHT = 0.3
    OVERFIT_PENALTY_WEIGHT = 0.15
    BUDGET_PENALTY_WEIGHT = 0.1
    
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS
    
    def extract_outfit_features(self, outfit: Outfit) -> Dict[str, Any]:
        """
        Extract features from an outfit for scoring.
        """
        features = {
            "colors": [],
            "fit": "regular",
            "formality": 0.5,
            "silhouette": "regular",
            "style": "casual",
            "patterns": [],
            "has_third_piece": False,
            "total_price": 0,
        }
        
        for item in outfit.items:
            # Collect colors
            if item.color_name:
                features["colors"].append(item.color_name)
            
            # Track formality (average)
            features["formality"] = (features["formality"] + item.formality) / 2
            
            # Track price
            features["total_price"] += item.price
            
            # Check for layers/third pieces
            if item.category in ["layer", "jacket", "blazer", "cardigan"]:
                features["has_third_piece"] = True
                features["third_piece"] = True
            
            # Fit from main pieces
            if item.category in ["top", "bottom"]:
                features["fit"] = item.fit
            
            # Pattern detection
            if item.pattern and item.pattern != "solid":
                features["patterns"].append(item.pattern)
        
        # Derived features
        features["multiple_patterns"] = len(features["patterns"]) > 1
        features["monochromatic"] = len(set(features["colors"])) <= 2
        
        # Bold detection
        bold_colors = {"red", "yellow", "orange", "pink", "purple", "gold"}
        features["bold_upper"] = any(
            item.category == "top" and item.color_name.lower() in bold_colors
            for item in outfit.items
        )
        features["bold_lower"] = any(
            item.category == "bottom" and item.color_name.lower() in bold_colors
            for item in outfit.items
        )
        
        # Neutral detection
        neutral_colors = {"black", "white", "grey", "beige", "navy", "cream", "tan"}
        features["neutral_outfit"] = all(
            item.color_name.lower() in neutral_colors
            for item in outfit.items if item.color_name
        )
        features["neutral_lower"] = any(
            item.category == "bottom" and item.color_name.lower() in neutral_colors
            for item in outfit.items
        )
        
        return features
    
    def compute_score(self, outfit: Outfit, user: FullUserVector) -> Dict[str, Any]:
        """
        Compute comprehensive outfit score.
        
        Returns:
            {
                "total_score": float,
                "scores": Dict[str, float],
                "penalties": Dict[str, float],
                "insights": List[str],
                "warnings": List[str]
            }
        """
        # Extract features
        outfit_features = self.extract_outfit_features(outfit)
        
        # Build user feature dicts for knowledge graph
        user_physical = {
            "skin_tone": user.physical.skin_tone,
            "undertone": user.physical.undertone.value,
            "body_shape_embedding": user.physical.body_shape_embedding,
            "height_normalized": user.physical.height_normalized,
            "shoulder_waist_ratio": user.physical.shoulder_waist_ratio,
        }
        
        user_psychology = {
            "confidence": user.psychology.confidence,
            "risk_tolerance": user.psychology.risk_tolerance,
            "trend_following": user.psychology.trend_following,
            "minimalism_score": user.psychology.minimalism_score,
        }
        
        user_context = {
            "occasion_type": user.context.occasion.occasion_type.value,
            "formality": user.context.occasion.formality,
            "temperature": user.context.climate.temperature_normalized,
            "season": user.context.time.season.value,
        }
        
        # Get knowledge graph evaluation
        kg_result = fashion_knowledge.evaluate_outfit(
            outfit_features, user_physical, user_psychology, user_context
        )
        
        # Map knowledge graph scores
        scores = {
            "color_harmony": kg_result["scores"]["color_harmony"],
            "skin_contrast": kg_result["scores"]["skin_contrast"],
            "body_proportion": kg_result["scores"]["body_proportion"],
            "style_balance": kg_result["scores"]["style_balance"],
            "confidence_fit": kg_result["scores"]["confidence_fit"],
            "context_fit": kg_result["scores"]["occasion_fit"],
            "trend_fit": kg_result["scores"]["trend_alignment"],
        }
        
        # Add history fit (personalization based on past behavior)
        scores["history_fit"] = self._compute_history_fit(outfit, user)
        
        # Compute weighted sum
        weighted_sum = sum(
            scores[k] * self.weights.get(k, 0.1)
            for k in scores
        )
        
        # Compute penalties
        penalties = {}
        
        # Clash penalty
        clash_penalty = self._compute_clash_penalty(outfit_features, user)
        penalties["clash"] = clash_penalty
        
        # Overfit penalty (too similar to recent suggestions)
        overfit_penalty = self._compute_overfit_penalty(outfit, user)
        penalties["overfit"] = overfit_penalty
        
        # Budget penalty
        budget_penalty = self._compute_budget_penalty(outfit_features, user)
        penalties["budget"] = budget_penalty
        
        # Total penalty
        total_penalty = sum(penalties.values())
        
        # Final score
        total_score = max(0.01, weighted_sum - total_penalty)
        
        return {
            "total_score": total_score,
            "scores": scores,
            "penalties": penalties,
            "insights": kg_result["insights"],
            "warnings": kg_result["warnings"]
        }
    
    def _compute_history_fit(self, outfit: Outfit, user: FullUserVector) -> float:
        """
        Score based on user's historical preferences.
        Uses revealed behavior to personalize.
        """
        if user.revealed_behavior.interaction_count < 3:
            return 0.5  # Not enough history, neutral score
        
        outfit_features = self.extract_outfit_features(outfit)
        score = 0.5
        
        # Check color history
        for color in outfit_features["colors"]:
            color_lower = color.lower()
            if color_lower in user.revealed_behavior.color_interactions:
                stats = user.revealed_behavior.color_interactions[color_lower]
                liked = stats.get("liked", 0)
                rejected = stats.get("rejected", 0)
                total = liked + rejected
                if total > 0:
                    acceptance_rate = liked / total
                    score += (acceptance_rate - 0.5) * 0.2  # Adjust score based on history
        
        # Check against stated preferences
        color_probs = user.stated_preferences.color_probs
        for color in outfit_features["colors"]:
            color_lower = color.lower()
            if color_lower in color_probs:
                prob = color_probs[color_lower]
                score += (prob - 0.5) * 0.1
        
        return max(0.1, min(1.0, score))
    
    def _compute_clash_penalty(self, features: Dict[str, Any], user: FullUserVector) -> float:
        """
        Penalty for style clashes.
        """
        penalty = 0.0
        
        # Double bold
        if features.get("bold_upper") and features.get("bold_lower"):
            # High confidence users can pull this off
            if user.psychology.confidence < 0.7:
                penalty += 0.15
        
        # Multiple patterns without skill
        if features.get("multiple_patterns"):
            if user.psychology.risk_tolerance < 0.6:
                penalty += 0.1
        
        # Formality mismatch
        if features.get("formal_top") and features.get("casual_bottom"):
            penalty += 0.1
        
        return penalty * self.CLASH_PENALTY_WEIGHT
    
    def _compute_overfit_penalty(self, outfit: Outfit, user: FullUserVector) -> float:
        """
        Penalty for showing too similar outfits.
        Prevents monotony.
        """
        # In production, compare against recently shown outfits
        # For now, return minimal penalty
        return 0.0
    
    def _compute_budget_penalty(self, features: Dict[str, Any], user: FullUserVector) -> float:
        """
        Penalty if outfit exceeds user's budget.
        """
        total_price = features.get("total_price") or 0.0
        max_budget = getattr(user.economic, "budget_max", 5000.0) or 5000.0
        
        if max_budget > 0 and total_price > max_budget:
            excess_ratio = (total_price - max_budget) / max_budget
            return min(0.3, excess_ratio * 0.3) * self.BUDGET_PENALTY_WEIGHT
        
        return 0.0
    
    def rank_outfits(
        self,
        outfits: List[Outfit],
        user: FullUserVector
    ) -> List[Tuple[Outfit, Dict[str, Any]]]:
        """
        Rank a list of outfits by score.
        Returns list of (outfit, score_details) tuples sorted by score.
        """
        scored = []
        for outfit in outfits:
            try:
                score_result = self.compute_score(outfit, user)
                scored.append((outfit, score_result))
            except Exception as e:
                logger.error(f"Error scoring outfit {outfit.outfit_id}: {e}")
                continue
        
        # Sort by total score descending
        scored.sort(key=lambda x: x[1]["total_score"], reverse=True)
        
        return scored


# Singleton instance
scoring_engine = ScoringEngine()
