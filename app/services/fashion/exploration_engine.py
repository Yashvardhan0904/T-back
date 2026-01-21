# Exploration Engine - UCB-based exploration vs exploitation
# Handles discovery of new styles and prevents monotony

from typing import Dict, List, Any, Tuple
import math
import random
from datetime import datetime
from app.models.user_vector import FullUserVector, Outfit
import logging

logger = logging.getLogger(__name__)


class ExplorationEngine:
    """
    Multi-Armed Bandit inspired exploration system.
    
    Uses Upper Confidence Bound (UCB) to balance:
    - Exploitation: Show what user probably likes
    - Exploration: Try new styles to discover preferences
    
    UCB_i = μ_i + c * sqrt(ln(N) / n_i)
    
    Where:
    - μ_i = average reward for style i
    - n_i = times style i was tried
    - N = total interactions
    - c = exploration constant
    """
    
    # Mixing ratios
    SAFE_RATIO = 0.70      # High-confidence matches
    ADJACENT_RATIO = 0.20  # Similar but different
    WILDCARD_RATIO = 0.10  # Creative risk for discovery
    
    # UCB parameters
    EXPLORATION_CONSTANT = 1.5
    
    # Arm categories
    STYLE_ARMS = [
        "streetwear", "formal", "casual", "minimalist", "maximalist",
        "ethnic", "western", "fusion", "vintage", "avant_garde"
    ]
    
    COLOR_ARMS = [
        "dark", "bright", "pastel", "neutral", "bold", "monochrome"
    ]
    
    def __init__(self):
        # In production, these would be loaded from database per user
        self.arm_stats: Dict[str, Dict[str, Any]] = {}
    
    def calculate_ucb(self, arm_id: str, total_interactions: int) -> float:
        """
        Calculate Upper Confidence Bound for an arm.
        
        UCB = μ + c * sqrt(ln(N) / n)
        """
        stats = self.arm_stats.get(arm_id, {"mean": 0.5, "count": 0})
        mean = stats["mean"]
        count = stats["count"]
        
        if count == 0:
            return float("inf")  # Always try unexplored arms
        
        if total_interactions == 0:
            return mean
        
        exploration_bonus = self.EXPLORATION_CONSTANT * math.sqrt(
            math.log(total_interactions) / count
        )
        
        return mean + exploration_bonus
    
    def select_outfits(
        self,
        scored_outfits: List[Tuple[Outfit, float]],
        user: FullUserVector,
        limit: int = 5
    ) -> List[Tuple[Outfit, float, str]]:
        """
        Select diverse mix of outfits using exploration strategy.
        
        Returns list of (outfit, score, selection_type) tuples.
        selection_type: "safe" | "adjacent" | "wildcard"
        """
        if not scored_outfits:
            return []
        
        # Calculate counts for each category
        safe_count = max(1, int(limit * self.SAFE_RATIO))
        adjacent_count = max(1, int(limit * self.ADJACENT_RATIO))
        wildcard_count = max(0, limit - safe_count - adjacent_count)
        
        # Sort by score
        sorted_outfits = sorted(scored_outfits, key=lambda x: x[1], reverse=True)
        
        selected = []
        used_indices = set()
        
        # 1. Select safe choices (top scored)
        for i, (outfit, score) in enumerate(sorted_outfits[:safe_count]):
            selected.append((outfit, score, "safe"))
            used_indices.add(i)
        
        # 2. Select adjacent styles (different but related)
        remaining = [(i, o, s) for i, (o, s) in enumerate(sorted_outfits) if i not in used_indices]
        
        if remaining and adjacent_count > 0:
            # Get dominant styles from safe selections
            safe_styles = set()
            for outfit, _, _ in selected:
                for item in outfit.items:
                    safe_styles.add(item.fit)
            
            # Find outfits with different but complementary styles
            adjacent_candidates = []
            for i, outfit, score in remaining:
                outfit_styles = {item.fit for item in outfit.items}
                # Different fit but still good score
                if outfit_styles != safe_styles and score > 0.4:
                    adjacent_candidates.append((i, outfit, score))
            
            # Take top adjacent
            for i, outfit, score in adjacent_candidates[:adjacent_count]:
                selected.append((outfit, score, "adjacent"))
                used_indices.add(i)
        
        # 3. Select wildcards (exploration)
        remaining = [(i, o, s) for i, (o, s) in enumerate(sorted_outfits) if i not in used_indices]
        
        if remaining and wildcard_count > 0:
            # For wildcards, use UCB scores instead of pure score
            total_interactions = user.revealed_behavior.interaction_count
            
            wildcard_candidates = []
            for i, outfit, base_score in remaining:
                # Get style category for the outfit
                style_arm = self._get_outfit_style_arm(outfit)
                ucb_score = self.calculate_ucb(style_arm, total_interactions)
                
                # Boost unexplored styles
                exploration_score = base_score * 0.5 + ucb_score * 0.5
                wildcard_candidates.append((i, outfit, exploration_score, base_score))
            
            # Sort by exploration score
            wildcard_candidates.sort(key=lambda x: x[2], reverse=True)
            
            for i, outfit, exp_score, base_score in wildcard_candidates[:wildcard_count]:
                selected.append((outfit, base_score, "wildcard"))
                used_indices.add(i)
        
        # Shuffle to avoid predictable order (safe first always)
        random.shuffle(selected)
        
        return selected
    
    def _get_outfit_style_arm(self, outfit: Outfit) -> str:
        """Categorize outfit into a style arm."""
        # Simple categorization based on formality
        formality = outfit.formality_score
        
        if formality > 0.7:
            return "formal"
        elif formality > 0.5:
            return "smart_casual"
        elif formality > 0.3:
            return "casual"
        else:
            return "streetwear"
    
    def _get_outfit_color_arm(self, outfit: Outfit) -> str:
        """Categorize outfit into a color arm."""
        colors = [item.color_name.lower() for item in outfit.items if item.color_name]
        
        dark_colors = {"black", "navy", "dark grey", "charcoal", "brown"}
        bright_colors = {"red", "yellow", "orange", "pink", "electric blue"}
        neutral_colors = {"white", "beige", "grey", "cream", "tan"}
        
        dark_count = sum(1 for c in colors if c in dark_colors)
        bright_count = sum(1 for c in colors if c in bright_colors)
        neutral_count = sum(1 for c in colors if c in neutral_colors)
        
        if dark_count > len(colors) / 2:
            return "dark"
        elif bright_count > 0:
            return "bright"
        elif neutral_count > len(colors) / 2:
            return "neutral"
        else:
            return "mixed"
    
    async def update_arm(
        self,
        user_id: str,
        arm_id: str,
        reward: float  # 1.0 for liked, -1.0 for rejected, 0 for skipped
    ) -> None:
        """
        Update arm statistics based on user feedback.
        
        Uses incremental mean update:
        μ_new = μ_old + (reward - μ_old) / n
        """
        if arm_id not in self.arm_stats:
            self.arm_stats[arm_id] = {"mean": 0.5, "count": 0}
        
        stats = self.arm_stats[arm_id]
        stats["count"] += 1
        
        # Normalize reward to [0, 1]
        normalized_reward = (reward + 1) / 2
        
        # Incremental mean update
        old_mean = stats["mean"]
        stats["mean"] = old_mean + (normalized_reward - old_mean) / stats["count"]
        
        logger.info(f"Updated arm {arm_id}: new_mean={stats['mean']:.3f}, count={stats['count']}")
    
    def get_exploration_recommendation(
        self,
        user: FullUserVector,
        mind_change_detected: bool = False
    ) -> Dict[str, Any]:
        """
        Get recommendation for exploration behavior based on user state.
        """
        interaction_count = user.revealed_behavior.interaction_count
        
        # New users need more exploration
        if interaction_count < 10:
            return {
                "mode": "discovery",
                "safe_ratio": 0.5,
                "adjacent_ratio": 0.3,
                "wildcard_ratio": 0.2,
                "reason": "New user - discovering preferences"
            }
        
        # Mind change detected - boost exploration
        if mind_change_detected:
            return {
                "mode": "re-discovery",
                "safe_ratio": 0.4,
                "adjacent_ratio": 0.35,
                "wildcard_ratio": 0.25,
                "reason": "Preference shift detected - exploring new options"
            }
        
        # Exploration mode from session
        if user.session.exploration_mode:
            return {
                "mode": "explorer",
                "safe_ratio": 0.3,
                "adjacent_ratio": 0.4,
                "wildcard_ratio": 0.3,
                "reason": "User in exploration mode"
            }
        
        # High risk tolerance
        if user.psychology.risk_tolerance > 0.7:
            return {
                "mode": "adventurous",
                "safe_ratio": 0.5,
                "adjacent_ratio": 0.3,
                "wildcard_ratio": 0.2,
                "reason": "High fashion risk tolerance"
            }
        
        # Default
        return {
            "mode": "normal",
            "safe_ratio": self.SAFE_RATIO,
            "adjacent_ratio": self.ADJACENT_RATIO,
            "wildcard_ratio": self.WILDCARD_RATIO,
            "reason": "Standard recommendation mix"
        }


# Singleton instance
exploration_engine = ExplorationEngine()
