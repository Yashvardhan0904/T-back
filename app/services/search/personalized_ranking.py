"""
Personalized Search Ranking System

This module implements profile-aware search result ranking that:
1. Integrates user preferences into search result scoring
2. Prioritizes exact matches for clear intent
3. Provides search result explanations
4. Considers user's style preferences, color preferences, brand preferences, and past behavior
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
import math

from app.models.context import UserProfile, StylePreference, ColorPreferences, BrandPreference

logger = logging.getLogger(__name__)


@dataclass
class RankingScore:
    """Detailed ranking score breakdown"""
    total_score: float
    base_relevance: float
    style_match: float
    color_preference: float
    brand_preference: float
    price_preference: float
    behavioral_boost: float
    exact_match_bonus: float
    explanation: str


class PersonalizedRanking:
    """
    Personalizes search results based on user profile and preferences.
    
    Scoring factors:
    1. Base relevance (text matching, category matching)
    2. Style preferences (fit, vibe, design)
    3. Color preferences (liked/disliked colors)
    4. Brand preferences (preferred/avoided brands)
    5. Price preferences (within preferred range)
    6. Behavioral signals (past interactions)
    7. Exact match bonuses (clear intent matching)
    """
    
    def __init__(self):
        self.scoring_weights = {
            "base_relevance": 0.3,
            "style_match": 0.25,
            "color_preference": 0.15,
            "brand_preference": 0.1,
            "price_preference": 0.1,
            "behavioral_boost": 0.05,
            "exact_match_bonus": 0.05
        }
    
    def rank_products(
        self,
        products: List[Dict[str, Any]],
        user_profile: UserProfile,
        query_intent: Dict[str, Any] = None,
        search_context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Rank products based on user profile and preferences.
        
        Args:
            products: List of product documents
            user_profile: User's profile with preferences
            query_intent: Intent extracted from user query
            search_context: Additional search context
            
        Returns:
            List of products with ranking scores and explanations
        """
        if not products:
            return []
        
        logger.info(f"[PersonalizedRanking] Ranking {len(products)} products for user profile")
        
        # Score each product
        scored_products = []
        for product in products:
            score = self._calculate_product_score(product, user_profile, query_intent, search_context)
            
            scored_product = {
                "product": product,
                "ranking_score": score.total_score,
                "score_breakdown": {
                    "base_relevance": score.base_relevance,
                    "style_match": score.style_match,
                    "color_preference": score.color_preference,
                    "brand_preference": score.brand_preference,
                    "price_preference": score.price_preference,
                    "behavioral_boost": score.behavioral_boost,
                    "exact_match_bonus": score.exact_match_bonus
                },
                "match_reasons": self._generate_match_reasons(score, product, user_profile),
                "explanation": score.explanation
            }
            scored_products.append(scored_product)
        
        # Sort by score (highest first)
        ranked_products = sorted(scored_products, key=lambda x: x["ranking_score"], reverse=True)
        
        logger.info(f"[PersonalizedRanking] Top product score: {ranked_products[0]['ranking_score']:.3f}")
        return ranked_products
    
    def _calculate_product_score(
        self,
        product: Dict[str, Any],
        user_profile: UserProfile,
        query_intent: Dict[str, Any] = None,
        search_context: Dict[str, Any] = None
    ) -> RankingScore:
        """Calculate detailed ranking score for a product"""
        
        # 1. Base relevance (always start with some base score)
        base_relevance = self._calculate_base_relevance(product, query_intent, search_context)
        
        # 2. Style match score
        style_match = self._calculate_style_match(product, user_profile)
        
        # 3. Color preference score
        color_preference = self._calculate_color_preference(product, user_profile)
        
        # 4. Brand preference score
        brand_preference = self._calculate_brand_preference(product, user_profile)
        
        # 5. Price preference score
        price_preference = self._calculate_price_preference(product, user_profile)
        
        # 6. Behavioral boost
        behavioral_boost = self._calculate_behavioral_boost(product, user_profile)
        
        # 7. Exact match bonus
        exact_match_bonus = self._calculate_exact_match_bonus(product, query_intent)
        
        # Calculate weighted total score
        total_score = (
            base_relevance * self.scoring_weights["base_relevance"] +
            style_match * self.scoring_weights["style_match"] +
            color_preference * self.scoring_weights["color_preference"] +
            brand_preference * self.scoring_weights["brand_preference"] +
            price_preference * self.scoring_weights["price_preference"] +
            behavioral_boost * self.scoring_weights["behavioral_boost"] +
            exact_match_bonus * self.scoring_weights["exact_match_bonus"]
        )
        
        # Generate explanation
        explanation = self._generate_score_explanation(
            base_relevance, style_match, color_preference, brand_preference,
            price_preference, behavioral_boost, exact_match_bonus, product, user_profile
        )
        
        return RankingScore(
            total_score=total_score,
            base_relevance=base_relevance,
            style_match=style_match,
            color_preference=color_preference,
            brand_preference=brand_preference,
            price_preference=price_preference,
            behavioral_boost=behavioral_boost,
            exact_match_bonus=exact_match_bonus,
            explanation=explanation
        )
    
    def _calculate_base_relevance(
        self,
        product: Dict[str, Any],
        query_intent: Dict[str, Any] = None,
        search_context: Dict[str, Any] = None
    ) -> float:
        """Calculate base relevance score (0.0 to 1.0)"""
        score = 0.5  # Base score for any product
        
        # Category match bonus
        if query_intent and query_intent.get("product_category"):
            product_category = product.get("category", "").lower()
            intent_category = query_intent["product_category"].lower()
            
            if intent_category in product_category or product_category in intent_category:
                score += 0.3
        
        # Search context relevance
        if search_context and search_context.get("search_terms"):
            product_name = product.get("name", "").lower()
            for term in search_context["search_terms"]:
                if term.lower() in product_name:
                    score += 0.1
        
        return min(1.0, score)
    
    def _calculate_style_match(self, product: Dict[str, Any], user_profile: UserProfile) -> float:
        """Calculate style preference match score (0.0 to 1.0)"""
        score = 0.5  # Neutral score
        
        # Check fit preferences
        product_fit = product.get("fit", "").lower()
        if product_fit:
            # Preferred fit bonus
            if user_profile.preferred_fit and user_profile.preferred_fit.lower() == product_fit:
                score += 0.3
            
            # Disliked fit penalty
            if user_profile.disliked_fits:
                for disliked_fit in user_profile.disliked_fits:
                    if disliked_fit.lower() == product_fit:
                        score -= 0.4
                        break
        
        # Check style vibe match
        product_style = product.get("style", "").lower()
        if product_style and user_profile.style_vibe:
            if user_profile.style_vibe.lower() in product_style or product_style in user_profile.style_vibe.lower():
                score += 0.2
        
        # Check design preference
        product_description = product.get("description", "").lower()
        if user_profile.design_preference:
            design_pref = user_profile.design_preference.lower()
            
            if design_pref == "clean" and any(word in product_description for word in ["minimal", "clean", "simple", "solid"]):
                score += 0.2
            elif design_pref == "loud" and any(word in product_description for word in ["printed", "graphic", "bold", "pattern"]):
                score += 0.2
        
        # Style preferences from profile
        for style_pref in user_profile.style_preferences:
            if style_pref.style.lower() in product_description or style_pref.style.lower() in product.get("name", "").lower():
                score += 0.1 * style_pref.confidence
        
        return max(0.0, min(1.0, score))
    
    def _calculate_color_preference(self, product: Dict[str, Any], user_profile: UserProfile) -> float:
        """Calculate color preference match score (0.0 to 1.0)"""
        score = 0.5  # Neutral score
        
        product_color = product.get("color", "").lower()
        if not product_color:
            return score
        
        color_prefs = user_profile.color_preferences
        
        # Liked colors bonus
        for liked_color in color_prefs.liked_colors:
            if liked_color.lower() in product_color:
                score += 0.3
                break
        
        # Disliked colors penalty
        for disliked_color in color_prefs.disliked_colors:
            if disliked_color.lower() in product_color:
                score -= 0.4
                break
        
        # Neutral colors (no penalty or bonus)
        for neutral_color in color_prefs.neutral_colors:
            if neutral_color.lower() in product_color:
                # Slight bonus for neutral colors (safe choice)
                score += 0.1
                break
        
        return max(0.0, min(1.0, score))
    
    def _calculate_brand_preference(self, product: Dict[str, Any], user_profile: UserProfile) -> float:
        """Calculate brand preference match score (0.0 to 1.0)"""
        score = 0.5  # Neutral score
        
        product_brand = product.get("brand", "").lower()
        if not product_brand:
            return score
        
        # Check brand preferences
        for brand_pref in user_profile.brand_preferences:
            if brand_pref.brand.lower() == product_brand:
                if brand_pref.preference_type == "like":
                    score += 0.3 * brand_pref.confidence
                elif brand_pref.preference_type == "dislike":
                    score -= 0.4 * brand_pref.confidence
                break
        
        return max(0.0, min(1.0, score))
    
    def _calculate_price_preference(self, product: Dict[str, Any], user_profile: UserProfile) -> float:
        """Calculate price preference match score (0.0 to 1.0)"""
        score = 0.5  # Neutral score
        
        product_price = product.get("price", 0)
        if not product_price or not user_profile.price_range:
            return score
        
        price_range = user_profile.price_range
        
        # Check if within preferred range
        within_range = True
        if price_range.min_price and product_price < price_range.min_price:
            within_range = False
        if price_range.max_price and product_price > price_range.max_price:
            within_range = False
        
        if within_range:
            score += 0.3
            
            # Bonus for being in the sweet spot (middle of range)
            if price_range.min_price and price_range.max_price:
                range_size = price_range.max_price - price_range.min_price
                middle = price_range.min_price + (range_size / 2)
                distance_from_middle = abs(product_price - middle) / range_size
                
                # Closer to middle = higher score
                sweet_spot_bonus = 0.2 * (1 - distance_from_middle)
                score += sweet_spot_bonus
        else:
            # Penalty for being outside range
            score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    def _calculate_behavioral_boost(self, product: Dict[str, Any], user_profile: UserProfile) -> float:
        """Calculate behavioral signal boost (0.0 to 1.0)"""
        score = 0.5  # Neutral score
        
        product_id = str(product.get("_id", ""))
        if not product_id:
            return score
        
        # Check interaction history
        for interaction in user_profile.interaction_history:
            if interaction.item_id == product_id:
                if interaction.interaction_type in ["like", "purchase"]:
                    score += 0.3
                elif interaction.interaction_type == "click":
                    score += 0.1
                elif interaction.interaction_type == "reject":
                    score -= 0.2
        
        # Similar product boost (same category, brand, style)
        similar_interactions = 0
        for interaction in user_profile.interaction_history:
            if interaction.interaction_type in ["like", "purchase"]:
                item_features = interaction.item_features
                
                # Same category
                if item_features.get("category") == product.get("category"):
                    similar_interactions += 1
                
                # Same brand
                if item_features.get("brand") == product.get("brand"):
                    similar_interactions += 1
                
                # Same style attributes
                if item_features.get("fit") == product.get("fit"):
                    similar_interactions += 1
        
        # Boost based on similar positive interactions
        if similar_interactions > 0:
            score += min(0.2, similar_interactions * 0.05)
        
        return max(0.0, min(1.0, score))
    
    def _calculate_exact_match_bonus(self, product: Dict[str, Any], query_intent: Dict[str, Any] = None) -> float:
        """Calculate exact match bonus for clear intent (0.0 to 1.0)"""
        score = 0.5  # Neutral score
        
        if not query_intent:
            return score
        
        # Exact category match
        if query_intent.get("product_category"):
            product_category = product.get("category", "").lower()
            intent_category = query_intent["product_category"].lower()
            
            if product_category == intent_category:
                score += 0.3
        
        # Exact filter matches
        filters = query_intent.get("filters", {})
        
        # Color match
        if filters.get("color"):
            product_color = product.get("color", "").lower()
            if filters["color"].lower() in product_color:
                score += 0.2
        
        # Brand match
        if filters.get("brand"):
            product_brand = product.get("brand", "").lower()
            if filters["brand"].lower() in product_brand:
                score += 0.2
        
        # Fit match
        if filters.get("fit"):
            product_fit = product.get("fit", "").lower()
            if filters["fit"].lower() == product_fit:
                score += 0.2
        
        return max(0.0, min(1.0, score))
    
    def _generate_score_explanation(
        self,
        base_relevance: float,
        style_match: float,
        color_preference: float,
        brand_preference: float,
        price_preference: float,
        behavioral_boost: float,
        exact_match_bonus: float,
        product: Dict[str, Any],
        user_profile: UserProfile
    ) -> str:
        """Generate human-readable explanation of ranking score"""
        
        explanations = []
        
        # Base relevance
        if base_relevance > 0.7:
            explanations.append("highly relevant to your search")
        elif base_relevance > 0.5:
            explanations.append("matches your search")
        
        # Style match
        if style_match > 0.7:
            explanations.append(f"perfect fit for your {user_profile.style_vibe} style")
        elif style_match > 0.6:
            explanations.append("matches your style preferences")
        elif style_match < 0.4:
            explanations.append("different from your usual style")
        
        # Color preference
        if color_preference > 0.7:
            explanations.append("in your favorite colors")
        elif color_preference < 0.3:
            explanations.append("in colors you typically avoid")
        
        # Brand preference
        if brand_preference > 0.7:
            explanations.append("from a brand you like")
        elif brand_preference < 0.3:
            explanations.append("from a brand you typically avoid")
        
        # Price preference
        if price_preference > 0.7:
            explanations.append("within your preferred price range")
        elif price_preference < 0.3:
            explanations.append("outside your usual price range")
        
        # Behavioral boost
        if behavioral_boost > 0.7:
            explanations.append("similar to items you've liked before")
        
        # Exact match
        if exact_match_bonus > 0.7:
            explanations.append("exactly what you're looking for")
        
        if not explanations:
            explanations.append("matches your search criteria")
        
        return ", ".join(explanations[:3])  # Limit to top 3 reasons
    
    def _generate_match_reasons(
        self,
        score: RankingScore,
        product: Dict[str, Any],
        user_profile: UserProfile
    ) -> List[str]:
        """Generate list of match reasons for display"""
        reasons = []
        
        # Add reasons based on score components
        if score.style_match > 0.6:
            if user_profile.preferred_fit and product.get("fit"):
                reasons.append(f"{user_profile.preferred_fit} fit")
            if user_profile.style_vibe:
                reasons.append(f"{user_profile.style_vibe} vibe")
        
        if score.color_preference > 0.6:
            product_color = product.get("color", "")
            if product_color:
                reasons.append(f"{product_color} color")
        
        if score.price_preference > 0.6:
            reasons.append("within budget")
        
        if score.brand_preference > 0.6:
            brand = product.get("brand", "")
            if brand:
                reasons.append(f"{brand} brand")
        
        if score.exact_match_bonus > 0.6:
            reasons.append("exact match")
        
        if score.behavioral_boost > 0.6:
            reasons.append("similar to liked items")
        
        # Ensure we have at least one reason
        if not reasons:
            reasons.append("matches search")
        
        return reasons[:4]  # Limit to top 4 reasons
    
    def explain_ranking_decision(
        self,
        ranked_products: List[Dict[str, Any]],
        user_profile: UserProfile
    ) -> str:
        """Generate explanation for why products were ranked in this order"""
        
        if not ranked_products:
            return "No products to rank"
        
        top_product = ranked_products[0]
        top_reasons = top_product.get("match_reasons", [])
        
        explanation = f"Top result chosen because it {', '.join(top_reasons[:2])}"
        
        if len(ranked_products) > 1:
            explanation += f". Ranked {len(ranked_products)} products by your style preferences"
        
        return explanation


# Singleton instance
personalized_ranking = PersonalizedRanking()