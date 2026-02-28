"""
Recommendation Agent - Specializes in ranking and personalization.

Input:
- Product list from SEARCH_AGENT
- User memory
- Current intent

Tasks:
- Score products based on user preferences
- Explain why each product matches
- Rank results

Constraints:
- Do NOT introduce new products
- Do NOT hallucinate attributes
- Prefer diversity when scores are similar
"""

from typing import List, Dict, Any, Optional
import logging
import random

logger = logging.getLogger(__name__)


class RecommendationAgent:
    """
    Ranks and personalizes products based on user memory and preferences.
    Builds complete OUTFITS (fits), not single items.
    Does NOT introduce new products or hallucinate attributes.
    """
    
    def build_outfits(
        self,
        products: List[Dict[str, Any]],
        user_memory: Dict[str, Any],
        intent: Dict[str, Any],
        trend_context: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Build complete outfits (fits) from products.
        
        Each outfit must include:
        - Top
        - Bottom
        - Footwear
        - Optional layer/accessory
        
        Returns ranked outfits with explanations.
        """
        if not products:
            return []
        
        # Group products by category
        tops = [p for p in products if self._is_top(p)]
        bottoms = [p for p in products if self._is_bottom(p)]
        footwear = [p for p in products if self._is_footwear(p)]
        layers = [p for p in products if self._is_layer(p)]
        
        # If we don't have enough categories, fall back to single product ranking
        if not (tops and bottoms and footwear):
            return self.rank_products(products, user_memory, intent)
        
        # Build outfit combinations
        outfits = []
        max_outfits = min(5, len(tops) * len(bottoms) * len(footwear))
        
        for _ in range(max_outfits):
            top = random.choice(tops)
            bottom = random.choice(bottoms)
            shoe = random.choice(footwear)
            layer_item = random.choice(layers) if layers else None
            
            outfit_items = [top, bottom, shoe]
            if layer_item:
                outfit_items.append(layer_item)
            
            # Score the outfit
            outfit_score, explanation = self._score_outfit(
                outfit_items, user_memory, intent, trend_context
            )
            
            outfits.append({
                "outfit_id": f"fit_{len(outfits) + 1}",
                "items": outfit_items,
                "score": outfit_score,
                "explanation": explanation,
                "total_price": sum(item.get("price", 0) for item in outfit_items),
                "trend_match": trend_context.get("aesthetic", "casual") if trend_context else "casual"
            })
        
        # Sort by score
        outfits.sort(key=lambda x: x["score"], reverse=True)
        return outfits[:5]  # Return top 5 fits
    
    def rank_products(
        self,
        products: List[Dict[str, Any]],
        user_memory: Dict[str, Any],
        intent: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Rank products based on user preferences and intent.
        
        Args:
            products: List of products from SEARCH_AGENT
            user_memory: Memory from MEMORY_AGENT
            intent: Intent from QUERY_AGENT
        
        Returns:
            Ranked list with explanations:
            [
                {
                    "product": {...},
                    "score": float,
                    "explanation": str,
                    "match_reasons": List[str]
                }
            ]
        """
        if not products:
            return []
        
        # Extract user preferences
        long_term = user_memory.get("long_term", {})
        preferred_price_range = long_term.get("preferred_price_range")
        interested_categories = long_term.get("interested_categories", [])
        preferred_brands = long_term.get("preferred_brands", [])
        style_preferences = long_term.get("style_preferences", {})
        preferred_fit = long_term.get("preferred_fit")
        disliked_fits = long_term.get("disliked_fits", [])
        
        # Extract intent filters
        intent_filters = intent.get("filters", {})
        intent_category = intent.get("product_category")
        
        # Score each product
        scored_products = []
        for product in products:
            score = 0.0
            match_reasons = []
            
            # Price matching (0-30 points)
            price = product.get("price", 0)
            if preferred_price_range:
                if isinstance(preferred_price_range, dict):
                    min_price = preferred_price_range.get("min", 0)
                    max_price = preferred_price_range.get("max", float('inf'))
                    if min_price <= price <= max_price:
                        score += 30
                        match_reasons.append("Matches your preferred price range")
                elif isinstance(preferred_price_range, str):
                    # Handle "under 2000" format
                    if "under" in preferred_price_range.lower():
                        max_price = int(''.join(filter(str.isdigit, preferred_price_range)))
                        if price <= max_price:
                            score += 30
                            match_reasons.append("Within your budget")
            
            # Category matching (0-25 points)
            product_category = (product.get("category") or "").lower()
            if intent_category and product_category == intent_category.lower():
                score += 25
                match_reasons.append(f"Matches requested category: {intent_category}")
            elif interested_categories:
                for cat in interested_categories:
                    if cat.lower() in product_category or product_category in cat.lower():
                        score += 20
                        match_reasons.append(f"Matches your interest in {cat}")
                        break
            
            # Brand matching (0-20 points)
            product_brand = (product.get("attributes", {}).get("brand") or "").lower()
            if preferred_brands:
                for brand in preferred_brands:
                    if brand.lower() in product_brand or product_brand in brand.lower():
                        score += 20
                        match_reasons.append(f"From your preferred brand: {brand}")
                        break
            
            # Fit preference matching (0-20 points) - CRITICAL for preference updates
            product_fit = (product.get("attributes", {}).get("fit") or "").lower()
            
            # Penalize disliked fits heavily
            if disliked_fits:
                for disliked_fit in disliked_fits:
                    if disliked_fit.lower() in product_fit:
                        score -= 50  # Heavy penalty
                        match_reasons.append(f"Doesn't match: you don't like {disliked_fit}")
                        break
            
            # Reward preferred fit
            if preferred_fit and preferred_fit.lower() in product_fit:
                score += 20
                match_reasons.append(f"Matches your preferred fit: {preferred_fit}")
            elif preferred_fit:
                # If user has a preferred fit but product doesn't match, small penalty
                score -= 5
            
            # Style matching (0-20 points)
            product_style = (product.get("attributes", {}).get("style") or "").lower()
            product_name_lower = (product.get("name") or "").lower()
            
            style_vibe = style_preferences.get("vibe", "").lower()
            if style_vibe and (style_vibe in product_style or style_vibe in product_name_lower):
                score += 20
                match_reasons.append(f"Matches your {style_vibe} vibe")
            
            # Design preference (0-15 points)
            design_pref = style_preferences.get("design_preference", "").lower()
            if design_pref:
                is_loud = any(kw in product_style or kw in product_name_lower for kw in ["printed", "graphic", "loud", "pattern", "strikes"])
                is_clean = any(kw in product_style or kw in product_name_lower for kw in ["solid", "plain", "clean", "minimal", "basic"])
                
                if design_pref in ["loud", "printed"] and is_loud:
                    score += 15
                    match_reasons.append("Matches your preference for bold designs")
                elif design_pref in ["clean", "simple", "plain", "solid"] and is_clean:
                    score += 15
                    match_reasons.append("Matches your clean & simple taste")
            
            # Occasion-Specific Logic (Phase 3 - THE "ALIVE" FEEL)
            # 1. College (Gen-Z preference for oversized fits)
            intent_occasion = (intent_filters.get("occasion") or intent.get("occasion") or "").lower()
            intent_sub = (intent_filters.get("sub_occasion") or intent.get("sub_occasion") or "").lower()
            
            if intent_occasion == "college":
                if product_fit == "oversized":
                    score += 25
                    match_reasons.append("Perfect oversized fit for that college vibe")
                elif product_style in ["streetwear", "casual", "relaxed"]:
                    score += 15
                    match_reasons.append("Great casual style for college")
            
            # 2. Party Sub-Types
            if intent_occasion == "party":
                if intent_sub == "clubbing":
                    if any(kw in product_style or kw in product_name_lower for kw in ["bold", "trendy", "shimmer", "party"]):
                        score += 25
                        match_reasons.append("Bold and trendy pick for clubbing")
                elif intent_sub == "wedding":
                    if any(kw in product_style or kw in product_name_lower for kw in ["ethnic", "luxury", "traditional", "formal"]):
                        score += 25
                        match_reasons.append("Elegant choice for a wedding")
                elif intent_sub == "house party":
                    if any(kw in product_style or kw in product_name_lower for kw in ["chill", "minimal", "casual", "relaxed"]):
                        score += 20
                        match_reasons.append("Chill and comfortable for a house party")
                else:
                    # General party boost
                    if any(kw in product_style or kw in product_name_lower for kw in ["elegant", "party", "festive"]):
                        score += 10
                        match_reasons.append("Good for party settings")
            
            # Intent filter matching (0-10 points)
            if intent_filters.get("color"):
                product_color = (product.get("attributes", {}).get("color") or "").lower()
                if intent_filters["color"].lower() in product_color:
                    score += 10
                    match_reasons.append(f"Matches requested color: {intent_filters['color']}")
            
            # Availability bonus (0-5 points)
            if product.get("availability", True):
                score += 5
            
            # RANKING JITTER (Phase 2)
            # Add small random variation to ensure fresh results each time
            import random
            score += random.uniform(-3, 3)
            
            # Normalize score to 0-100 (but allow negative for filtering)
            score = max(-100, min(100, score))
            
            # Filter out products with very negative scores (disliked)
            if score < -20:
                continue
            
            scored_products.append({
                "product": product,
                "score": round(score, 2),
                "explanation": self._generate_explanation(score, match_reasons),
                "match_reasons": match_reasons
            })
        
        # Sort by score descending
        scored_products.sort(key=lambda x: x["score"], reverse=True)
        
        # Apply diversity: if scores are very similar, prefer different categories
        if len(scored_products) > 1:
            top_score = scored_products[0]["score"]
            # If top 3 have similar scores (within 5 points), diversify
            if top_score - scored_products[2]["score"] < 5:
                scored_products = self._apply_diversity(scored_products)
        
        return scored_products
    
    def _generate_explanation(self, score: float, reasons: List[str]) -> str:
        """Generate explanation for why product matches."""
        if score >= 70:
            return f"Excellent match! {', '.join(reasons[:2])}"
        elif score >= 50:
            return f"Good match. {reasons[0] if reasons else 'Relevant to your preferences'}"
        elif score >= 30:
            return f"Relevant option. {reasons[0] if reasons else 'May interest you'}"
        else:
            return "Available option"
    
    def _apply_diversity(self, scored_products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply diversity to avoid showing too many similar products."""
        if len(scored_products) <= 3:
            return scored_products
        
        # Group by category
        seen_categories = set()
        diverse_products = []
        
        for item in scored_products:
            category = item["product"].get("category", "")
            if category not in seen_categories or len(diverse_products) < 3:
                diverse_products.append(item)
                seen_categories.add(category)
            if len(diverse_products) >= 5:
                break
        
        # Add remaining high-scoring items
        for item in scored_products:
            if item not in diverse_products and len(diverse_products) < 5:
                diverse_products.append(item)
        
        return diverse_products
    
    def _is_top(self, product: Dict[str, Any]) -> bool:
        """Check if product is a top (shirt, t-shirt, hoodie, etc.)"""
        category = product.get("category", "").lower()
        name = product.get("name", "").lower()
        top_keywords = ["shirt", "tee", "t-shirt", "hoodie", "sweater", "top", "blouse", "polo"]
        return any(kw in category or kw in name for kw in top_keywords)
    
    def _is_bottom(self, product: Dict[str, Any]) -> bool:
        """Check if product is a bottom (jeans, pants, cargos, etc.)"""
        category = product.get("category", "").lower()
        name = product.get("name", "").lower()
        bottom_keywords = ["jean", "pant", "cargo", "trouser", "short", "skirt"]
        return any(kw in category or kw in name for kw in bottom_keywords)
    
    def _is_footwear(self, product: Dict[str, Any]) -> bool:
        """Check if product is footwear"""
        category = product.get("category", "").lower()
        name = product.get("name", "").lower()
        footwear_keywords = ["shoe", "sneaker", "boot", "sandal", "slipper"]
        return any(kw in category or kw in name for kw in footwear_keywords)
    
    def _is_layer(self, product: Dict[str, Any]) -> bool:
        """Check if product is a layer (jacket, coat, etc.)"""
        category = product.get("category", "").lower()
        name = product.get("name", "").lower()
        layer_keywords = ["jacket", "coat", "blazer", "cardigan", "vest"]
        return any(kw in category or kw in name for kw in layer_keywords)
    
    def _score_outfit(
        self,
        items: List[Dict[str, Any]],
        user_memory: Dict[str, Any],
        intent: Dict[str, Any],
        trend_context: Dict[str, Any] = None
    ) -> tuple:
        """Score an outfit based on trend relevance, user vibe match, color harmony, and fit compatibility."""
        score = 0.0
        reasons = []
        
        # Trend relevance (0-30 points)
        if trend_context:
            aesthetic = trend_context.get("aesthetic", "")
            color_palette = trend_context.get("color_palette", [])
            
            # Check if outfit colors match trend palette
            outfit_colors = [item.get("attributes", {}).get("color", "").lower() for item in items]
            matching_colors = sum(1 for oc in outfit_colors for cp in color_palette if cp.lower() in oc or oc in cp.lower())
            if matching_colors > 0:
                score += 20
                reasons.append(f"Matches {aesthetic} aesthetic")
        
        # User vibe match (0-25 points)
        long_term = user_memory.get("long_term", {})
        user_vibe = long_term.get("style_vibe", "casual")
        preferred_fit = long_term.get("preferred_fit", "regular")
        
        # Check fit compatibility
        fits = [item.get("attributes", {}).get("fit", "").lower() for item in items]
        if preferred_fit.lower() in " ".join(fits):
            score += 15
            reasons.append(f"Matches your {preferred_fit} fit preference")
        
        # Color harmony (0-20 points)
        colors = [item.get("attributes", {}).get("color", "") for item in items if item.get("attributes", {}).get("color")]
        if len(set(colors)) <= 3:  # Good color harmony (not too many different colors)
            score += 15
            reasons.append("Good color harmony")
        
        # Price compatibility (0-15 points)
        total_price = sum(item.get("price", 0) for item in items)
        preferred_price_range = long_term.get("preferred_price_range")
        if preferred_price_range:
            if isinstance(preferred_price_range, str) and "under" in preferred_price_range.lower():
                max_price = int(''.join(filter(str.isdigit, preferred_price_range)))
                if total_price <= max_price:
                    score += 15
                    reasons.append("Within your budget")
        
        # Completeness bonus (0-10 points)
        if len(items) >= 3:  # Has top, bottom, footwear
            score += 10
            reasons.append("Complete outfit")
        
        score = min(100, score)
        explanation = ". ".join(reasons) if reasons else "Well-balanced outfit"
        
        return score, explanation


# Singleton instance
recommendation_agent = RecommendationAgent()
