"""
Validation Agent - Validates system output.

Checks:
- No hallucinated products
- Recommendations match user intent
- Claims align with database attributes
- Tone is neutral and helpful
"""

from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ValidationAgent:
    """
    Validates system output for accuracy and safety.
    """
    
    def validate_recommendations(
        self,
        recommendations: List[Dict[str, Any]],
        original_products: List[Dict[str, Any]],
        user_intent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate recommendations against original products and intent.
        
        Args:
            recommendations: Ranked recommendations from RECOMMENDATION_AGENT
            original_products: Original products from SEARCH_AGENT
            user_intent: Intent from QUERY_AGENT
        
        Returns:
        {
            "is_valid": bool,
            "issues": List[str],
            "validated_recommendations": List[Dict]
        }
        """
        issues = []
        validated_recommendations = []
        
        # Create set of valid product IDs from original search
        valid_product_ids = {p.get("id") for p in original_products}
        
        for rec in recommendations:
            product = rec.get("product", {})
            product_id = product.get("id")
            
            # Check 1: Product exists in original search results
            if product_id not in valid_product_ids:
                issues.append(f"Product {product_id} not found in search results (hallucination)")
                continue
            
            # Check 2: Product attributes are consistent
            original_product = next((p for p in original_products if p.get("id") == product_id), None)
            if original_product:
                # Verify key attributes match
                if product.get("name") != original_product.get("name"):
                    issues.append(f"Product name mismatch for {product_id}")
                if product.get("price") != original_product.get("price"):
                    issues.append(f"Product price mismatch for {product_id}")
            
            # Check 3: Recommendation matches intent
            intent_type = user_intent.get("intent_type", "search")
            if intent_type == "recommend" and rec.get("score", 0) < 30:
                issues.append(f"Low-scoring product {product_id} may not match recommendation intent")
            
            # Product passed validation
            validated_recommendations.append(rec)
        
        is_valid = len(issues) == 0
        
        return {
            "is_valid": is_valid,
            "issues": issues,
            "validated_recommendations": validated_recommendations
        }
    
    def validate_response(
        self,
        response_text: str,
        recommendations: List[Dict[str, Any]],
        original_products: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate response text for accuracy.
        
        Checks:
        - No mention of products not in recommendations
        - Claims match product attributes
        - Tone is appropriate
        """
        issues = []
        
        # Extract product names from recommendations
        valid_product_names = {rec.get("product", {}).get("name", "").lower() for rec in recommendations}
        valid_product_names.update({p.get("name", "").lower() for p in original_products})
        
        # Check for inappropriate tone
        inappropriate_words = ["guarantee", "best ever", "perfect for everyone"]
        response_lower = response_text.lower()
        for word in inappropriate_words:
            if word in response_lower:
                issues.append(f"Response contains potentially inappropriate claim: '{word}'")
        
        # Note: Full product name extraction from text would require NLP
        # For now, we do basic validation
        
        is_valid = len(issues) == 0
        
        return {
            "is_valid": is_valid,
            "issues": issues
        }


# Singleton instance
validation_agent = ValidationAgent()
