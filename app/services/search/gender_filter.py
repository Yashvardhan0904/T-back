"""
Mandatory Gender Filter System

This module implements strict gender filtering that NEVER falls back to opposite gender products.
Key principles:
1. When user specifies gender, it becomes a mandatory constraint
2. No results is better than wrong gender results
3. Gender inference from product categories (e.g., "sarees" → female)
4. Comprehensive logging for debugging and verification
"""

from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import logging
import re

logger = logging.getLogger(__name__)


class Gender(Enum):
    """Supported gender categories"""
    MALE = "male"
    FEMALE = "female"
    UNISEX = "unisex"


class GenderInferenceConfidence(Enum):
    """Confidence levels for gender inference"""
    HIGH = "high"      # 95%+ confidence (e.g., "saree", "bra")
    MEDIUM = "medium"  # 70-95% confidence (e.g., "kurta", "dress")
    LOW = "low"        # 50-70% confidence (e.g., "shirt", "jeans")
    NONE = "none"      # No gender inference possible


# Gender-specific product categories and terms
GENDER_SPECIFIC_TERMS = {
    Gender.FEMALE: {
        "high_confidence": [
            "saree", "sarees", "lehenga", "lehengas", "kurti", "kurtis", 
            "salwar", "churidar", "dupatta", "blouse", "bra", "bras",
            "panty", "panties", "lingerie", "dress", "dresses", "skirt", "skirts",
            "crop top", "crop tops", "palazzo", "palazzos", "anarkali"
        ],
        "medium_confidence": [
            "kurta", "kurtas", "top", "tops", "tunic", "tunics",
            "ethnic wear", "traditional wear", "party wear"
        ],
        "low_confidence": [
            "shirt", "shirts", "jeans", "pants", "trousers"
        ]
    },
    Gender.MALE: {
        "high_confidence": [
            "dhoti", "dhotis", "lungi", "lungis", "vest", "vests",
            "brief", "briefs", "boxer", "boxers", "underwear",
            "sherwani", "sherwanis", "nehru jacket", "nehru jackets"
        ],
        "medium_confidence": [
            "kurta", "kurtas", "shirt", "shirts", "polo", "polos",
            "formal wear", "business wear"
        ],
        "low_confidence": [
            "jeans", "pants", "trousers", "shorts", "t-shirt", "tshirt"
        ]
    },
    Gender.UNISEX: {
        "high_confidence": [
            "sneakers", "shoes", "footwear", "sandals", "slippers",
            "watch", "watches", "bag", "bags", "backpack", "backpacks",
            "cap", "caps", "hat", "hats", "sunglasses", "accessories"
        ],
        "medium_confidence": [
            "hoodie", "hoodies", "sweatshirt", "sweatshirts", "jacket", "jackets",
            "tracksuit", "tracksuits", "sportswear", "activewear"
        ],
        "low_confidence": [
            "t-shirt", "tshirt", "jeans", "shorts"
        ]
    }
}

# Database gender field mappings
GENDER_DB_MAPPINGS = {
    Gender.MALE: ["male", "men", "mens", "man", "boy", "boys", "masculine"],
    Gender.FEMALE: ["female", "women", "womens", "woman", "girl", "girls", "feminine", "ladies"],
    Gender.UNISEX: ["unisex", "gender neutral", "both", "all", "neutral"]
}


class GenderFilter:
    """
    Mandatory gender filtering system that never compromises on gender constraints.
    """
    
    def __init__(self):
        self.filter_stats = {
            "total_queries": 0,
            "gender_filtered_queries": 0,
            "gender_inferences": 0,
            "no_results_due_to_gender": 0
        }
    
    def infer_gender_from_query(self, query: str) -> Tuple[Optional[Gender], GenderInferenceConfidence]:
        """
        Infer gender from query text based on product categories and terms.
        
        Args:
            query: Search query text
            
        Returns:
            Tuple of (inferred_gender, confidence_level)
        """
        if not query:
            return None, GenderInferenceConfidence.NONE
        
        query_lower = query.lower().strip()
        
        # Check for explicit gender mentions first
        for gender in Gender:
            for db_term in GENDER_DB_MAPPINGS[gender]:
                if db_term in query_lower:
                    logger.info(f"[GenderFilter] Explicit gender found in query: {gender.value}")
                    return gender, GenderInferenceConfidence.HIGH
        
        # Check for gender-specific product terms
        best_match = None
        best_confidence = GenderInferenceConfidence.NONE
        
        for gender in Gender:
            for confidence_level in ["high_confidence", "medium_confidence", "low_confidence"]:
                terms = GENDER_SPECIFIC_TERMS[gender][confidence_level]
                
                for term in terms:
                    if term in query_lower:
                        confidence_map = {
                            "high_confidence": GenderInferenceConfidence.HIGH,
                            "medium_confidence": GenderInferenceConfidence.MEDIUM,
                            "low_confidence": GenderInferenceConfidence.LOW
                        }
                        
                        current_confidence = confidence_map[confidence_level]
                        
                        # Use the highest confidence match
                        if best_confidence == GenderInferenceConfidence.NONE or \
                           (current_confidence.value == "high" and best_confidence.value != "high"):
                            best_match = gender
                            best_confidence = current_confidence
                            logger.info(f"[GenderFilter] Inferred gender from '{term}': {gender.value} ({current_confidence.value})")
        
        if best_match:
            self.filter_stats["gender_inferences"] += 1
        
        return best_match, best_confidence
    
    def build_mandatory_gender_filter(
        self,
        gender_preference: Optional[str],
        inferred_gender: Optional[Gender] = None,
        confidence: GenderInferenceConfidence = GenderInferenceConfidence.NONE
    ) -> Optional[Dict[str, Any]]:
        """
        Build MongoDB filter for mandatory gender constraint.
        
        Args:
            gender_preference: User's stated gender preference
            inferred_gender: Gender inferred from query
            confidence: Confidence level of inference
            
        Returns:
            MongoDB filter dict or None if no gender constraint
        """
        target_gender = None
        
        # Priority: explicit preference > high-confidence inference > medium-confidence inference
        if gender_preference:
            # Map user preference to Gender enum
            gender_lower = gender_preference.lower()
            for gender in Gender:
                if gender_lower in [g.lower() for g in GENDER_DB_MAPPINGS[gender]]:
                    target_gender = gender
                    break
            
            if not target_gender:
                # Try direct mapping
                if gender_lower in ["male", "men", "mens"]:
                    target_gender = Gender.MALE
                elif gender_lower in ["female", "women", "womens"]:
                    target_gender = Gender.FEMALE
                elif gender_lower in ["unisex", "neutral"]:
                    target_gender = Gender.UNISEX
        
        elif inferred_gender and confidence in [GenderInferenceConfidence.HIGH, GenderInferenceConfidence.MEDIUM]:
            target_gender = inferred_gender
        
        if not target_gender:
            return None
        
        # Build MongoDB filter
        db_gender_terms = GENDER_DB_MAPPINGS[target_gender]
        
        # Create case-insensitive regex pattern for all valid terms
        pattern = "|".join([re.escape(term) for term in db_gender_terms])
        
        gender_filter = {
            "gender": {
                "$regex": f"^({pattern})$",
                "$options": "i"
            }
        }
        
        logger.info(f"[GenderFilter] Built mandatory filter for {target_gender.value}: {gender_filter}")
        self.filter_stats["gender_filtered_queries"] += 1
        
        return gender_filter
    
    def apply_mandatory_gender_filter(
        self,
        mongo_query: Dict[str, Any],
        gender_preference: Optional[str] = None,
        query_text: Optional[str] = None
    ) -> Tuple[Dict[str, Any], bool]:
        """
        Apply mandatory gender filter to MongoDB query.
        
        Args:
            mongo_query: Existing MongoDB query
            gender_preference: User's explicit gender preference
            query_text: Original query text for inference
            
        Returns:
            Tuple of (updated_query, gender_filter_applied)
        """
        self.filter_stats["total_queries"] += 1
        
        # Infer gender from query if not explicitly provided
        inferred_gender, confidence = self.infer_gender_from_query(query_text) if query_text else (None, GenderInferenceConfidence.NONE)
        
        # Build gender filter
        gender_filter = self.build_mandatory_gender_filter(
            gender_preference=gender_preference,
            inferred_gender=inferred_gender,
            confidence=confidence
        )
        
        if not gender_filter:
            logger.info("[GenderFilter] No gender constraint applied")
            return mongo_query, False
        
        # Apply filter to query
        updated_query = mongo_query.copy()
        updated_query.update(gender_filter)
        
        logger.info(f"[GenderFilter] Applied mandatory gender filter: {gender_filter}")
        return updated_query, True
    
    def validate_gender_compliance(
        self,
        products: List[Dict[str, Any]],
        expected_gender: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Validate that returned products comply with gender constraints.
        
        Args:
            products: List of product documents
            expected_gender: Expected gender constraint
            
        Returns:
            Tuple of (compliant_products, non_compliant_products)
        """
        if not expected_gender:
            return products, []
        
        compliant = []
        non_compliant = []
        
        # Map expected gender to valid terms
        expected_terms = []
        for gender in Gender:
            if expected_gender.lower() in [g.lower() for g in GENDER_DB_MAPPINGS[gender]]:
                expected_terms = [term.lower() for term in GENDER_DB_MAPPINGS[gender]]
                break
        
        if not expected_terms:
            logger.warning(f"[GenderFilter] Unknown gender constraint: {expected_gender}")
            return products, []
        
        for product in products:
            product_gender = product.get("gender", "").lower()
            
            if any(term in product_gender for term in expected_terms):
                compliant.append(product)
            else:
                non_compliant.append(product)
                logger.warning(f"[GenderFilter] Non-compliant product: {product.get('name')} (gender: {product_gender})")
        
        if non_compliant:
            logger.error(f"[GenderFilter] Found {len(non_compliant)} non-compliant products out of {len(products)}")
        
        return compliant, non_compliant
    
    def log_no_results_due_to_gender(
        self,
        gender_constraint: str,
        original_query: str,
        total_products_without_filter: int = 0
    ):
        """Log when gender filter results in no products"""
        self.filter_stats["no_results_due_to_gender"] += 1
        
        logger.info(
            f"[GenderFilter] No results due to gender constraint '{gender_constraint}' "
            f"for query '{original_query}'. "
            f"Products available without filter: {total_products_without_filter}"
        )
    
    def get_filter_stats(self) -> Dict[str, Any]:
        """Get gender filter statistics"""
        stats = self.filter_stats.copy()
        
        if stats["total_queries"] > 0:
            stats["gender_filter_rate"] = stats["gender_filtered_queries"] / stats["total_queries"]
            stats["inference_rate"] = stats["gender_inferences"] / stats["total_queries"]
            stats["no_results_rate"] = stats["no_results_due_to_gender"] / stats["total_queries"]
        
        return stats
    
    def explain_gender_constraint(
        self,
        gender_preference: Optional[str],
        query_text: Optional[str] = None
    ) -> str:
        """
        Generate human-readable explanation of gender constraint.
        
        Returns:
            Explanation string for user
        """
        if gender_preference:
            gender_map = {
                "male": "men's",
                "men": "men's", 
                "mens": "men's",
                "female": "women's",
                "women": "women's",
                "womens": "women's",
                "unisex": "unisex"
            }
            
            display_gender = gender_map.get(gender_preference.lower(), gender_preference)
            return f"Showing only {display_gender} products as requested"
        
        elif query_text:
            inferred_gender, confidence = self.infer_gender_from_query(query_text)
            
            if inferred_gender and confidence in [GenderInferenceConfidence.HIGH, GenderInferenceConfidence.MEDIUM]:
                gender_display = {
                    Gender.MALE: "men's",
                    Gender.FEMALE: "women's", 
                    Gender.UNISEX: "unisex"
                }.get(inferred_gender, inferred_gender.value)
                
                return f"Showing {gender_display} products based on your search"
        
        return "No gender constraint applied"


# Singleton instance
gender_filter = GenderFilter()