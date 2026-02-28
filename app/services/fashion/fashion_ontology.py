"""
Fashion Ontology System - Domain Intelligence Layer

This is the missing fashion knowledge layer that understands:
1. What people wear for different occasions
2. Cultural context (Indian fashion norms)
3. Outfit composition rules
4. Category appropriateness

This prevents disasters like suggesting socks for party wear.
"""

from typing import Dict, List, Set, Optional, Tuple, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class OccasionType(Enum):
    """Types of occasions"""
    PARTY = "party"
    WEDDING = "wedding"
    OFFICE = "office"
    CASUAL = "casual"
    FORMAL = "formal"
    TRADITIONAL = "traditional"
    CLUB = "club"
    DATE = "date"
    FESTIVAL = "festival"
    INTERVIEW = "interview"


class CulturalContext(Enum):
    """Cultural contexts"""
    INDIAN = "indian"
    WESTERN = "western"
    FUSION = "fusion"


class Gender(Enum):
    """Gender categories"""
    WOMEN = "women"
    MEN = "men"
    UNISEX = "unisex"


class FashionOntology:
    """
    Fashion domain knowledge system that understands:
    - What to wear for different occasions
    - Cultural appropriateness
    - Outfit composition rules
    - Category filtering logic
    """
    
    def __init__(self):
        self.occasion_wear = self._build_occasion_wear_map()
        self.blocked_categories = self._build_blocked_categories()
        self.outfit_templates = self._build_outfit_templates()
        self.cultural_preferences = self._build_cultural_preferences()
        self.formality_levels = self._build_formality_levels()
    
    def _build_occasion_wear_map(self) -> Dict[OccasionType, Dict[Gender, Dict[CulturalContext, List[str]]]]:
        """Build comprehensive occasion-appropriate wear mapping"""
        return {
            OccasionType.PARTY: {
                Gender.WOMEN: {
                    CulturalContext.WESTERN: [
                        "bodycon dress", "mini dress", "midi dress", "jumpsuit", 
                        "co-ord set", "crop top", "skirt", "party top", "gown",
                        "cocktail dress", "off-shoulder dress", "wrap dress"
                    ],
                    CulturalContext.INDIAN: [
                        "saree", "lehenga", "anarkali", "sharara", "palazzo suit",
                        "kurti", "indo-western dress", "fusion wear", "ethnic gown"
                    ],
                    CulturalContext.FUSION: [
                        "indo-western dress", "fusion kurti", "dhoti pants with top",
                        "cape dress", "jacket with ethnic bottom"
                    ]
                },
                Gender.MEN: {
                    CulturalContext.WESTERN: [
                        "shirt", "polo", "blazer", "chinos", "formal pants",
                        "party shirt", "dress shirt", "casual blazer"
                    ],
                    CulturalContext.INDIAN: [
                        "kurta", "nehru jacket", "bandhgala", "sherwani",
                        "ethnic shirt", "pathani suit"
                    ],
                    CulturalContext.FUSION: [
                        "kurta with jeans", "nehru jacket with chinos",
                        "ethnic shirt with formal pants"
                    ]
                }
            },
            
            OccasionType.WEDDING: {
                Gender.WOMEN: {
                    CulturalContext.INDIAN: [
                        "saree", "lehenga", "anarkali", "sharara", "gharara",
                        "heavy kurti", "silk saree", "designer lehenga"
                    ],
                    CulturalContext.WESTERN: [
                        "gown", "cocktail dress", "formal dress", "maxi dress"
                    ]
                },
                Gender.MEN: {
                    CulturalContext.INDIAN: [
                        "sherwani", "bandhgala", "kurta", "nehru jacket",
                        "dhoti kurta", "silk kurta"
                    ],
                    CulturalContext.WESTERN: [
                        "suit", "blazer", "formal shirt", "tuxedo"
                    ]
                }
            },
            
            OccasionType.OFFICE: {
                Gender.WOMEN: {
                    CulturalContext.WESTERN: [
                        "shirt", "blouse", "formal pants", "pencil skirt",
                        "blazer", "formal dress", "trousers"
                    ],
                    CulturalContext.INDIAN: [
                        "formal kurti", "saree", "formal salwar"
                    ]
                },
                Gender.MEN: {
                    CulturalContext.WESTERN: [
                        "formal shirt", "trousers", "blazer", "suit",
                        "formal pants", "dress shirt"
                    ],
                    CulturalContext.INDIAN: [
                        "formal kurta", "nehru jacket"
                    ]
                }
            },
            
            OccasionType.CASUAL: {
                Gender.WOMEN: {
                    CulturalContext.WESTERN: [
                        "t-shirt", "jeans", "casual dress", "top", "shorts",
                        "casual kurti", "jeggings", "casual pants"
                    ],
                    CulturalContext.INDIAN: [
                        "kurti", "casual saree", "salwar", "palazzo"
                    ]
                },
                Gender.MEN: {
                    CulturalContext.WESTERN: [
                        "t-shirt", "jeans", "casual shirt", "polo",
                        "shorts", "casual pants", "hoodie"
                    ],
                    CulturalContext.INDIAN: [
                        "casual kurta", "ethnic t-shirt"
                    ]
                }
            }
        }
    
    def _build_blocked_categories(self) -> Dict[OccasionType, List[str]]:
        """Categories that should NEVER be suggested for specific occasions"""
        return {
            OccasionType.PARTY: [
                "socks", "innerwear", "undergarments", "basic tee", 
                "gym wear", "sleepwear", "loungewear", "sportswear",
                "activewear", "workout clothes", "pajamas"
            ],
            OccasionType.WEDDING: [
                "casual t-shirt", "shorts", "flip-flops", "gym wear",
                "sportswear", "basic tee", "loungewear"
            ],
            OccasionType.OFFICE: [
                "crop top", "mini skirt", "shorts", "flip-flops",
                "gym wear", "party wear", "club wear", "sleepwear"
            ],
            OccasionType.FORMAL: [
                "casual t-shirt", "shorts", "flip-flops", "gym wear",
                "sportswear", "loungewear", "crop top"
            ]
        }
    
    def _build_outfit_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """Complete outfit templates for different occasions"""
        return {
            "western_party_women": {
                "primary": ["dress", "jumpsuit", "co-ord set"],
                "secondary": ["top", "skirt"],
                "accessories": ["heels", "clutch", "jewelry"],
                "avoid": ["socks", "sneakers", "casual shoes"]
            },
            "indian_party_women": {
                "primary": ["saree", "lehenga", "anarkali"],
                "secondary": ["kurti", "palazzo"],
                "accessories": ["ethnic jewelry", "ethnic footwear", "clutch"],
                "avoid": ["western accessories", "casual footwear"]
            },
            "office_women": {
                "primary": ["formal shirt", "blazer", "formal dress"],
                "secondary": ["formal pants", "pencil skirt"],
                "accessories": ["formal shoes", "handbag"],
                "avoid": ["crop tops", "mini skirts", "party wear"]
            },
            "casual_women": {
                "primary": ["t-shirt", "casual dress", "kurti"],
                "secondary": ["jeans", "casual pants", "palazzo"],
                "accessories": ["casual shoes", "handbag", "casual jewelry"],
                "avoid": ["formal wear", "party wear"]
            }
        }
    
    def _build_cultural_preferences(self) -> Dict[str, Dict[str, float]]:
        """Cultural preference weights for Indian context"""
        return {
            "indian_context": {
                "traditional_occasions": {
                    "saree": 0.9,
                    "lehenga": 0.8,
                    "kurti": 0.7,
                    "anarkali": 0.8,
                    "western_dress": 0.3
                },
                "party_occasions": {
                    "saree": 0.8,
                    "western_dress": 0.7,
                    "indo_western": 0.9,
                    "lehenga": 0.6
                }
            }
        }
    
    def _build_formality_levels(self) -> Dict[str, int]:
        """Formality levels for different categories (1-10 scale)"""
        return {
            # Very Formal (8-10)
            "gown": 10,
            "tuxedo": 10,
            "suit": 9,
            "formal dress": 8,
            "blazer": 8,
            
            # Semi-Formal (5-7)
            "cocktail dress": 7,
            "dress shirt": 6,
            "midi dress": 6,
            "formal kurti": 6,
            "saree": 7,
            
            # Casual (2-4)
            "jeans": 3,
            "t-shirt": 2,
            "casual dress": 4,
            "kurti": 4,
            
            # Very Casual (1)
            "shorts": 1,
            "flip-flops": 1,
            "gym wear": 1
        }
    
    def get_appropriate_categories(
        self, 
        occasion: str, 
        gender: str, 
        cultural_context: str = "indian"
    ) -> Tuple[List[str], List[str]]:
        """
        Get appropriate and blocked categories for an occasion.
        
        Returns:
            Tuple of (appropriate_categories, blocked_categories)
        """
        # Map string inputs to enums
        try:
            occasion_enum = OccasionType(occasion.lower())
            gender_enum = Gender(gender.lower())
            culture_enum = CulturalContext(cultural_context.lower())
        except ValueError as e:
            logger.warning(f"Invalid enum value: {e}")
            return [], []
        
        # Get appropriate categories
        appropriate = []
        if occasion_enum in self.occasion_wear:
            if gender_enum in self.occasion_wear[occasion_enum]:
                if culture_enum in self.occasion_wear[occasion_enum][gender_enum]:
                    appropriate = self.occasion_wear[occasion_enum][gender_enum][culture_enum]
        
        # Get blocked categories
        blocked = self.blocked_categories.get(occasion_enum, [])
        
        logger.info(f"[FashionOntology] {occasion} {gender} {cultural_context}: "
                   f"{len(appropriate)} appropriate, {len(blocked)} blocked")
        
        return appropriate, blocked
    
    def is_category_appropriate(
        self, 
        category: str, 
        occasion: str, 
        gender: str, 
        cultural_context: str = "indian"
    ) -> bool:
        """Check if a category is appropriate for an occasion"""
        appropriate, blocked = self.get_appropriate_categories(occasion, gender, cultural_context)
        
        category_lower = category.lower()
        
        # Check if explicitly blocked
        if any(blocked_cat.lower() in category_lower for blocked_cat in blocked):
            return False
        
        # Check if in appropriate list
        if any(app_cat.lower() in category_lower for app_cat in appropriate):
            return True
        
        # Default to neutral (not blocked, not explicitly appropriate)
        return True
    
    def get_outfit_template(self, occasion: str, gender: str, cultural_context: str = "western") -> Dict[str, List[str]]:
        """Get complete outfit template for an occasion"""
        template_key = f"{cultural_context}_{occasion}_{gender}"
        return self.outfit_templates.get(template_key, {})
    
    def infer_occasion_from_query(self, query: str) -> Optional[str]:
        """Infer occasion from user query"""
        query_lower = query.lower()
        
        occasion_keywords = {
            "party": ["party", "parties", "celebration", "birthday", "anniversary"],
            "wedding": ["wedding", "marriage", "shaadi", "reception", "sangeet", "mehendi"],
            "office": ["office", "work", "professional", "meeting", "corporate"],
            "formal": ["formal", "business", "interview", "presentation"],
            "casual": ["casual", "everyday", "daily", "regular", "comfortable"],
            "traditional": ["traditional", "festival", "puja", "religious", "ethnic"],
            "club": ["club", "clubbing", "nightout", "dancing"],
            "date": ["date", "romantic", "dinner", "movie"]
        }
        
        for occasion, keywords in occasion_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                logger.info(f"[FashionOntology] Inferred occasion '{occasion}' from query: {query}")
                return occasion
        
        return None
    
    def infer_cultural_context(self, query: str, user_location: str = "india") -> str:
        """Infer cultural context from query and user location"""
        query_lower = query.lower()
        
        indian_keywords = ["saree", "kurti", "lehenga", "traditional", "ethnic", "indian", "desi"]
        western_keywords = ["dress", "western", "modern", "contemporary"]
        
        if any(keyword in query_lower for keyword in indian_keywords):
            return "indian"
        elif any(keyword in query_lower for keyword in western_keywords):
            return "western"
        elif user_location.lower() == "india":
            return "indian"  # Default for Indian users
        else:
            return "western"
    
    def get_formality_score(self, category: str) -> int:
        """Get formality score for a category (1-10)"""
        return self.formality_levels.get(category.lower(), 5)  # Default to medium formality
    
    def suggest_alternatives(self, blocked_category: str, occasion: str, gender: str) -> List[str]:
        """Suggest appropriate alternatives when a category is blocked"""
        appropriate, _ = self.get_appropriate_categories(occasion, gender)
        
        # Return top 3 alternatives
        return appropriate[:3] if appropriate else []
    
    def validate_outfit_coherence(self, categories: List[str], occasion: str) -> Dict[str, Any]:
        """Validate if selected categories make a coherent outfit"""
        template = self.get_outfit_template(occasion, "women")  # Default to women for now
        
        if not template:
            return {"coherent": True, "issues": [], "suggestions": []}
        
        issues = []
        suggestions = []
        
        # Check if primary categories are present
        primary_present = any(
            any(cat.lower() in item.lower() for cat in categories)
            for item in template.get("primary", [])
        )
        
        if not primary_present:
            issues.append("Missing primary garment")
            suggestions.extend(template.get("primary", [])[:2])
        
        # Check for blocked items
        avoid_items = template.get("avoid", [])
        for category in categories:
            if any(avoid.lower() in category.lower() for avoid in avoid_items):
                issues.append(f"Inappropriate item: {category}")
        
        return {
            "coherent": len(issues) == 0,
            "issues": issues,
            "suggestions": suggestions
        }


# Singleton instance
fashion_ontology = FashionOntology()