"""
Outfit Stylist Agent - Fashion Intelligence Layer

This agent provides the missing fashion reasoning that converts user intent into outfit ideas.
It prevents disasters like suggesting socks for party wear by understanding:
1. Occasion appropriateness
2. Outfit composition
3. Cultural context
4. Fashion logic

This is the intermediate reasoning layer between user intent and product search.
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
from app.services.fashion.fashion_ontology import fashion_ontology, OccasionType, Gender, CulturalContext

logger = logging.getLogger(__name__)


class OutfitStyler:
    """
    Fashion intelligence agent that converts user intent into outfit recommendations.
    
    This agent acts as the fashion brain that understands:
    - What people wear for different occasions
    - How to compose complete outfits
    - Cultural appropriateness
    - Style coherence
    """
    
    def __init__(self):
        self.fashion_knowledge = fashion_ontology
    
    def analyze_fashion_intent(
        self,
        user_query: str,
        user_memory: Dict[str, Any],
        conversation_history: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze user query to extract fashion intent with domain intelligence.
        
        This is the core method that prevents fashion disasters by understanding context.
        """
        logger.info(f"[OutfitStyler] Analyzing fashion intent: {user_query}")
        
        # Extract basic info
        gender = self._extract_gender(user_memory, user_query)
        occasion = self._extract_occasion(user_query, conversation_history)
        cultural_context = self._extract_cultural_context(user_query, user_memory)
        emotional_state = self._extract_emotional_state(user_query)
        
        # Build fashion context
        fashion_context = {
            "gender": gender,
            "occasion": occasion,
            "cultural_context": cultural_context,
            "emotional_state": emotional_state,
            "formality_level": self._determine_formality_level(occasion, user_query),
            "confidence_level": self._assess_confidence_needs(user_query, emotional_state),
            "style_preference": user_memory.get("long_term", {}).get("style_vibe", "casual")
        }
        
        logger.info(f"[OutfitStyler] Fashion context: {fashion_context}")
        
        return fashion_context
    
    def generate_outfit_strategy(
        self,
        fashion_context: Dict[str, Any],
        user_query: str
    ) -> Dict[str, Any]:
        """
        Generate outfit strategy based on fashion context.
        
        This prevents random product suggestions by creating a coherent outfit plan.
        """
        gender = fashion_context.get("gender", "women")
        occasion = fashion_context.get("occasion", "casual")
        cultural_context = fashion_context.get("cultural_context", "indian")
        
        # Get appropriate categories
        appropriate_categories, blocked_categories = self.fashion_knowledge.get_appropriate_categories(
            occasion=occasion,
            gender=gender,
            cultural_context=cultural_context
        )
        
        # Get outfit template
        outfit_template = self.fashion_knowledge.get_outfit_template(
            occasion=occasion,
            gender=gender,
            cultural_context=cultural_context
        )
        
        # Generate outfit ideas
        outfit_ideas = self._generate_outfit_ideas(
            appropriate_categories, 
            outfit_template, 
            fashion_context
        )
        
        # Create search strategy
        search_strategy = {
            "primary_categories": outfit_template.get("primary", appropriate_categories[:3]),
            "secondary_categories": outfit_template.get("secondary", []),
            "blocked_categories": blocked_categories,
            "outfit_ideas": outfit_ideas,
            "search_priority": "outfit_first",  # Search for complete outfits, not random items
            "cultural_bias": cultural_context,
            "formality_filter": fashion_context.get("formality_level", "medium")
        }
        
        logger.info(f"[OutfitStyler] Generated strategy: {len(outfit_ideas)} outfit ideas, "
                   f"{len(blocked_categories)} blocked categories")
        
        return search_strategy
    
    def _extract_gender(self, user_memory: Dict[str, Any], user_query: str) -> str:
        """Extract gender with memory persistence"""
        # Check memory first
        memory_gender = user_memory.get("long_term", {}).get("gender")
        if memory_gender:
            # Map search system gender format to fashion ontology format
            gender_mapping = {
                "male": "men",
                "female": "women", 
                "unisex": "unisex",
                # Also handle if already in correct format
                "men": "men",
                "women": "women"
            }
            return gender_mapping.get(memory_gender.lower(), "women")
        
        # Check query
        query_lower = user_query.lower()
        if any(word in query_lower for word in ["woman", "women", "girl", "female", "she", "her"]):
            return "women"
        elif any(word in query_lower for word in ["man", "men", "boy", "male", "he", "his"]):
            return "men"
        
        # Default based on common patterns (can be improved)
        return "women"  # Default assumption for Indian context
    
    def _extract_occasion(self, user_query: str, conversation_history: List[Dict[str, Any]] = None) -> str:
        """Extract occasion with context awareness"""
        # Direct occasion inference
        occasion = self.fashion_knowledge.infer_occasion_from_query(user_query)
        if occasion:
            return occasion
        
        # Check conversation history for context
        if conversation_history:
            for msg in reversed(conversation_history[-5:]):  # Last 5 messages
                content = msg.get("content", "").lower()
                occasion = self.fashion_knowledge.infer_occasion_from_query(content)
                if occasion:
                    logger.info(f"[OutfitStyler] Found occasion '{occasion}' in conversation history")
                    return occasion
        
        # Default to casual if no clear occasion
        return "casual"
    
    def _extract_cultural_context(self, user_query: str, user_memory: Dict[str, Any]) -> str:
        """Extract cultural context"""
        # Check user location/preferences
        user_location = user_memory.get("long_term", {}).get("location", "india")
        
        return self.fashion_knowledge.infer_cultural_context(user_query, user_location)
    
    def _extract_emotional_state(self, user_query: str) -> str:
        """Extract emotional state from query"""
        query_lower = user_query.lower()
        
        confusion_indicators = ["idk", "don't know", "confused", "help", "not sure", "what to wear"]
        excitement_indicators = ["excited", "can't wait", "love", "amazing"]
        stress_indicators = ["urgent", "need", "asap", "quickly", "stressed"]
        
        if any(indicator in query_lower for indicator in confusion_indicators):
            return "confused"
        elif any(indicator in query_lower for indicator in excitement_indicators):
            return "excited"
        elif any(indicator in query_lower for indicator in stress_indicators):
            return "stressed"
        else:
            return "neutral"
    
    def _determine_formality_level(self, occasion: str, user_query: str) -> str:
        """Determine formality level"""
        formality_map = {
            "wedding": "very_formal",
            "office": "formal",
            "interview": "very_formal",
            "party": "semi_formal",
            "casual": "casual",
            "club": "semi_formal"
        }
        
        return formality_map.get(occasion, "casual")
    
    def _assess_confidence_needs(self, user_query: str, emotional_state: str) -> str:
        """Assess user's confidence needs"""
        if emotional_state == "confused":
            return "needs_guidance"
        elif "confident" in user_query.lower():
            return "high_confidence"
        elif "comfortable" in user_query.lower():
            return "comfort_focused"
        else:
            return "moderate"
    
    def _generate_outfit_ideas(
        self,
        appropriate_categories: List[str],
        outfit_template: Dict[str, List[str]],
        fashion_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate specific outfit ideas"""
        outfit_ideas = []
        
        occasion = fashion_context.get("occasion", "casual")
        cultural_context = fashion_context.get("cultural_context", "indian")
        gender = fashion_context.get("gender", "women")
        
        # Generate outfit ideas based on occasion and culture
        if occasion == "party" and gender == "women":
            if cultural_context == "indian":
                outfit_ideas = [
                    {
                        "name": "Elegant Saree Look",
                        "description": "Classic saree with matching blouse and accessories",
                        "primary_items": ["saree", "blouse"],
                        "accessories": ["ethnic jewelry", "ethnic footwear", "clutch"],
                        "vibe": "traditional elegance",
                        "confidence_level": "high"
                    },
                    {
                        "name": "Designer Lehenga Set",
                        "description": "Stunning lehenga with crop top and dupatta",
                        "primary_items": ["lehenga", "crop top", "dupatta"],
                        "accessories": ["ethnic jewelry", "ethnic footwear"],
                        "vibe": "festive glamour",
                        "confidence_level": "very_high"
                    },
                    {
                        "name": "Indo-Western Fusion",
                        "description": "Modern kurti with palazzo and jacket",
                        "primary_items": ["kurti", "palazzo", "jacket"],
                        "accessories": ["fusion jewelry", "heels"],
                        "vibe": "contemporary chic",
                        "confidence_level": "moderate"
                    }
                ]
            else:  # Western context
                outfit_ideas = [
                    {
                        "name": "Cocktail Dress Look",
                        "description": "Elegant midi or mini dress with heels",
                        "primary_items": ["cocktail dress", "midi dress"],
                        "accessories": ["heels", "clutch", "jewelry"],
                        "vibe": "sophisticated party",
                        "confidence_level": "high"
                    },
                    {
                        "name": "Chic Co-ord Set",
                        "description": "Matching top and bottom set",
                        "primary_items": ["co-ord set", "matching top", "matching bottom"],
                        "accessories": ["heels", "handbag", "jewelry"],
                        "vibe": "trendy coordination",
                        "confidence_level": "moderate"
                    },
                    {
                        "name": "Jumpsuit Elegance",
                        "description": "Stylish jumpsuit with accessories",
                        "primary_items": ["jumpsuit"],
                        "accessories": ["heels", "clutch", "statement jewelry"],
                        "vibe": "modern sophistication",
                        "confidence_level": "high"
                    }
                ]
        
        elif occasion == "office" and gender == "women":
            outfit_ideas = [
                {
                    "name": "Professional Blazer Set",
                    "description": "Blazer with formal pants or skirt",
                    "primary_items": ["blazer", "formal pants", "formal shirt"],
                    "accessories": ["formal shoes", "handbag"],
                    "vibe": "corporate professional",
                    "confidence_level": "high"
                },
                {
                    "name": "Elegant Formal Dress",
                    "description": "Knee-length formal dress",
                    "primary_items": ["formal dress"],
                    "accessories": ["formal shoes", "handbag", "minimal jewelry"],
                    "vibe": "polished elegance",
                    "confidence_level": "moderate"
                }
            ]
        
        elif occasion == "casual":
            outfit_ideas = [
                {
                    "name": "Comfortable Casual",
                    "description": "Relaxed and comfortable everyday wear",
                    "primary_items": ["kurti", "jeans"] if cultural_context == "indian" else ["t-shirt", "jeans"],
                    "accessories": ["casual shoes", "handbag"],
                    "vibe": "relaxed comfort",
                    "confidence_level": "moderate"
                }
            ]
        
        # Fallback: generate basic outfit ideas from appropriate categories
        if not outfit_ideas and appropriate_categories:
            outfit_ideas = [
                {
                    "name": f"{occasion.title()} Look",
                    "description": f"Stylish {occasion} outfit",
                    "primary_items": appropriate_categories[:2],
                    "accessories": ["shoes", "bag"],
                    "vibe": f"{occasion} appropriate",
                    "confidence_level": "moderate"
                }
            ]
        
        return outfit_ideas
    
    def validate_search_results(
        self,
        products: List[Dict[str, Any]],
        search_strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate search results against fashion logic"""
        blocked_categories = search_strategy.get("blocked_categories", [])
        appropriate_categories = search_strategy.get("primary_categories", [])
        
        valid_products = []
        blocked_products = []
        
        for product in products:
            product_category = product.get("category", "").lower()
            product_name = product.get("name", "").lower()
            
            # Check if product is blocked
            is_blocked = any(
                blocked_cat.lower() in product_category or blocked_cat.lower() in product_name
                for blocked_cat in blocked_categories
            )
            
            if is_blocked:
                blocked_products.append({
                    "product": product,
                    "reason": "Inappropriate for occasion"
                })
            else:
                valid_products.append(product)
        
        validation_result = {
            "valid_products": valid_products,
            "blocked_products": blocked_products,
            "validation_passed": len(blocked_products) == 0,
            "issues": [f"Blocked {len(blocked_products)} inappropriate products"] if blocked_products else []
        }
        
        if blocked_products:
            logger.warning(f"[OutfitStyler] Blocked {len(blocked_products)} inappropriate products")
            for blocked in blocked_products[:3]:  # Log first 3
                logger.warning(f"[OutfitStyler] Blocked: {blocked['product'].get('name')} - {blocked['reason']}")
        
        return validation_result
    
    def generate_outfit_explanation(
        self,
        outfit_idea: Dict[str, Any],
        fashion_context: Dict[str, Any]
    ) -> str:
        """Generate explanation for why this outfit works"""
        occasion = fashion_context.get("occasion", "casual")
        cultural_context = fashion_context.get("cultural_context", "indian")
        emotional_state = fashion_context.get("emotional_state", "neutral")
        
        outfit_name = outfit_idea.get("name", "Outfit")
        vibe = outfit_idea.get("vibe", "stylish")
        
        explanations = []
        
        # Occasion appropriateness
        explanations.append(f"Perfect for {occasion} occasions")
        
        # Cultural context
        if cultural_context == "indian":
            explanations.append("Respects Indian fashion sensibilities")
        
        # Emotional support
        if emotional_state == "confused":
            explanations.append("Easy to style and wear confidently")
        
        # Style coherence
        explanations.append(f"Creates a {vibe} look")
        
        return f"{outfit_name}: {', '.join(explanations)}"


# Singleton instance
outfit_stylist = OutfitStyler()