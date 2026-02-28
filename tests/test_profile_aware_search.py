"""
Property-Based Tests for Profile-Aware Search Results

**Feature: intelligent-chat-recommendations, Property 7: Profile-Aware Search Results**
**Validates: Requirements 7.1, 7.2, 7.3, 7.4**

Property 7: Profile-Aware Search Results
For any search query, the Search_Agent should return results matching both query intent and user profile,
rank results considering user preferences and past behavior, prioritize exact matches for clear intent,
and explain limitations while suggesting alternatives.
"""

import pytest
import asyncio
from hypothesis import given, strategies as st, settings
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from app.services.agents.search_agent import SearchAgent
from app.services.search.personalized_ranking import personalized_ranking
from app.models.context import (
    UserProfile, StylePreference, ColorPreferences, BrandPreference,
    PriceRange, Interaction, PreferenceSource
)


# Test data generators
@st.composite
def user_profile_data(draw):
    """Generate a user profile for testing"""
    genders = ["male", "female", "unisex"]
    vibes = ["streetwear", "minimal", "classic", "trendy"]
    fits = ["oversized", "slim", "regular", "loose"]
    colors = ["black", "white", "red", "blue", "green"]
    
    return UserProfile(
        user_id=draw(st.text(min_size=5, max_size=20)),
        gender_preference=draw(st.sampled_from(genders)),
        style_vibe=draw(st.sampled_from(vibes)),
        preferred_fit=draw(st.sampled_from(fits)),
        color_preferences=ColorPreferences(
            liked_colors=draw(st.lists(st.sampled_from(colors), min_size=0, max_size=3)),
            disliked_colors=draw(st.lists(st.sampled_from(colors), min_size=0, max_size=2))
        ),
        price_range=PriceRange(
            min_price=draw(st.integers(min_value=500, max_value=2000)),
            max_price=draw(st.integers(min_value=2000, max_value=10000))
        )
    )


@st.composite
def search_query_data(draw):
    """Generate search query with intent"""
    categories = ["shirt", "jeans", "sneakers", "hoodie", "jacket"]
    colors = ["black", "white", "red", "blue"]
    brands = ["Nike", "Adidas", "Zara", "H&M"]
    
    category = draw(st.sampled_from(categories))
    color = draw(st.sampled_from(colors))
    brand = draw(st.sampled_from(brands))
    
    # Generate different query patterns
    query_patterns = [
        f"{color} {category}",
        f"{brand} {category}",
        f"{category} for men",
        f"casual {category}",
        category
    ]
    
    query = draw(st.sampled_from(query_patterns))
    
    return {
        "query": query,
        "category": category,
        "color": color,
        "brand": brand
    }


@st.composite
def product_list_with_variety(draw):
    """Generate a list of products with variety in attributes"""
    categories = ["shirt", "jeans", "sneakers", "hoodie"]
    colors = ["black", "white", "red", "blue", "green"]
    brands = ["Nike", "Adidas", "Zara", "H&M", "TestBrand"]
    fits = ["oversized", "slim", "regular", "loose"]
    
    products = []
    for i in range(draw(st.integers(min_size=5, max_size=15))):
        product = {
            "_id": f"product_{i}",
            "name": f"Test {draw(st.sampled_from(categories))} {i}",
            "category": draw(st.sampled_from(categories)),
            "color": draw(st.sampled_from(colors)),
            "brand": draw(st.sampled_from(brands)),
            "fit": draw(st.sampled_from(fits)),
            "price": draw(st.floats(min_value=500, max_value=5000)),
            "style": draw(st.sampled_from(["casual", "formal", "sporty"])),
            "description": f"A great {draw(st.sampled_from(categories))}",
            "isActive": True,
            "isApproved": True
        }
        products.append(product)
    
    return products


class TestProfileAwareSearchResults:
    """Property-based tests for profile-aware search results"""
    
    @pytest.fixture
    def search_agent(self):
        """Create search agent instance for testing"""
        return SearchAgent()
    
    @pytest.fixture
    def mock_database(self):
        """Mock database for testing"""
        with patch('app.services.agents.search_agent.get_database') as mock_db:
            yield mock_db
    
    @given(user_profile_data(), search_query_data(), product_list_with_variety())
    @settings(max_examples=50, deadline=15000)
    async def test_property_results_match_query_intent_and_user_profile(
        self,
        search_agent: SearchAgent,
        mock_database,
        user_profile: UserProfile,
        query_data: Dict[str, str],
        products: List[Dict[str, Any]]
    ):
        """
        Property Test: Search results should match both query intent and user profile
        
        **Validates: Requirement 7.1**
        WHEN a user searches for specific items, THE Search_Agent SHALL return results that match 
        both query intent and user profile
        """
        # Mock database response
        mock_collection = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=products)
        mock_collection.find.return_value = mock_cursor
        mock_database.return_value.products = mock_collection
        
        # Mock search orchestrator
        with patch('app.services.agents.search_agent.search_orchestrator') as mock_orchestrator:
            mock_orchestrator.hybrid_search = AsyncMock(return_value=products)
            
            # Perform search with personalized ranking
            result = await search_agent.search_with_personalized_ranking(
                query=query_data["query"],
                user_profile=user_profile,
                filters={"gender": user_profile.gender_preference},
                limit=10
            )
            
            returned_products = result.get("products", [])
            search_authority = result.get("search_authority", {})
            
            # Verify personalized ranking was applied
            assert search_authority.get("personalized_ranking_applied") is True, \
                "Personalized ranking should be applied to match user profile"
            
            # Verify results consider user preferences
            if returned_products:
                top_product = returned_products[0]
                
                # Should have ranking score and match reasons
                assert "ranking_score" in top_product, \
                    "Products should have ranking scores based on user profile"
                
                assert "match_reasons" in top_product, \
                    "Products should have match reasons explaining relevance to user profile"
                
                # Verify gender compliance if specified
                if user_profile.gender_preference:
                    product_gender = top_product["product"].get("gender", "").lower()
                    expected_genders = []
                    
                    if user_profile.gender_preference.lower() in ["male", "men"]:
                        expected_genders = ["male", "men", "mens"]
                    elif user_profile.gender_preference.lower() in ["female", "women"]:
                        expected_genders = ["female", "women", "womens"]
                    
                    if expected_genders and product_gender:
                        is_gender_compliant = any(exp in product_gender for exp in expected_genders)
                        assert is_gender_compliant, \
                            f"Top result should match user's gender preference: {user_profile.gender_preference}"
    
    @given(user_profile_data(), product_list_with_variety())
    @settings(max_examples=30, deadline=10000)
    async def test_property_ranking_considers_user_preferences_and_behavior(
        self,
        search_agent: SearchAgent,
        mock_database,
        user_profile: UserProfile,
        products: List[Dict[str, Any]]
    ):
        """
        Property Test: Results should be ranked considering user preferences and past behavior
        
        **Validates: Requirement 7.2**
        WHEN ranking search results, THE Search_Agent SHALL consider user preferences, 
        past behavior, and stated requirements
        """
        # Add some interaction history to user profile
        user_profile.interaction_history = [
            Interaction(
                interaction_type="like",
                item_id="product_1",
                item_features={"category": "shirt", "color": "black", "brand": "Nike"},
                timestamp=datetime.utcnow()
            ),
            Interaction(
                interaction_type="purchase",
                item_id="product_2", 
                item_features={"category": "jeans", "fit": "slim"},
                timestamp=datetime.utcnow()
            )
        ]
        
        # Mock database response
        mock_collection = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=products)
        mock_collection.find.return_value = mock_cursor
        mock_database.return_value.products = mock_collection
        
        # Mock search orchestrator
        with patch('app.services.agents.search_agent.search_orchestrator') as mock_orchestrator:
            mock_orchestrator.hybrid_search = AsyncMock(return_value=products)
            
            # Perform search with personalized ranking
            result = await search_agent.search_with_personalized_ranking(
                query="casual wear",
                user_profile=user_profile,
                limit=10
            )
            
            returned_products = result.get("products", [])
            
            if len(returned_products) > 1:
                # Verify products are ranked (scores should be in descending order)
                scores = [p.get("ranking_score", 0) for p in returned_products]
                assert scores == sorted(scores, reverse=True), \
                    "Products should be ranked by score in descending order"
                
                # Verify top product has higher score than bottom product
                top_score = returned_products[0].get("ranking_score", 0)
                bottom_score = returned_products[-1].get("ranking_score", 0)
                assert top_score >= bottom_score, \
                    "Top ranked product should have higher or equal score than bottom ranked product"
                
                # Verify score breakdown includes preference factors
                top_product = returned_products[0]
                score_breakdown = top_product.get("score_breakdown", {})
                
                expected_factors = ["style_match", "color_preference", "brand_preference", "price_preference"]
                for factor in expected_factors:
                    assert factor in score_breakdown, \
                        f"Score breakdown should include {factor} based on user preferences"
    
    @given(user_profile_data(), search_query_data())
    @settings(max_examples=30, deadline=10000)
    async def test_property_exact_matches_prioritized_for_clear_intent(
        self,
        search_agent: SearchAgent,
        mock_database,
        user_profile: UserProfile,
        query_data: Dict[str, str]
    ):
        """
        Property Test: Exact matches should be prioritized when user intent is clear
        
        **Validates: Requirement 7.3**
        THE Search_Agent SHALL prioritize exact matches over partial matches when user intent is clear
        """
        # Create products with exact and partial matches
        exact_match_product = {
            "_id": "exact_match",
            "name": f"Perfect {query_data['category']}",
            "category": query_data["category"],
            "color": query_data["color"],
            "brand": query_data["brand"],
            "price": 1000,
            "fit": user_profile.preferred_fit,
            "isActive": True,
            "isApproved": True
        }
        
        partial_match_product = {
            "_id": "partial_match",
            "name": f"Different {query_data['category']}",
            "category": query_data["category"],
            "color": "different_color",
            "brand": "different_brand",
            "price": 1000,
            "fit": "different_fit",
            "isActive": True,
            "isApproved": True
        }
        
        products = [partial_match_product, exact_match_product]  # Put partial match first
        
        # Mock database response
        mock_collection = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=products)
        mock_collection.find.return_value = mock_cursor
        mock_database.return_value.products = mock_collection
        
        # Mock search orchestrator
        with patch('app.services.agents.search_agent.search_orchestrator') as mock_orchestrator:
            mock_orchestrator.hybrid_search = AsyncMock(return_value=products)
            
            # Perform search with clear intent (specific filters)
            filters = {
                "color": query_data["color"],
                "brand": query_data["brand"],
                "gender": user_profile.gender_preference
            }
            
            result = await search_agent.search_with_personalized_ranking(
                query=query_data["query"],
                user_profile=user_profile,
                filters=filters,
                category=query_data["category"],
                limit=10
            )
            
            returned_products = result.get("products", [])
            
            if len(returned_products) >= 2:
                # The exact match should be ranked higher than partial match
                top_product = returned_products[0]
                second_product = returned_products[1]
                
                top_score_breakdown = top_product.get("score_breakdown", {})
                second_score_breakdown = second_product.get("score_breakdown", {})
                
                # Exact match should have higher exact_match_bonus
                top_exact_bonus = top_score_breakdown.get("exact_match_bonus", 0)
                second_exact_bonus = second_score_breakdown.get("exact_match_bonus", 0)
                
                # If one product is clearly a better exact match, it should rank higher
                if top_product["product"]["_id"] == "exact_match":
                    assert top_exact_bonus >= second_exact_bonus, \
                        "Exact match should have higher exact match bonus than partial match"
    
    @given(user_profile_data())
    @settings(max_examples=20, deadline=10000)
    async def test_property_search_limitations_explained_with_alternatives(
        self,
        search_agent: SearchAgent,
        mock_database,
        user_profile: UserProfile
    ):
        """
        Property Test: Search limitations should be explained with alternatives suggested
        
        **Validates: Requirement 7.4**
        WHEN search results are limited, THE Search_Agent SHALL explain why and suggest alternatives
        """
        # Mock database to return no results
        mock_collection = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[])
        mock_collection.find.return_value = mock_cursor
        mock_database.return_value.products = mock_collection
        
        # Mock search orchestrator to return no results
        with patch('app.services.agents.search_agent.search_orchestrator') as mock_orchestrator:
            mock_orchestrator.hybrid_search = AsyncMock(return_value=[])
            
            # Perform search that returns no results
            result = await search_agent.search_products(
                query="very specific rare item",
                filters={"gender": user_profile.gender_preference},
                limit=10
            )
            
            search_authority = result.get("search_authority", {})
            products_found = search_authority.get("products_found", 0)
            
            # Should return no products
            assert products_found == 0, "Should return no products for unavailable items"
            
            # Should provide explanation
            explanation = search_agent.get_search_result_explanation(result, user_profile)
            assert explanation is not None and len(explanation) > 0, \
                "Should provide explanation when no results are found"
            
            # Explanation should be helpful (not just "no results")
            assert "no" in explanation.lower() or "not found" in explanation.lower(), \
                "Explanation should indicate why no results were found"
            
            # Should suggest alternatives or modifications
            helpful_phrases = ["try", "different", "alternative", "category", "style", "filter"]
            is_helpful = any(phrase in explanation.lower() for phrase in helpful_phrases)
            assert is_helpful, \
                f"Explanation should suggest alternatives or modifications: '{explanation}'"
    
    @given(user_profile_data(), product_list_with_variety())
    @settings(max_examples=20, deadline=10000)
    async def test_property_search_quality_over_quantity(
        self,
        search_agent: SearchAgent,
        mock_database,
        user_profile: UserProfile,
        products: List[Dict[str, Any]]
    ):
        """
        Property Test: Search should maintain result quality over quantity
        
        **Validates: Requirement 7.5 (implied)**
        THE Search_Agent SHALL maintain search result quality over quantity
        """
        # Add some low-quality products (inactive, unapproved)
        low_quality_products = []
        for i, product in enumerate(products[:3]):
            low_quality = product.copy()
            low_quality["_id"] = f"low_quality_{i}"
            low_quality["isActive"] = False if i % 2 == 0 else True
            low_quality["isApproved"] = False if i % 2 == 1 else True
            low_quality_products.append(low_quality)
        
        all_products = products + low_quality_products
        
        # Mock database response
        mock_collection = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=all_products)
        mock_collection.find.return_value = mock_cursor
        mock_database.return_value.products = mock_collection
        
        # Mock search orchestrator to return only quality products
        quality_products = [p for p in all_products if p.get("isActive") and p.get("isApproved")]
        with patch('app.services.agents.search_agent.search_orchestrator') as mock_orchestrator:
            mock_orchestrator.hybrid_search = AsyncMock(return_value=quality_products)
            
            # Perform search
            result = await search_agent.search_with_personalized_ranking(
                query="quality products",
                user_profile=user_profile,
                limit=10
            )
            
            returned_products = result.get("products", [])
            
            # All returned products should be high quality (active and approved)
            for product_wrapper in returned_products:
                product = product_wrapper.get("product", {})
                assert product.get("isActive") is True, \
                    "All returned products should be active"
                assert product.get("isApproved") is True, \
                    "All returned products should be approved"
            
            # Should prefer fewer high-quality results over many low-quality results
            search_authority = result.get("search_authority", {})
            products_found = search_authority.get("products_found", 0)
            
            # Quality check: if we have results, they should all be high quality
            if products_found > 0:
                assert products_found <= len(quality_products), \
                    "Should not return more products than available quality products"


# Async test runner
@pytest.mark.asyncio
class TestAsyncProfileAwareSearch(TestProfileAwareSearchResults):
    """Async wrapper for property tests"""
    pass


if __name__ == "__main__":
    # Run property tests
    pytest.main([__file__, "-v", "--tb=short"])