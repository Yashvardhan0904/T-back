"""
Property-Based Tests for Mandatory Gender Filtering

**Feature: intelligent-chat-recommendations, Property 1: Mandatory Gender Filtering**
**Validates: Requirements 1.1, 1.2, 1.3, 1.4**

Property 1: Mandatory Gender Filtering
For any search query with a specified gender preference, the Search_Agent should apply the gender as a mandatory filter 
and never return products from the opposite gender, even when no results are found.
"""

import pytest
import asyncio
from hypothesis import given, strategies as st, settings
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.agents.search_agent import SearchAgent
from app.services.search.gender_filter import gender_filter, Gender, GenderInferenceConfidence


# Test data generators
@st.composite
def gender_preference(draw):
    """Generate a gender preference"""
    genders = ["male", "female", "unisex", "men", "women", "mens", "womens"]
    return draw(st.sampled_from(genders))


@st.composite
def search_query_with_gender(draw):
    """Generate search query with gender context"""
    base_queries = ["activewear", "jeans", "t-shirt", "sneakers", "jacket", "hoodie"]
    gender_terms = ["mens", "womens", "male", "female", "men", "women"]
    
    base_query = draw(st.sampled_from(base_queries))
    gender_term = draw(st.sampled_from(gender_terms))
    
    # Sometimes put gender first, sometimes after
    if draw(st.booleans()):
        query = f"{gender_term} {base_query}"
    else:
        query = f"{base_query} for {gender_term}"
    
    return query, gender_term


@st.composite
def product_with_gender(draw):
    """Generate a product with gender field"""
    genders = ["Male", "Female", "Unisex", "Men", "Women", "male", "female"]
    names = ["Test Shirt", "Sample Jeans", "Demo Sneakers", "Example Hoodie"]
    
    return {
        "_id": draw(st.text(min_size=24, max_size=24, alphabet="0123456789abcdef")),
        "name": draw(st.sampled_from(names)),
        "gender": draw(st.sampled_from(genders)),
        "category": "clothing",
        "price": draw(st.floats(min_value=100, max_value=5000)),
        "brand": "TestBrand",
        "isActive": True,
        "isApproved": True,
        "isVisible": True
    }


@st.composite
def mixed_gender_product_list(draw):
    """Generate a list of products with mixed genders"""
    products = draw(st.lists(product_with_gender(), min_size=5, max_size=20))
    return products


class TestMandatoryGenderFiltering:
    """Property-based tests for mandatory gender filtering"""
    
    @pytest.fixture
    def search_agent(self):
        """Create search agent instance for testing"""
        return SearchAgent()
    
    @pytest.fixture
    def mock_database(self):
        """Mock database for testing"""
        with patch('app.services.agents.search_agent.get_database') as mock_db:
            yield mock_db
    
    @given(gender_preference(), st.text(min_size=1, max_size=50))
    @settings(max_examples=100, deadline=10000)
    async def test_property_gender_preference_applied_as_mandatory_filter(
        self,
        search_agent: SearchAgent,
        mock_database,
        gender_pref: str,
        query: str
    ):
        """
        Property Test: Gender preference should be applied as mandatory filter
        
        **Validates: Requirement 1.1**
        WHEN a user specifies a gender preference, THE Search_Agent SHALL apply the gender as a mandatory filter
        """
        # Mock database response
        mock_collection = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[])
        mock_collection.find.return_value = mock_cursor
        mock_database.return_value.products = mock_collection
        
        # Mock search orchestrator to avoid external dependencies
        with patch('app.services.agents.search_agent.search_orchestrator') as mock_orchestrator:
            mock_orchestrator.hybrid_search = AsyncMock(return_value=[])
            
            # Perform search with gender preference
            filters = {"gender": gender_pref}
            result = await search_agent.search_products(
                query=query,
                filters=filters,
                limit=10
            )
            
            # Verify gender filter was applied
            search_authority = result.get("search_authority", {})
            assert search_authority.get("gender_filter_applied") is True, \
                f"Gender filter should be applied when gender preference '{gender_pref}' is specified"
            
            assert search_authority.get("gender_constraint") is not None, \
                f"Gender constraint should be set when gender preference '{gender_pref}' is specified"
    
    @given(mixed_gender_product_list(), gender_preference())
    @settings(max_examples=50, deadline=10000)
    async def test_property_never_return_opposite_gender_products(
        self,
        search_agent: SearchAgent,
        mock_database,
        products: List[Dict[str, Any]],
        gender_pref: str
    ):
        """
        Property Test: Never return products from opposite gender
        
        **Validates: Requirement 1.2**
        WHEN no products match the user's gender preference, THE Search_Agent SHALL return no results 
        rather than fallback to opposite gender products
        """
        # Mock database to return mixed gender products
        mock_collection = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=products)
        mock_collection.find.return_value = mock_cursor
        mock_database.return_value.products = mock_collection
        
        # Perform search with gender preference
        filters = {"gender": gender_pref}
        result = await search_agent.search_products(
            query="test query",
            filters=filters,
            limit=20
        )
        
        returned_products = result.get("products", [])
        
        # Map gender preference to expected gender values
        expected_genders = []
        gender_lower = gender_pref.lower()
        if gender_lower in ["male", "men", "mens"]:
            expected_genders = ["male", "men", "mens", "masculine"]
        elif gender_lower in ["female", "women", "womens"]:
            expected_genders = ["female", "women", "womens", "feminine", "ladies"]
        elif gender_lower in ["unisex", "neutral"]:
            expected_genders = ["unisex", "gender neutral", "both", "neutral"]
        
        # Verify no opposite gender products are returned
        for product in returned_products:
            product_gender = product.get("attributes", {}).get("gender", "").lower()
            
            # If we have a gender constraint, verify compliance
            if expected_genders:
                is_compliant = any(expected in product_gender for expected in expected_genders)
                assert is_compliant, \
                    f"Product with gender '{product_gender}' should not be returned when user requested '{gender_pref}'"
    
    @given(search_query_with_gender())
    @settings(max_examples=50, deadline=10000)
    async def test_property_gender_inference_from_query(
        self,
        search_agent: SearchAgent,
        mock_database,
        query_data: tuple
    ):
        """
        Property Test: Gender should be inferred from product categories
        
        **Validates: Requirement 1.4**
        IF a user requests a gender-specific item, THE Search_Agent SHALL infer and apply the appropriate gender filter
        """
        query, gender_term = query_data
        
        # Mock database response
        mock_collection = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[])
        mock_collection.find.return_value = mock_cursor
        mock_database.return_value.products = mock_collection
        
        # Mock search orchestrator
        with patch('app.services.agents.search_agent.search_orchestrator') as mock_orchestrator:
            mock_orchestrator.hybrid_search = AsyncMock(return_value=[])
            
            # Perform search without explicit gender filter (should infer from query)
            result = await search_agent.search_products(
                query=query,
                filters={},
                limit=10
            )
            
            # Check if gender was inferred and applied
            search_authority = result.get("search_authority", {})
            
            # If the query contains explicit gender terms, filter should be applied
            if any(term in query.lower() for term in ["mens", "womens", "male", "female", "men", "women"]):
                assert search_authority.get("gender_filter_applied") is True, \
                    f"Gender filter should be inferred and applied for query '{query}'"
                
                assert search_authority.get("gender_constraint") is not None, \
                    f"Gender constraint should be inferred from query '{query}'"
    
    @given(st.lists(st.text(min_size=1, max_size=20), min_size=1, max_size=10))
    @settings(max_examples=30, deadline=10000)
    async def test_property_gender_filter_logging_and_verification(
        self,
        search_agent: SearchAgent,
        mock_database,
        queries: List[str]
    ):
        """
        Property Test: Gender filter application should be logged for debugging
        
        **Validates: Requirement 1.5**
        WHEN gender filters are applied, THE Search_Agent SHALL log the filter application for debugging and verification
        """
        # Mock database response
        mock_collection = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[])
        mock_collection.find.return_value = mock_cursor
        mock_database.return_value.products = mock_collection
        
        # Mock search orchestrator
        with patch('app.services.agents.search_agent.search_orchestrator') as mock_orchestrator:
            mock_orchestrator.hybrid_search = AsyncMock(return_value=[])
            
            # Capture log messages
            with patch('app.services.agents.search_agent.logger') as mock_logger:
                for query in queries:
                    # Test with explicit gender preference
                    await search_agent.search_products(
                        query=query,
                        filters={"gender": "male"},
                        limit=10
                    )
                
                # Verify logging occurred
                # Should have at least one log call for gender filter application
                log_calls = mock_logger.info.call_args_list
                gender_log_found = any(
                    "gender filter" in str(call).lower() or "mandatory" in str(call).lower()
                    for call in log_calls
                )
                
                assert gender_log_found, \
                    "Gender filter application should be logged for debugging and verification"
    
    @given(gender_preference())
    @settings(max_examples=30, deadline=10000)
    async def test_property_user_profile_gender_automatically_applied(
        self,
        search_agent: SearchAgent,
        mock_database,
        stored_gender: str
    ):
        """
        Property Test: Stored gender preference should be automatically applied
        
        **Validates: Requirement 1.3**
        WHEN a user's gender preference is stored in User_Profile, THE Search_Agent SHALL automatically apply this filter
        """
        # Mock database response
        mock_collection = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=[])
        mock_collection.find.return_value = mock_cursor
        mock_database.return_value.products = mock_collection
        
        # Mock search orchestrator
        with patch('app.services.agents.search_agent.search_orchestrator') as mock_orchestrator:
            mock_orchestrator.hybrid_search = AsyncMock(return_value=[])
            
            # Simulate stored gender preference in filters
            filters = {"gender": stored_gender}
            
            result = await search_agent.search_products(
                query="casual wear",
                filters=filters,
                limit=10
            )
            
            # Verify gender filter was applied from stored preference
            search_authority = result.get("search_authority", {})
            assert search_authority.get("gender_filter_applied") is True, \
                f"Stored gender preference '{stored_gender}' should be automatically applied"
            
            assert search_authority.get("gender_constraint") == stored_gender, \
                f"Gender constraint should match stored preference '{stored_gender}'"
    
    @given(st.lists(product_with_gender(), min_size=0, max_size=5))
    @settings(max_examples=20, deadline=10000)
    async def test_property_no_results_better_than_wrong_gender(
        self,
        search_agent: SearchAgent,
        mock_database,
        products: List[Dict[str, Any]]
    ):
        """
        Property Test: No results is better than wrong gender results
        
        **Validates: Core principle of mandatory filtering**
        The system should return empty results rather than products that don't match gender constraints
        """
        # Create products that are all female when user wants male
        female_products = []
        for product in products:
            female_product = product.copy()
            female_product["gender"] = "Female"
            female_products.append(female_product)
        
        # Mock database to return only female products
        mock_collection = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.to_list = AsyncMock(return_value=female_products)
        mock_collection.find.return_value = mock_cursor
        mock_database.return_value.products = mock_collection
        
        # Search for male products
        result = await search_agent.search_products(
            query="male clothing",
            filters={"gender": "male"},
            limit=10
        )
        
        returned_products = result.get("products", [])
        
        # Should return no products rather than wrong gender products
        for product in returned_products:
            product_gender = product.get("attributes", {}).get("gender", "").lower()
            assert "female" not in product_gender and "women" not in product_gender, \
                f"Should not return female products when user requested male products"
    
    def test_gender_inference_confidence_levels(self):
        """
        Test gender inference with different confidence levels
        """
        # High confidence terms
        high_confidence_cases = [
            ("saree for wedding", Gender.FEMALE, GenderInferenceConfidence.HIGH),
            ("mens formal shirt", Gender.MALE, GenderInferenceConfidence.HIGH),
            ("unisex sneakers", Gender.UNISEX, GenderInferenceConfidence.HIGH)
        ]
        
        for query, expected_gender, expected_confidence in high_confidence_cases:
            inferred_gender, confidence = gender_filter.infer_gender_from_query(query)
            
            assert inferred_gender == expected_gender, \
                f"Should infer {expected_gender.value} from '{query}', got {inferred_gender}"
            
            assert confidence == expected_confidence, \
                f"Should have {expected_confidence.value} confidence for '{query}', got {confidence}"


# Async test runner
@pytest.mark.asyncio
class TestAsyncMandatoryGenderFiltering(TestMandatoryGenderFiltering):
    """Async wrapper for property tests"""
    pass


if __name__ == "__main__":
    # Run property tests
    pytest.main([__file__, "-v", "--tb=short"])