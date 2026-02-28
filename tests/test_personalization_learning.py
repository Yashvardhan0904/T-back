"""
Property-Based Tests for Personalization Learning

**Property 6: Personalization Learning**
**Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5**

Tests that the system learns from user interactions and improves recommendations over time.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from hypothesis import given, strategies as st, settings
from typing import Dict, List, Any

from app.services.agents.brain_orchestrator import brain_orchestrator
from app.services.agents.memory_agent import memory_agent


# Test data generators
@st.composite
def user_preference_data(draw):
    """Generate user preference data for testing"""
    genders = ["male", "female", "unisex"]
    vibes = ["streetwear", "minimal", "classic", "trendy", "luxury"]
    fits = ["oversized", "slim", "regular", "loose"]
    colors = ["black", "white", "blue", "red", "green", "navy"]
    brands = ["Nike", "Adidas", "Zara", "H&M", "Uniqlo"]
    
    return {
        "user_id": draw(st.text(min_size=5, max_size=20)),
        "email": draw(st.emails()),
        "gender": draw(st.sampled_from(genders)),
        "style_vibe": draw(st.sampled_from(vibes)),
        "preferred_fit": draw(st.sampled_from(fits)),
        "preferred_colors": draw(st.lists(st.sampled_from(colors), min_size=1, max_size=3)),
        "preferred_brands": draw(st.lists(st.sampled_from(brands), min_size=0, max_size=2)),
        "disliked_fits": draw(st.lists(st.sampled_from(fits), min_size=0, max_size=2)),
        "disliked_colors": draw(st.lists(st.sampled_from(colors), min_size=0, max_size=2))
    }


@st.composite
def user_interaction_sequence(draw):
    """Generate a sequence of user interactions for learning"""
    interactions = []
    
    # Initial preference statement
    preference_statements = [
        "I prefer {fit} fits",
        "I like {color} colors",
        "I don't like {disliked_fit}",
        "I love {brand} products",
        "My style is {vibe}"
    ]
    
    fits = ["oversized", "slim", "regular", "loose"]
    colors = ["black", "white", "blue", "red"]
    vibes = ["streetwear", "minimal", "classic"]
    brands = ["Nike", "Adidas", "Zara"]
    
    num_interactions = draw(st.integers(min_value=2, max_value=5))
    
    for i in range(num_interactions):
        statement = draw(st.sampled_from(preference_statements))
        
        if "{fit}" in statement:
            fit = draw(st.sampled_from(fits))
            statement = statement.format(fit=fit)
        elif "{color}" in statement:
            color = draw(st.sampled_from(colors))
            statement = statement.format(color=color)
        elif "{disliked_fit}" in statement:
            disliked_fit = draw(st.sampled_from(fits))
            statement = statement.format(disliked_fit=disliked_fit)
        elif "{brand}" in statement:
            brand = draw(st.sampled_from(brands))
            statement = statement.format(brand=brand)
        elif "{vibe}" in statement:
            vibe = draw(st.sampled_from(vibes))
            statement = statement.format(vibe=vibe)
        
        interactions.append(statement)
    
    return interactions


class TestPersonalizationLearning:
    """Test personalization learning properties"""
    
    @given(user_preference_data())
    @settings(max_examples=50, deadline=10000)
    async def test_property_preferences_persist_across_sessions(
        self,
        user_data: Dict[str, Any]
    ):
        """
        **Property 6.1: Preference Persistence**
        
        GIVEN a user expresses preferences in one session
        WHEN they start a new session
        THEN their preferences should be remembered and applied
        
        **Validates: Requirements 6.1, 6.2**
        """
        with patch('app.services.agents.memory_agent.memory_agent.retrieve_memory') as mock_retrieve, \
             patch('app.services.agents.memory_agent.memory_agent.update_memory') as mock_update, \
             patch('app.services.memory.service.memory_service.get_user_memory') as mock_get_memory, \
             patch('app.services.agents.search_agent.search_agent.search_products') as mock_search:
            
            # Setup: User expresses preferences in first session
            preference_input = f"I prefer {user_data['preferred_fit']} fits and {user_data['style_vibe']} style"
            
            # Mock empty initial memory
            mock_retrieve.return_value = {"long_term": {}, "short_term": {}}
            mock_get_memory.return_value = {"context": {"conversationHistory": []}}
            mock_search.return_value = {"products": [], "search_authority": {"search_status": "EMPTY", "products_found": 0}}
            
            # First session: Express preferences
            await brain_orchestrator.process_request(
                user_input=preference_input,
                user_id=user_data["user_id"],
                email=user_data["email"]
            )
            
            # Verify preferences were saved
            assert mock_update.called, "Preferences should be saved to memory"
            
            # Extract saved preferences from mock calls
            saved_preferences = {}
            for call in mock_update.call_args_list:
                if call[1].get('data'):
                    saved_preferences.update(call[1]['data'])
            
            # Setup: Second session with saved preferences in memory
            mock_retrieve.return_value = {
                "long_term": saved_preferences,
                "short_term": {}
            }
            mock_update.reset_mock()
            
            # Second session: Make a search request
            search_input = "show me some t-shirts"
            
            result = await brain_orchestrator.process_request(
                user_input=search_input,
                user_id=user_data["user_id"],
                email=user_data["email"]
            )
            
            # Verify preferences are applied (system doesn't ask for already known info)
            response = result.get("response", "").lower()
            
            # Should not ask for preferences that were already saved
            if user_data['preferred_fit'] in saved_preferences.get('preferred_fit', ''):
                assert "what fit" not in response, "Should not ask for fit when already known"
            
            if user_data['style_vibe'] in saved_preferences.get('style_vibe', ''):
                assert "what vibe" not in response, "Should not ask for vibe when already known"
    
    @given(user_interaction_sequence())
    @settings(max_examples=30, deadline=10000)
    async def test_property_learning_improves_over_interactions(
        self,
        interactions: List[str]
    ):
        """
        **Property 6.2: Learning Improvement**
        
        GIVEN a user has multiple interactions with preference expressions
        WHEN the system processes each interaction
        THEN the memory should accumulate and refine preferences over time
        
        **Validates: Requirements 6.2, 6.3, 6.4**
        """
        with patch('app.services.agents.memory_agent.memory_agent.retrieve_memory') as mock_retrieve, \
             patch('app.services.agents.memory_agent.memory_agent.update_memory') as mock_update, \
             patch('app.services.memory.service.memory_service.get_user_memory') as mock_get_memory, \
             patch('app.services.agents.search_agent.search_agent.search_products') as mock_search:
            
            user_id = "test_learning_user"
            email = "learning@test.com"
            accumulated_memory = {"long_term": {}, "short_term": {}}
            
            mock_get_memory.return_value = {"context": {"conversationHistory": []}}
            mock_search.return_value = {"products": [], "search_authority": {"search_status": "EMPTY", "products_found": 0}}
            
            memory_updates = []
            
            # Process each interaction
            for i, interaction in enumerate(interactions):
                # Setup current memory state
                mock_retrieve.return_value = accumulated_memory.copy()
                
                # Process interaction
                result = await brain_orchestrator.process_request(
                    user_input=interaction,
                    user_id=user_id,
                    email=email
                )
                
                # Capture memory updates
                if mock_update.called:
                    for call in mock_update.call_args_list:
                        if call[1].get('data'):
                            memory_updates.append(call[1]['data'])
                            # Simulate memory accumulation
                            accumulated_memory["long_term"].update(call[1]['data'])
                
                mock_update.reset_mock()
            
            # Verify learning occurred
            assert len(memory_updates) > 0, "System should learn from user interactions"
            
            # Verify memory accumulation
            final_memory = accumulated_memory["long_term"]
            
            # Check that preferences were extracted and stored
            preference_fields = ["preferred_fit", "style_vibe", "disliked_fits", "preferred_brands", "gender"]
            learned_preferences = [field for field in preference_fields if field in final_memory]
            
            assert len(learned_preferences) > 0, f"Should learn at least one preference from interactions: {interactions}"
    
    @given(user_preference_data())
    @settings(max_examples=30, deadline=10000)
    async def test_property_conflicting_preferences_resolved(
        self,
        user_data: Dict[str, Any]
    ):
        """
        **Property 6.3: Conflict Resolution**
        
        GIVEN a user expresses conflicting preferences over time
        WHEN the system processes these preferences
        THEN it should resolve conflicts by prioritizing recent preferences
        
        **Validates: Requirements 6.4, 6.5**
        """
        with patch('app.services.agents.memory_agent.memory_agent.retrieve_memory') as mock_retrieve, \
             patch('app.services.agents.memory_agent.memory_agent.update_memory') as mock_update, \
             patch('app.services.memory.service.memory_service.get_user_memory') as mock_get_memory, \
             patch('app.services.agents.search_agent.search_agent.search_products') as mock_search:
            
            user_id = user_data["user_id"]
            email = user_data["email"]
            
            mock_get_memory.return_value = {"context": {"conversationHistory": []}}
            mock_search.return_value = {"products": [], "search_authority": {"search_status": "EMPTY", "products_found": 0}}
            
            # First preference
            first_fit = "oversized"
            mock_retrieve.return_value = {"long_term": {}, "short_term": {}}
            
            await brain_orchestrator.process_request(
                user_input=f"I prefer {first_fit} fits",
                user_id=user_id,
                email=email
            )
            
            # Simulate first preference saved
            first_memory = {"long_term": {"preferred_fit": first_fit}, "short_term": {}}
            
            # Conflicting preference
            second_fit = "slim"
            mock_retrieve.return_value = first_memory
            mock_update.reset_mock()
            
            await brain_orchestrator.process_request(
                user_input=f"Actually, I prefer {second_fit} fits now",
                user_id=user_id,
                email=email
            )
            
            # Verify conflict resolution
            if mock_update.called:
                # Check that the newer preference is saved
                latest_update = None
                for call in mock_update.call_args_list:
                    if call[1].get('data') and 'preferred_fit' in call[1]['data']:
                        latest_update = call[1]['data']['preferred_fit']
                
                if latest_update:
                    assert latest_update == second_fit, f"Should prioritize recent preference '{second_fit}' over old '{first_fit}'"
    
    @given(user_preference_data())
    @settings(max_examples=20, deadline=10000)
    async def test_property_implicit_preferences_extracted(
        self,
        user_data: Dict[str, Any]
    ):
        """
        **Property 6.4: Implicit Learning**
        
        GIVEN a user makes implicit preference statements
        WHEN the system processes these statements
        THEN it should extract and store implicit preferences
        
        **Validates: Requirements 6.3, 6.5**
        """
        with patch('app.services.agents.memory_agent.memory_agent.retrieve_memory') as mock_retrieve, \
             patch('app.services.agents.memory_agent.memory_agent.update_memory') as mock_update, \
             patch('app.services.memory.service.memory_service.get_user_memory') as mock_get_memory, \
             patch('app.services.agents.search_agent.search_agent.search_products') as mock_search:
            
            user_id = user_data["user_id"]
            email = user_data["email"]
            
            mock_retrieve.return_value = {"long_term": {}, "short_term": {}}
            mock_get_memory.return_value = {"context": {"conversationHistory": []}}
            mock_search.return_value = {"products": [], "search_authority": {"search_status": "EMPTY", "products_found": 0}}
            
            # Implicit preference statements
            implicit_statements = [
                f"I'm a {user_data['gender']} looking for clothes",  # Gender
                f"I love {user_data['style_vibe']} style",  # Style vibe
                f"I hate {user_data['disliked_fits'][0]} clothes" if user_data['disliked_fits'] else "I hate tight clothes",  # Dislikes
                f"My budget is under 2000"  # Price preference
            ]
            
            for statement in implicit_statements:
                await brain_orchestrator.process_request(
                    user_input=statement,
                    user_id=user_id,
                    email=email
                )
            
            # Verify implicit preferences were extracted
            assert mock_update.called, "Should extract preferences from implicit statements"
            
            # Collect all saved preferences
            all_saved_data = {}
            for call in mock_update.call_args_list:
                if call[1].get('data'):
                    all_saved_data.update(call[1]['data'])
            
            # Verify specific implicit extractions
            expected_extractions = []
            
            if user_data['gender']:
                expected_extractions.append(("gender", user_data['gender']))
            
            if user_data['style_vibe']:
                expected_extractions.append(("style_vibe", user_data['style_vibe']))
            
            # At least one implicit preference should be extracted
            extracted_count = sum(1 for field, expected_value in expected_extractions 
                                if field in all_saved_data and expected_value.lower() in str(all_saved_data[field]).lower())
            
            assert extracted_count > 0, f"Should extract at least one implicit preference. Saved: {all_saved_data}, Expected: {expected_extractions}"


# Run the tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])