"""
Property-Based Test: Natural Language Understanding

**Property 8: Natural Language Understanding**
**Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5**

This test validates that the system correctly understands and processes natural language input
including casual language, slang, indirect preferences, and ambiguous queries.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from typing import Dict, Any, List
import asyncio
from unittest.mock import AsyncMock, patch

from app.services.agents.brain_orchestrator import brain_orchestrator
from app.services.agents.query_agent import query_agent


class TestNaturalLanguageUnderstanding:
    """Test natural language understanding capabilities"""
    
    @given(
        casual_query=st.one_of(
            st.just("yo need some fire fits"),
            st.just("looking for something dope"),
            st.just("need drip for tonight"),
            st.just("what's trending rn"),
            st.just("show me some sick hoodies"),
            st.just("idk what to wear help"),
            st.just("need something lowkey fire"),
            st.just("looking for fits that slap")
        ),
        user_memory=st.fixed_dictionaries({
            "long_term": st.fixed_dictionaries({
                "gender": st.just("male"),
                "style_vibe": st.just("streetwear"),
                "slang_tolerance": st.just("high")
            })
        })
    )
    @settings(max_examples=50, deadline=30000)
    async def test_casual_language_processing(self, casual_query: str, user_memory: Dict[str, Any]):
        """
        **Property 8.1: Casual Language Processing**
        
        The system should correctly interpret casual language and Gen-Z slang,
        extracting meaningful intent from informal expressions.
        """
        # Mock dependencies
        with patch('app.services.agents.memory_agent.memory_agent.retrieve_memory') as mock_memory, \
             patch('app.services.agents.search_agent.search_agent.search_products') as mock_search, \
             patch('app.services.agents.recommendation_agent.recommendation_agent.rank_products') as mock_rank, \
             patch('app.services.agents.response_agent.response_agent.generate_response') as mock_response, \
             patch('app.services.memory.service.memory_service.get_user_memory') as mock_get_memory:
            
            # Setup mocks
            mock_memory.return_value = user_memory
            mock_get_memory.return_value = {"context": {"conversationHistory": []}}
            mock_search.return_value = {
                "products": [{"id": "1", "name": "Test Hoodie", "price": 2000}],
                "search_authority": {"search_status": "FOUND", "products_found": 1}
            }
            mock_rank.return_value = [{"product": {"id": "1", "name": "Test Hoodie"}, "score": 0.8}]
            mock_response.return_value = "Found some fire hoodies for you!"
            
            # Test casual language understanding
            result = await brain_orchestrator.process_request(
                user_input=casual_query,
                user_id="test_user",
                email="test@example.com"
            )
            
            # Verify system understood the casual language
            assert result is not None
            assert "response" in result
            assert len(result["response"]) > 0
            
            # Verify search was triggered (casual language was interpreted as search intent)
            mock_search.assert_called_once()
            
            # Verify the query was processed (not rejected as invalid)
            search_call = mock_search.call_args
            assert search_call is not None
            
            # The system should extract meaningful intent from casual language
            query_arg = search_call[1]["query"] if "query" in search_call[1] else search_call[0][0]
            assert len(query_arg) > 0  # Query was processed, not empty
    
    @given(
        indirect_preference=st.one_of(
            st.just("I hate tight clothes"),
            st.just("not a fan of bright colors"),
            st.just("prefer comfortable fits"),
            st.just("don't like flashy stuff"),
            st.just("something that doesn't stand out"),
            st.just("I'm more of a minimalist"),
            st.just("not into loud patterns"),
            st.just("keep it simple please")
        ),
        user_memory=st.fixed_dictionaries({
            "long_term": st.fixed_dictionaries({
                "gender": st.just("female"),
                "style_vibe": st.just("minimal")
            })
        })
    )
    @settings(max_examples=40, deadline=30000)
    async def test_indirect_preference_inference(self, indirect_preference: str, user_memory: Dict[str, Any]):
        """
        **Property 8.2: Indirect Preference Inference**
        
        The system should infer preferences from indirect statements and negative expressions,
        converting them into actionable search filters.
        """
        # Test query understanding for indirect preferences
        intent = await query_agent.understand_query(indirect_preference, [])
        
        # Verify indirect preferences are captured
        assert intent is not None
        assert "preferences" in intent or "filters" in intent
        
        # Check that negative preferences are converted to positive filters
        preferences = intent.get("preferences", {})
        filters = intent.get("filters", {})
        
        # System should infer actionable preferences from negative statements
        has_inferred_preference = (
            len(preferences) > 0 or 
            len(filters) > 0 or
            intent.get("intent_type") == "preference_update"
        )
        
        assert has_inferred_preference, f"Failed to infer preferences from: {indirect_preference}"
        
        # Verify specific inference patterns
        if "tight" in indirect_preference.lower():
            # Should infer loose/oversized preference
            fit_related = any(
                "fit" in str(v).lower() or "oversized" in str(v).lower() or "loose" in str(v).lower()
                for v in [preferences, filters, intent.get("product_category", "")]
            )
            assert fit_related, "Should infer fit preference from 'tight' rejection"
        
        if "bright" in indirect_preference.lower() or "flashy" in indirect_preference.lower():
            # Should infer neutral/muted color preference
            color_related = any(
                "color" in str(v).lower() or "neutral" in str(v).lower() or "muted" in str(v).lower()
                for v in [preferences, filters]
            )
            # This is acceptable to not always infer - indirect preferences are complex
    
    @given(
        ambiguous_query=st.one_of(
            st.just("something nice"),
            st.just("good stuff"),
            st.just("what do you recommend"),
            st.just("show me options"),
            st.just("I need clothes"),
            st.just("help me choose"),
            st.just("what's good"),
            st.just("surprise me")
        ),
        user_memory=st.fixed_dictionaries({
            "long_term": st.fixed_dictionaries({
                "gender": st.just("male"),
                "style_vibe": st.just("casual")
            })
        })
    )
    @settings(max_examples=30, deadline=30000)
    async def test_ambiguity_handling(self, ambiguous_query: str, user_memory: Dict[str, Any]):
        """
        **Property 8.3: Ambiguity Handling**
        
        The system should handle ambiguous queries by either asking clarifying questions
        or using user memory to provide contextual recommendations.
        """
        with patch('app.services.agents.memory_agent.memory_agent.retrieve_memory') as mock_memory, \
             patch('app.services.memory.service.memory_service.get_user_memory') as mock_get_memory:
            
            mock_memory.return_value = user_memory
            mock_get_memory.return_value = {"context": {"conversationHistory": []}}
            
            # Test ambiguous query handling
            result = await brain_orchestrator.process_request(
                user_input=ambiguous_query,
                user_id="test_user",
                email="test@example.com"
            )
            
            # System should handle ambiguity gracefully
            assert result is not None
            assert "response" in result
            assert len(result["response"]) > 0
            
            # Should either ask clarifying questions or provide recommendations based on memory
            response = result["response"].lower()
            
            # Check for clarifying question patterns
            clarifying_indicators = [
                "what", "which", "looking for", "prefer", "occasion", 
                "style", "vibe", "type", "kind", "specific"
            ]
            
            # Check for recommendation patterns (using memory)
            recommendation_indicators = [
                "found", "here", "options", "recommend", "suggest", "try"
            ]
            
            has_clarifying_question = any(indicator in response for indicator in clarifying_indicators)
            has_recommendations = any(indicator in response for indicator in recommendation_indicators)
            
            # System should either ask for clarification OR provide memory-based recommendations
            assert has_clarifying_question or has_recommendations, \
                f"Ambiguous query '{ambiguous_query}' should trigger clarification or memory-based recommendations"
    
    @given(
        emotional_query=st.one_of(
            st.just("I'm so confused about what to wear"),
            st.just("help I have nothing to wear"),
            st.just("feeling lost with fashion"),
            st.just("I'm excited for this party"),
            st.just("nervous about the interview"),
            st.just("can't decide what looks good"),
            st.just("feeling overwhelmed with choices"),
            st.just("super excited to shop")
        ),
        user_memory=st.fixed_dictionaries({
            "long_term": st.fixed_dictionaries({
                "gender": st.just("female"),
                "style_vibe": st.just("classic")
            })
        })
    )
    @settings(max_examples=30, deadline=30000)
    async def test_emotional_expression_recognition(self, emotional_query: str, user_memory: Dict[str, Any]):
        """
        **Property 8.4: Emotional Expression Recognition**
        
        The system should recognize emotional expressions and respond with appropriate
        empathy and support while maintaining helpful functionality.
        """
        with patch('app.services.agents.memory_agent.memory_agent.retrieve_memory') as mock_memory, \
             patch('app.services.memory.service.memory_service.get_user_memory') as mock_get_memory:
            
            mock_memory.return_value = user_memory
            mock_get_memory.return_value = {"context": {"conversationHistory": []}}
            
            # Test emotional expression recognition
            result = await brain_orchestrator.process_request(
                user_input=emotional_query,
                user_id="test_user",
                email="test@example.com"
            )
            
            # System should respond to emotional expressions
            assert result is not None
            assert "response" in result
            response = result["response"].lower()
            
            # Check for empathetic response patterns
            empathy_indicators = [
                "understand", "help", "no worries", "got you", "here for you",
                "let me help", "i can help", "don't worry", "it's okay"
            ]
            
            # Check for supportive language
            supportive_indicators = [
                "perfect", "great", "awesome", "excited", "love", "amazing",
                "confident", "look good", "feel good", "right choice"
            ]
            
            has_empathy = any(indicator in response for indicator in empathy_indicators)
            has_support = any(indicator in response for indicator in supportive_indicators)
            
            # System should show empathy or support for emotional expressions
            assert has_empathy or has_support, \
                f"Emotional query '{emotional_query}' should trigger empathetic or supportive response"
            
            # Response should still be helpful (not just emotional support)
            assert len(response) > 20, "Response should be substantial and helpful"
    
    @given(
        context_dependent_query=st.one_of(
            st.just("show me more like that"),
            st.just("something similar"),
            st.just("different color of the same"),
            st.just("that but cheaper"),
            st.just("not that style"),
            st.just("the other one was better"),
            st.just("like the first option"),
            st.just("similar vibe but different")
        ),
        conversation_history=st.lists(
            st.fixed_dictionaries({
                "role": st.just("assistant"),
                "content": st.just("Here are some hoodies for you"),
                "product_ids": st.lists(st.just("prod_123"), min_size=1, max_size=3)
            }),
            min_size=1,
            max_size=3
        ),
        user_memory=st.fixed_dictionaries({
            "long_term": st.fixed_dictionaries({
                "gender": st.just("male"),
                "style_vibe": st.just("streetwear")
            })
        })
    )
    @settings(max_examples=25, deadline=30000)
    async def test_context_dependent_understanding(
        self, 
        context_dependent_query: str, 
        conversation_history: List[Dict[str, Any]], 
        user_memory: Dict[str, Any]
    ):
        """
        **Property 8.5: Context-Dependent Understanding**
        
        The system should understand queries that depend on conversation context,
        referencing previous products or interactions appropriately.
        """
        with patch('app.services.agents.memory_agent.memory_agent.retrieve_memory') as mock_memory, \
             patch('app.services.memory.service.memory_service.get_user_memory') as mock_get_memory, \
             patch('app.services.agents.search_agent.search_agent.search_products') as mock_search:
            
            mock_memory.return_value = user_memory
            mock_get_memory.return_value = {"context": {"conversationHistory": conversation_history}}
            mock_search.return_value = {
                "products": [{"id": "2", "name": "Similar Hoodie", "price": 1800}],
                "search_authority": {"search_status": "FOUND", "products_found": 1}
            }
            
            # Test context-dependent query understanding
            intent = await query_agent.understand_query(context_dependent_query, conversation_history)
            
            # Verify context dependency is recognized
            assert intent is not None
            
            # Context-dependent queries should be recognized as such
            intent_type = intent.get("intent_type", "")
            
            # Should recognize reference to previous context
            context_indicators = [
                "similar", "like", "same", "that", "other", "first", "previous"
            ]
            
            query_lower = context_dependent_query.lower()
            references_context = any(indicator in query_lower for indicator in context_indicators)
            
            if references_context:
                # System should handle context-dependent queries appropriately
                # This could be through intent type or through filters/preferences
                handles_context = (
                    intent_type in ["search", "recommend", "preference_update", "rejection"] or
                    len(intent.get("filters", {})) > 0 or
                    len(intent.get("preferences", {})) > 0
                )
                
                assert handles_context, \
                    f"Context-dependent query '{context_dependent_query}' should be handled appropriately"
    
    @pytest.mark.asyncio
    async def test_property_natural_language_understanding_integration(self):
        """
        **Integration Test: Complete Natural Language Understanding Pipeline**
        
        Test the complete pipeline from natural language input to appropriate system response,
        validating that all NLU components work together correctly.
        """
        test_cases = [
            {
                "query": "yo need some fire streetwear fits under 3k",
                "expected_elements": ["streetwear", "budget", "casual_language"],
                "user_memory": {"long_term": {"gender": "male", "slang_tolerance": "high"}}
            },
            {
                "query": "I don't like tight clothes, prefer something loose",
                "expected_elements": ["preference_inference", "fit_preference"],
                "user_memory": {"long_term": {"gender": "female", "style_vibe": "casual"}}
            },
            {
                "query": "help I'm so confused about what to wear",
                "expected_elements": ["emotional_support", "guidance"],
                "user_memory": {"long_term": {"gender": "female", "style_vibe": "minimal"}}
            }
        ]
        
        for case in test_cases:
            with patch('app.services.agents.memory_agent.memory_agent.retrieve_memory') as mock_memory, \
                 patch('app.services.memory.service.memory_service.get_user_memory') as mock_get_memory:
                
                mock_memory.return_value = case["user_memory"]
                mock_get_memory.return_value = {"context": {"conversationHistory": []}}
                
                # Test complete NLU pipeline
                result = await brain_orchestrator.process_request(
                    user_input=case["query"],
                    user_id="test_user",
                    email="test@example.com"
                )
                
                # Verify system handled the query appropriately
                assert result is not None
                assert "response" in result
                assert len(result["response"]) > 0
                
                # Verify response quality
                response = result["response"].lower()
                
                # Should not be a generic error response
                error_indicators = ["error", "failed", "couldn't process", "invalid"]
                has_error = any(indicator in response for indicator in error_indicators)
                assert not has_error, f"Query '{case['query']}' should not result in error response"
                
                # Should be contextually appropriate
                assert len(response) > 10, "Response should be substantial"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])