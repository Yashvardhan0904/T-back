"""
Context Management Service

This package provides the enhanced context management system with three-tier memory architecture:
1. Working Memory: Current conversation context
2. Session Memory: Temporary session preferences  
3. Long-term Memory: Persistent user profile

Key components:
- ContextManager: Central coordinator for all memory tiers
- Context Models: Data structures for user context and preferences
"""

from .manager import context_manager, ContextManager

__all__ = ['context_manager', 'ContextManager']