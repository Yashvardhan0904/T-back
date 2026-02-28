"""
Preference Hierarchy and Conflict Resolution System

This module handles preference conflicts and maintains hierarchy:
1. Explicit statements override inferred preferences
2. Most recent statements override older ones
3. Source confidence scoring for preference reliability
4. Conflict resolution with user confirmation when needed
"""

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import logging

from app.models.context import PreferenceSource, StylePreference

logger = logging.getLogger(__name__)


class ConflictResolutionStrategy(Enum):
    """Strategy for resolving preference conflicts"""
    MOST_RECENT = "most_recent"  # Use most recent preference
    HIGHEST_CONFIDENCE = "highest_confidence"  # Use highest confidence preference
    EXPLICIT_ONLY = "explicit_only"  # Only use explicit preferences
    ASK_USER = "ask_user"  # Ask user to resolve conflict


@dataclass
class PreferenceConflict:
    """Represents a conflict between preferences"""
    preference_key: str
    conflicting_values: List[Dict[str, Any]]
    resolution_strategy: ConflictResolutionStrategy
    resolved_value: Optional[Any] = None
    confidence: float = 0.0
    requires_user_confirmation: bool = False


class PreferenceResolver:
    """
    Handles preference hierarchy and conflict resolution.
    
    Priority order:
    1. Source type: explicit > inferred > behavioral
    2. Recency: newer > older
    3. Confidence: higher > lower
    """
    
    def __init__(self):
        self.source_priority = {
            PreferenceSource.EXPLICIT: 3,
            PreferenceSource.INFERRED: 2,
            PreferenceSource.BEHAVIORAL: 1
        }
    
    def resolve_preference_conflicts(
        self,
        preferences: List[Dict[str, Any]],
        strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.MOST_RECENT
    ) -> Dict[str, PreferenceConflict]:
        """
        Resolve conflicts between preferences.
        
        Args:
            preferences: List of preference dictionaries
            strategy: Resolution strategy to use
            
        Returns:
            Dictionary mapping preference keys to resolved conflicts
        """
        # Group preferences by key
        preference_groups = self._group_preferences_by_key(preferences)
        
        conflicts = {}
        
        for pref_key, pref_list in preference_groups.items():
            if len(pref_list) > 1:
                # We have a conflict
                conflict = self._resolve_single_conflict(pref_key, pref_list, strategy)
                conflicts[pref_key] = conflict
            else:
                # No conflict, just wrap in conflict object for consistency
                pref = pref_list[0]
                conflicts[pref_key] = PreferenceConflict(
                    preference_key=pref_key,
                    conflicting_values=[pref],
                    resolution_strategy=strategy,
                    resolved_value=pref.get("value"),
                    confidence=pref.get("confidence", 1.0)
                )
        
        return conflicts
    
    def _group_preferences_by_key(self, preferences: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group preferences by their key/type"""
        groups = {}
        
        for pref in preferences:
            key = pref.get("type") or pref.get("key", "unknown")
            if key not in groups:
                groups[key] = []
            groups[key].append(pref)
        
        return groups
    
    def _resolve_single_conflict(
        self,
        pref_key: str,
        conflicting_prefs: List[Dict[str, Any]],
        strategy: ConflictResolutionStrategy
    ) -> PreferenceConflict:
        """Resolve a single preference conflict"""
        
        if strategy == ConflictResolutionStrategy.MOST_RECENT:
            return self._resolve_by_recency(pref_key, conflicting_prefs)
        elif strategy == ConflictResolutionStrategy.HIGHEST_CONFIDENCE:
            return self._resolve_by_confidence(pref_key, conflicting_prefs)
        elif strategy == ConflictResolutionStrategy.EXPLICIT_ONLY:
            return self._resolve_explicit_only(pref_key, conflicting_prefs)
        else:
            # Default to most recent
            return self._resolve_by_recency(pref_key, conflicting_prefs)
    
    def _resolve_by_recency(self, pref_key: str, conflicting_prefs: List[Dict[str, Any]]) -> PreferenceConflict:
        """Resolve conflict by choosing most recent preference"""
        
        # Sort by timestamp (most recent first)
        sorted_prefs = sorted(
            conflicting_prefs,
            key=lambda p: p.get("timestamp", datetime.min),
            reverse=True
        )
        
        # Apply source priority as tiebreaker
        if len(sorted_prefs) > 1:
            # Check if top preferences have same timestamp
            top_timestamp = sorted_prefs[0].get("timestamp", datetime.min)
            same_time_prefs = [p for p in sorted_prefs if p.get("timestamp", datetime.min) == top_timestamp]
            
            if len(same_time_prefs) > 1:
                # Use source priority as tiebreaker
                sorted_prefs = sorted(
                    same_time_prefs,
                    key=lambda p: self._get_source_priority(p.get("source", "inferred")),
                    reverse=True
                )
        
        winner = sorted_prefs[0]
        
        return PreferenceConflict(
            preference_key=pref_key,
            conflicting_values=conflicting_prefs,
            resolution_strategy=ConflictResolutionStrategy.MOST_RECENT,
            resolved_value=winner.get("value"),
            confidence=winner.get("confidence", 0.8)
        )
    
    def _resolve_by_confidence(self, pref_key: str, conflicting_prefs: List[Dict[str, Any]]) -> PreferenceConflict:
        """Resolve conflict by choosing highest confidence preference"""
        
        # Sort by confidence (highest first)
        sorted_prefs = sorted(
            conflicting_prefs,
            key=lambda p: p.get("confidence", 0.5),
            reverse=True
        )
        
        winner = sorted_prefs[0]
        
        return PreferenceConflict(
            preference_key=pref_key,
            conflicting_values=conflicting_prefs,
            resolution_strategy=ConflictResolutionStrategy.HIGHEST_CONFIDENCE,
            resolved_value=winner.get("value"),
            confidence=winner.get("confidence", 0.5)
        )
    
    def _resolve_explicit_only(self, pref_key: str, conflicting_prefs: List[Dict[str, Any]]) -> PreferenceConflict:
        """Resolve conflict by only considering explicit preferences"""
        
        # Filter to explicit preferences only
        explicit_prefs = [
            p for p in conflicting_prefs 
            if p.get("source") == "explicit" or p.get("source") == PreferenceSource.EXPLICIT
        ]
        
        if not explicit_prefs:
            # No explicit preferences, fall back to most recent
            return self._resolve_by_recency(pref_key, conflicting_prefs)
        
        if len(explicit_prefs) == 1:
            winner = explicit_prefs[0]
        else:
            # Multiple explicit preferences, use most recent
            winner = max(explicit_prefs, key=lambda p: p.get("timestamp", datetime.min))
        
        return PreferenceConflict(
            preference_key=pref_key,
            conflicting_values=conflicting_prefs,
            resolution_strategy=ConflictResolutionStrategy.EXPLICIT_ONLY,
            resolved_value=winner.get("value"),
            confidence=winner.get("confidence", 0.9)
        )
    
    def _get_source_priority(self, source: Any) -> int:
        """Get priority score for preference source"""
        if isinstance(source, str):
            source_map = {
                "explicit": 3,
                "inferred": 2,
                "behavioral": 1
            }
            return source_map.get(source.lower(), 1)
        elif isinstance(source, PreferenceSource):
            return self.source_priority.get(source, 1)
        else:
            return 1
    
    def calculate_preference_confidence(
        self,
        preference: Dict[str, Any],
        context: Dict[str, Any] = None
    ) -> float:
        """
        Calculate confidence score for a preference based on various factors.
        
        Factors:
        - Source type (explicit > inferred > behavioral)
        - Recency (newer > older)
        - Consistency with other preferences
        - User interaction patterns
        
        Returns:
            Confidence score between 0.0 and 1.0
        """
        base_confidence = 0.5
        
        # Source type bonus
        source = preference.get("source", "inferred")
        source_bonus = {
            "explicit": 0.4,
            "inferred": 0.2,
            "behavioral": 0.1
        }.get(source.lower() if isinstance(source, str) else str(source).lower(), 0.1)
        
        # Recency bonus (preferences from last hour get bonus)
        timestamp = preference.get("timestamp", datetime.min)
        if isinstance(timestamp, datetime):
            hours_old = (datetime.utcnow() - timestamp).total_seconds() / 3600
            recency_bonus = max(0, 0.2 - (hours_old * 0.01))  # Decay over time
        else:
            recency_bonus = 0
        
        # Consistency bonus (if provided context supports this preference)
        consistency_bonus = 0
        if context:
            # Check if preference is consistent with user's other preferences
            # This is a simplified version - could be more sophisticated
            similar_prefs = context.get("similar_preferences", [])
            if any(p.get("value") == preference.get("value") for p in similar_prefs):
                consistency_bonus = 0.1
        
        # Calculate final confidence
        confidence = min(1.0, base_confidence + source_bonus + recency_bonus + consistency_bonus)
        
        return confidence
    
    def should_ask_user_for_confirmation(
        self,
        conflict: PreferenceConflict,
        threshold: float = 0.3
    ) -> bool:
        """
        Determine if user confirmation is needed for conflict resolution.
        
        Args:
            conflict: The preference conflict
            threshold: Confidence threshold below which to ask user
            
        Returns:
            True if user confirmation is recommended
        """
        # Ask user if confidence is low
        if conflict.confidence < threshold:
            return True
        
        # Ask user if conflicting values are very different
        values = [v.get("value") for v in conflict.conflicting_values]
        unique_values = set(values)
        
        # If we have many different values, might want confirmation
        if len(unique_values) > 2:
            return True
        
        # Ask user if we have recent explicit conflicts
        explicit_prefs = [
            v for v in conflict.conflicting_values 
            if v.get("source") == "explicit"
        ]
        
        if len(explicit_prefs) > 1:
            # Multiple explicit preferences conflict - ask user
            return True
        
        return False
    
    def generate_conflict_resolution_message(
        self,
        conflict: PreferenceConflict
    ) -> str:
        """Generate a natural language message about conflict resolution"""
        
        pref_key = conflict.preference_key
        resolved_value = conflict.resolved_value
        
        if conflict.requires_user_confirmation:
            values = [v.get("value") for v in conflict.conflicting_values]
            unique_values = list(set(values))
            
            if len(unique_values) == 2:
                return f"I noticed you mentioned both {unique_values[0]} and {unique_values[1]} for {pref_key}. Which one would you prefer?"
            else:
                return f"I'm seeing different preferences for {pref_key}. Could you clarify which one you'd like me to use?"
        
        else:
            # Just inform about resolution
            strategy_messages = {
                ConflictResolutionStrategy.MOST_RECENT: f"Got it! Using your most recent preference: {resolved_value}",
                ConflictResolutionStrategy.HIGHEST_CONFIDENCE: f"Noted! Going with {resolved_value} based on your clear preference",
                ConflictResolutionStrategy.EXPLICIT_ONLY: f"Perfect! I'll use {resolved_value} as you explicitly mentioned"
            }
            
            return strategy_messages.get(
                conflict.resolution_strategy,
                f"Understood! I'll go with {resolved_value} for {pref_key}"
            )
    
    def merge_preferences(
        self,
        session_preferences: Dict[str, Any],
        profile_preferences: Dict[str, Any],
        conversation_preferences: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Merge preferences from different sources with proper hierarchy.
        
        Priority: conversation > session > profile
        
        Args:
            session_preferences: Current session preferences
            profile_preferences: Long-term profile preferences  
            conversation_preferences: Preferences from current conversation
            
        Returns:
            Merged preferences dictionary
        """
        merged = {}
        
        # Start with profile preferences (lowest priority)
        merged.update(profile_preferences)
        
        # Override with session preferences
        merged.update(session_preferences)
        
        # Override with conversation preferences (highest priority)
        for conv_pref in conversation_preferences:
            key = conv_pref.get("type") or conv_pref.get("key")
            if key:
                merged[key] = conv_pref.get("value")
        
        return merged


# Singleton instance
preference_resolver = PreferenceResolver()