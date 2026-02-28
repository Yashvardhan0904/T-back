"""
Acknowledgment Engine - Generates conversational acknowledgments for user preferences

This engine transforms robotic responses into natural, enthusiastic acknowledgments.
Key features:
1. Pattern recognition for user preferences and statements
2. Casual, Gen-Z friendly language generation
3. Conversation flow management with smooth transitions
4. Variety patterns to avoid repetitive responses
5. Context-aware acknowledgments based on conversation history
"""

from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import random
import re
import logging

logger = logging.getLogger(__name__)


class AcknowledgmentType(Enum):
    """Types of acknowledgments"""
    PREFERENCE_STATED = "preference_stated"  # User states a preference
    DISCOVERY_ANSWER = "discovery_answer"    # User answers discovery question
    APPRECIATION = "appreciation"            # User likes something
    REJECTION = "rejection"                  # User rejects suggestions
    CLARIFICATION = "clarification"          # User provides clarification
    GREETING = "greeting"                    # User greets or starts conversation


class SlangLevel(Enum):
    """Slang intensity levels"""
    LOW = "low"        # Minimal slang, professional
    MEDIUM = "medium"  # Moderate slang, friendly
    HIGH = "high"      # Heavy slang, very casual


class AcknowledgmentEngine:
    """
    Generates natural, conversational acknowledgments for user interactions.
    Transforms robotic responses into enthusiastic, Gen-Z friendly acknowledgments.
    """
    
    def __init__(self):
        self.acknowledgment_patterns = self._build_acknowledgment_patterns()
        self.transition_phrases = self._build_transition_phrases()
        self.recent_acknowledgments = []  # Track recent acknowledgments to avoid repetition
        self.max_recent_history = 10
    
    def _build_acknowledgment_patterns(self) -> Dict[AcknowledgmentType, Dict[SlangLevel, List[str]]]:
        """Build acknowledgment patterns for different types and slang levels"""
        return {
            AcknowledgmentType.PREFERENCE_STATED: {
                SlangLevel.LOW: [
                    "Got it! {preference} is a great choice.",
                    "Perfect! I'll keep {preference} in mind.",
                    "Noted! {preference} it is.",
                    "Excellent choice with {preference}.",
                    "I'll make sure to focus on {preference}."
                ],
                SlangLevel.MEDIUM: [
                    "Got it! {preference} is definitely the vibe.",
                    "Nice choice! {preference} is solid.",
                    "Perfect! {preference} is a great pick.",
                    "Love that! {preference} is fire.",
                    "Bet! {preference} is the way to go."
                ],
                SlangLevel.HIGH: [
                    "Yooo! {preference} is absolutely fire! 🔥",
                    "Bet! {preference} hits different fr 👌",
                    "That's the vibe! {preference} is clean af",
                    "Periodt! {preference} is the move 💯",
                    "No cap, {preference} is elite! ✨"
                ]
            },
            AcknowledgmentType.DISCOVERY_ANSWER: {
                SlangLevel.LOW: [
                    "Perfect! {answer} is exactly what I needed to know.",
                    "Great! {answer} helps me find the right options.",
                    "Excellent! {answer} gives me good direction.",
                    "Thanks! {answer} is very helpful.",
                    "Got it! {answer} is noted."
                ],
                SlangLevel.MEDIUM: [
                    "Perfect! {answer} is exactly what I was looking for.",
                    "Nice! {answer} helps me narrow things down.",
                    "Great choice! {answer} is solid.",
                    "Love it! {answer} gives me good vibes to work with.",
                    "Bet! {answer} is the direction we're going."
                ],
                SlangLevel.HIGH: [
                    "Yesss! {answer} is the energy we need! ⚡",
                    "That's what I'm talking about! {answer} hits different 🎯",
                    "Periodt! {answer} is the vibe check we needed ✨",
                    "No cap! {answer} is exactly the direction 💯",
                    "Fire choice! {answer} is about to be iconic 🔥"
                ]
            },
            AcknowledgmentType.APPRECIATION: {
                SlangLevel.LOW: [
                    "I'm glad you like it! That's a great choice.",
                    "Excellent taste! That piece is really nice.",
                    "Great pick! You have good style sense.",
                    "Perfect! That's definitely a winner.",
                    "Nice choice! That's a solid option."
                ],
                SlangLevel.MEDIUM: [
                    "Yes! You've got great taste.",
                    "Love that! You picked a winner.",
                    "Nice eye! That's definitely fire.",
                    "Great choice! You know what's good.",
                    "Perfect pick! That's going to look amazing."
                ],
                SlangLevel.HIGH: [
                    "Yooo your taste is immaculate! 👑",
                    "That's fire! You got the vision fr 🔥",
                    "Periodt! Your style sense is unmatched ✨",
                    "No cap, you picked the best one! 💯",
                    "Elite taste! That's going to be iconic 👌"
                ]
            },
            AcknowledgmentType.REJECTION: {
                SlangLevel.LOW: [
                    "No problem! Let me find something different.",
                    "Understood! I'll look for other options.",
                    "Got it! Let me try a different approach.",
                    "No worries! I'll find something more suitable.",
                    "Alright! Let me search for alternatives."
                ],
                SlangLevel.MEDIUM: [
                    "No worries! Let me find something that hits different.",
                    "Got it! Let me switch up the vibe.",
                    "All good! Let me try a different direction.",
                    "Bet! Let me find something more your style.",
                    "Cool! Let me look for something else."
                ],
                SlangLevel.HIGH: [
                    "No cap, let me find something that actually slaps! 💯",
                    "Bet! Let me switch the whole vibe up 🔄",
                    "Say less! I'll find options that actually hit 🎯",
                    "Fr! Let me get you something fire instead 🔥",
                    "Understood the assignment! Different energy coming up ⚡"
                ]
            },
            AcknowledgmentType.CLARIFICATION: {
                SlangLevel.LOW: [
                    "Thanks for clarifying! That helps a lot.",
                    "Perfect! That makes much more sense.",
                    "Got it! That's exactly what I needed to know.",
                    "Excellent! That clears things up perfectly.",
                    "Thank you! That's very helpful."
                ],
                SlangLevel.MEDIUM: [
                    "Ah, got it! That makes way more sense.",
                    "Perfect! That's exactly what I needed.",
                    "Nice! That clears everything up.",
                    "Great! That helps me understand the vibe.",
                    "Cool! That's much clearer now."
                ],
                SlangLevel.HIGH: [
                    "Ohhh bet! That makes so much more sense now! 💡",
                    "Say less! I totally get the vision now 👀",
                    "Facts! That's the clarity I needed fr 🎯",
                    "No cap! Everything clicks now ✨",
                    "Periodt! The assignment is clear now 💯"
                ]
            },
            AcknowledgmentType.GREETING: {
                SlangLevel.LOW: [
                    "Hello! How can I help you today?",
                    "Hi there! What are you looking for?",
                    "Good to see you! How can I assist?",
                    "Welcome! What can I help you find?",
                    "Hello! Ready to find something great?"
                ],
                SlangLevel.MEDIUM: [
                    "Hey! What's up? Ready to find some great stuff?",
                    "Hi there! What are we shopping for today?",
                    "Hey! What can I help you discover?",
                    "What's good! What are you in the mood for?",
                    "Hey! Ready to find something awesome?"
                ],
                SlangLevel.HIGH: [
                    "Yooo! What's the vibe today? 👋",
                    "Hey bestie! What are we hunting for? ✨",
                    "What's good! Ready to find some fire fits? 🔥",
                    "Heyy! What's the energy we're going for? ⚡",
                    "Yo! What's the mission today? 🎯"
                ]
            }
        }
    
    def _build_transition_phrases(self) -> Dict[SlangLevel, List[str]]:
        """Build transition phrases for smooth conversation flow"""
        return {
            SlangLevel.LOW: [
                "Now, let me find some options for you.",
                "Let me search for some great choices.",
                "I'll look for some perfect matches.",
                "Let me find some excellent options.",
                "Now I can search for exactly what you need."
            ],
            SlangLevel.MEDIUM: [
                "Now let me find some fire options for you.",
                "Let me search for some great picks.",
                "I'll hunt down some perfect matches.",
                "Let me find some solid choices.",
                "Now I can get you exactly what you're looking for."
            ],
            SlangLevel.HIGH: [
                "Now let me find some absolute bangers for you! 🔥",
                "Time to hunt down some fire options! 🎯",
                "Let me get you some elite picks! ✨",
                "About to find some options that absolutely slap! 💯",
                "Time to serve you some iconic choices! 👑"
            ]
        }
    
    def detect_acknowledgment_type(
        self,
        user_input: str,
        conversation_history: List[Dict[str, Any]] = None,
        user_intent: Dict[str, Any] = None
    ) -> Tuple[AcknowledgmentType, Dict[str, Any]]:
        """
        Detect what type of acknowledgment is needed based on user input and context.
        
        Returns:
            Tuple of (acknowledgment_type, context_data)
        """
        user_lower = user_input.lower().strip()
        
        # Check for greetings
        greeting_patterns = [
            r'\b(hi|hello|hey|yo|sup|what\'s up|whats up)\b',
            r'^(good morning|good afternoon|good evening)',
            r'\b(how are you|how\'s it going)\b'
        ]
        
        if any(re.search(pattern, user_lower) for pattern in greeting_patterns):
            return AcknowledgmentType.GREETING, {"greeting_detected": True}
        
        # Check for appreciation/likes
        appreciation_patterns = [
            r'\b(love|like|fire|amazing|perfect|great|awesome|nice|cool)\b.*\b(this|that|it)\b',
            r'\b(this is|that\'s|it\'s)\b.*(fire|amazing|perfect|great|awesome|nice|cool)',
            r'\b(i love|i like|love this|like this)\b'
        ]
        
        if any(re.search(pattern, user_lower) for pattern in appreciation_patterns):
            return AcknowledgmentType.APPRECIATION, {"appreciation_detected": True}
        
        # Check for rejections
        rejection_patterns = [
            r'\b(no|nah|not|don\'t like|dont like|hate|dislike)\b',
            r'\b(something else|different|other options|alternatives)\b',
            r'\b(not my style|not for me|not really)\b'
        ]
        
        if any(re.search(pattern, user_lower) for pattern in rejection_patterns):
            return AcknowledgmentType.REJECTION, {"rejection_detected": True}
        
        # Check if this is answering a discovery question
        if conversation_history:
            last_assistant_message = None
            for msg in reversed(conversation_history):
                if msg.get("role") == "assistant":
                    last_assistant_message = msg.get("content", "").lower()
                    break
            
            if last_assistant_message:
                # Check if last message was a question
                question_indicators = ["?", "what", "which", "how", "do you", "are you"]
                if any(indicator in last_assistant_message for indicator in question_indicators):
                    return AcknowledgmentType.DISCOVERY_ANSWER, {
                        "answer": user_input,
                        "question_context": last_assistant_message[:100]
                    }
        
        # Check for explicit preferences
        preference_patterns = [
            r'\b(i prefer|i like|i want|i need)\b',
            r'\b(my style is|my vibe is|i\'m into)\b',
            r'\b(size|color|fit|style|vibe).*\b(is|are)\b',
            r'\b(oversized|slim|regular|streetwear|minimal|classic)\b'
        ]
        
        if any(re.search(pattern, user_lower) for pattern in preference_patterns):
            # Extract the preference
            preference_match = None
            for pattern in preference_patterns:
                match = re.search(pattern, user_lower)
                if match:
                    # Get the part after the pattern
                    start_pos = match.end()
                    preference_match = user_input[start_pos:].strip()[:50]
                    break
            
            return AcknowledgmentType.PREFERENCE_STATED, {
                "preference": preference_match or user_input,
                "preference_type": "general"
            }
        
        # Check for clarifications
        clarification_patterns = [
            r'\b(i mean|what i meant|actually|to clarify)\b',
            r'\b(let me explain|more specifically)\b'
        ]
        
        if any(re.search(pattern, user_lower) for pattern in clarification_patterns):
            return AcknowledgmentType.CLARIFICATION, {"clarification": user_input}
        
        # Default to preference stated if none of the above
        return AcknowledgmentType.PREFERENCE_STATED, {"preference": user_input}
    
    def generate_acknowledgment(
        self,
        acknowledgment_type: AcknowledgmentType,
        context_data: Dict[str, Any],
        slang_level: SlangLevel = SlangLevel.MEDIUM,
        include_transition: bool = True
    ) -> str:
        """
        Generate a natural acknowledgment based on type and context.
        
        Args:
            acknowledgment_type: Type of acknowledgment needed
            context_data: Context data from detection
            slang_level: Level of casual language to use
            include_transition: Whether to include transition phrase
            
        Returns:
            Generated acknowledgment string
        """
        # Get patterns for this type and slang level
        patterns = self.acknowledgment_patterns.get(acknowledgment_type, {})
        level_patterns = patterns.get(slang_level, patterns.get(SlangLevel.MEDIUM, []))
        
        if not level_patterns:
            # Fallback
            return "Got it! Let me help you with that."
        
        # Filter out recently used patterns to avoid repetition
        available_patterns = [p for p in level_patterns if p not in self.recent_acknowledgments]
        if not available_patterns:
            available_patterns = level_patterns  # Reset if all used
        
        # Select random pattern
        selected_pattern = random.choice(available_patterns)
        
        # Track usage
        self.recent_acknowledgments.append(selected_pattern)
        if len(self.recent_acknowledgments) > self.max_recent_history:
            self.recent_acknowledgments.pop(0)
        
        # Fill in context data
        try:
            if acknowledgment_type == AcknowledgmentType.PREFERENCE_STATED:
                preference = context_data.get("preference", "that")
                acknowledgment = selected_pattern.format(preference=preference)
            elif acknowledgment_type == AcknowledgmentType.DISCOVERY_ANSWER:
                answer = context_data.get("answer", "that")
                acknowledgment = selected_pattern.format(answer=answer)
            else:
                acknowledgment = selected_pattern
        except KeyError:
            # Fallback if formatting fails
            acknowledgment = selected_pattern
        
        # Add transition if requested
        if include_transition and acknowledgment_type != AcknowledgmentType.GREETING:
            transition_patterns = self.transition_phrases.get(slang_level, [])
            if transition_patterns:
                transition = random.choice(transition_patterns)
                acknowledgment = f"{acknowledgment} {transition}"
        
        return acknowledgment
    
    def get_slang_level_from_user_memory(self, user_memory: Dict[str, Any]) -> SlangLevel:
        """Extract slang tolerance from user memory"""
        slang_tolerance = user_memory.get("long_term", {}).get("slang_tolerance", "medium")
        
        slang_map = {
            "low": SlangLevel.LOW,
            "medium": SlangLevel.MEDIUM,
            "high": SlangLevel.HIGH
        }
        
        return slang_map.get(slang_tolerance, SlangLevel.MEDIUM)
    
    def create_conversational_acknowledgment(
        self,
        user_input: str,
        user_memory: Dict[str, Any],
        conversation_history: List[Dict[str, Any]] = None,
        user_intent: Dict[str, Any] = None
    ) -> str:
        """
        Main method to create a conversational acknowledgment.
        
        This is the primary interface for generating acknowledgments.
        """
        # Detect acknowledgment type
        ack_type, context_data = self.detect_acknowledgment_type(
            user_input, conversation_history, user_intent
        )
        
        # Get user's slang preference
        slang_level = self.get_slang_level_from_user_memory(user_memory)
        
        # Generate acknowledgment
        acknowledgment = self.generate_acknowledgment(
            acknowledgment_type=ack_type,
            context_data=context_data,
            slang_level=slang_level,
            include_transition=True
        )
        
        logger.info(f"[AcknowledgmentEngine] Generated {ack_type.value} acknowledgment: {acknowledgment[:50]}...")
        
        return acknowledgment
    
    def enhance_response_with_acknowledgment(
        self,
        base_response: str,
        user_input: str,
        user_memory: Dict[str, Any],
        conversation_history: List[Dict[str, Any]] = None
    ) -> str:
        """
        Enhance an existing response by prepending a natural acknowledgment.
        
        This is useful for adding conversational flow to existing responses.
        """
        acknowledgment = self.create_conversational_acknowledgment(
            user_input, user_memory, conversation_history
        )
        
        # Combine acknowledgment with base response
        enhanced_response = f"{acknowledgment}\n\n{base_response}"
        
        return enhanced_response


# Singleton instance
acknowledgment_engine = AcknowledgmentEngine()