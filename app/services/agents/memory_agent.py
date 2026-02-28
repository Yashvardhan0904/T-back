"""
Memory Agent - Manages long-term and short-term user memory.

Responsibilities:
- Store user preferences (brand, price range, category, style)
- Store behavioral signals (clicks, purchases, rejections)
- Summarize conversation context
- Retrieve relevant memory when requested
"""

from typing import Dict, List, Any, Optional
from app.db.mongodb import get_database
from app.services.memory.service import memory_service
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


class MemoryAgent:
    """
    Manages user memory with three types:
    - SHORT_TERM: current session intent
    - LONG_TERM: stable preferences
    - BEHAVIORAL: actions & feedback
    """
    
    async def retrieve_memory(
        self,
        user_id: str,
        email: Optional[str] = None,
        chat_id: Optional[str] = None,
        memory_types: List[str] = None
    ) -> Dict[str, Any]:
        """
        Retrieve user memory in structured JSON format.
        
        Returns:
        {
            "short_term": {...},
            "long_term": {...},
            "behavioral": {...},
            "conversation_summary": "..."
        }
        """
        if memory_types is None:
            memory_types = ["SHORT_TERM", "LONG_TERM", "BEHAVIORAL"]
        
        # Get user data from existing memory service
        user_data = await memory_service.get_user_memory(user_id, email, chat_id)
        memory = user_data.get("memory", {})
        context = user_data.get("context", {})
        
        # Extract structured memory
        preferences = memory.get("preferences", {}) if memory else {}
        style = memory.get("style", {}) if memory else {}
        behavior = memory.get("behavior", {}) if memory else {}
        
        result = {
            "short_term": {
                "current_intent": context.get("current_intent"),
                "session_preferences": context.get("session_preferences", {}),
                "recent_queries": context.get("recent_queries", []),
                "buying_type": context.get("buying_type")
            },
            "long_term": {
                "preferred_price_range": preferences.get("price_range"),
                "interested_categories": preferences.get("categories", []),
                "preferred_brands": preferences.get("brands", []),
                "style_preferences": style,
                "behavioral_patterns": behavior,
                "user_name": memory.get("userName") if memory else None,
                "gender": (memory.get("physical") or {}).get("gender") if memory else None,
                # Gen-Z fashion memory fields
                "fashion_aesthetics": style.get("aesthetics", []),
                "preferred_fit": style.get("fit", "regular"),
                "disliked_fits": style.get("disliked_fits", []),
                "style_vibe": style.get("vibe", "casual"),
                "slang_tolerance": behavior.get("slang_tolerance", "medium"),
                "occasion_preference": behavior.get("occasions", []),
                # NEW: Size fields
                "size": (memory.get("physical") or {}).get("size") if memory else None,
                "shoe_size": (memory.get("physical") or {}).get("shoeSize") if memory else None,
                "design_preference": style.get("design_preference"),
                # NEW: Likes/Dislikes with reasons
                "likes": (memory.get("revealed") or {}).get("likes", []) if memory else [],
                "dislikes": (memory.get("revealed") or {}).get("dislikes", []) if memory else []
            },
            "behavioral": {
                "recent_clicks": [],
                "recent_purchases": [],
                "recent_rejections": []
            },
            "conversation_summary": self._summarize_conversation(context.get("conversationHistory", []))
        }
        
        # Load behavioral data from database
        db = get_database()
        try:
            # Get recent interactions
            interactions = await db.fashion_interactions.find({
                "$or": [
                    {"userId": user_id},
                    {"userEmail": email}
                ]
            }).sort("timestamp", -1).limit(10).to_list(length=10)
            
            for interaction in interactions:
                action = interaction.get("action", "")
                if action == "liked" or action == "purchased":
                    result["behavioral"]["recent_clicks"].append({
                        "item_id": interaction.get("outfitId"),
                        "action": action,
                        "timestamp": interaction.get("timestamp")
                    })
                elif action == "rejected":
                    result["behavioral"]["recent_rejections"].append({
                        "item_id": interaction.get("outfitId"),
                        "timestamp": interaction.get("timestamp")
                    })
        except Exception as e:
            logger.error(f"Error loading behavioral data: {e}")
        
        return result
    
    async def update_memory(
        self,
        user_id: str,
        email: Optional[str] = None,
        memory_type: str = "LONG_TERM",
        data: Dict[str, Any] = None
    ) -> bool:
        """
        Update user memory with new preferences.
        
        Args:
            memory_type: "SHORT_TERM" | "LONG_TERM" | "BEHAVIORAL"
            data: Dictionary of preferences to update
        """
        if not data:
            return False
        
        try:
            db = get_database()
            query = {"$or": []}
            if email:
                query["$or"].append({"userEmail": email})
            if user_id and user_id != "guest":
                query["$or"].append({"userId": user_id})
            
            if not query["$or"]:
                return False
            
            update_dict = {"$set": {"lastUpdated": datetime.utcnow()}}
            
            # Map memory updates to UserMemory schema
            if memory_type == "LONG_TERM":
                # Initialize $addToSet if needed
                if "$addToSet" not in update_dict:
                    update_dict["$addToSet"] = {}
                
                # Update long-term preferences
                if "user_name" in data and data["user_name"]:
                    update_dict["$set"]["userName"] = data["user_name"]
                
                if "gender" in data and data["gender"]:
                    update_dict["$set"]["physical.gender"] = data["gender"]

                if "preferred_price_range" in data:
                    price_range = data["preferred_price_range"]
                    if isinstance(price_range, str):
                        # Parse "under 2000" -> max: 2000
                        if "under" in price_range.lower():
                            try:
                                max_price = int(''.join(filter(str.isdigit, price_range)))
                                update_dict["$set"]["preferences.price_range.max"] = max_price
                            except ValueError:
                                pass
                        elif "over" in price_range.lower():
                            try:
                                min_price = int(''.join(filter(str.isdigit, price_range)))
                                update_dict["$set"]["preferences.price_range.min"] = min_price
                            except ValueError:
                                pass
                
                if "interested_categories" in data:
                    categories = data["interested_categories"]
                    if isinstance(categories, list) and categories:
                        update_dict["$addToSet"]["preferences.categories"] = {"$each": categories}
                
                if "preferred_brands" in data:
                    brands = data["preferred_brands"]
                    if isinstance(brands, list) and brands:
                        update_dict["$addToSet"]["preferences.brands"] = {"$each": brands}
            
            # Handle fit preferences
            if "preferred_fit" in data:
                update_dict["$set"]["style.fit"] = data["preferred_fit"]
            
            if "disliked_fits" in data:
                disliked_fits = data["disliked_fits"]
                if isinstance(disliked_fits, list) and disliked_fits:
                    update_dict["$addToSet"]["style.disliked_fits"] = {"$each": disliked_fits}
            
            # Use existing memory service for preference extraction
            if data:
                # Extract preferences from data dict
                message = json.dumps(data)
                await memory_service.extract_preferences(user_id, email, message)
            
            # Execute update if we have changes
            if "$set" in update_dict and len(update_dict["$set"]) > 1:  # More than just lastUpdated
                await db.usermemories.update_one(query, update_dict, upsert=True)
            
            # Step 1.7: Update session context for short-term data (buying_type)
            if "buying_type" in data and data["buying_type"]:
                chat_session_id = None
                if chat_id:
                    try: chat_session_id = ObjectId(chat_id)
                    except: pass
                
                if chat_session_id:
                    await db.aicontexts.update_one(
                        {"chatSessionId": chat_session_id},
                        {"$set": {"buying_type": data["buying_type"]}},
                        upsert=True
                    )
                    logger.info(f"Updated buying_type='{data['buying_type']}' for session {chat_id}")
            
            logger.info(f"Updated {memory_type} memory for {email or user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating memory: {e}")
            return False
    
    def _summarize_conversation(self, history: List[Dict[str, Any]]) -> str:
        """Summarize conversation history for context."""
        if not history:
            return "No previous conversation."
        
        recent = history[-5:]  # Last 5 turns
        summary = "Recent conversation: "
        for turn in recent:
            role = turn.get("role", "user")
            content = turn.get("content", "")[:50]  # First 50 chars
            summary += f"{role}: {content}... "
        
        return summary
    
    async def store_behavioral_signal(
        self,
        user_id: str,
        email: Optional[str] = None,
        action: str = "clicked",
        item_id: str = None,
        features: Dict[str, Any] = None
    ) -> bool:
        """Store behavioral signal (click, purchase, rejection)."""
        try:
            db = get_database()
            await db.fashion_interactions.insert_one({
                "userId": user_id,
                "userEmail": email,
                "outfitId": item_id or "unknown",
                "action": action,
                "features": features or {},
                "timestamp": datetime.utcnow()
            })
            return True
        except Exception as e:
            logger.error(f"Error storing behavioral signal: {e}")
            return False


# Singleton instance
memory_agent = MemoryAgent()
