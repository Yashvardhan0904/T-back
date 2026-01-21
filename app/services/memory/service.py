from app.db.mongodb import get_database
from bson import ObjectId
from datetime import datetime
import logging
from groq import Groq
from app.core.config import get_settings
import json

settings = get_settings()
logger = logging.getLogger(__name__)

class MemoryService:
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)

    async def get_user_memory(self, user_id: str, email: str = None, chat_id: str = None) -> dict:
        """
        Fetches UserMemory (DNA) and Full Chat Session History.
        """
        db = get_database()
        logger.info(f"Loading memory for user_id={user_id}, email={email}, chat_id={chat_id}")
        
        # 1. Prepare DNA query
        dna_query = {"$or": []}
        if email: dna_query["$or"].append({"userEmail": email})
        if user_id and user_id != "guest": dna_query["$or"].append({"userId": user_id})
        
        # 2. Fetch UserMemory (Style DNA)
        memory = None
        if dna_query["$or"]:
            try:
                memory = await db.usermemories.find_one(dna_query)
                if memory: 
                    memory["_id"] = str(memory["_id"])
                    logger.info(f"Loaded DNA for {email or user_id}")
            except Exception as e:
                logger.error(f"Error fetching UserMemory: {e}")

        # 3. Fetch Session History (Entire Chat Session)
        session_history = []
        if chat_id and isinstance(chat_id, str) and chat_id not in ["null", "undefined", ""]:
            try:
                # chat_id from frontend is usually an ObjectId string
                if len(chat_id) == 24:
                    cursor = db.messages.find({"chatSessionId": ObjectId(chat_id)}).sort("createdAt", 1)
                    msgs = await cursor.to_list(length=100) # Entire session, up to 100 turns
                    for m in msgs:
                        session_history.append({
                            "role": m.get("role", "user"),
                            "content": m.get("content", "")
                        })
                    logger.info(f"Loaded {len(session_history)} messages from session {chat_id}")
                else:
                    logger.warning(f"Invalid chatId format: {chat_id}")
            except Exception as e:
                logger.error(f"Failed to fetch session messages for {chat_id}: {e}")

        # 4. Fallback: If no chat_id, use global AIContext turns
        if not session_history and dna_query["$or"]:
            try:
                context = await db.aicontexts.find_one(dna_query, sort=[("lastUpdated", -1)])
                if context:
                    session_history = context.get("conversationHistory", [])
                    logger.info(f"Fallback: Loaded {len(session_history)} messages from AIContext")
            except Exception as e:
                logger.error(f"Error fetching AIContext: {e}")

        return {
            "memory": memory,
            "context": {"conversationHistory": session_history}
        }

    async def save_interaction(self, user_id: str, email: str, role: str, content: str, intent: str = None):
        """
        Appends a new turn to the AIContext.
        """
        db = get_database()
        if not email and (not user_id or user_id == "guest"): return

        query = {"$or": []}
        if email: query["$or"].append({"userEmail": email})
        if user_id and user_id != "guest": query["$or"].append({"userId": user_id})

        new_turn = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow()
        }

        try:
            # Update history and lastUpdated
            await db.aicontexts.update_one(
                query,
                {
                    "$push": {
                        "conversationHistory": {"$each": [new_turn], "$slice": -20}, # Keep last 20
                        "intentHistory": {"$each": [intent] if intent else [], "$slice": -20}
                    },
                    "$set": {
                        "lastUpdated": datetime.utcnow(),
                        "userEmail": email,
                        "userId": user_id
                    }
                },
                upsert=True
            )
        except Exception as e:
            logger.error(f"Failed to save context: {e}")

    async def extract_preferences(self, user_id: str, email: str, message: str):
        """
        Uses Groq to see if the user expressed a preference and updates UserMemory.
        """
        db = get_database()
        if not email and (not user_id or user_id == "guest"): return

        prompt = f"""
        Extract user style preferences from this message for a shopping AI DNA.
        Message: "{message}"

        Return JSON ONLY:
        - gender: (e.g. "male", "female", "non-binary")
        - style_identity: (e.g. "masculine", "feminine", "androgynous")
        - style_fit: (e.g. "oversized", "slim")
        - vibe: (e.g. "streetwear", "minimalist")
        - nickname: (string, the name user wants to be called)

        Rules:
        1. Only include fields where a preference was explicitly or strongly implied.
        2. If user says "I am a boy/man/guy", set gender="male", style_identity="masculine".
        3. If user says "I am a girl/woman", set gender="female", style_identity="feminine".
        4. If user says "Don't call me X" or "Call me Y", set nickname.
        5. NEVER set nickname to Brand names (Trendora, etc).
        6. Use null for fields with no new evidence.
        """

        try:
            completion = self.client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            data = json.loads(completion.choices[0].message.content)
            
            updates = {}
            if data.get("gender"): updates["behavior.gender"] = data["gender"]
            if data.get("style_identity"): updates["style.identity"] = data["style_identity"]
            if data.get("style_fit"): updates["style.fit"] = data["style_fit"]
            if data.get("vibe"): updates["style.vibe"] = data["vibe"]
            if data.get("nickname"): updates["userName"] = data["nickname"]

            if updates:
                query = {"$or": []}
                if email: query["$or"].append({"userEmail": email})
                if user_id and user_id != "guest": query["$or"].append({"userId": user_id})

                # Construct the actual mongo update
                mongo_update = {"$set": {"lastUpdated": datetime.utcnow()}}
                
                # Handle $each for color/avoids
                for k, v in updates.items():
                    if isinstance(v, dict) and "$each" in v:
                        if "$addToSet" not in mongo_update: mongo_update["$addToSet"] = {}
                        mongo_update["$addToSet"][k] = v
                    else:
                        mongo_update["$set"][k] = v

                await db.usermemories.update_one(query, mongo_update, upsert=True)
                logger.info(f"Updated DNA for {email or user_id}: {updates}")

        except Exception as e:
            logger.error(f"Preference Extraction Error: {e}")

memory_service = MemoryService()
