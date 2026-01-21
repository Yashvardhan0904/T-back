from app.db.mongodb import get_database
import asyncio
from bson import ObjectId

async def cleanup_names():
    db = get_database()
    # Find all memories where userName is Trediora or Trendora
    corrupted = await db.usermemories.find({
        "userName": {"$in": ["Trediora", "Trendora", "trediora", "trendora"]}
    }).to_list(length=100)
    
    print(f"Found {len(corrupted)} corrupted user memories.")
    
    for mem in corrupted:
        # If we have a userEmail, we can try to find their real name from the users collection
        # But for now, let's just reset it to None/Gorgeous so LLM fallbacks kick in
        await db.usermemories.update_one(
            {"_id": mem["_id"]},
            {"$set": {"userName": "Gorgeous"}} # Reset to safe fallback
        )
        print(f"Reset name for memory ID: {mem['_id']}")

if __name__ == "__main__":
    from app.db.mongodb import connect_to_mongo, close_mongo_connection
    async def run():
        await connect_to_mongo()
        await cleanup_names()
        await close_mongo_connection()
    asyncio.run(run())
