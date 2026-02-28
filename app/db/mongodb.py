from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import get_settings
import logging

settings = get_settings()
logger = logging.getLogger(__name__)

class MongoDB:
    client: AsyncIOMotorClient = None
    db = None

db = MongoDB()

async def connect_to_mongo():
    db.client = AsyncIOMotorClient(settings.MONGODB_URI)
    # Use Yashvardhan database (override DB_NAME if it's set to admin)
    db_name = settings.DB_NAME if settings.DB_NAME != "admin" else "Yashvardhan"
    db.db = db.client[db_name]
    try:
        # Verify connection
        await db.client.admin.command('ping')
        logger.info("✅ Connected to MongoDB: %s", db_name)
    except Exception as e:
        logger.error("❌ Could not connect to MongoDB: %s", e)
        raise e

async def close_mongo_connection():
    if db.client:
        db.client.close()
        logger.info("🔌 Closed MongoDB connection")

def get_database():
    return db.db
