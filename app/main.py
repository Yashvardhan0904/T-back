from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import chat, intent, search
from app.db.mongodb import connect_to_mongo, close_mongo_connection
from app.core.config import get_settings
import logging

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

settings = None  # global, initialized on startup


@app.on_event("startup")
async def startup_event():
    global settings
    settings = get_settings()

    logger.error("USING MONGODB_URI = %s", settings.MONGODB_URI)

    # Update app metadata AFTER settings load
    app.title = settings.PROJECT_NAME
    app.version = settings.VERSION

    await connect_to_mongo()


@app.on_event("shutdown")
async def shutdown_event():
    await close_mongo_connection()


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(chat.router, prefix="/api", tags=["Chat"])
app.include_router(intent.router, prefix="/api", tags=["Debug"])
app.include_router(search.router, prefix="/api", tags=["Debug"])


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Trendora AI Orchestrator"}
