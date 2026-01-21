from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import model_validator
from functools import lru_cache
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    PROJECT_NAME: str = "Trendora AI AI Orchestrator"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # MongoDB
    MONGODB_URI: str
    DB_NAME: str
    
    # Server
    PORT: int = 8000

    # AI Providers
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_MODEL_SMALL: str = "llama-3.1-8b-instant"
    
    # External Embeddings (Hugging Face / Jina)
    # Using free Hugging Face Inference API for deployment
    HF_TOKEN: str = ""
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EXT_EMBEDDING_API_URL: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )

    @model_validator(mode='after')
    def compute_embedding_url(self):
        if not self.EXT_EMBEDDING_API_URL:
            self.EXT_EMBEDDING_API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.EMBEDDING_MODEL}"
        return self

@lru_cache()
def get_settings():
    return Settings()
