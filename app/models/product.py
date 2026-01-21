from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class Product(BaseModel):
    id: Optional[str] = Field(alias="_id")
    name: str
    description: str
    price: float
    category: str
    gender: str
    color: Optional[str] = None
    fabric: Optional[str] = None
    fit: Optional[str] = None
    occasion: Optional[str] = None
    images: List[str] = []
    tags: List[str] = []
    sellerId: Optional[str] = None
    stock: int = 0
    embeddings: Optional[List[float]] = None
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
