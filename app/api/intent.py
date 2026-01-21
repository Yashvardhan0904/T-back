from fastapi import APIRouter
from pydantic import BaseModel
from app.services.intent.service import intent_service

router = APIRouter()

class IntentRequest(BaseModel):
    message: str

@router.post("/intent")
async def debug_intent(request: IntentRequest):
    intent = await intent_service.classify_intent(request.message)
    return {"message": request.message, "detected_intent": intent}
