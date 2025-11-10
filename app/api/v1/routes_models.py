from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def list_models():
    return {"message": "List of models"}
