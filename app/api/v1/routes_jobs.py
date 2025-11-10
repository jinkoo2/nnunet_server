from fastapi import APIRouter

router = APIRouter()

@router.get("/")
async def list_jobs():
    return {"message": "List of jobs"}

@router.post("/")
async def submit_job():
    return {"message": "Job submitted"}
