from fastapi import APIRouter
from celery.result import AsyncResult
from pydantic import BaseModel
from celery_app import long_task  # Import your Celery app

# Define the router
router = APIRouter()

@router.get("/celery/task-status/{task_id}")
async def get_task_status(task_id: str):
    """Check the status of a task."""
    task_result = AsyncResult(task_id)
    if task_result.state == "PENDING":
        return {"task_id": task_id, "status": "pending"}
    elif task_result.state == "SUCCESS":
        return {"task_id": task_id, "status": "completed", "result": task_result.result}
    elif task_result.state == "FAILURE":
        return {"task_id": task_id, "status": "failed", "error": str(task_result.info)}
    else:
        return {"task_id": task_id, "status": task_result.state}
