from fastapi import APIRouter
from celery.result import AsyncResult
from pydantic import BaseModel
from celery_app import long_task  # Import your Celery app

# Define the router
router = APIRouter()

# Define the Pydantic model
class TaskRequest(BaseModel):
    data: str

@router.post("/submit-task/")
async def submit_task(request: TaskRequest):
    """Submit a lengthy task for processing."""
    # Extract the `data` field from the TaskRequest object
    data = request.data
    print(f"data={data}")
    task = long_task.apply_async(args=[data])  # Offload task to Celery
    return {"task_id": task.id, "status": "submitted"}

@router.get("/task-status/{task_id}")
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
