from logger import logger

from fastapi import FastAPI
from nnunet_dataset_json_routes import router as dataset_router
from nnunet_dataset_images_and_labels_routes import router as dataset_images_labels_router
from nnunet_plan_and_preprocess_routes import router as plan_and_preprocess_router
from nnunet_predictions_routes import router as predictions_router
app = FastAPI()

app.include_router(dataset_router)
app.include_router(dataset_images_labels_router)
app.include_router(plan_and_preprocess_router)
app.include_router(predictions_router)

@app.get("/ping")
async def ping():
    logger.info('/ping')
    return {"msg": "Pong"}

@app.get("/")
async def root():
    logger.info('/')
    return {"msg": "Welcome to the FastAPI Celery example 2!"}

from fastapi import Request
import time

# Configure logging
from logger import logger

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware to log request details before processing the request."""
    start_time = time.time()

    client_ip = request.client.host if request.client else "Unknown IP"
    user_agent = request.headers.get("User-Agent", "Unknown User-Agent")
    method = request.method
    url = request.url.path
    query_params = dict(request.query_params)

    logger.info(f"Incoming request: {method} {url} from {client_ip}, User-Agent: {user_agent}, Query: {query_params}")

    # Process request and measure response time
    response = await call_next(request)
    duration = time.time() - start_time

    logger.info(f"Completed request: {method} {url} from {client_ip} in {duration:.3f}s with status {response.status_code}")

    return response

