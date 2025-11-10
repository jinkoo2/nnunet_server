from fastapi import FastAPI,Request
from app.core.config import settings
import logging
import time

# logging
from app.core.logging_config import setup_logging, get_logger
setup_logging(settings.LOG_LEVEL)
logger = get_logger(__name__)

# app
logger.info(f"APP_NAME={settings.APP_NAME}")
app = FastAPI(title=settings.APP_NAME, version="1.0.0")

# Middleware to log each request
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000  # ms
    logger.info(
        f"{request.method} {request.url.path} - status: {response.status_code} - {process_time:.2f}ms"
    )
    return response


@app.get("/")
def read_root():
    logger.debug("Root endpoint called")
    return {"app_name": settings.APP_NAME, "log_level": settings.LOG_LEVEL}

# routes
from app.api.v1 import routes_jobs, routes_models, routes_status, routes_raw_dataset_json
app.include_router(routes_raw_dataset_json.router, prefix="/api/v1/raw/datasets", tags=["RawDatasets"])
app.include_router(routes_jobs.router, prefix="/api/v1/jobs", tags=["Jobs"])
app.include_router(routes_models.router, prefix="/api/v1/models", tags=["Models"])
app.include_router(routes_status.router, prefix="/api/v1/status", tags=["Status"])

@app.on_event("startup")
def startup_event():
    logger.info("Starting nnUNet Server...")
    logger.info(f"NNUNet raw dir: {routes_raw_dataset_json.nnunet_raw_dir}")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

