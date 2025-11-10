from fastapi import FastAPI
from app.api.v1 import routes_jobs, routes_models, routes_status

app = FastAPI(title="nnUNet Server", version="1.0.0")

app.include_router(routes_jobs.router, prefix="/api/v1/jobs", tags=["Jobs"])
app.include_router(routes_models.router, prefix="/api/v1/models", tags=["Models"])
app.include_router(routes_status.router, prefix="/api/v1/status", tags=["Status"])
