import uvicorn

def run_dev():
    uvicorn.run("app.main:app", host="0.0.0.0", port=9000, reload=True)
