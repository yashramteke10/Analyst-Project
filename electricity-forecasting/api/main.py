from fastapi import FastAPI
from api.routes import upload, train, predict

app = FastAPI(
    title="Electricity Forecasting API",
    version="1.0.0",
    description="Optional REST API for the electricity forecasting project.",
)

app.include_router(upload.router)
app.include_router(train.router)
app.include_router(predict.router)

@app.get("/")
def root():
    return {
        "project": "Electricity Consumption Forecasting",
        "status": "API running",
    }
