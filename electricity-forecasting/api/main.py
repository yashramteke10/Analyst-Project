"""Optional FastAPI entry point; install fastapi and uvicorn to use it."""
try:
    from fastapi import FastAPI
    app = FastAPI(title="Electricity Forecasting API")
    @app.get("/health")
    def health():
        return {"status": "ok"}
except ImportError:
    app = None
