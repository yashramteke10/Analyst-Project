from fastapi import APIRouter

router = APIRouter(prefix="/predict", tags=["Prediction"])

@router.get("/{run_id}")
def predict(run_id: int):
    return {
        "run_id": run_id,
        "message": "Prediction endpoint registered. Use the saved model artifact for inference.",
    }
