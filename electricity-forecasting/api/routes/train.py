from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/train", tags=["Training"])

class TrainRequest(BaseModel):
    dataset_id: int
    model: str = "LSTM"

@router.post("")
def train_model(request: TrainRequest):
    # Full API orchestration can reuse the same modules as app.py.
    # The Streamlit MVP remains the primary application entry point.
    return {
        "message": "Training endpoint registered.",
        "dataset_id": request.dataset_id,
        "model": request.model,
        "note": "Connect this endpoint to the training service for decoupled deployment.",
    }
