from fastapi import APIRouter, File, UploadFile, HTTPException
import pandas as pd
from io import BytesIO

from src.data_pipeline import process_dataframe
from src.db import create_dataset

router = APIRouter(prefix="/upload", tags=["Upload"])

@router.post("")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    try:
        content = await file.read()
        raw = pd.read_csv(BytesIO(content))
        cleaned, report = process_dataframe(raw)
        dataset_id = create_dataset(
            file.filename,
            len(cleaned),
            cleaned["datetime"].min(),
            cleaned["datetime"].max(),
        )
        return {
            "dataset_id": dataset_id,
            "filename": file.filename,
            "rows": len(cleaned),
            "report": report,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
