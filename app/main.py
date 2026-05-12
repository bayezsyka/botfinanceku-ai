from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.model import (
    CATEGORIES,
    MODEL_VERSION,
    add_feedback,
    predict_category,
    train_model,
)


app = FastAPI(
    title="BotFinanceku AI",
    description="AI mini untuk memprediksi kategori pengeluaran.",
    version="0.2.0",
)


class PredictRequest(BaseModel):
    subject: str = Field(..., min_length=1, examples=["kopken"])
    amount: int | None = Field(default=None, examples=[19000])


class FeedbackRequest(BaseModel):
    subject: str = Field(..., min_length=1, examples=["servis kipas"])
    amount: int | None = Field(default=None, examples=[25000])
    correct_category: str = Field(..., examples=["operasional"])


@app.get("/")
def root():
    return {
        "message": "BotFinanceku AI aktif",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "categories": CATEGORIES,
    }


@app.post("/predict")
def predict(payload: PredictRequest):
    result = predict_category(payload.subject)

    return {
        "subject": payload.subject,
        "amount": payload.amount,
        "predicted_category": result["predicted_category"],
        "confidence": result["confidence"],
        "is_confident": result["is_confident"],
        "candidates": result["candidates"],
        "model_version": result["model_version"],
    }


@app.post("/feedback")
def feedback(payload: FeedbackRequest):
    try:
        return add_feedback(
            subject=payload.subject,
            amount=payload.amount,
            correct_category=payload.correct_category,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.post("/train")
def train():
    return train_model()