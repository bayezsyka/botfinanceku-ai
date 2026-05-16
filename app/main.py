from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.model import (
    CATEGORIES,
    MODEL_VERSION,
    add_feedback,
    predict_category,
    train_model,
    normalize_category,
)
from app.supabase_client import update_expense_confirmation


app = FastAPI(
    title="BotFinanceku AI",
    description="AI mini untuk memprediksi kategori pengeluaran.",
    version="0.3.0",
)


class PredictRequest(BaseModel):
    subject: str = Field(..., min_length=1, examples=["beli kopi"])
    amount: int | None = Field(default=None, examples=[18000])


class FeedbackRequest(BaseModel):
    subject: str = Field(..., min_length=1, examples=["servis motor"])
    amount: int | None = Field(default=None, examples=[50000])
    correct_category: str = Field(..., examples=["transportasi"])


class ConfirmRequest(BaseModel):
    expense_id: str = Field(..., min_length=1)
    subject: str = Field(..., min_length=1)
    amount: int
    correct_category: str = Field(..., examples=["jajan"])


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


@app.post("/confirm")
def confirm(payload: ConfirmRequest):
    norm_category = normalize_category(payload.correct_category)
    if norm_category not in CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"Kategori tidak valid: {payload.correct_category}",
        )

    try:
        updated_row = update_expense_confirmation(
            expense_id=payload.expense_id,
            category=norm_category,
        )

        feedback_result = add_feedback(
            subject=payload.subject,
            amount=payload.amount,
            correct_category=norm_category,
        )

        return {
            "message": "Transaksi dikonfirmasi dan AI dilatih.",
            "updated_row": updated_row,
            "feedback_result": feedback_result,
        }
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/train")
def train():
    return train_model()
