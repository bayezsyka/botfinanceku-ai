import re
from pathlib import Path

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


MODEL_VERSION = "local-dev-0.2"

DATA_DIR = Path("data")
TRAINING_FILE = DATA_DIR / "training_data.csv"

CATEGORIES = [
    "makan & minum",
    "jajan",
    "operasional",
    "dan lain-lain",
]


INITIAL_DATA = [
    {"subject": "sarapan", "amount": 14000, "category": "makan & minum"},
    {"subject": "sarapan subuh", "amount": 22000, "category": "makan & minum"},
    {"subject": "ketoprak", "amount": 14000, "category": "makan & minum"},
    {"subject": "makan malam", "amount": 19000, "category": "makan & minum"},
    {"subject": "makan siang dan minum", "amount": 18000, "category": "makan & minum"},
    {"subject": "indomie telor dan es", "amount": 15000, "category": "makan & minum"},

    {"subject": "kopken", "amount": 19000, "category": "jajan"},
    {"subject": "es teler", "amount": 13000, "category": "jajan"},

    {"subject": "parkir", "amount": 2000, "category": "operasional"},
    {"subject": "beli galon", "amount": 23000, "category": "operasional"},

    {"subject": "beli softlens", "amount": 76000, "category": "dan lain-lain"},
]


model = None
training_df = None


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_training_data() -> pd.DataFrame:
    base_df = pd.DataFrame(INITIAL_DATA)

    if TRAINING_FILE.exists():
        feedback_df = pd.read_csv(TRAINING_FILE)
        combined_df = pd.concat([base_df, feedback_df], ignore_index=True)
    else:
        combined_df = base_df

    combined_df["clean_subject"] = combined_df["subject"].apply(clean_text)

    return combined_df


def build_model(df: pd.DataFrame):
    new_model = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        )),
    ])

    new_model.fit(df["clean_subject"], df["category"])

    return new_model


def train_model():
    global model, training_df

    training_df = load_training_data()
    model = build_model(training_df)

    return {
        "message": "Model berhasil dilatih ulang.",
        "total_training_data": len(training_df),
        "model_version": MODEL_VERSION,
    }


def predict_category(subject: str):
    if model is None:
        train_model()

    clean_subject = clean_text(subject)

    predicted_category = model.predict([clean_subject])[0]
    probabilities = model.predict_proba([clean_subject])[0]

    candidates = []

    for category, probability in zip(model.classes_, probabilities):
        candidates.append({
            "category": category,
            "confidence": round(float(probability), 4),
        })

    candidates = sorted(
        candidates,
        key=lambda item: item["confidence"],
        reverse=True,
    )

    top_confidence = candidates[0]["confidence"]

    return {
        "subject": subject,
        "clean_subject": clean_subject,
        "predicted_category": predicted_category,
        "confidence": top_confidence,
        "is_confident": top_confidence >= 0.5,
        "candidates": candidates,
        "model_version": MODEL_VERSION,
    }


def add_feedback(subject: str, amount: int | None, correct_category: str):
    if correct_category not in CATEGORIES:
        raise ValueError(f"Kategori tidak valid: {correct_category}")

    DATA_DIR.mkdir(exist_ok=True)

    new_row = pd.DataFrame([{
        "subject": subject,
        "amount": amount or 0,
        "category": correct_category,
    }])

    if TRAINING_FILE.exists():
        old_df = pd.read_csv(TRAINING_FILE)
        updated_df = pd.concat([old_df, new_row], ignore_index=True)
    else:
        updated_df = new_row

    updated_df.to_csv(TRAINING_FILE, index=False)

    train_result = train_model()
    prediction_after_training = predict_category(subject)

    return {
        "message": "Feedback disimpan dan model dilatih ulang.",
        "saved_feedback": {
            "subject": subject,
            "amount": amount,
            "correct_category": correct_category,
        },
        "train_result": train_result,
        "prediction_after_training": prediction_after_training,
    }


train_model()