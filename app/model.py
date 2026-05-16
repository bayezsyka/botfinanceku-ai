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
    "makan",
    "jajan",
    "kebutuhan_kos",
    "tagihan",
    "laundry",
    "transportasi",
    "kesehatan",
    "hiburan",
    "sosial",
    "belanja_pribadi",
    "edukasi",
    "lain_lain",
]


INITIAL_DATA = [
    {"subject": "sarapan", "amount": 15000, "category": "makan"},
    {"subject": "makan siang", "amount": 20000, "category": "makan"},
    {"subject": "makan malam", "amount": 20000, "category": "makan"},
    {"subject": "makan sore", "amount": 15000, "category": "makan"},
    {"subject": "makan penyetan", "amount": 18000, "category": "makan"},
    {"subject": "ketoprak", "amount": 14000, "category": "makan"},
    {"subject": "nasi ayam", "amount": 15000, "category": "makan"},
    {"subject": "ayam geprek", "amount": 16000, "category": "makan"},
    {"subject": "indomie telor", "amount": 12000, "category": "makan"},
    {"subject": "indomie telor dan es", "amount": 15000, "category": "makan"},
    {"subject": "nasi padang", "amount": 20000, "category": "makan"},
    {"subject": "warteg", "amount": 15000, "category": "makan"},
    {"subject": "makan di burjo", "amount": 15000, "category": "makan"},

    {"subject": "beli kopi", "amount": 18000, "category": "jajan"},
    {"subject": "kopken", "amount": 19000, "category": "jajan"},
    {"subject": "gooday cappucino", "amount": 6000, "category": "jajan"},
    {"subject": "es gooday", "amount": 5000, "category": "jajan"},
    {"subject": "mocafrio", "amount": 15000, "category": "jajan"},
    {"subject": "es teler", "amount": 13000, "category": "jajan"},
    {"subject": "snack", "amount": 10000, "category": "jajan"},
    {"subject": "cemilan", "amount": 12000, "category": "jajan"},
    {"subject": "minuman manis", "amount": 10000, "category": "jajan"},
    {"subject": "roti", "amount": 8000, "category": "jajan"},
    {"subject": "gorengan", "amount": 5000, "category": "jajan"},
    {"subject": "boba", "amount": 15000, "category": "jajan"},

    {"subject": "beli galon", "amount": 23000, "category": "kebutuhan_kos"},
    {"subject": "galon", "amount": 23000, "category": "kebutuhan_kos"},
    {"subject": "sabun", "amount": 15000, "category": "kebutuhan_kos"},
    {"subject": "spons", "amount": 5000, "category": "kebutuhan_kos"},
    {"subject": "penyerap lembap", "amount": 18000, "category": "kebutuhan_kos"},
    {"subject": "sendok garpu", "amount": 10000, "category": "kebutuhan_kos"},
    {"subject": "alat kamar", "amount": 25000, "category": "kebutuhan_kos"},
    {"subject": "pewangi kamar", "amount": 15000, "category": "kebutuhan_kos"},
    {"subject": "tisu", "amount": 12000, "category": "kebutuhan_kos"},
    {"subject": "deterjen", "amount": 20000, "category": "kebutuhan_kos"},
    {"subject": "rak kecil", "amount": 50000, "category": "kebutuhan_kos"},

    {"subject": "listrik", "amount": 100000, "category": "tagihan"},
    {"subject": "bayar listrik", "amount": 100000, "category": "tagihan"},
    {"subject": "internet", "amount": 150000, "category": "tagihan"},
    {"subject": "wifi", "amount": 150000, "category": "tagihan"},
    {"subject": "air", "amount": 50000, "category": "tagihan"},
    {"subject": "iuran kos", "amount": 50000, "category": "tagihan"},
    {"subject": "tagihan kos", "amount": 50000, "category": "tagihan"},
    {"subject": "pulsa", "amount": 50000, "category": "tagihan"},
    {"subject": "paket data", "amount": 100000, "category": "tagihan"},

    {"subject": "laundry", "amount": 30000, "category": "laundry"},
    {"subject": "laundry kiloan", "amount": 30000, "category": "laundry"},
    {"subject": "cuci pakaian", "amount": 25000, "category": "laundry"},
    {"subject": "setrika", "amount": 20000, "category": "laundry"},
    {"subject": "cuci sepatu", "amount": 35000, "category": "laundry"},

    {"subject": "isi bensin", "amount": 30000, "category": "transportasi"},
    {"subject": "bensin", "amount": 30000, "category": "transportasi"},
    {"subject": "parkir", "amount": 2000, "category": "transportasi"},
    {"subject": "ojek", "amount": 15000, "category": "transportasi"},
    {"subject": "gojek", "amount": 20000, "category": "transportasi"},
    {"subject": "grab", "amount": 20000, "category": "transportasi"},
    {"subject": "tol", "amount": 15000, "category": "transportasi"},
    {"subject": "servis motor", "amount": 150000, "category": "transportasi"},
    {"subject": "service motor", "amount": 150000, "category": "transportasi"},
    {"subject": "tambal ban", "amount": 15000, "category": "transportasi"},
    {"subject": "cuci motor", "amount": 20000, "category": "transportasi"},

    {"subject": "obat", "amount": 25000, "category": "kesehatan"},
    {"subject": "beli obat", "amount": 30000, "category": "kesehatan"},
    {"subject": "urut", "amount": 100000, "category": "kesehatan"},
    {"subject": "vitamin", "amount": 40000, "category": "kesehatan"},
    {"subject": "softlens", "amount": 76000, "category": "kesehatan"},
    {"subject": "dokter", "amount": 150000, "category": "kesehatan"},
    {"subject": "periksa", "amount": 100000, "category": "kesehatan"},
    {"subject": "klinik", "amount": 80000, "category": "kesehatan"},
    {"subject": "minyak kayu putih", "amount": 15000, "category": "kesehatan"},
    {"subject": "masker", "amount": 10000, "category": "kesehatan"},
    {"subject": "potong rambut", "amount": 35000, "category": "kesehatan"},

    {"subject": "futsal", "amount": 40000, "category": "hiburan"},
    {"subject": "nonton", "amount": 50000, "category": "hiburan"},
    {"subject": "game", "amount": 100000, "category": "hiburan"},
    {"subject": "nongkrong", "amount": 50000, "category": "hiburan"},
    {"subject": "billiard", "amount": 75000, "category": "hiburan"},
    {"subject": "konser", "amount": 300000, "category": "hiburan"},
    {"subject": "jalan-jalan", "amount": 150000, "category": "hiburan"},
    {"subject": "karaoke", "amount": 80000, "category": "hiburan"},

    {"subject": "bayar dimas", "amount": 50000, "category": "sosial"},
    {"subject": "bayar teman", "amount": 100000, "category": "sosial"},
    {"subject": "transfer teman", "amount": 50000, "category": "sosial"},
    {"subject": "patungan", "amount": 30000, "category": "sosial"},
    {"subject": "traktir", "amount": 150000, "category": "sosial"},
    {"subject": "kas kelas", "amount": 20000, "category": "sosial"},
    {"subject": "sumbangan", "amount": 50000, "category": "sosial"},
    {"subject": "kado", "amount": 100000, "category": "sosial"},
    {"subject": "urunan", "amount": 25000, "category": "sosial"},

    {"subject": "baju", "amount": 150000, "category": "belanja_pribadi"},
    {"subject": "sandal", "amount": 50000, "category": "belanja_pribadi"},
    {"subject": "sepatu", "amount": 300000, "category": "belanja_pribadi"},
    {"subject": "tas", "amount": 200000, "category": "belanja_pribadi"},
    {"subject": "aksesoris", "amount": 50000, "category": "belanja_pribadi"},
    {"subject": "jam tangan", "amount": 250000, "category": "belanja_pribadi"},
    {"subject": "parfum", "amount": 150000, "category": "belanja_pribadi"},
    {"subject": "dompet", "amount": 100000, "category": "belanja_pribadi"},
    {"subject": "celana", "amount": 150000, "category": "belanja_pribadi"},
    {"subject": "hoodie", "amount": 200000, "category": "belanja_pribadi"},

    {"subject": "buku", "amount": 80000, "category": "edukasi"},
    {"subject": "beli buku", "amount": 80000, "category": "edukasi"},
    {"subject": "kursus", "amount": 250000, "category": "edukasi"},
    {"subject": "software belajar", "amount": 100000, "category": "edukasi"},
    {"subject": "alat produktivitas", "amount": 50000, "category": "edukasi"},
    {"subject": "langganan belajar", "amount": 150000, "category": "edukasi"},
    {"subject": "print tugas", "amount": 10000, "category": "edukasi"},
    {"subject": "fotokopi", "amount": 5000, "category": "edukasi"},
    {"subject": "alat tulis", "amount": 20000, "category": "edukasi"},
    {"subject": "pulpen", "amount": 5000, "category": "edukasi"},

    {"subject": "tes", "amount": 1000, "category": "lain_lain"},
    {"subject": "pengeluaran lain", "amount": 50000, "category": "lain_lain"},
    {"subject": "biaya tak dikenal", "amount": 25000, "category": "lain_lain"},
    {"subject": "transaksi tidak jelas", "amount": 30000, "category": "lain_lain"},
    {"subject": "random", "amount": 15000, "category": "lain_lain"},
    {"subject": "lain lain", "amount": 10000, "category": "lain_lain"},
]


model = None
training_df = None


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_category(category: str) -> str:
    cat = str(category).lower().strip()
    mapping = {
        "makan & minum": "makan",
        "makan dan minum": "makan",
        "operasional": "kebutuhan_kos",
        "dan lain-lain": "lain_lain",
        "lain-lain": "lain_lain",
        "lain lain": "lain_lain"
    }
    return mapping.get(cat, cat)


def load_training_data() -> pd.DataFrame:
    base_df = pd.DataFrame(INITIAL_DATA)

    if TRAINING_FILE.exists():
        try:
            feedback_df = pd.read_csv(TRAINING_FILE)
            combined_df = pd.concat([base_df, feedback_df], ignore_index=True)
        except Exception:
            combined_df = base_df
    else:
        combined_df = base_df

    if "category" in combined_df.columns:
        combined_df["category"] = combined_df["category"].apply(normalize_category)
        combined_df = combined_df[combined_df["category"].isin(CATEGORIES)]

    combined_df["clean_subject"] = combined_df["subject"].apply(clean_text)

    if combined_df.empty:
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

    raw_predicted_category = model.predict([clean_subject])[0]
    predicted_category = normalize_category(raw_predicted_category)
    if predicted_category not in CATEGORIES:
        predicted_category = "lain_lain"

    probabilities = model.predict_proba([clean_subject])[0]

    candidates_dict = {}

    for category, probability in zip(model.classes_, probabilities):
        norm_cat = normalize_category(category)
        if norm_cat not in CATEGORIES:
            continue
        prob = float(probability)
        if norm_cat not in candidates_dict or prob > candidates_dict[norm_cat]:
            candidates_dict[norm_cat] = prob

    candidates = [
        {"category": cat, "confidence": round(prob, 4)}
        for cat, prob in candidates_dict.items()
    ]

    candidates = sorted(
        candidates,
        key=lambda item: item["confidence"],
        reverse=True,
    )

    top_confidence = candidates[0]["confidence"] if candidates else 0.0

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
    norm_category = normalize_category(correct_category)
    if norm_category not in CATEGORIES:
        raise ValueError(f"Kategori tidak valid: {correct_category}")

    DATA_DIR.mkdir(exist_ok=True)

    new_row = pd.DataFrame([{
        "subject": subject,
        "amount": amount or 0,
        "category": norm_category,
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