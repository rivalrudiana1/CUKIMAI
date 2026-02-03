# auto_train_groq.py
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import time
import os
from groq_client import generate_solution  # pastikan ini import Groq client kamu

DATA_FILE = "log_dataset.csv"
MODEL_KATEGORI_FILE = "model_kategori.pkl"
MODEL_TINGKAT_FILE = "model_tingkat.pkl"
VECTORIZER_FILE = "vectorizer.pkl"

def train_model():
    if not os.path.exists(DATA_FILE):
        print("Dataset tidak ditemukan.")
        return None, None, None

    data = pd.read_csv(DATA_FILE)
    X = data["teks_masalah"]
    y_kategori = data["kategori_utama"]
    y_tingkat = data["tingkat"]

    vectorizer = TfidfVectorizer(max_features=500)
    X_vect = vectorizer.fit_transform(X)

    model_kategori = LogisticRegression(max_iter=500)
    model_kategori.fit(X_vect, y_kategori)

    model_tingkat = LogisticRegression(max_iter=500)
    model_tingkat.fit(X_vect, y_tingkat)

    # Save model
    with open(MODEL_KATEGORI_FILE, "wb") as f:
        pickle.dump(model_kategori, f)
    with open(MODEL_TINGKAT_FILE, "wb") as f:
        pickle.dump(model_tingkat, f)
    with open(VECTORIZER_FILE, "wb") as f:
        pickle.dump(vectorizer, f)

    print("Training selesai dan model tersimpan.")
    return model_kategori, model_tingkat, vectorizer

def update_groq_model(model_kategori, model_tingkat, vectorizer):
    """
    Fungsi ini akan dipanggil setiap selesai retrain.
    Bisa untuk update prompt / parameter di Groq agar AI selalu pakai model terbaru.
    """
    print("Groq AI siap menggunakan model terbaru...")

# Loop otomatis setiap 60 detik
while True:
    kategori_model, tingkat_model, vectorizer_model = train_model()
    if kategori_model and tingkat_model:
        update_groq_model(kategori_model, tingkat_model, vectorizer_model)
    print("Menunggu 60 detik sebelum retrain lagi...")
    time.sleep(60)
