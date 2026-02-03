import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import time
import os

DATA_FILE = "log_dataset.csv"
MODEL_UTAMA_FILE = "model_utama.pkl"
MODEL_PENDUKUNG_FILE = "model_pendukung.pkl"
MODEL_TINGKAT_FILE = "model_tingkat.pkl"
VECTORIZER_FILE = "vectorizer.pkl"

def train_model():
    if not os.path.exists(DATA_FILE):
        print("Dataset tidak ditemukan.")
        return

    data = pd.read_csv(DATA_FILE)
    X = data["teks_masalah"]
    y_utama = data["kategori_utama"]
    y_pendukung = data["kategori_pendukung"]
    y_tingkat = data["tingkat"]

    vectorizer = TfidfVectorizer(max_features=500)
    X_vec = vectorizer.fit_transform(X)

    model_utama = LogisticRegression(max_iter=200)
    model_utama.fit(X_vec, y_utama)

    model_pendukung = LogisticRegression(max_iter=200)
    model_pendukung.fit(X_vec, y_pendukung)

    model_tingkat = LogisticRegression(max_iter=200)
    model_tingkat.fit(X_vec, y_tingkat)

    # Simpan model
    with open(MODEL_UTAMA_FILE, "wb") as f:
        pickle.dump(model_utama, f)
    with open(MODEL_PENDUKUNG_FILE, "wb") as f:
        pickle.dump(model_pendukung, f)
    with open(MODEL_TINGKAT_FILE, "wb") as f:
        pickle.dump(model_tingkat, f)
    with open(VECTORIZER_FILE, "wb") as f:
        pickle.dump(vectorizer, f)

    print("Training selesai, model terbaru tersimpan.")

# Loop otomatis
while True:
    train_model()
    print("Menunggu 60 detik sebelum retrain lagi...")
    time.sleep(60)
