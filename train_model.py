import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("dataset.csv")  # pastikan kolom: teks_masalah,kategori_utama,kategori_pendukung,tingkat

X = data["teks_masalah"]
y_utama = data["kategori_utama"]
y_pendukung = data["kategori_pendukung"]
y_tingkat = data["tingkat"]

# Stopwords bahasa Indonesia sederhana
stopwords_id = ["yang","dan","di","ke","dari","pada","untuk","dengan","atau","ini","itu","saya","aku","kamu"]

# TF-IDF
vectorizer = TfidfVectorizer(max_features=500, stop_words=stopwords_id)
X_vec = vectorizer.fit_transform(X)

# Label encoder untuk kategori & tingkat
encoder_utama = LabelEncoder()
y_utama_enc = encoder_utama.fit_transform(y_utama)

encoder_pendukung = LabelEncoder()
y_pendukung_enc = encoder_pendukung.fit_transform(y_pendukung)

encoder_tingkat = LabelEncoder()
y_tingkat_enc = encoder_tingkat.fit_transform(y_tingkat)

# Logistic Regression untuk multi-output
model_utama = LogisticRegression(max_iter=200)
model_utama.fit(X_vec, y_utama_enc)

model_pendukung = LogisticRegression(max_iter=200)
model_pendukung.fit(X_vec, y_pendukung_enc)

model_tingkat = LogisticRegression(max_iter=200)
model_tingkat.fit(X_vec, y_tingkat_enc)

# Simpan model dan encoder
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("model_utama.pkl", "wb") as f:
    pickle.dump(model_utama, f)

with open("encoder_utama.pkl", "wb") as f:
    pickle.dump(encoder_utama, f)

with open("model_pendukung.pkl", "wb") as f:
    pickle.dump(model_pendukung, f)

with open("encoder_pendukung.pkl", "wb") as f:
    pickle.dump(encoder_pendukung, f)

with open("model_tingkat.pkl", "wb") as f:
    pickle.dump(model_tingkat, f)

with open("encoder_tingkat.pkl", "wb") as f:
    pickle.dump(encoder_tingkat, f)

print("Training selesai, semua model dan encoder tersimpan.")
