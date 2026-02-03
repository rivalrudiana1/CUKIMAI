import streamlit as st
import pickle
import os
from groq_client import generate_solution  # pastikan Groq client siap

# File model
VECTORIZER_FILE = "vectorizer.pkl"
MODEL_UTAMA_FILE = "model_utama.pkl"
ENCODER_UTAMA_FILE = "encoder_utama.pkl"
MODEL_PENDUKUNG_FILE = "model_pendukung.pkl"
ENCODER_PENDUKUNG_FILE = "encoder_pendukung.pkl"
MODEL_TINGKAT_FILE = "model_tingkat.pkl"
ENCODER_TINGKAT_FILE = "encoder_tingkat.pkl"

# Load model dan encoder
def load_models():
    files = [VECTORIZER_FILE, MODEL_UTAMA_FILE, ENCODER_UTAMA_FILE,
             MODEL_PENDUKUNG_FILE, ENCODER_PENDUKUNG_FILE,
             MODEL_TINGKAT_FILE, ENCODER_TINGKAT_FILE]
    for file in files:
        if not os.path.exists(file):
            st.error(f"File {file} tidak ditemukan. Jalankan train_model.py dulu.")
            return [None]*7

    with open(VECTORIZER_FILE, "rb") as f:
        vectorizer = pickle.load(f)
    with open(MODEL_UTAMA_FILE, "rb") as f:
        model_utama = pickle.load(f)
    with open(ENCODER_UTAMA_FILE, "rb") as f:
        encoder_utama = pickle.load(f)
    with open(MODEL_PENDUKUNG_FILE, "rb") as f:
        model_pendukung = pickle.load(f)
    with open(ENCODER_PENDUKUNG_FILE, "rb") as f:
        encoder_pendukung = pickle.load(f)
    with open(MODEL_TINGKAT_FILE, "rb") as f:
        model_tingkat = pickle.load(f)
    with open(ENCODER_TINGKAT_FILE, "rb") as f:
        encoder_tingkat = pickle.load(f)

    return vectorizer, model_utama, encoder_utama, model_pendukung, encoder_pendukung, model_tingkat, encoder_tingkat

# Fungsi klasifikasi multi-output
def klasifikasi(teks, vectorizer, model_utama, encoder_utama,
                model_pendukung, encoder_pendukung, model_tingkat, encoder_tingkat):
    X_vec = vectorizer.transform([teks])
    kategori_utama = encoder_utama.inverse_transform(model_utama.predict(X_vec))[0]
    kategori_pendukung = encoder_pendukung.inverse_transform(model_pendukung.predict(X_vec))[0]
    tingkat = encoder_tingkat.inverse_transform(model_tingkat.predict(X_vec))[0]
    return kategori_utama, kategori_pendukung, tingkat

# Streamlit config
st.set_page_config(page_title="CUKIMAI", page_icon="🎓", layout="centered")
st.title("🎓 CUKIMAI")
st.subheader("AI Konselor Mahasiswa: Analisis Masalah, Solusi, dan Motivasi")
st.write(
    "Tuliskan masalahmu sebagai mahasiswa. "
    "CUKIMAI akan menganalisis kategori utama & pendukung, tingkat masalah, solusi, to do list, dan motivasi."
)

teks = st.text_area(
    "Masukkan masalah kamu",
    placeholder="Contoh: saya sulit membagi waktu antara akademik dan organisasi"
)

vectorizer, model_utama, encoder_utama, model_pendukung, encoder_pendukung, model_tingkat, encoder_tingkat = load_models()

if st.button("Analisis Masalah"):
    if teks.strip() == "":
        st.warning("Masukkan permasalahan terlebih dahulu.")
    elif not vectorizer:
        st.error("Model belum tersedia.")
    else:
        with st.spinner("CUKIMAI sedang menganalisis..."):
            utama, pendukung, tingkat = klasifikasi(teks, vectorizer, model_utama, encoder_utama,
                                                    model_pendukung, encoder_pendukung, model_tingkat, encoder_tingkat)
            solusi = generate_solution(teks, utama, pendukung, tingkat)

        st.success("Analisis selesai")
        st.markdown("### 🧠 Hasil Analisis AI")
        st.write(f"**Kategori Utama:** {utama}")
        st.write(f"**Kategori Pendukung:** {pendukung}")
        st.write(f"**Tingkat Masalah:** {tingkat}")

        st.markdown("### 💡 Solusi & To Do List + Motivasi")
        st.write(solusi)

st.markdown("---")
st.caption("CUKIMAI | AI Konselor Mahasiswa | Support System untuk Mahasiswa")
