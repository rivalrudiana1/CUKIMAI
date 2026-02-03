import streamlit as st
import pickle
from groq_client import generate_solution

# Fungsi load model terbaru
def load_models():
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("model_utama.pkl", "rb") as f:
        model_utama = pickle.load(f)
    with open("encoder_utama.pkl", "rb") as f:
        encoder_utama = pickle.load(f)
    with open("model_pendukung.pkl", "rb") as f:
        model_pendukung = pickle.load(f)
    with open("encoder_pendukung.pkl", "rb") as f:
        encoder_pendukung = pickle.load(f)
    with open("model_tingkat.pkl", "rb") as f:
        model_tingkat = pickle.load(f)
    with open("encoder_tingkat.pkl", "rb") as f:
        encoder_tingkat = pickle.load(f)
    return vectorizer, model_utama, encoder_utama, model_pendukung, encoder_pendukung, model_tingkat, encoder_tingkat

st.set_page_config(page_title="CUKIMAI", page_icon="🎓", layout="centered")
st.title("🎓 CUKIMAI")
st.subheader("Konsultasi Mahasiswa dengan AI")

teks = st.text_area("Masukkan permasalahan kamu", placeholder="Contoh: saya sulit membagi waktu antara akademik dan organisasi")

def klasifikasi(teks, vectorizer, model_utama, encoder_utama, model_pendukung, encoder_pendukung, model_tingkat, encoder_tingkat):
    X_vec = vectorizer.transform([teks])
    utama = encoder_utama.inverse_transform(model_utama.predict(X_vec))[0]
    pendukung = encoder_pendukung.inverse_transform(model_pendukung.predict(X_vec))[0]
    tingkat = encoder_tingkat.inverse_transform(model_tingkat.predict(X_vec))[0]
    return utama, pendukung, tingkat

if st.button("Analisis Masalah"):
    if teks.strip() == "":
        st.warning("Masukkan permasalahan terlebih dahulu.")
    else:
        with st.spinner("CUKIMAI sedang berpikir..."):
            vectorizer, model_utama, encoder_utama, model_pendukung, encoder_pendukung, model_tingkat, encoder_tingkat = load_models()
            utama, pendukung, tingkat = klasifikasi(teks, vectorizer, model_utama, encoder_utama, model_pendukung, encoder_pendukung, model_tingkat, encoder_tingkat)
            solusi = generate_solution(teks, utama, pendukung, tingkat)

        st.success("Analisis selesai")
        st.markdown("### 🧠 Hasil Analisis AI")
        st.write(f"**Kategori Utama:** {utama}")
        st.write(f"**Kategori Pendukung:** {pendukung}")
        st.write(f"**Tingkat Permasalahan:** {tingkat}")
        st.markdown("### 💡 Solusi & To Do List")
        st.write(solusi)
