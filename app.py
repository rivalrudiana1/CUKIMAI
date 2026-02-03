import streamlit as st
import pickle
import numpy as np
from groq import Groq
from groq_client import generate_solution


client = Groq(api_key="ISI_API_KEY_KAMU")

st.set_page_config(page_title="CUKIMAI", page_icon="🎓")

with open("model_knn.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

st.title("CUKIMAI")
st.write("Consultation and Understanding of Kampus Issues using Machine Learning and Artificial Intelligence")

teks = st.text_area("Ceritakan masalah kamu")

def groq_solusi(teks, utama, pendukung):
    prompt = f"""
Kamu adalah konselor mahasiswa.
Masalah utama: {utama}
Masalah pendukung: {pendukung}
Curhat: {teks}

Berikan solusi singkat dan to do list prioritas.
"""
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

if st.button("Analisis"):
    if teks.strip() == "":
        st.warning("Masukkan teks terlebih dahulu")
    else:
        X = vectorizer.transform([teks])
        probs = model.predict_proba(X)[0]

        top2 = np.argsort(probs)[-2:][::-1]
        utama = encoder.inverse_transform([top2[0]])[0]
        pendukung = encoder.inverse_transform([top2[1]])[0]

        st.subheader("Hasil Analisis AI")
        st.write("Masalah Utama:", utama)
        st.write("Masalah Pendukung:", pendukung)

        with st.spinner("AI sedang menyusun solusi"):
            solusi = generate_solution(teks, utama, pendukung)

        st.subheader("Solusi dan Skala Prioritas")
        st.write(solusi)
