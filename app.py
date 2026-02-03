import streamlit as st
import pickle
import numpy as np

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

if st.button("Analisis"):
    if teks.strip() == "":
        st.warning("Masukkan teks terlebih dahulu")
    else:
        X = vectorizer.transform([teks])
        probs = model.predict_proba(X)[0]

        top2_idx = np.argsort(probs)[-2:][::-1]
        utama = encoder.inverse_transform([top2_idx[0]])[0]
        pendukung = encoder.inverse_transform([top2_idx[1]])[0]

        st.success("Hasil Analisis")
        st.write("Masalah Utama:", utama)
        st.write("Masalah Pendukung:", pendukung)
