import streamlit as st
import pickle
import numpy as np

with open("model_knn.pkl", "rb") as f:
    model = pickle.load(f)

with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

solusi = {
"Akademik": [
"Susun jadwal belajar mingguan",
"Konsultasi dengan dosen wali",
"Kurangi aktivitas non-akademik"
],
"Keuangan": [
"Catat pengeluaran bulanan",
"Cari program beasiswa",
"Pertimbangkan kerja paruh waktu"
],
"Mental": [
"Manfaatkan layanan konseling kampus",
"Atur waktu istirahat",
"Kurangi tekanan tugas berlebihan"
],
"Fasilitas": [
"Laporkan kendala ke unit layanan kampus",
"Gunakan fasilitas alternatif",
"Koordinasi dengan pihak terkait"
],
"Manajemen waktu": [
"Gunakan teknik time blocking",
"Tentukan prioritas tugas",
"Hindari multitasking berlebihan"
]
}

st.title("MahaSense")
st.write("Analisis Permasalahan Mahasiswa Berbasis AI")

beban = st.slider("Beban tugas", 1, 5)
tekanan = st.slider("Tekanan akademik", 1, 5)
keuangan = st.slider("Masalah keuangan", 1, 5)
fasilitas = st.slider("Fasilitas kampus", 1, 5)

if st.button("Analisis"):
    data_input = np.array([[beban, tekanan, keuangan, fasilitas]])
    pred = model.predict(data_input)
    hasil = encoder.inverse_transform(pred)[0]

st.subheader("Hasil Analisis AI")
st.success(hasil)

st.subheader("Rekomendasi Solusi")
for s in solusi[hasil]:
    st.write("- " + s)
