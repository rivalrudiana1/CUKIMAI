import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_solution(teks_masalah, kategori_utama, kategori_pendukung, tingkat):
    prompt = f"""
Kamu adalah AI konselor mahasiswa.

Masalah: {teks_masalah}

Kategori Utama: {kategori_utama}
Kategori Pendukung: {kategori_pendukung}
Tingkat Masalah: {tingkat}

Tugas kamu:
1. Berikan solusi konkret dan singkat.
2. Buat skala prioritas (to do list) yang harus dikerjakan mahasiswa.
3. Berikan kata motivasi/semangat dari tokoh inspiratif.
4. Format jawaban:
Kategori Utama: ...
Kategori Pendukung: ...
Solusi: ...
To Do List:
- ...
- ...
Motivasi: ...
Gunakan bahasa santai mahasiswa.
"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
