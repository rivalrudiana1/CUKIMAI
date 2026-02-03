import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def generate_solution(problem_text, kategori_utama, kategori_pendukung):
    prompt = f"""
Kamu adalah AI konselor mahasiswa.

Curhat mahasiswa:
{problem_text}

Kategori masalah utama: {kategori_utama}
Kategori masalah pendukung: {kategori_pendukung}

Tugas kamu:
1. Berikan solusi yang realistis dan bisa dilakukan mahasiswa.
2. Buat to do list prioritas maksimal 5 poin.
3. Urutkan dari yang paling penting.
4. Gunakan bahasa sederhana dan langsung.
5. Jangan terlalu panjang.
"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.4
    )

    return response.choices[0].message.content.strip()
