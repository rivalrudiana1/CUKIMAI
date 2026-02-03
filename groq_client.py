import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def generate_solution(problem_text, kategori, tingkat):
    prompt = f"""
Kamu adalah AI konselor mahasiswa.

Masalah:
{problem_text}

Kategori masalah: {kategori}
Tingkat masalah: {tingkat}

Tugas kamu:
1. Berikan solusi konkret dan singkat.
2. Buat skala prioritas dalam bentuk to do list.
Gunakan bahasa mahasiswa.
"""

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content
