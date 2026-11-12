# -*- coding: utf-8 -*-
import re
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os
import requests

# Load environment variables dari .env
load_dotenv()

# ============================================================================
# PENDEKATAN 1: KNOWLEDGE BASE LOKAL (Rule-based Pattern Matching)
# Untuk pertanyaan FAQ yang sering muncul - jawaban cepat, pasti, terverifikasi
# ============================================================================

knowledge_base = {
    "greeting": {
        "patterns": ["halo", "hai", "selamat pagi", "selamat siang"],
        "responses": ["Halo! Ada yang bisa saya bantu terkait informasi kampus?"]
    },
    "jadwal_kuliah": {
        "patterns": ["jadwal kuliah", "jadwal kelas", "kapan kuliah"],
        "responses": ["Jadwal kuliah dapat dilihat di portal akademik pada menu 'Jadwal Kuliah'."]
    },
    "jadwal_ai": {
        "patterns": ["jadwal ai", "kuliah ai", "jadwal kecerdasan buatan", "kelas ai"],
        "responses": ["Mata kuliah Artificial Intelligence (Kecerdasan Buatan) diadakan setiap hari Selasa pukul 10:00 - 12:30 di Ruang Lab Komputer 3."]
    },
    "lokasi_kampus": {
        "patterns": ["lokasi kampus", "alamat kampus", "dimana kampus"],
        "responses": ["Kampus berlokasi di Jl. Pendidikan No. 1, dekat halte busway terdekat."]
    },
    "kontak_admin": {
        "patterns": ["kontak admin", "hubungi admin", "nomor admin"],
        "responses": ["Admin akademik dapat dihubungi di admin@kampus.ac.id atau (021) 555-0100."]
    },
    "terima_kasih": {
        "patterns": ["terima kasih", "makasih", "thanks"],
        "responses": ["Sama-sama! Senang bisa membantu."]
    },
    "goodbye": {
        "patterns": ["bye", "sampai jumpa", "keluar", "exit"],
        "responses": ["Sampai jumpa! Semoga harimu menyenangkan."]
    }
}

FALLBACK_RESPONSE = "Maaf, saya tidak tahu jawabannya karena informasi tidak tersedia di database kampus."

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

def match_intent(user_input):
    processed = preprocess(user_input)
    for intent, data in knowledge_base.items():
        for pattern in data["patterns"]:
            if pattern in processed:
                return intent
    return None

def generate_response(intent):
    if intent is None:
        return FALLBACK_RESPONSE
    responses = knowledge_base[intent]["responses"]
    return random.choice(responses)

# ============================================================================
# PENDEKATAN 2: TF-IDF + COSINE SIMILARITY (Modul 5)
# Untuk pertanyaan yang bervariasi tapi masih dalam KB
# ============================================================================

training_examples = {
    "jadwal_kuliah": ["jadwal kuliah", "jadwal kelas", "kapan kuliah", "jam kuliah"],
    "lokasi_kampus": ["lokasi kampus", "alamat kampus", "letak kampus"],
    "kontak_admin": ["kontak admin", "hubungi admin", "nomor admin", "email admin"],
}

corpus, labels = [], []
for intent, examples in training_examples.items():
    for ex in examples:
        corpus.append(ex)
        labels.append(intent)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

def match_intent_nlp(user_input, threshold=0.3):
    user_vec = vectorizer.transform([preprocess(user_input)])
    similarities = cosine_similarity(user_vec, tfidf_matrix)[0]
    best_idx = similarities.argmax()
    if similarities[best_idx] >= threshold:
        return labels[best_idx], similarities[best_idx]
    return None, similarities[best_idx]

# ============================================================================
# SETUP LLM API (Cloudflare Workers AI)
# ============================================================================

CLOUDFLARE_API_KEY = os.getenv('CLOUDFLARE_API_KEY')
CLOUDFLARE_ACCOUNT_ID = os.getenv('CLOUDFLARE_ACCOUNT_ID')

if CLOUDFLARE_API_KEY and CLOUDFLARE_ACCOUNT_ID:
    print("✓ Cloudflare Workers AI terhubung!")
else:
    print("⚠ Peringatan: CLOUDFLARE_API_KEY atau CLOUDFLARE_ACCOUNT_ID tidak ditemukan di .env")

# ============================================================================
# PENDEKATAN 3: RAG (Retrieval-Augmented Generation)
# LLM HANYA boleh menjawab dari konteks KB, TIDAK BOLEH dari pengetahuan sendiri
# ============================================================================

rag_corpus = []
rag_metadata = []

for intent, data in knowledge_base.items():
    for pattern in data["patterns"]:
        rag_corpus.append(pattern)
        rag_metadata.append({
            "intent": intent,
            "response": data["responses"][0]
        })

vectorizer_rag = TfidfVectorizer()
tfidf_matrix_rag = vectorizer_rag.fit_transform(rag_corpus)

def retrieve_context(user_input, top_k=3):
    user_vec = vectorizer_rag.transform([preprocess(user_input)])
    similarities = cosine_similarity(user_vec, tfidf_matrix_rag)[0]
    top_indices = similarities.argsort()[::-1][:top_k]

    contexts = []
    max_score = similarities[top_indices[0]] if len(top_indices) > 0 else 0.0

    for idx in top_indices:
        score = similarities[idx]
        if score > 0.0:
            meta = rag_metadata[idx]
            contexts.append(f"Q: '{rag_corpus[idx]}' -> A: '{meta['response']}'")

    return "\n".join(contexts), max_score

def ask_llm_free(user_input):
    """LLM bebas untuk pertanyaan di luar KB - boleh jawab dari pengetahuan sendiri"""
    if not CLOUDFLARE_API_KEY or not CLOUDFLARE_ACCOUNT_ID:
        return "Koneksi API LLM tidak aktif."

    system_prompt = "Anda adalah asisten virtual kampus yang ramah dan membantu. Jawab pertanyaan user dengan lengkap dan informatif."
    
    try:
        url = f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/run/@cf/meta/llama-3.1-8b-instruct"
        
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {CLOUDFLARE_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                "max_tokens": 500
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'result' in result and 'response' in result['result']:
                return result['result']['response'].strip()
            elif 'result' in result and 'choices' in result['result']:
                msg = result['result']['choices'][0].get('message', {})
                content = msg.get('content') or msg.get('reasoning', '')
                return content.strip() if content else "Error: model tidak mengembalikan content"
            else:
                return f"Format response tidak sesuai: {result}"
        else:
            return f"Error API ({response.status_code}): {response.text[:200]}"
            
    except Exception as e:
        return f"Error: {e}"

def ask_llm_rag(user_input, context):
    if not CLOUDFLARE_API_KEY or not CLOUDFLARE_ACCOUNT_ID:
        return "Koneksi API LLM tidak aktif."

    system_instruction = (
        "Anda adalah asisten virtual akademik kampus yang JUJUR dan DISIPLIN. "
        "ATURAN KETAT: "
        "1. Anda HANYA boleh menjawab berdasarkan KONTEKS di bawah ini "
        "2. Jika KONTEKS tidak relevan atau tidak cukup untuk menjawab, "
        "   Anda WAJIB jawab: 'Maaf, s"
        "aya tidak tahu jawabannya karena informasi tidak tersedia di database kampus.' "
        "3. JANGAN PERNAH menggunakan pengetahuan luar atau berasumsi "
        "4. Jawab singkat, jelas, ramah, berdasarkan konteks saja"
    )

    prompt = f"KONTEKS DARI DATABASE KAMPUS:\n{context}\n\nPertanyaan: {user_input}\n\nJawab berdasarkan konteks di atas saja:"

    try:
        url = f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/run/@cf/meta/llama-3.1-8b-instruct"
        
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {CLOUDFLARE_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 500
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'result' in result and 'response' in result['result']:
                return result['result']['response'].strip()
            elif 'result' in result and 'choices' in result['result']:
                msg = result['result']['choices'][0].get('message', {})
                content = msg.get('content') or msg.get('reasoning', '')
                return content.strip() if content else "Error: model tidak mengembalikan content"
            else:
                return f"Format response tidak sesuai: {result}"
        else:
            return f"Error API ({response.status_code}): {response.text[:200]}"
            
    except Exception as e:
        return f"Error: {e}"

# ============================================================================
# FUNGSI UTAMA: HYBRID SYSTEM (3 Pendekatan)
# ============================================================================

def get_chatbot_response(user_input):
    # PENDEKATAN 1: Pattern matching langsung
    intent = match_intent(user_input)
    if intent is not None:
        return generate_response(intent)
    
    # PENDEKATAN 2: TF-IDF dengan threshold tinggi (>= 0.7)
    intent_nlp, score = match_intent_nlp(user_input)
    if intent_nlp is not None and score >= 0.7:
        return generate_response(intent_nlp)
    
    # Cek similarity dengan KB untuk routing
    context, max_score = retrieve_context(user_input, top_k=3)
    
    # PENDEKATAN 3a: RAG - Pertanyaan terkait KB (0.3 <= score < 0.7)
    # LLM jawab HANYA dari konteks KB, tidak boleh pengetahuan sendiri
    if context and max_score >= 0.3:
        return ask_llm_rag(user_input, context)
    
    # PENDEKATAN 3b: LLM Bebas - Pertanyaan di luar KB (score < 0.3)
    # LLM boleh jawab dari pengetahuan sendiri
    return ask_llm_free(user_input)

# ============================================================================
# CHATBOT INTERFACE
# ============================================================================

def run_chatbot():
    print("=" * 60)
    print("CHATBOT LAYANAN AKADEMIK PRODI")
    print("Hybrid System: KB Lokal + TF-IDF + RAG")
    print("Model: Llama 3.1 8B via Cloudflare Workers AI")
    print("Ketik 'keluar' untuk berhenti")
    print("=" * 60)
    
    while True:
        user_input = input("\nAnda: ").strip()
        
        if not user_input:
            continue
            
        response = get_chatbot_response(user_input)
        print(f"Bot: {response}")
        
        if match_intent(user_input) == "goodbye":
            break

if __name__ == "__main__":
    run_chatbot()
