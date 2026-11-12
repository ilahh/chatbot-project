# Chatbot Layanan Akademik Prodi

## 👥 Anggota Kelompok

> Arif Fadillah (25343105)
> M. Ichsan Nurdin (26016698)
> Fajar Rizky Kurniawan (25343009)

---

**Tugas Kelompok Mata Kuliah Artificial Intelligence**

Proyek ini dibuat untuk keperluan tugas kuliah Artificial Intelligence.

Chatbot hybrid untuk layanan akademik kampus yang menggabungkan 3 pendekatan berbeda: **Knowledge Base Lokal**, **TF-IDF + Cosine Similarity**, dan **LLM dengan RAG**.

---

## 📌 Fitur Utama

- **3 Pendekatan Cerdas**
  1. **Knowledge Base Lokal** — Pattern matching untuk FAQ umum (cepat & pasti)
  2. **TF-IDF + Cosine Similarity** — Deteksi intent pada pertanyaan bervariasi
  3. **LLM (Large Language Model)** — Menjawab pertanyaan kompleks di luar KB

- **RAG (Retrieval-Augmented Generation)**
  - LLM hanya menjawab berdasarkan konteks Knowledge Base
  - Mencegah LLM berhalusinasi atau menjawab dari pengetahuan liar

- **LLM Bebas**
  - Untuk pertanyaan umum yang benar-benar di luar topik kampus
  - LLM boleh menggunakan pengetahuan umumnya

- **Model LLM**: Llama 3.1 8B Instruct via **Cloudflare Workers AI**
  - Gratis (dengan batasan harian)
  - Cepat & tidak memerlukan GPU lokal

---

## Alur Kerja Chatbot

```
Input User
    │
    ▼
Pattern Match (Knowledge Base)
    │
    ├─ Cocok ──▶ Jawaban langsung dari KB
    │
    ▼
TF-IDF Similarity >= 0.7?
    │
    ├─ Ya ─────▶ Jawaban langsung dari KB
    │
    ▼
TF-IDF Similarity >= 0.3?
    │
    ├─ Ya ─────▶ RAG (LLM + Konteks KB)
    │              LLM hanya jawab dari konteks
    │
    ▼
    └─ Tidak ──▶ LLM Bebas
                  LLM jawab dari pengetahuan umum
```

---

## ⚙️ Cara Menjalankan

### 1. Clone Repository

```bash
git clone <url-repo>
cd Chatbot
```

### 2. Install Dependencies

```bash
pip install python-dotenv scikit-learn numpy requests
```

Atau jika ada `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Setup Environment Variables

Copy file `.env.example` menjadi `.env`:

```bash
cp .env.example .env
```

Isi `.env` dengan API Key dan Account ID Cloudflare kamu:

```env
CLOUDFLARE_API_KEY=cfut_xxxxxxxxxx
CLOUDFLARE_ACCOUNT_ID=xxxxxxxxxx
```

> **Cara mendapatkan Cloudflare API Key:**
> 1. Login ke [Cloudflare Dashboard](https://dash.cloudflare.com/)
> 2. Pergi ke **AI > Workers AI**
> 3. Buat atau copy API Token
> 4. Copy Account ID dari halaman Workers AI

### 4. Jalankan Chatbot

```bash
python chatbot.py
```

---

## 💬 Contoh Interaksi

```
============================================================
CHATBOT LAYANAN AKADEMIK PRODI
Hybrid System: KB Lokal + TF-IDF + RAG
Model: Llama 3.1 8B via Cloudflare Workers AI
Ketik 'keluar' untuk berhenti
============================================================

Anda: halo
Bot: Halo! Ada yang bisa saya bantu terkait informasi kampus?

Anda: dimana lokasi kampus
Bot: Kampus berlokasi di Jl. Pendidikan No. 1, dekat halte busway terdekat.

Anda: jadwal kuliah AI kapan?
Bot: Mata kuliah Artificial Intelligence ... Selasa pukul 10:00 - 12:30

Anda: bagaimana cuaca hari ini?
Bot: [Jawaban dari LLM bebas]

Anda: keluar
Bot: Sampai jumpa! Semoga harimu menyenangkan.
```

---

## 📚 Knowledge Base

Chatbot memiliki pengetahuan tentang:

- **Greeting** — halo, hai, selamat pagi/siang
- **Jadwal Kuliah** — jadwal kuliah umum
- **Jadwal AI** — jadwal mata kuliah Kecerdasan Buatan
- **Lokasi Kampus** — alamat kampus
- **Kontak Admin** — email dan nomor telepon admin
- **Terima Kasih** — response sopan
- **Goodbye** — sampai jumpa, keluar

Knowledge Base dapat dikembangkan dengan menambahkan intent baru di variabel `knowledge_base` pada file `chatbot.py`.

---

## 🧪 Testing

Beberapa contoh pertanyaan untuk menguji masing-masing pendekatan:

| Pertanyaan | Pendekatan yang Digunakan |
|------------|---------------------------|
| "halo" | Knowledge Base Lokal |
| "dimana lokasi kampus" | Knowledge Base Lokal |
| "kapan kuliah AI diadakan?" | TF-IDF / RAG |
| "bagaimana cuaca hari ini?" | LLM Bebas |
| "siapa presiden Indonesia?" | LLM Bebas |

---

