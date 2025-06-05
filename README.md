# DocuMentor AI: Document Summarizer & QnA Bot

DocuMentor AI adalah aplikasi berbasis Python dan Streamlit yang memungkinkan pengguna untuk:
1. Mengunggah dokumen (PDF, TXT, DOCX).
2. Mendapatkan ringkasan otomatis dari isi dokumen.
3. Mengajukan pertanyaan dan mendapatkan jawaban berdasarkan konten dokumen.

Aplikasi ini menggunakan model AI dari Hugging Face Transformers untuk peringkasan dan LangChain untuk membangun sistem QnA dengan pendekatan RAG (Retrieval Augmented Generation).

## Fitur Utama
- Ekstraksi teks dari file PDF, TXT, dan DOCX.
- Peringkasan teks otomatis menggunakan model Hugging Face.
- Sistem Tanya Jawab (QnA) berbasis RAG:
    - Chunking teks.
    - Pembuatan embedding menggunakan Sentence Transformers.
    - Penyimpanan vektor menggunakan FAISS.
    - Pilihan antara LLM generatif lokal (misal, FLAN-T5) atau model QnA ekstraktif untuk menjawab pertanyaan.
- Antarmuka pengguna interaktif dibangun dengan Streamlit.
- Pengaturan mode QnA (generatif atau ekstraktif) melalui sidebar.

## Teknologi yang Digunakan
- **Bahasa:** Python 3.x
- **Frontend:** Streamlit
- **Backend & AI:**
    - LangChain
    - Hugging Face Transformers (untuk model summarization, embedding, QnA, LLM)
    - Sentence Transformers
    - FAISS (untuk vector store)
    - PyMuPDF (untuk ekstraksi PDF)
    - python-docx (untuk ekstraksi DOCX)

## Lisensi
Proyek ini dilisensikan di bawah [LISENSI_ANDA, misal: MIT License].

## Dikembangkan Oleh
[fredli4qoni](https://github.com/fredli4qoni)