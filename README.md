# 🤖 DocuMentor AI: Document Summarizer & QnA Bot

<p align="center">
  <em>Asisten cerdas untuk meringkas dokumen dan menjawab pertanyaan Anda .</em>
</p>

<p align="center">
  <!-- Badge Teknologi -->
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.9+"/>
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/LangChain-AI%20Framework-F7902D?style=for-the-badge" alt="LangChain"/> 
  <img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Transformers-yellow?style=for-the-badge&logo=huggingface&logoColor=white" alt="Hugging Face Transformers"/>
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch"/>
  <br/>
  <a href="https://github.com/fredli4qooni/DocuMentor-AI-Document-Summarizer-QnA-/stargazers"><img src="https://img.shields.io/github/stars/fredli4qooni/DocuMentor-AI-Document-Summarizer-QnA-?style=social" alt="GitHub Stars"/></a>
  <a href="https://github.com/fredli4qooni/DocuMentor-AI-Document-Summarizer-QnA-/network/members"><img src="https://img.shields.io/github/forks/fredli4qooni/DocuMentor-AI-Document-Summarizer-QnA-?style=social" alt="GitHub Forks"/></a>
  <img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" alt="License: MIT"/> 
</p>

---

DocuMentor AI adalah aplikasi berbasis Python dan Streamlit yang dirancang untuk merevolusi cara Anda berinteraksi dengan dokumen panjang. Unggah file PDF, TXT, atau DOCX Anda, dan biarkan AI bekerja untuk Anda!

## ✨ Fitur Utama

- 📄 **Ekstraksi Teks Universal**: Mendukung format PDF, TXT, dan DOCX.
- ✍️ **Peringkasan Cerdas**: Dapatkan ringkasan otomatis yang akurat dan ringkas dari konten dokumen Anda menggunakan model AI canggih dari Hugging Face.
- 💬 **Tanya Jawab Interaktif (QnA)**: Ajukan pertanyaan dalam bahasa alami dan dapatkan jawaban yang relevan berdasarkan isi dokumen. Didukung oleh RAG (Retrieval Augmented Generation) menggunakan LangChain:
  - Pembagian teks menjadi _chunks_ yang optimal.
  - Pembuatan _embedding_ semantik dengan Sentence Transformers.
  - Pencarian kemiripan cepat menggunakan _vector store_ FAISS.
  - Pilihan antara LLM generatif lokal atau model QnA ekstraktif untuk fleksibilitas dan performa.
- 🎨 **Antarmuka Pengguna Intuitif**: Dibangun dengan Streamlit untuk pengalaman pengguna yang mulus dan interaktif.
- 🔧 **Konfigurasi Fleksibel**: Pilih mode QnA (generatif atau ekstraktif) sesuai kebutuhan dan sumber daya Anda.

## 🚀 Demo Singkat

Berikut adalah beberapa tampilan aplikasi DocuMentor AI:

<details>
<summary><strong>▶️ Tampilan Utama (Klik untuk lihat)</strong></summary>
<p align="center">
  <img src="docs/images/tampilan1.png" alt="Tampilan Utama DocuMentor AI" width="700"/>
  <em>Tampilan antarmuka utama dengan opsi unggah dokumen di sidebar.</em>
</p>
</details>

<details>
<summary><strong>▶️ Fitur Hasil Ringkasan (Klik untuk lihat)</strong></summary>
<p align="center">
  <img src="docs/images/fitur_summary.png" alt="Contoh Ringkasan" width="700"/>
  <em>Contoh ringkasan yang dihasilkan dari sebuah dokumen.</em>
</p>
</details>

<details>
<summary><strong>▶️ Interaksi QnA (Klik untuk lihat)</strong></summary>
<p align="center">
  <img src="docs/images/fitur_qna.png" alt="Contoh QnA" width="700"/>
  <em>Pengguna bertanya dan mendapatkan jawaban berdasarkan konten dokumen.</em>
</p>
</details>

## 🛠️ Teknologi yang Digunakan

- **Bahasa Pemrograman**: Python 3.9+
- **Framework Web/UI**: Streamlit
- **AI & NLP**:
  - LangChain
  - Hugging Face Transformers (Summarization, Embedding, QnA Models, LLMs)
  - Sentence Transformers (untuk Embedding)
  - FAISS (Vector Store)
- **Pemrosesan Dokumen**:
  - PyMuPDF (untuk PDF)
  - python-docx (untuk DOCX)

## 🧑‍💻 Dikembangkan Oleh

**Fredli4Qoni**

[![GitHub](https://img.shields.io/badge/GitHub-fredli4qoni-blue?style=flat-square&logo=github)](https://github.com/fredli4qooni)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Fredli%20Agusta%20Qoni-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/fredli-fourqoni/)

---
