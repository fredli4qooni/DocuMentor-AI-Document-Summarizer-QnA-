# app.py
import streamlit as st
from core.document_processor import (
    extract_text_from_pdf, extract_text_from_txt, extract_text_from_docx,
    get_text_chunks
)
from core.summarizer import summarize_text
from core.qna_handler import (
    create_vector_store_from_chunks,
    get_conversational_qa_chain,
    get_answer_from_chain,
    get_extractive_answer,
    initialize_llm,
    llm_type as qna_llm_type
)
import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="DocuMentor AI", 
    page_icon="🤖",           
    layout="wide",              
    initial_sidebar_state="expanded" 
)

# --- Sidebar ---
with st.sidebar:
    st.title("DocuMentor AI")
    st.markdown("Asisten Cerdas untuk Dokumen Anda")
    st.markdown("---")
    st.subheader("📤 Unggah Dokumen")
    uploaded_file = st.file_uploader(
        "Pilih file (PDF, TXT, DOCX)",
        type=["pdf", "txt", "docx"],
        label_visibility="collapsed" 
    )
    st.markdown("---")
    st.subheader("⚙️ Pengaturan Model QnA")
    use_generative = st.checkbox(
        "Gunakan LLM Generatif (FLAN-T5)",
        value=False, # Default ke ekstraktif
        help="Lebih interaktif & kontekstual, namun butuh resource lebih. Jika tidak dicentang, model QnA ekstraktif (lebih cepat) akan digunakan."
    )

    # Kredit Developer di Sidebar
    st.markdown("---")
    st.markdown(" Dev ❤️ by **fredli4qoni**")
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Profile-blue?style=flat-square&logo=github)](https://github.com/fredli4qooni)") 


# --- Fungsi Cache untuk LLM ---
@st.cache_resource
def load_llm_config(use_gen_llm):
    initialize_llm(use_generative_llm=use_gen_llm)
    return qna_llm_type

current_llm_type = load_llm_config(use_generative)
st.sidebar.info(f"Mode QnA: **{str(current_llm_type).upper()}**")


# --- Inisialisasi Session State ---
if 'raw_text' not in st.session_state: st.session_state.raw_text = None
if 'text_chunks' not in st.session_state: st.session_state.text_chunks = None
if 'vector_store' not in st.session_state: st.session_state.vector_store = None
if 'qna_chain' not in st.session_state: st.session_state.qna_chain = None
if 'uploaded_file_name' not in st.session_state: st.session_state.uploaded_file_name = None
if 'summary' not in st.session_state: st.session_state.summary = None
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'doc_processed' not in st.session_state: st.session_state.doc_processed = False


# --- Konten Utama Aplikasi ---
st.header("🤖 DocuMentor AI: Ringkas & Tanya Jawab Dokumen Anda")
st.markdown("Unggah dokumen Anda di panel sebelah kiri, lalu klik tombol 'Proses Dokumen' di bawah ini.")

# Tombol Proses Dokumen, lebih menonjol
process_button_placeholder = st.empty() 
if uploaded_file:
    if st.session_state.uploaded_file_name != uploaded_file.name:
        # Jika file baru diunggah, reset status proses
        st.session_state.doc_processed = False
        # Reset juga state lainnya untuk file baru
        st.session_state.raw_text = None
        st.session_state.text_chunks = None
        st.session_state.summary = None
        st.session_state.vector_store = None
        st.session_state.qna_chain = None
        st.session_state.chat_history = []

    process_button = st.button("✨ Proses Dokumen Sekarang!", type="primary", use_container_width=True, disabled=(uploaded_file is None or st.session_state.doc_processed))
else:
    process_button = st.button("✨ Proses Dokumen Sekarang!", type="primary", use_container_width=True, disabled=True)


if process_button and uploaded_file is not None:
    st.session_state.uploaded_file_name = uploaded_file.name
    st.session_state.doc_processed = False # Set ulang status proses

    # Menggunakan progress bar
    progress_bar = st.progress(0, text="Memulai pemrosesan...")

    try:
        # 1. Ekstraksi Teks
        progress_bar.progress(10, text=f"Membaca file: {uploaded_file.name}...")
        extracted_text = ""
        if uploaded_file.type == "application/pdf": extracted_text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "text/plain": extracted_text = extract_text_from_txt(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document": extracted_text = extract_text_from_docx(uploaded_file)

        if not extracted_text or not extracted_text.strip():
            st.error(f"Tidak ada teks yang dapat diekstrak dari '{uploaded_file.name}'.")
            progress_bar.progress(100, text="Gagal mengekstrak teks.")
            st.stop() # Hentikan jika tidak ada teks

        st.session_state.raw_text = extracted_text
        progress_bar.progress(25, text="Teks berhasil diekstrak!")

        # 2. Pembuatan Ringkasan
        progress_bar.progress(30, text="Membuat ringkasan...")
        summary_result = summarize_text(st.session_state.raw_text)
        st.session_state.summary = summary_result
        if "Error:" in summary_result:
            st.warning(f"Peringatan terkait ringkasan: {summary_result}")
        progress_bar.progress(50, text="Ringkasan selesai.")

        # 3. Persiapan QnA
        progress_bar.progress(55, text="Memecah teks untuk QnA...")
        st.session_state.text_chunks = get_text_chunks(st.session_state.raw_text)
        if not st.session_state.text_chunks:
            st.warning("Tidak ada potongan teks dihasilkan untuk QnA.")
            progress_bar.progress(100, text="Gagal mempersiapkan QnA.")
            st.session_state.doc_processed = True # Tandai sudah diproses (meskipun QnA gagal)
            st.stop()

        progress_bar.progress(70, text="Membuat index pencarian QnA...")
        st.session_state.vector_store = create_vector_store_from_chunks(st.session_state.text_chunks)
        if st.session_state.vector_store is None:
            st.error("Gagal membuat basis data vektor untuk QnA.")
            progress_bar.progress(100, text="Gagal membuat index QnA.")
            st.session_state.doc_processed = True
            st.stop()
        
        progress_bar.progress(85, text="Menyiapkan model QnA...")
        if current_llm_type == 'generative':
            st.session_state.qna_chain = get_conversational_qa_chain(st.session_state.vector_store)
        elif current_llm_type == 'extractive':
            st.session_state.qna_chain = "extractive_qa_mode"
        else:
            st.session_state.qna_chain = None
            st.warning("Mode QnA tidak terdefinisi.")
        
        progress_bar.progress(100, text="Pemrosesan dokumen selesai!")
        st.success(f"Dokumen '{uploaded_file.name}' berhasil diproses!")
        st.session_state.doc_processed = True # Tandai dokumen sudah diproses

    except Exception as e:
        st.error(f"Terjadi kesalahan fatal saat memproses dokumen: {e}")
        progress_bar.progress(100, text="Error!")
        # Reset state jika error
        st.session_state.raw_text = None; st.session_state.summary = None
        st.session_state.text_chunks = None; st.session_state.vector_store = None
        st.session_state.qna_chain = None; st.session_state.doc_processed = False


# --- Tampilkan Hasil Jika Dokumen Sudah Diproses ---
if st.session_state.doc_processed and uploaded_file is not None:
    # Menggunakan Tabs untuk Ringkasan dan QnA
    tab1, tab2, tab3 = st.tabs(["📝 Ringkasan", "💬 Tanya Jawab (QnA)", "📜 Teks Asli"])

    with tab1:
        st.subheader("Ringkasan Dokumen")
        if st.session_state.summary:
            st.markdown(st.session_state.summary)
            # Info tambahan
            if st.session_state.raw_text:
                original_words = len(st.session_state.raw_text.split())
                summary_words = len(st.session_state.summary.split())
                if original_words > 0:
                    reduction = 100 - (summary_words / original_words * 100)
                    st.caption(f"Panjang asli: {original_words} kata. Panjang ringkasan: {summary_words} kata (Pengurangan ~{reduction:.0f}%).")
        else:
            st.info("Ringkasan tidak tersedia atau belum dibuat.")

    with tab2:
        st.subheader("Tanya Jawab (QnA) berdasarkan Dokumen")
        if st.session_state.vector_store and (st.session_state.qna_chain or current_llm_type == 'extractive'):
            # Tampilkan histori chat jika mode generatif
            if current_llm_type == 'generative' and st.session_state.chat_history:
                st.markdown("**Riwayat Percakapan:**")
                for i, (q, a) in enumerate(st.session_state.chat_history[-5:]): # Tampilkan 5 interaksi terakhir
                    st.info(f"🤔 **Anda:** {q}")
                    st.success(f"🤖 **DocuMentor:** {a}")
                if len(st.session_state.chat_history) > 5:
                    st.caption("Menampilkan 5 interaksi terakhir...")
                st.markdown("---")


            user_question = st.text_input("Ketik pertanyaan Anda di sini:", key="user_qna_input")

            if user_question:
                with st.spinner("🤖 DocuMentor sedang berpikir..."):
                    answer = ""
                    source_docs = []
                    formatted_chat_history = [] # Untuk mode generatif

                    if current_llm_type == 'generative' and st.session_state.qna_chain and st.session_state.qna_chain != "extractive_qa_mode":
                        for q_hist, a_hist in st.session_state.chat_history:
                            formatted_chat_history.append((q_hist, a_hist))
                        answer, source_docs = get_answer_from_chain(
                            st.session_state.qna_chain, user_question, chat_history=formatted_chat_history
                        )
                        if "Error:" not in answer and answer.strip():
                            st.session_state.chat_history.append((user_question, answer))
                    
                    elif current_llm_type == 'extractive':
                        answer, source_docs = get_extractive_answer(user_question, st.session_state.vector_store)
                        # Untuk mode ekstraktif, kita tidak maintain chat history yang kompleks
                        if "Error:" not in answer and answer.strip():
                             st.session_state.chat_history.append((user_question, answer))


                    st.markdown(f"🤖 **DocuMentor:**")
                    st.info(f"{answer}") # Menggunakan st.info untuk jawaban bot

                    if source_docs:
                        with st.expander("Lihat sumber/konteks yang digunakan (klik untuk buka)"):
                            for i, doc in enumerate(source_docs):
                                st.markdown(f"**Konteks {i+1}:**")
                                st.caption(f"...{doc.page_content[:300]}...") # Tampilkan 300 char pertama
                                st.markdown("---")
        else:
            st.warning("Fitur QnA belum siap. Pastikan dokumen sudah diproses dan model QnA berhasil dimuat.")

    with tab3:
        st.subheader("Teks Asli Dokumen")
        if st.session_state.raw_text:
            st.text_area("Isi Dokumen Lengkap:", st.session_state.raw_text, height=400, key="raw_text_display")
            st.caption(f"Total {len(st.session_state.raw_text.split())} kata, {len(st.session_state.text_chunks)} potongan teks (chunks) untuk QnA.")
        else:
            st.info("Tidak ada teks dokumen yang ditampilkan.")

# Placeholder jika belum ada file diunggah
if not uploaded_file:
    st.info("👋 Selamat datang! Silakan unggah dokumen Anda di panel sebelah kiri untuk memulai.")
    st.markdown("""
    **DocuMentor AI dapat membantu Anda untuk:**
    - **Meringkas** dokumen panjang dengan cepat.
    - **Menjawab pertanyaan** spesifik berdasarkan isi dokumen.

    Cukup unggah file PDF, TXT, atau DOCX Anda!
    """)