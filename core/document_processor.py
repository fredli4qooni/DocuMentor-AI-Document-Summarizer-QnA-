# core/document_processor.py

import fitz  # PyMuPDF
from docx import Document as DocxDocument 
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text_from_pdf(pdf_file_object):
    """
    Mengekstrak teks dari objek file PDF yang diunggah.
    pdf_file_object adalah objek file yang didapat dari st.file_uploader.
    """
    try:
        # PyMuPDF perlu bytes, jadi kita baca dari file object
        pdf_bytes = pdf_file_object.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text += page.get_text("text") # "text" untuk plain text
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None


def extract_text_from_txt(txt_file_object):
    """
    Mengekstrak teks dari objek file TXT yang diunggah.
    """
    try:
        # Objek file dari Streamlit perlu dibaca dan didecode
        return txt_file_object.read().decode("utf-8")
    except Exception as e:
        print(f"Error extracting text from TXT: {e}")
        return None

def extract_text_from_docx(docx_file_object):

    """
    Mengekstrak teks dari objek file DOCX yang diunggah.
    """
    try:
        doc = DocxDocument(docx_file_object) # python-docx bisa langsung handle file object
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return None
    
def get_text_chunks(raw_text):
    """
    Membagi teks mentah menjadi chunks yang lebih kecil.
    """
    if not raw_text or not raw_text.strip():
        return []
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  
            chunk_overlap=200, 
            length_function=len,
            add_start_index=True, 
        )
        chunks = text_splitter.split_text(raw_text)
        
        return chunks # 
    except Exception as e:
        print(f"Error splitting text into chunks: {e}")
        return []