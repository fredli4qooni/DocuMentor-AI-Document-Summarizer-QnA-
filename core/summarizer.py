# core/summarizer.py
from transformers import pipeline, AutoTokenizer
import torch
import gc
import re
from typing import Optional, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

class TextSummarizer:
    def __init__(self):
        self.summarizer_pipeline = None
        self.tokenizer = None
        self.model_max_token_limit = 0
        self.device = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Inisialisasi model dengan error handling yang lebih baik"""
        try:
            # Deteksi device yang optimal
            if torch.cuda.is_available():
                self.device = 0
                logger.info(f"CUDA tersedia. GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            else:
                self.device = -1
                logger.info("Menggunakan CPU untuk inference")
            
            # Load tokenizer terlebih dahulu
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_NAME,
                cache_dir="./model_cache",  # Cache lokal untuk mempercepat loading
                local_files_only=False
            )
            
            # Load pipeline dengan konfigurasi optimal
            self.summarizer_pipeline = pipeline(
                "summarization",
                model=MODEL_NAME,
                tokenizer=self.tokenizer,
                device=self.device,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,  # Gunakan fp16 untuk GPU
                model_kwargs={
                    "cache_dir": "./model_cache",
                    "low_cpu_mem_usage": True,  # Optimasi memory
                }
            )
            
            # Set batas token yang aman
            self.model_max_token_limit = min(self.tokenizer.model_max_length, 1024)
            
            logger.info(f"Model {MODEL_NAME} berhasil dimuat pada {'GPU' if self.device == 0 else 'CPU'}")
            logger.info(f"Token limit: {self.model_max_token_limit}")
            
        except Exception as e:
            logger.error(f"Gagal memuat model: {e}")
            self.summarizer_pipeline = None
            self.tokenizer = None
            self.model_max_token_limit = 0
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocessing teks untuk hasil yang lebih baik"""
        if not text:
            return ""
        
        # Normalisasi whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Hapus karakter yang tidak perlu
        text = re.sub(r'[^\w\s\.,!?;:()\-"]', '', text)
        
        # Batasi panjang paragraf yang terlalu panjang
        sentences = text.split('.')
        if len(sentences) > 50:  # Jika terlalu banyak kalimat
            text = '. '.join(sentences[:50]) + '.'
        
        return text
    
    def _chunk_text(self, text: str, max_chunk_tokens: int = 800) -> list:
        """Membagi teks menjadi chunk yang lebih kecil jika diperlukan"""
        # Hitung token dari teks
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        if len(tokens) <= max_chunk_tokens:
            return [text]
        
        # Bagi berdasarkan kalimat
        sentences = text.split('.')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            test_chunk = current_chunk + sentence + "."
            test_tokens = self.tokenizer.encode(test_chunk, add_special_tokens=False)
            
            if len(test_tokens) <= max_chunk_tokens:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + "."
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _optimize_summary_params(self, text_length: int) -> Dict[str, Any]:
        """Optimasi parameter summary berdasarkan panjang teks"""
        if text_length < 200:
            return {"max_length": 80, "min_length": 20}
        elif text_length < 500:
            return {"max_length": 130, "min_length": 30}
        elif text_length < 1000:
            return {"max_length": 200, "min_length": 50}
        else:
            return {"max_length": 300, "min_length": 80}
    
    def summarize_text(self, text_to_summarize: str, max_length: Optional[int] = None, 
                      min_length: Optional[int] = None) -> str:
        """Fungsi utama untuk meringkas teks dengan optimasi"""
        
        # Validasi awal
        if not self._is_model_ready():
            return "Error: Model tidak tersedia. Silakan restart aplikasi."
        
        if not text_to_summarize or not text_to_summarize.strip():
            return "Tidak ada teks untuk diringkas."
        
        try:
            # Preprocessing
            clean_text = self._preprocess_text(text_to_summarize)
            
            if len(clean_text) < 50:
                return "Teks terlalu pendek untuk diringkas secara efektif."
            
            # Optimasi parameter
            params = self._optimize_summary_params(len(clean_text))
            if max_length:
                params["max_length"] = max_length
            if min_length:
                params["min_length"] = min_length
            
            # Chunking jika diperlukan
            chunks = self._chunk_text(clean_text)
            
            if len(chunks) == 1:
                return self._summarize_single_chunk(chunks[0], params)
            else:
                return self._summarize_multiple_chunks(chunks, params)
                
        except Exception as e:
            logger.error(f"Error during summarization: {e}")
            return self._handle_summarization_error(e)
    
    def _summarize_single_chunk(self, text: str, params: Dict[str, Any]) -> str:
        """Meringkas satu chunk teks"""
        try:
            # Pastikan input tidak melebihi batas model
            inputs = self.tokenizer.encode(
                text, 
                return_tensors="pt",
                max_length=self.model_max_token_limit - 50,  # Sediakan buffer
                truncation=True,
                padding=False
            )
            
            truncated_text = self.tokenizer.decode(inputs[0], skip_special_tokens=True)
            
            # Generate summary
            summary_result = self.summarizer_pipeline(
                truncated_text,
                max_length=params["max_length"],
                min_length=params["min_length"],
                do_sample=False,
                clean_up_tokenization_spaces=True,
                no_repeat_ngram_size=3,  # Hindari pengulangan
                early_stopping=True
            )
            
            return summary_result[0]['summary_text'].strip()
            
        except Exception as e:
            raise Exception(f"Error in single chunk summarization: {e}")
    
    def _summarize_multiple_chunks(self, chunks: list, params: Dict[str, Any]) -> str:
        """Meringkas multiple chunks dan gabungkan hasilnya"""
        try:
            summaries = []
            
            for i, chunk in enumerate(chunks):
                logger.info(f"Meringkas chunk {i+1}/{len(chunks)}")
                
                # Kurangi parameter untuk chunk individual
                chunk_params = {
                    "max_length": max(params["max_length"] // len(chunks), 50),
                    "min_length": max(params["min_length"] // len(chunks), 10)
                }
                
                chunk_summary = self._summarize_single_chunk(chunk, chunk_params)
                summaries.append(chunk_summary)
                
                # Garbage collection untuk memory management
                if i % 3 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Gabungkan summary
            combined_summary = " ".join(summaries)
            
            # Jika gabungan masih terlalu panjang, ringkas sekali lagi
            if len(combined_summary.split()) > params["max_length"]:
                final_summary = self._summarize_single_chunk(combined_summary, params)
                return final_summary
            
            return combined_summary
            
        except Exception as e:
            raise Exception(f"Error in multiple chunk summarization: {e}")
    
    def _is_model_ready(self) -> bool:
        """Check apakah model siap digunakan"""
        return all([
            self.summarizer_pipeline is not None,
            self.tokenizer is not None,
            self.model_max_token_limit > 0
        ])
    
    def _handle_summarization_error(self, error: Exception) -> str:
        """Handle berbagai jenis error dengan pesan yang informatif"""
        error_str = str(error).lower()
        
        if "cuda out of memory" in error_str:
            return ("Error: GPU memory tidak cukup. Coba dengan teks yang lebih pendek "
                   "atau restart aplikasi untuk membersihkan memory.")
        elif "sequence length" in error_str or "too long" in error_str:
            return ("Error: Teks terlalu panjang. Coba bagi menjadi bagian yang lebih kecil "
                   "atau gunakan teks yang lebih pendek.")
        elif "index" in error_str:
            return ("Error: Terjadi masalah internal. Coba dengan format teks yang berbeda "
                   "atau restart aplikasi.")
        else:
            return f"Error tidak terduga saat meringkas: {error}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Mendapatkan informasi tentang model"""
        return {
            "model_name": MODEL_NAME,
            "device": "GPU" if self.device == 0 else "CPU",
            "max_tokens": self.model_max_token_limit,
            "is_ready": self._is_model_ready(),
            "cuda_available": torch.cuda.is_available()
        }
    
    def cleanup(self):
        """Cleanup memory"""
        if self.summarizer_pipeline:
            del self.summarizer_pipeline
        if self.tokenizer:
            del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Global instance
summarizer = TextSummarizer()

# Backward compatibility functions
def summarize_text(text_to_summarize: str, max_length: int = 150, min_length: int = 30) -> str:
    """Fungsi wrapper untuk backward compatibility"""
    return summarizer.summarize_text(text_to_summarize, max_length, min_length)

def get_model_info() -> Dict[str, Any]:
    """Get model information"""
    return summarizer.get_model_info()

# Cleanup function untuk dipanggil saat aplikasi ditutup
def cleanup_model():
    """Cleanup model resources"""
    summarizer.cleanup()