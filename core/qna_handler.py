# core/qna_handler.py
import os
import gc
import logging
import time
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum
import warnings

# Suppress LangChain deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    import numpy as np
except ImportError:
    # Fallback jika numpy tidak tersedia
    class np:
        @staticmethod
        def mean(arr):
            return sum(arr) / len(arr) if arr else 0

import torch

# Updated imports untuk LangChain yang lebih baru
try:
    from langchain_community.embeddings import SentenceTransformerEmbeddings, HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.llms import HuggingFacePipeline
except ImportError:
    # Fallback untuk versi lama
    from langchain.embeddings import SentenceTransformerEmbeddings, HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.llms import HuggingFacePipeline

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.schema import Document
from langchain.prompts import PromptTemplate

from transformers import (
    AutoModelForSeq2SeqLM, 
    AutoTokenizer, 
    pipeline as hf_pipeline,
    AutoModelForQuestionAnswering
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration constants
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
QA_MODEL_NAME = "deepset/roberta-base-squad2"

class OptimizedQnAHandler:
    def __init__(self):
        self.embeddings = None
        self.llm_for_qna = None
        self.llm_type = None
        self.qa_pipeline = None
        self.vector_store = None
        self.conversation_chain = None
        self.memory = None
        self.device = self._get_optimal_device()
        
        # Performance tracking
        self.response_times = []
        self.cache = {}
        self._extractive_qa_pipeline = None
        
        self._initialize_components()
    
    def _get_optimal_device(self) -> str:
        """Menentukan device optimal untuk inference"""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU tersedia dengan {gpu_memory:.1f}GB memory")
            return "cuda"
        else:
            logger.info("Menggunakan CPU untuk inference")
            return "cpu"
    
    def _initialize_components(self):
        """Inisialisasi komponen dasar"""
        try:
            self._load_embeddings()
            self._initialize_memory()
            logger.info("QnA Handler berhasil diinisialisasi")
        except Exception as e:
            logger.error(f"Error during initialization: {e}")
    
    def _load_embeddings(self):
        """Load embedding model dengan fallback yang robust"""
        try:
            # Coba berbagai cara loading embeddings
            embedding_options = [
                lambda: SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME),
                lambda: HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME),
                lambda: SentenceTransformerEmbeddings(
                    model_name=EMBEDDING_MODEL_NAME,
                    model_kwargs={'device': 'cpu'}
                ),
            ]
            
            for i, embedding_loader in enumerate(embedding_options):
                try:
                    self.embeddings = embedding_loader()
                    logger.info(f"Embedding model '{EMBEDDING_MODEL_NAME}' loaded successfully (method {i+1})")
                    return
                except Exception as e:
                    logger.warning(f"Embedding loading method {i+1} failed: {e}")
                    continue
            
            raise Exception("Semua metode loading embedding gagal")
                
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            self.embeddings = None
    
    def _initialize_memory(self):
        """Inisialisasi memory untuk conversation"""
        try:
            self.memory = ConversationBufferMemory(
                memory_key='chat_history',
                return_messages=True,
                output_key='answer'
            )
            logger.info("Conversation memory initialized")
        except Exception as e:
            logger.error(f"Error initializing memory: {e}")
    
    def initialize_llm(self, use_generative_llm=False, repo_id="google/flan-t5-base"):
        """Inisialisasi LLM dengan backward compatibility"""
        global llm_for_qna, llm_type
        
        if self.llm_for_qna is not None:
            logger.info(f"LLM ({self.llm_type}) already initialized.")
            return True

        if use_generative_llm:
            return self._load_generative_llm(repo_id)
        else:
            return self._load_extractive_qa()
    
    def _load_generative_llm(self, model_name: str) -> bool:
        """Load model generatif dengan optimasi"""
        try:
            logger.info(f"Loading generative model: {model_name}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cuda":
                model = model.to("cuda")
            
            pipe = hf_pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=512,
                temperature=0.7,
                device=0 if self.device == "cuda" else -1
            )
            
            self.llm_for_qna = HuggingFacePipeline(pipeline=pipe)
            self.llm_type = 'generative'
            
            logger.info(f"Generative LLM '{model_name}' loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load generative LLM: {e}")
            self.llm_for_qna = None
            self.llm_type = None
            return False
    
    def _load_extractive_qa(self) -> bool:
        """Load model QA ekstraktif"""
        try:
            logger.info(f"Loading extractive QA model: {QA_MODEL_NAME}")
            
            # Test pipeline availability
            test_pipeline = hf_pipeline("question-answering", model=QA_MODEL_NAME)
            self.llm_type = 'extractive'
            
            logger.info(f"Extractive QA model '{QA_MODEL_NAME}' is available")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load extractive QA model: {e}")
            self.llm_type = None
            return False
    
    def create_vector_store_from_chunks(self, text_chunks):
        """Membuat vector store dari chunks dengan backward compatibility"""
        if not self.embeddings:
            logger.error("Embedding model not loaded. Cannot create vector store.")
            return None
        
        if not text_chunks:
            logger.error("No text chunks provided to create vector store.")
            return None
        
        try:
            # Preprocessing chunks
            processed_chunks = []
            for chunk in text_chunks:
                clean_chunk = chunk.strip()
                if len(clean_chunk) > 20:  # Skip chunks yang terlalu pendek
                    processed_chunks.append(clean_chunk)
            
            if not processed_chunks:
                logger.error("No valid chunks after preprocessing")
                return None
            
            logger.info(f"Creating vector store from {len(processed_chunks)} chunks...")
            self.vector_store = FAISS.from_texts(texts=processed_chunks, embedding=self.embeddings)
            
            logger.info("Vector store created successfully using FAISS.")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            return None
    
    def get_conversational_qa_chain(self, vector_store):
        """Setup conversational chain dengan backward compatibility"""
        if vector_store is None:
            logger.error("Vector store is None. Cannot create QnA chain.")
            return None
        
        if self.llm_type is None:
            logger.error("No LLM or QA model available. Cannot create QnA chain.")
            return None
        
        self.vector_store = vector_store
        
        try:
            if self.llm_type == 'generative' and self.llm_for_qna:
                retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                
                self.conversation_chain = ConversationalRetrievalChain.from_llm(
                    llm=self.llm_for_qna,
                    retriever=retriever,
                    memory=self.memory,
                    return_source_documents=True
                )
                
                logger.info("Conversational QnA chain (generative) created.")
                return self.conversation_chain
            
            elif self.llm_type == 'extractive':
                logger.info("Using extractive QA. ConversationalRetrievalChain not applicable.")
                return "extractive_qa_mode"
            
            else:
                logger.error("LLM type undefined. Cannot create QnA chain.")
                return None
                
        except Exception as e:
            logger.error(f"Error creating QnA chain: {e}")
            return None
    
    def get_answer_from_chain(self, qa_chain_object, query, chat_history=[]):
        """Mendapatkan jawaban dengan backward compatibility"""
        if qa_chain_object is None:
            return "Error: QnA chain belum diinisialisasi.", []
        
        if isinstance(qa_chain_object, str) and qa_chain_object == "extractive_qa_mode":
            return "Extractive QnA mode: Process in app.py", []
        
        try:
            start_time = time.time()
            
            result = qa_chain_object({"question": query, "chat_history": chat_history})
            answer = result.get("answer", "Tidak ada jawaban yang ditemukan.")
            source_documents = result.get("source_documents", [])
            
            # Track performance
            response_time = time.time() - start_time
            self.response_times.append(response_time)
            
            return answer, source_documents
            
        except Exception as e:
            logger.error(f"Error getting answer from chain: {e}")
            return f"Error saat memproses pertanyaan: {e}", []
    
    def get_extractive_answer(self, query, vector_store):
        """QA ekstraktif dengan optimasi"""
        if vector_store is None:
            return "Vector store belum ada.", []
        
        if self._extractive_qa_pipeline is None:
            try:
                self._extractive_qa_pipeline = hf_pipeline(
                    "question-answering", 
                    model=QA_MODEL_NAME,
                    device=0 if self.device == "cuda" else -1
                )
                logger.info(f"Extractive QA pipeline '{QA_MODEL_NAME}' initialized for direct use.")
            except Exception as e:
                logger.error(f"Failed to initialize extractive QA pipeline: {e}")
                return f"Gagal memuat model QA ekstraktif: {e}", []
        
        try:
            start_time = time.time()
            
            # Retrieve relevant documents
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            relevant_docs = retriever.get_relevant_documents(query)
            
            if not relevant_docs:
                return "Tidak ditemukan informasi relevan dalam dokumen untuk pertanyaan ini.", []
            
            # Combine context dengan batasan panjang
            contexts = [doc.page_content for doc in relevant_docs]
            context = " ".join(contexts)
            
            # Batasi panjang context untuk menghindari error
            if len(context) > 2000:
                context = context[:2000] + "..."
            
            # QA Pipeline
            qa_result = self._extractive_qa_pipeline(question=query, context=context)
            answer = qa_result['answer']
            
            # Track performance
            response_time = time.time() - start_time
            self.response_times.append(response_time)
            
            return answer, relevant_docs
            
        except Exception as e:
            logger.error(f"Error during extractive QnA: {e}")
            return f"Error saat melakukan QnA ekstraktif: {e}", []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Mendapatkan statistik performa"""
        return {
            "total_queries": len(self.response_times),
            "avg_response_time": np.mean(self.response_times) if self.response_times else 0,
            "cache_size": len(self.cache),
            "vector_store_size": self.vector_store.index.ntotal if self.vector_store else 0,
            "llm_type": self.llm_type,
            "device": self.device,
            "embeddings_loaded": self.embeddings is not None
        }
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.llm_for_qna:
                del self.llm_for_qna
            if self._extractive_qa_pipeline:
                del self._extractive_qa_pipeline
            if self.vector_store:
                del self.vector_store
            
            self.cache.clear()
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("QnA Handler cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Global instance untuk backward compatibility
qna_handler = OptimizedQnAHandler()

# Inisialisasi default
qna_handler.initialize_llm(use_generative_llm=False)

# Expose global variables untuk backward compatibility
llm_for_qna = qna_handler.llm_for_qna
llm_type = qna_handler.llm_type
embeddings = qna_handler.embeddings

# Backward compatibility functions
def initialize_llm(use_generative_llm=False, repo_id="google/flan-t5-base"):
    """Backward compatibility function"""
    global llm_for_qna, llm_type
    success = qna_handler.initialize_llm(use_generative_llm, repo_id)
    llm_for_qna = qna_handler.llm_for_qna
    llm_type = qna_handler.llm_type
    return success

def create_vector_store_from_chunks(text_chunks):
    """Backward compatibility function"""
    return qna_handler.create_vector_store_from_chunks(text_chunks)

def get_conversational_qa_chain(vector_store):
    """Backward compatibility function"""
    return qna_handler.get_conversational_qa_chain(vector_store)

def get_answer_from_chain(qa_chain_object, query, chat_history=[]):
    """Backward compatibility function"""
    return qna_handler.get_answer_from_chain(qa_chain_object, query, chat_history)

def get_extractive_answer(query, vector_store):
    """Backward compatibility function"""
    return qna_handler.get_extractive_answer(query, vector_store)

def get_statistics():
    """Get performance statistics"""
    return qna_handler.get_statistics()

def cleanup_qna():
    """Cleanup QnA resources"""
    qna_handler.cleanup()