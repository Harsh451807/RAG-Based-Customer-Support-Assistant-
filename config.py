# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # ── LLM Settings ────────────────────────────────────────
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    LLM_MODEL = "llama-3.1-8b-instant"        # ✅ CHANGED THIS
    LLM_TEMPERATURE = 0.2
    LLM_MAX_TOKENS = 500

    # ── Embedding Settings ───────────────────────────────────
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    # ── Chunking Settings ────────────────────────────────────
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50

    # ── Retrieval Settings ───────────────────────────────────
    TOP_K = 4
    MIN_SIMILARITY_SCORE = 0.35

    # ── ChromaDB Settings ────────────────────────────────────
    CHROMA_PERSIST_DIR = "./data/chroma_db"
    COLLECTION_NAME = "support_knowledge_base"

    # ── HITL Settings ────────────────────────────────────────
    LLM_CONFIDENCE_THRESHOLD = 0.5
    TICKETS_FILE = "./data/tickets.json"

    # ── Document Settings ────────────────────────────────────
    DOCUMENTS_DIR = "./data/documents"

    # ── System Prompt ────────────────────────────────────────
    SYSTEM_PROMPT = """You are a helpful customer support assistant.
Your role is to answer customer questions accurately and concisely
based ONLY on the provided context from our knowledge base.

Rules:
1. Answer ONLY based on the provided context
2. If you cannot find the answer in the context, say exactly:
   I don't have enough information to answer this question accurately.
3. Keep answers clear and concise (3-5 sentences maximum)
4. List steps when explaining procedures
5. Always maintain a helpful, professional tone
6. Reference the source document when possible"""