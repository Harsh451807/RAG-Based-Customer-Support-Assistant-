# ingest.py

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ingestion.document_loader import DocumentLoader
from ingestion.chunker import DocumentChunker
from ingestion.embedder import EmbeddingEngine
from storage.vector_store import VectorStoreManager
from config import Config        # ✅ import Config class

def run_ingestion():
    print("="*50)
    print("   RAG Ingestion Pipeline Starting...")
    print("="*50)
    
    docs_dir = Config.DOCUMENTS_DIR              # ✅ Config.DOCUMENTS_DIR
    
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        print(f"Please add PDF files to: {docs_dir}")
        return
    
    pdf_files = [f for f in os.listdir(docs_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in: {docs_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s): {pdf_files}")
    
    print("\nStep 1: Loading PDF files...")
    loader = DocumentLoader()
    documents = loader.load_directory(docs_dir)
    
    print("\nStep 2: Splitting into chunks...")
    chunker = DocumentChunker()
    chunks = chunker.chunk_documents(documents)
    
    print("\nStep 3: Creating embeddings...")
    embedder = EmbeddingEngine()
    embedded_chunks = embedder.embed_chunks(chunks)
    
    print("\nStep 4: Storing in ChromaDB...")
    vector_store = VectorStoreManager()
    vector_store.store_embeddings(embedded_chunks)
    
    print("\n" + "="*50)
    print("✅ Ingestion Complete!")
    print(f"   Total chunks stored: {vector_store.get_count()}")
    print("   You can now run: python main.py")
    print("="*50)

if __name__ == "__main__":
    run_ingestion()