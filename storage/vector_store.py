# storage/vector_store.py

import chromadb
from typing import List
from config import Config        # ✅ import Config class

class VectorStoreManager:
    
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=Config.CHROMA_PERSIST_DIR       # ✅ Config.CHROMA_PERSIST_DIR
        )
        self.collection = self.client.get_or_create_collection(
            name=Config.COLLECTION_NAME,         # ✅ Config.COLLECTION_NAME
            metadata={"hnsw:space": "cosine"}
        )
        print(f"  ChromaDB ready. Count: {self.collection.count()}")
    
    def store_embeddings(self, embedded_chunks: List) -> None:
        if not embedded_chunks:
            print("  No chunks to store")
            return
        
        batch_size = 100
        total = 0
        
        for i in range(0, len(embedded_chunks), batch_size):
            batch = embedded_chunks[i:i + batch_size]
            
            ids = [ec.chunk.chunk_id for ec in batch]
            embeddings = [ec.embedding for ec in batch]
            documents = [ec.chunk.text for ec in batch]
            metadatas = [ec.chunk.metadata for ec in batch]
            
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            total += len(batch)
            print(f"  Stored {total}/{len(embedded_chunks)} chunks")
    
    def query(self, query_embedding: List[float], top_k: int = 4):
        count = self.collection.count()
        if count == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"]
        )
        return results
    
    def get_count(self) -> int:
        return self.collection.count()
    
    def reset_collection(self) -> None:
        self.client.delete_collection(Config.COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=Config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print("Collection reset")