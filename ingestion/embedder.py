# ingestion/embedder.py

from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from typing import List
from config import Config        # ✅ import Config class

@dataclass
class EmbeddedChunk:
    chunk: object
    embedding: List[float]

class EmbeddingEngine:
    
    def __init__(self):
        print(f"  Loading embedding model: {Config.EMBEDDING_MODEL}")
        self.model = SentenceTransformer(Config.EMBEDDING_MODEL)
        print("  Embedding model loaded!")
    
    def embed_chunks(self, chunks: List) -> List[EmbeddedChunk]:
        texts = [chunk.text for chunk in chunks]
        
        print(f"  Creating embeddings for {len(chunks)} chunks...")
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        
        embedded_chunks = [
            EmbeddedChunk(
                chunk=chunk,
                embedding=embedding.tolist()
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]
        
        return embedded_chunks
    
    def embed_query(self, query: str) -> List[float]:
        embedding = self.model.encode(
            query,
            normalize_embeddings=True
        )
        return embedding.tolist()