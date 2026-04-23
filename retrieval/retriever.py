# retrieval/retriever.py

from dataclasses import dataclass
from typing import List
from ingestion.embedder import EmbeddingEngine
from storage.vector_store import VectorStoreManager
from config import Config        # ✅ import Config class

@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int
    similarity_score: float
    rank: int

@dataclass
class RetrievalResult:
    query: str
    chunks: List[RetrievedChunk]
    top_score: float
    is_confident: bool
    context_text: str

class ContextRetriever:
    
    def __init__(self, embedder: EmbeddingEngine, vector_store: VectorStoreManager):
        self.embedder = embedder
        self.vector_store = vector_store
        self.min_score = Config.MIN_SIMILARITY_SCORE   # ✅ Config.MIN_SIMILARITY_SCORE
    
    def retrieve(self, query: str) -> RetrievalResult:
        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)
        
        # Search ChromaDB
        raw_results = self.vector_store.query(
            query_embedding, 
            Config.TOP_K                               # ✅ Config.TOP_K
        )
        
        if not raw_results["documents"][0]:
            return RetrievalResult(
                query=query,
                chunks=[],
                top_score=0.0,
                is_confident=False,
                context_text=""
            )
        
        # Process results
        chunks = []
        for idx, (doc, meta, distance) in enumerate(zip(
            raw_results["documents"][0],
            raw_results["metadatas"][0],
            raw_results["distances"][0]
        )):
            similarity = 1 - (distance / 2)
            chunks.append(RetrievedChunk(
                text=doc,
                source=meta.get("source", "Unknown"),
                page=meta.get("page", 0),
                similarity_score=round(similarity, 3),
                rank=idx + 1
            ))
        
        top_score = chunks[0].similarity_score if chunks else 0.0
        is_confident = top_score >= self.min_score
        
        print(f"  Top score: {top_score:.3f} | Confident: {is_confident}")
        
        return RetrievalResult(
            query=query,
            chunks=chunks,
            top_score=top_score,
            is_confident=is_confident,
            context_text=self._format_context(chunks)
        )
    
    def _format_context(self, chunks: List[RetrievedChunk]) -> str:
        if not chunks:
            return ""
        
        context_parts = []
        for chunk in chunks:
            context_parts.append(
                f"[Source: {chunk.source}, Page {chunk.page}]\n"
                f"{chunk.text}"
            )
        
        return "\n\n---\n\n".join(context_parts)