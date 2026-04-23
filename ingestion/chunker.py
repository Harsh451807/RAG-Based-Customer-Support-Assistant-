# ingestion/chunker.py

from langchain_text_splitters import RecursiveCharacterTextSplitter
from dataclasses import dataclass
from typing import List
from config import Config        # ✅ import Config class

@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: dict
    token_count: int

class DocumentChunker:
    
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,        # ✅ Config.CHUNK_SIZE
            chunk_overlap=Config.CHUNK_OVERLAP,  # ✅ Config.CHUNK_OVERLAP
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk_documents(self, documents) -> List[Chunk]:
        all_chunks = []
        
        for doc in documents:
            splits = self.splitter.split_text(doc.raw_text)
            
            for idx, split_text in enumerate(splits):
                chunk = Chunk(
                    chunk_id=f"{doc.doc_id}_chunk_{idx}",
                    text=split_text,
                    metadata={
                        **doc.metadata,
                        "chunk_index": idx,
                        "chunk_id": f"{doc.doc_id}_chunk_{idx}"
                    },
                    token_count=len(split_text.split())
                )
                all_chunks.append(chunk)
        
        return all_chunks