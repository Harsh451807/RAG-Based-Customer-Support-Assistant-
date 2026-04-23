# ingestion/document_loader.py

import fitz   # PyMuPDF
import os
from dataclasses import dataclass
from typing import List

@dataclass
class RawDocument:
    doc_id: str
    source_file: str
    page_number: int
    raw_text: str
    metadata: dict

class DocumentLoader:
    
    def load_pdf(self, file_path: str) -> List[RawDocument]:
        """Load a single PDF file"""
        documents = []
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return []
        
        try:
            pdf = fitz.open(file_path)
            file_name = os.path.basename(file_path)
            doc_id = file_name.replace(".pdf", "").replace(" ", "_").lower()
            
            print(f"  Opening: {file_name} ({pdf.page_count} pages)")
            
            for page_num in range(pdf.page_count):
                page = pdf[page_num]
                text = page.get_text("text").strip()
                
                if not text:
                    continue
                
                documents.append(RawDocument(
                    doc_id=f"{doc_id}_p{page_num + 1}",
                    source_file=file_name,
                    page_number=page_num + 1,
                    raw_text=text,
                    metadata={
                        "source": file_name,
                        "page": page_num + 1,
                        "total_pages": pdf.page_count,
                        "doc_id": doc_id
                    }
                ))
            
            pdf.close()
            return documents
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []
    
    def load_directory(self, directory: str) -> List[RawDocument]:
        """Load all PDFs from a directory"""
        all_documents = []
        
        if not os.path.exists(directory):
            print(f"Directory not found: {directory}")
            return []
        
        pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
        
        if not pdf_files:
            print(f"No PDFs found in {directory}")
            return []
        
        for pdf_file in pdf_files:
            file_path = os.path.join(directory, pdf_file)
            docs = self.load_pdf(file_path)
            all_documents.extend(docs)
        
        print(f"Total pages loaded: {len(all_documents)}")
        return all_documents