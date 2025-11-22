"""Document loading and chunking utilities."""

import os
from typing import List, Dict
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import config

class DocumentProcessor:
    """Handles document loading and chunking."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_pdf(self, file_path: str) -> str:
        """Load text from PDF file."""
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"❌ Error loading PDF {file_path}: {e}")
            return ""
    
    def load_txt(self, file_path: str) -> str:
        """Load text from TXT file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"❌ Error loading TXT {file_path}: {e}")
            return ""
    
    def load_documents(self, data_dir: str = config.DATA_DIR) -> List[Dict]:
        """Load all documents from the data directory."""
        documents = []
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"📁 Created {data_dir} directory. Please add your documents.")
            return documents
        
        for filename in os.listdir(data_dir):
            if filename.startswith('.') or filename == 'README.md':
                continue
                
            file_path = os.path.join(data_dir, filename)
            
            if not os.path.isfile(file_path):
                continue
            
            if filename.endswith('.pdf'):
                text = self.load_pdf(file_path)
            elif filename.endswith('.txt'):
                text = self.load_txt(file_path)
            else:
                print(f"⚠  Skipping unsupported file: {filename}")
                continue
            
            if text.strip():
                documents.append({
                    'filename': filename,
                    'content': text
                })
                print(f"✅ Loaded: {filename}")
        
        return documents
    
    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """Split documents into chunks."""
        all_chunks = []
        
        for doc in documents:
            chunks = self.text_splitter.split_text(doc['content'])
            
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    'text': chunk,
                    'source': doc['filename'],
                    'chunk_id': i
                })
        
        print(f"📄 Created {len(all_chunks)} chunks from {len(documents)} documents")
        return all_chunks
