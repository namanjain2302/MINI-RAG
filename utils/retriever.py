# utils/retriever.py
"""Vector database and retrieval utilities."""

import chromadb
from chromadb.config import Settings
import config
from typing import List, Dict
import traceback  # For detailed error logging


class VectorRetriever:
    """Handles vector storage and retrieval with Chroma."""
    
    def __init__(self):
        print(f"🔄 Initializing ChromaDB at {config.CHROMA_PERSIST_DIR}")
        
        self.client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
        
        # Get or create collection with proper exception handling
        try:
            self.collection = self.client.get_collection(name=config.COLLECTION_NAME)
            count = self.collection.count()
            print(f"✅ Loaded existing collection with {count} chunks")
        except Exception as e:
            print(f"⚠ Warning: Could not load collection: {e}")
            traceback.print_exc()  # Optional: prints detailed stack trace to console
            # Create new collection as fallback
            self.collection = self.client.create_collection(
                name=config.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            print("✅ Created new collection")
    
    def add_chunks(self, chunks: List[Dict], embeddings):
        """Add chunks and their embeddings to the database."""
        ids = [f"{chunk['source']}_{chunk['chunk_id']}" for chunk in chunks]
        documents = [chunk['text'] for chunk in chunks]
        metadatas = [{'source': chunk['source'], 'chunk_id': chunk['chunk_id']} 
                     for chunk in chunks]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas
        )
        
        print(f"✅ Added {len(chunks)} chunks to the database")
    
    def search(self, query_embedding, top_k: int = config.TOP_K_CHUNKS):
        """Search for similar chunks."""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        return results
    
    def get_collection_count(self):
        """Get the number of documents in the collection."""
        return self.collection.count()
    
    def reset_collection(self):
        """Delete and recreate the collection."""
        try:
            self.client.delete_collection(name=config.COLLECTION_NAME)
            self.collection = self.client.create_collection(
                name=config.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            print("✅ Collection reset successfully")
        except Exception as e:
            print(f"❌ Error resetting collection: {e}")
            traceback.print_exc()