"""Embedding generation utilities."""

from sentence_transformers import SentenceTransformer
import config
import os

class Embedder:
    """Handles text embedding generation."""
    
    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        print(f"🔄 Loading embedding model: {model_name}")
        
        # Set HF token if available
        if config.HF_API_KEY:
            os.environ['HUGGINGFACE_HUB_TOKEN'] = config.HF_API_KEY
        
        try:
            self.model = SentenceTransformer(model_name)
            print(f"✅ Embedding model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading embedding model: {e}")
            raise
    
    def embed_text(self, text: str):
        """Generate embedding for a single text."""
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_batch(self, texts: list):
        """Generate embeddings for multiple texts."""
        print(f"🔄 Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(
            texts, 
            convert_to_numpy=True, 
            show_progress_bar=True,
            batch_size=32
        )
        print(f"✅ Embeddings generated")
        return embeddings