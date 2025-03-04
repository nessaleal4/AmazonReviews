import numpy as np
from typing import List, Dict, Any, Union
import pandas as pd
from tqdm import tqdm

class EmbeddingGenerator:
    """Utility class for generating simple embeddings from text"""
    
    def __init__(self):
        """Initialize with a simple model"""
        self.model_loaded = True
        self.vector_size = 384  # Default vector size
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate a simple embedding for a text using hash-based approach"""
        if not text or not isinstance(text, str):
            return np.zeros(self.vector_size)
        
        # Clean text
        text = text.lower().strip()
        
        # Create a deterministic embedding based on text content
        # This is not a real semantic embedding but provides a placeholder
        # until we can fix the PyTorch issues
        
        # Generate a seed from the text
        seed = sum(ord(c) for c in text) % 10000
        np.random.seed(seed)
        
        # Generate a random vector with fixed seed
        embedding = np.random.randn(self.vector_size)
        
        # Normalize to unit length
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    
    def generate_embeddings_batch(self, texts: List[str], batch_size=32) -> List[np.ndarray]:
        """Generate embeddings for a list of texts"""
        all_embeddings = []
        
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = [self.generate_embedding(text) for text in batch_texts]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def process_reviews_dataframe(self, df: pd.DataFrame, text_column='text',
                                  category=None, batch_size=32) -> Dict[str, Any]:
        """Process a dataframe of reviews, returning embeddings and payloads"""
        if df.empty:
            return {"embeddings": [], "payloads": []}
        
        # Extract text for embedding
        texts = df[text_column].fillna('').tolist()
        
        # Generate embeddings
        embeddings = self.generate_embeddings_batch(texts, batch_size)
        
        # Prepare payloads
        payloads = []
        for i, row in df.iterrows():
            # Extract rating and convert to sentiment
            rating = row.get('overall', 0)
            if rating >= 4:
                sentiment = 'positive'
            elif rating <= 2:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            # Create payload
            payload = {
                "review_id": row.get('reviewerID', str(i)),
                "product_id": row.get('asin', ''),
                "category": category or '',
                "product_title": row.get('title', 'Unknown Product'),
                "text": row.get(text_column, ''),
                "sentiment": sentiment,
                "rating": rating
            }
            payloads.append(payload)
        
        return {
            "embeddings": embeddings,
            "payloads": payloads
        }

# Example usage
if __name__ == "__main__":
    # Test the embedding generator
    generator = EmbeddingGenerator()
    
    # Test with a single text
    text = "This product is amazing! I love it and would recommend it to everyone."
    embedding = generator.generate_embedding(text)
    print(f"Generated embedding shape: {embedding.shape}")
    
    # Test with a batch
    texts = [
        "Great product, works as expected.",
        "Terrible quality, broke after one use.",
        "Average product for the price."
    ]
    embeddings = generator.generate_embeddings_batch(texts)
    print(f"Generated {len(embeddings)} embeddings")
