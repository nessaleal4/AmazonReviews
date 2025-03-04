import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List, Dict, Any, Union
import pandas as pd
from tqdm import tqdm

class EmbeddingGenerator:
    """Utility class for generating embeddings from text using pretrained models"""
    
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize with a pretrained model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(self.device)
            self.model_loaded = True
        except Exception as e:
            print(f"Error loading embedding model: {str(e)}")
            self.model_loaded = False
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        if not self.model_loaded or not text:
            return np.zeros(384)  # Default size for the MiniLM model
        
        # Preprocess text
        if isinstance(text, str):
            text = text.replace('\n', ' ')
        else:
            text = ''
        
        # Tokenize and generate embedding
        try:
            inputs = self.tokenizer(
                text, 
                return_tensors='pt', 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Use mean pooling to get sentence embedding
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            
            # Mask padding tokens
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            
            # Sum token embeddings and normalize
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.sum(input_mask_expanded, 1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            
            # Calculate mean
            embeddings = (sum_embeddings / sum_mask).squeeze()
            
            # Convert to numpy and return
            return embeddings.cpu().numpy()
        except Exception as e:
            print(f"Error generating embedding: {str(e)}")
            return np.zeros(384)  # Default size for the MiniLM model
    
    def generate_embeddings_batch(self, texts: List[str], batch_size=32) -> List[np.ndarray]:
        """Generate embeddings for a list of texts in batches"""
        if not self.model_loaded:
            return [np.zeros(384) for _ in texts]
        
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
        if not self.model_loaded or df.empty:
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
    if generator.model_loaded:
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
