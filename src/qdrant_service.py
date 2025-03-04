import os
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import streamlit as st
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import json

class QdrantService:
    """Service for interacting with Qdrant vector database"""
    
    def __init__(self, collection_name="amazon_reviews"):
        """Initialize Qdrant client with API key from environment or Streamlit secrets"""
        self.collection_name = collection_name
        
        # Try to get credentials from Streamlit secrets
        try:
            self.api_key = st.secrets["qdrant"]["api_key"]
            self.url = st.secrets["qdrant"]["url"]
            self.client = QdrantClient(url=self.url, api_key=self.api_key)
            self.is_connected = True
        except Exception as e:
            print(f"Error connecting to Qdrant: {str(e)}")
            self.is_connected = False
            self.client = None
    
    def is_collection_exists(self) -> bool:
        """Check if collection exists"""
        if not self.is_connected:
            return False
            
        try:
            collections = self.client.get_collections()
            return self.collection_name in [c.name for c in collections.collections]
        except Exception as e:
            print(f"Error checking collection: {str(e)}")
            return False
    
    def create_collection(self, vector_size=768):
        """Create a new collection for Amazon reviews"""
        if not self.is_connected:
            return False
            
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            
            # Create payload indexes for efficient filtering
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="category",
                field_schema="keyword"
            )
            
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="product_id",
                field_schema="keyword"
            )
            
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="sentiment",
                field_schema="keyword"
            )
            
            return True
        except Exception as e:
            print(f"Error creating collection: {str(e)}")
            return False
    
    def store_embeddings(self, embeddings: List[np.ndarray], payloads: List[Dict], batch_size=100):
        """Store embeddings and payloads in batches"""
        if not self.is_connected:
            return False
            
        if not self.is_collection_exists():
            self.create_collection(vector_size=len(embeddings[0]))
        
        try:
            # Process in batches to avoid timeout issues
            for i in range(0, len(embeddings), batch_size):
                batch_embeddings = embeddings[i:i + batch_size]
                batch_payloads = payloads[i:i + batch_size]
                
                points = [
                    PointStruct(
                        id=i + idx, 
                        vector=embedding.tolist(), 
                        payload=payload
                    )
                    for idx, (embedding, payload) in enumerate(zip(batch_embeddings, batch_payloads))
                ]
                
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
            
            return True
        except Exception as e:
            print(f"Error storing embeddings: {str(e)}")
            return False
    
    def search_similar(self, query_embedding: np.ndarray, category: str = None, 
                       product_id: str = None, sentiment: str = None, limit: int = 10):
        """Search for similar reviews with optional filters"""
        if not self.is_connected:
            return []
            
        try:
            # Build filter conditions
            filter_conditions = []
            if category:
                filter_conditions.append(
                    models.FieldCondition(
                        key="category",
                        match=models.MatchValue(value=category)
                    )
                )
            
            if product_id:
                filter_conditions.append(
                    models.FieldCondition(
                        key="product_id",
                        match=models.MatchValue(value=product_id)
                    )
                )
                
            if sentiment:
                filter_conditions.append(
                    models.FieldCondition(
                        key="sentiment",
                        match=models.MatchValue(value=sentiment)
                    )
                )
            
            # Create filter if conditions exist
            search_filter = None
            if filter_conditions:
                search_filter = models.Filter(
                    must=filter_conditions
                )
            
            # Perform search
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=limit,
                filter=search_filter
            )
            
            return [{"score": hit.score, "payload": hit.payload} for hit in results]
        except Exception as e:
            print(f"Error searching: {str(e)}")
            return []
    
    def get_product_list(self, category: str) -> List[Dict[str, str]]:
        """Get list of products for a specific category"""
        if not self.is_connected:
            return []
            
        try:
            # Get unique product_ids with their titles for the given category
            response = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="category",
                            match=models.MatchValue(value=category)
                        )
                    ]
                ),
                limit=10000,  # Adjust as needed
                with_payload=["product_id", "product_title"],
            )
            
            # Extract unique products
            products = {}
            for point in response[0]:
                product_id = point.payload.get("product_id")
                product_title = point.payload.get("product_title", "Unknown Product")
                if product_id and product_id not in products:
                    products[product_id] = product_title
            
            # Return as list of dicts
            return [{"id": pid, "title": title} for pid, title in products.items()]
        except Exception as e:
            print(f"Error getting product list: {str(e)}")
            return []

# Example usage
if __name__ == "__main__":
    # Test the service
    service = QdrantService()
    if service.is_connected:
        print("Connected to Qdrant")
        
        # Example: Create a test embedding
        test_embedding = np.random.rand(768)  # Example embedding size
        
        # Example payload
        test_payload = {
            "review_id": "test123",
            "category": "Electronics",
            "product_id": "B01MQVHWCK",
            "product_title": "Test Product",
            "text": "This is a test review",
            "sentiment": "positive",
            "rating": 5
        }
        
        # Store the embedding
        service.store_embeddings([test_embedding], [test_payload])
        
        # Search for similar reviews
        results = service.search_similar(test_embedding, category="Electronics")
        print(f"Found {len(results)} results")
        for result in results:
            print(f"Score: {result['score']}, Product: {result['payload']['product_title']}")
