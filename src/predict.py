import os
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import json

class SentimentPredictor:
    def __init__(self, model_dir="pretrained"):
        self.model_dir = model_dir
        self.models = {}
        self.tokenizers = {}
        self.label_encoders = {}
        self.metadata = {}
        self._load_available_models()
    
    def _load_available_models(self):
        """Load all available pretrained models."""
        if not os.path.exists(self.model_dir):
            print(f"Model directory {self.model_dir} does not exist.")
            return
        
        categories = ["Electronics", "Books", "Beauty_and_Personal_Care", "Home_and_Kitchen"]
        
        for category in categories:
            model_path = os.path.join(self.model_dir, category)
            encoder_path = os.path.join(self.model_dir, f"{category}_label_encoder.pkl")
            metadata_path = os.path.join(self.model_dir, f"{category}_metadata.json")
            
            if os.path.exists(model_path) and os.path.exists(encoder_path):
                try:
                    # Load MiniLM model and tokenizer
                    model = AutoModelForSequenceClassification.from_pretrained(model_path)
                    tokenizer = AutoTokenizer.from_pretrained(model_path)
                    
                    # Load label encoder
                    with open(encoder_path, 'rb') as f:
                        label_encoder = pickle.load(f)
                    
                    # Store in dictionaries
                    self.models[category] = model
                    self.tokenizers[category] = tokenizer
                    self.label_encoders[category] = label_encoder
                    
                    # Load metadata if available
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'r') as f:
                            self.metadata[category] = json.load(f)
                    
                    print(f"Loaded model for {category}")
                except Exception as e:
                    print(f"Error loading model for {category}: {str(e)}")
    
    def get_available_categories(self):
        """Get list of available product categories with trained models."""
        return list(self.models.keys())
    
    def get_available_model_types(self, category):
        """Return available model types (only minilm for now)."""
        if category in self.models:
            return ["minilm"]
        return []
    
    def predict(self, text, category, model_type="minilm"):
        """Predict sentiment for text using the specified model."""
        if category not in self.models:
            raise ValueError(f"No model available for category: {category}")
        
        # Get model, tokenizer and label encoder
        model = self.models[category]
        tokenizer = self.tokenizers[category]
        label_encoder = self.label_encoders[category]
        
        # Prepare the text
        encoding = tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Get prediction
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        with torch.no_grad():
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
        
        # Convert prediction to sentiment label
        sentiment = label_encoder.inverse_transform([prediction])[0]
        
        # Get probability for each class
        probabilities = {}
        for i, prob in enumerate(probs[0].cpu().numpy()):
            label = label_encoder.inverse_transform([i])[0]
            probabilities[label] = float(prob)
        
        # Prepare result
        result = {
            'sentiment': sentiment,
            'probabilities': probabilities,
            'model_info': self.metadata.get(category, {})
        }
        
        return result

if __name__ == "__main__":
    # Example usage
    predictor = SentimentPredictor()
    
    # Print available categories
    print("Available categories:", predictor.get_available_categories())
    
    # Example prediction
    if predictor.get_available_categories():
        category = predictor.get_available_categories()[0]
        result = predictor.predict("This product is amazing! I love it.", category)
        print(f"Prediction for {category}:", result)
