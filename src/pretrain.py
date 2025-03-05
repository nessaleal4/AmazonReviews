import os
import argparse
import pandas as pd
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import json
import gzip
from io import BytesIO
import requests

# Define Dropbox URLs
DROPBOX_LINKS = {
    "Books": "https://www.dropbox.com/s/312wv7jtm1tpxeo/Books.jsonl.gz?dl=1",
    "Beauty_and_Personal_Care": "https://www.dropbox.com/s/w2bg91ewpziaaa3/Beauty_and_Personal_Care.jsonl.gz?dl=1",
    "Electronics": "https://www.dropbox.com/s/st07mgrwzazitru/Electronics.jsonl.gz?dl=1",
    "Home_and_Kitchen": "https://www.dropbox.com/s/oxn45ntlkxo8ju5/Home_and_Kitchen.jsonl.gz?dl=1"
}

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data_from_dropbox(category, sample_size=10000):
    """Load a sample of reviews directly from Dropbox."""
    url = DROPBOX_LINKS.get(category)
    if not url:
        print(f"No URL found for category: {category}")
        return None
    
    try:
        print(f"Downloading data for {category}...")
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            print(f"Failed to download data: HTTP {response.status_code}")
            return None
        
        print(f"Processing data...")
        reviews = []
        with gzip.GzipFile(fileobj=BytesIO(response.content)) as f:
            for i, line in enumerate(f):
                if i >= sample_size:
                    break
                try:
                    review = json.loads(line)
                    
                    # Extract text and rating
                    text = review.get('reviewText', '')
                    if not text:
                        text = review.get('summary', 'No text available')
                    
                    rating = review.get('overall', 0)
                    
                    # Add sentiment based on rating
                    if rating >= 4:
                        sentiment = 'positive'
                    elif rating <= 2:
                        sentiment = 'negative'
                    else:
                        sentiment = 'neutral'
                    
                    reviews.append({
                        'text': text,
                        'sentiment': sentiment,
                        'rating': rating
                    })
                except json.JSONDecodeError:
                    continue
        
        # Convert to DataFrame
        df = pd.DataFrame(reviews)
        print(f"Loaded {len(df)} reviews from {category}")
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def train_minilm(category, output_dir="pretrained", batch_size=16, epochs=3, sample_size=10000):
    # Load data
    df = load_data_from_dropbox(category, sample_size)
    if df is None or df.empty:
        print(f"Failed to load data for {category}")
        return
    
    # Encode sentiment labels
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['sentiment'])
    
    # Save label encoder
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{category}_label_encoder.pkl"), 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Initialize tokenizer and model
    model_name = 'microsoft/MiniLM-L6-H384-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=len(label_encoder.classes_)
    )
    
    # Prepare dataset
    dataset = ReviewDataset(df['text'].values, df['label'].values, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Average loss: {avg_loss:.4f}")
    
    # Save the model
    model_path = os.path.join(output_dir, category)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    print(f"Model and tokenizer saved to {model_path}")
    
    # Create and save metadata
    metadata = {
        'category': category,
        'num_samples': len(df),
        'label_mapping': {i: label for i, label in enumerate(label_encoder.classes_)},
        'model_type': 'minilm',
        'base_model': model_name,
        'class_distribution': df['sentiment'].value_counts().to_dict()
    }
    
    with open(os.path.join(output_dir, f"{category}_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to {os.path.join(output_dir, f'{category}_metadata.json')}")
    
    return model, tokenizer, label_encoder

def main():
    parser = argparse.ArgumentParser(description="Pretrain MiniLM models on Amazon reviews")
    parser.add_argument("--categories", nargs="+", default=["Electronics", "Books", "Beauty_and_Personal_Care", "Home_and_Kitchen"])
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--sample_size", type=int, default=10000, help="Number of reviews to use (10% of dataset)")
    args = parser.parse_args()
    
    for category in args.categories:
        print(f"Training model for {category}...")
        train_minilm(
            category, 
            batch_size=args.batch_size, 
            epochs=args.epochs,
            sample_size=args.sample_size
        )

if __name__ == "__main__":
    main()
