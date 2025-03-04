import os
import argparse
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    """Perform additional text preprocessing for sentiment analysis."""
    if not isinstance(text, str):
        return ""
    
    # Tokenize
    tokens = nltk.word_tokenize(text.lower())
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return " ".join(tokens)

def create_sentiment_labels(df):
    """Create sentiment labels based on review ratings."""
    df['sentiment'] = df['overall'].apply(lambda x: 
                                         'positive' if x >= 4 
                                         else ('negative' if x <= 2 
                                               else 'neutral'))
    return df

def train_model(category, model_type='logistic', input_dir="data/sampled", output_dir="models"):
    """Train a sentiment analysis model on the given category data."""
    # Load training data
    train_path = os.path.join(input_dir, f"{category}_train.parquet")
    val_path = os.path.join(input_dir, f"{category}_val.parquet")
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print(f"Train or validation data doesn't exist for {category}")
        return
    
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    
    # Create sentiment labels
    train_df = create_sentiment_labels(train_df)
    val_df = create_sentiment_labels(val_df)
    
    # Apply additional text preprocessing
    print("Preprocessing text...")
    train_df['processed_text'] = train_df['text'].apply(preprocess_text)
    val_df['processed_text'] = val_df['text'].apply(preprocess_text)
    
    # Define feature and target variables
    X_train = train_df['processed_text']
    y_train = train_df['sentiment']
    X_val = val_df['processed_text']
    y_val = val_df['sentiment']
    
    # Create and train the model
    print(f"Training {model_type} model for {category}...")
    
    if model_type == 'logistic':
        model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000)),
            ('classifier', LogisticRegression(random_state=42, max_iter=1000))
        ])
    elif model_type == 'random_forest':
        model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000)),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
    elif model_type == 'svm':
        model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000)),
            ('classifier', LinearSVC(random_state=42))
        ])
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred)
    
    print(f"Model accuracy: {accuracy:.4f}")
    print(f"Classification report:\n{report}")
    
    # Save the model
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{category}_{model_type}_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {model_path}")
    
    # Create and save metadata about the model
    metadata = {
        'accuracy': accuracy,
        'category': category,
        'model_type': model_type,
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'class_distribution': dict(train_df['sentiment'].value_counts())
    }
    
    metadata_path = os.path.join(output_dir, f"{category}_{model_type}_metadata.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"Model metadata saved to {metadata_path}")
    
    return model, metadata

def main():
    parser = argparse.ArgumentParser(description="Train sentiment analysis models")
    parser.add_argument("--categories", nargs="+", default=["Electronics", "Books", "Beauty_and_Personal_Care", "Home_and_Kitchen"])
    parser.add_argument("--model", type=str, default="logistic", choices=["logistic", "random_forest", "svm"])
    args = parser.parse_args()
    
    for category in args.categories:
        train_model(category, model_type=args.model)

if __name__ == "__main__":
    main()
