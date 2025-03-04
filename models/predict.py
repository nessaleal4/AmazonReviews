import os
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

def preprocess_text(text):
    """Perform text preprocessing for sentiment analysis."""
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

class SentimentPredictor:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.models = {}
        self.metadata = {}
        self._load_available_models()
    
    def _load_available_models(self):
        """Load all available models from the model directory."""
        if not os.path.exists(self.model_dir):
            print(f"Model directory {self.model_dir} does not exist.")
            return
        
        for filename in os.listdir(self.model_dir):
            if filename.endswith("_model.pkl"):
                model_path = os.path.join(self.model_dir, filename)
                # Extract category and model type from filename
                parts = os.path.splitext(filename)[0].split("_")
                category = parts[0]
                model_type = parts[1]
                
                # Load the model
                try:
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    
                    # Create a key for the model
                    model_key = f"{category}_{model_type}"
                    self.models[model_key] = model
                    
                    # Try to load corresponding metadata
                    metadata_path = os.path.join(self.model_dir, f"{category}_{model_type}_metadata.pkl")
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'rb') as f:
                            self.metadata[model_key] = pickle.load(f)
                    
                    print(f"Loaded model: {model_key}")
                except Exception as e:
                    print(f"Error loading model {filename}: {str(e)}")
    
    def get_available_categories(self):
        """Get list of available product categories."""
        categories = set()
        for model_key in self.models.keys():
            category = model_key.split("_")[0]
            categories.add(category)
        return sorted(list(categories))
    
    def get_available_model_types(self, category):
        """Get list of available model types for a category."""
        model_types = []
        for model_key in self.models.keys():
            parts = model_key.split("_")
            if parts[0] == category:
                model_types.append(parts[1])
        return sorted(model_types)
    
    def predict(self, text, category, model_type="logistic"):
        """Predict sentiment for the given text using the specified model."""
        model_key = f"{category}_{model_type}"
        
        if model_key not in self.models:
            raise ValueError(f"Model for {model_key} not found.")
        
        # Preprocess the text
        processed_text = preprocess_text(text)
        
        # Make prediction
        model = self.models[model_key]
        sentiment = model.predict([processed_text])[0]
        
        # Get prediction probabilities if the model supports it
        probabilities = {}
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba([processed_text])[0]
            for i, label in enumerate(model.classes_):
                probabilities[label] = proba[i]
        
        result = {
            'sentiment': sentiment,
            'probabilities': probabilities,
            'model_info': self.metadata.get(model_key, {})
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
