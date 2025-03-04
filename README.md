# Amazon Reviews Sentiment Analysis

A Streamlit application for analyzing sentiment in Amazon product reviews across multiple categories with vector search capabilities.

## Project Structure

```
AmazonReviews/
├── data/                  # Data directory
│   ├── raw/               # Raw data from Dropbox
│   ├── processed/         # Processed data
│   └── sampled/           # Train/val/test splits
├── models/                # Trained models
├── src/                   # Source code
│   ├── __init__.py        # Package initialization
│   ├── preprocess.py      # Data preprocessing
│   ├── split_data.py      # Data splitting
│   ├── train_model.py     # Model training
│   ├── predict.py         # Prediction functionality
│   ├── qdrant_service.py  # Vector database integration
│   └── embedding_util.py  # Text embedding generation
├── streamlit/             # Streamlit app
│   ├── __init__.py        # Package initialization
│   ├── app.py             # Main app
│   └── .streamlit/        # Streamlit configuration
│       └── config.toml    # Streamlit configuration
├── .gitignore             # Git ignore file
├── .streamlit/            # Streamlit secrets (not committed)
│   └── secrets.toml       # Qdrant API credentials
├── requirements.txt       # Full development requirements
└── requirements-streamlit.txt  # Streamlit Cloud requirements
```

## Features

- **Sentiment Analysis**: Analyze the sentiment of product reviews using machine learning models
- **Data Explorer**: Visualize ratings and sentiment distributions
- **Model Training**: Train custom sentiment analysis models on Amazon review data
- **Vector Search**: Semantically search for similar reviews using vector embeddings
- **Product Filtering**: Browse specific products and their reviews within each category

## Data Sources

The application uses Amazon product review data stored in Dropbox. Each category is stored as a compressed JSONL file:

- Books.jsonl.gz
- Electronics.jsonl.gz
- Beauty_and_Personal_Care.jsonl.gz
- Home_and_Kitchen.jsonl.gz

Additionally, metadata files are available for each category:

- meta_Books.jsonl.gz
- meta_Electronics.jsonl.gz
- meta_Beauty_and_Personal_Care.jsonl.gz
- meta_Home_and_Kitchen.jsonl.gz

The app uses these Dropbox URLs to read data directly without requiring local storage.

## Vector Database Integration

The application integrates with Qdrant vector database for fast semantic searching:

1. Review text is converted to vector embeddings using pretrained models
2. Embeddings are stored in Qdrant for efficient similarity search
3. The vector search tab enables finding semantically similar reviews
4. Product-specific search filters help focus on relevant products

To use the vector search functionality:
1. Set up a Qdrant account at [cloud.qdrant.io](https://cloud.qdrant.io/)
2. Create a cluster and get your API key
3. Add your Qdrant credentials to `.streamlit/secrets.toml`:
   ```toml
   [qdrant]
   url = "your-qdrant-cluster-url"
   api_key = "your-qdrant-api-key"
   ```
4. For Streamlit Cloud deployment, add these same credentials in the app settings

## Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/nessaleal4/AmazonReviews.git
   cd AmazonReviews
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up Qdrant credentials in `.streamlit/secrets.toml`

5. Run the Streamlit app:
   ```bash
   streamlit run streamlit/app.py
   ```

## Streamlit Cloud Deployment

1. Fork or push this repository to your GitHub account

2. Log in to [Streamlit Cloud](https://streamlit.io/cloud)

3. Create a new app with the following settings:
   - Repository: `https://github.com/your-username/AmazonReviews`
   - Branch: `main`
   - Main file path: `streamlit/app.py`

4. Advanced settings:
   - Python version: 3.9
   - Packages: Use the requirements-streamlit.txt file

5. Add your Qdrant credentials in the Streamlit Cloud app settings under "Secrets"

6. Deploy!

## Working with the Data

This project handles large Amazon review datasets (22GB+) by:

1. Storing data in Dropbox instead of GitHub
2. Reading data directly from Dropbox URLs
3. Processing only samples of data for demonstration purposes
4. Providing functionality to train models on full datasets when needed
5. Using vector embeddings for efficient semantic search

## Models and Vectorization Techniques

### Text Vectorization

The application uses **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorization to convert review text into numerical features that machine learning models can process:

- **Term Frequency (TF)**: Measures how frequently a term appears in a document (review)
- **Inverse Document Frequency (IDF)**: Downweights terms that appear in many documents, giving more importance to distinctive terms
- **TF-IDF Score**: Combines these two metrics, highlighting important words unique to a specific review

The TF-IDF vectorization process:
1. Tokenizes each review into individual words
2. Computes term frequencies for each word in each review
3. Computes the inverse document frequency to identify discriminative words
4. Creates a sparse matrix where each row represents a review and each column represents a word in the vocabulary
5. Each cell contains the TF-IDF score for that word in that review

### Classification Models

The application supports three different classification models, each with different strengths:

1. **Logistic Regression** (default):
   - Fast training and prediction
   - Provides probability estimates for each sentiment class
   - Works well with high-dimensional sparse data like TF-IDF vectors
   - Interpretable coefficients that can identify important words for each sentiment

2. **Random Forest**:
   - Ensemble method that combines multiple decision trees
   - Less prone to overfitting than individual decision trees
   - Can capture non-linear relationships in the data
   - Provides feature importance metrics

3. **Support Vector Machine (SVM)**:
   - Particularly well-suited for text classification with high-dimensional data
   - Effective at finding the optimal boundary between sentiment classes
   - Robust to overfitting in high-dimensional spaces
   - Implementation uses LinearSVC for faster processing with large datasets

Each model is trained using a pipeline that combines TF-IDF vectorization and the classification algorithm, ensuring consistent preprocessing during both training and prediction.

### Vector Embeddings for Semantic Search

For the vector search functionality, the application uses a different approach:

- **Sentence Transformers**: Pre-trained transformer models convert review text into dense vector embeddings
- These embeddings capture semantic meaning rather than just word frequencies
- Similar reviews will have similar vector representations, even if they use different words
- The embeddings are stored in Qdrant vector database for efficient similarity search

The combination of TF-IDF for classification and transformer embeddings for semantic search provides both accurate sentiment analysis and powerful search capabilities.
