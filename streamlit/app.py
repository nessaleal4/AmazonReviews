import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import json
import gzip
import requests
from io import BytesIO

# Add the project root directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import your project modules
try:
    from src.predict import SentimentPredictor
    from src.preprocess import download_from_dropbox, process_reviews
    from src.split_data import split_dataset
    from src.qdrant_service import QdrantService
    from src.embedding_util import EmbeddingGenerator
except ImportError as e:
    st.error(f"Error importing project modules: {str(e)}")
    # Alternative import approach for Streamlit Cloud
    try:
        sys.path.append('/mount/src/amazonreviews')
        from src.predict import SentimentPredictor
        from src.preprocess import download_from_dropbox, process_reviews
        from src.split_data import split_dataset
        from src.qdrant_service import QdrantService
        from src.embedding_util import EmbeddingGenerator
        st.success("Modules imported using alternative path")
    except ImportError as e:
        st.error(f"Still unable to import modules: {str(e)}")

# Set page configuration
st.set_page_config(
    page_title="Amazon Reviews Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    st.warning(f"Unable to download NLTK resources: {str(e)}")

# Initialize session state variables if they don't exist
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'selected_category' not in st.session_state:
    st.session_state.selected_category = None
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "logistic"
if 'sample_df' not in st.session_state:
    st.session_state.sample_df = None
if 'product_list' not in st.session_state:
    st.session_state.product_list = []
if 'selected_product' not in st.session_state:
    st.session_state.selected_product = None
if 'qdrant_service' not in st.session_state:
    st.session_state.qdrant_service = QdrantService()
if 'embedding_generator' not in st.session_state:
    st.session_state.embedding_generator = EmbeddingGenerator()

# Title
st.title("📊 Amazon Reviews Sentiment Analysis")

# Sidebar
st.sidebar.header("Options")

# Category selection
categories = ["Electronics", "Books", "Beauty_and_Personal_Care", "Home_and_Kitchen"]
selected_category = st.sidebar.selectbox("Select Product Category", categories)

# Initialize Qdrant connection status
qdrant_status = "Connected" if st.session_state.qdrant_service.is_connected else "Not Connected"
st.sidebar.text(f"Vector Database: {qdrant_status}")

# Product selection (if Qdrant is connected)
if st.session_state.qdrant_service.is_connected:
    # Only fetch products if category changed
    if selected_category != st.session_state.selected_category:
        with st.sidebar.spinner("Loading products..."):
            products = st.session_state.qdrant_service.get_product_list(selected_category)
            st.session_state.product_list = products
            st.session_state.selected_category = selected_category
    
    # Create product dropdown if products are available
    if st.session_state.product_list:
        product_options = ["All Products"] + [p["title"][:50] + "..." if len(p["title"]) > 50 else p["title"] 
                                             for p in st.session_state.product_list]
        product_index = st.sidebar.selectbox("Select Product", range(len(product_options)), 
                                            format_func=lambda x: product_options[x])
        
        if product_index > 0:
            st.session_state.selected_product = st.session_state.product_list[product_index-1]["id"]
        else:
            st.session_state.selected_product = None
    else:
        st.sidebar.text("No products available. Import data first.")
        st.session_state.selected_product = None

# Initialize predictor if needed
@st.cache_resource
def load_predictor():
    return SentimentPredictor()

try:
    predictor = load_predictor()
    available_models = predictor.get_available_model_types(selected_category)
    
    if available_models:
        selected_model = st.sidebar.selectbox("Select Model Type", available_models)
    else:
        selected_model = st.sidebar.selectbox("Select Model Type", ["logistic", "random_forest", "svm"])
        st.sidebar.warning(f"No trained models found for {selected_category}. You'll need to train a model first.")
except Exception as e:
    st.sidebar.error(f"Error loading models: {str(e)}")
    selected_model = st.sidebar.selectbox("Select Model Type", ["logistic", "random_forest", "svm"])

# Update session state
st.session_state.selected_category = selected_category
st.session_state.selected_model = selected_model
st.session_state.predictor = predictor if 'predictor' in locals() else None

# Add tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Sentiment Analysis", "Data Explorer", "Model Training", "Vector Search", "About"])

# Tab 1: Sentiment Analysis
with tab1:
    st.header("Review Sentiment Analyzer")
    
    # Input for sentiment analysis
    user_input = st.text_area("Enter a product review to analyze", height=150)
    
    if st.button("Analyze Sentiment"):
        if user_input and st.session_state.predictor:
            try:
                result = st.session_state.predictor.predict(
                    user_input, 
                    st.session_state.selected_category,
                    st.session_state.selected_model
                )
                
                # Display result with appropriate styling
                sentiment = result['sentiment']
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if sentiment == "positive":
                        st.success("Positive Sentiment")
                    elif sentiment == "negative":
                        st.error("Negative Sentiment")
                    else:
                        st.info("Neutral Sentiment")
                
                with col2:
                    if 'probabilities' in result and result['probabilities']:
                        probs = result['probabilities']
                        fig, ax = plt.subplots()
                        ax.bar(probs.keys(), probs.values())
                        ax.set_ylabel('Probability')
                        ax.set_title('Sentiment Probabilities')
                        st.pyplot(fig)
                
                # Show model information
                if 'model_info' in result and result['model_info']:
                    with st.expander("Model Information"):
                        st.json(result['model_info'])
                
                # If Qdrant is connected, show similar reviews
                if st.session_state.qdrant_service.is_connected and st.session_state.embedding_generator.model_loaded:
                    with st.spinner("Finding similar reviews..."):
                        # Generate embedding for user input
                        query_embedding = st.session_state.embedding_generator.generate_embedding(user_input)
                        
                        # Search for similar reviews
                        similar_reviews = st.session_state.qdrant_service.search_similar(
                            query_embedding, 
                            category=selected_category,
                            product_id=st.session_state.selected_product,
                            sentiment=sentiment,
                            limit=5
                        )
                        
                        if similar_reviews:
                            st.subheader("Similar Reviews")
                            for i, review in enumerate(similar_reviews):
                                payload = review["payload"]
                                with st.expander(f"Review {i+1}: {payload.get('product_title', 'Product')}"):
                                    st.write(f"**Text:** {payload.get('text', 'No text')}")
                                    st.write(f"**Rating:** {payload.get('rating', 'N/A')}")
                                    st.write(f"**Similarity Score:** {review['score']:.4f}")
                        else:
                            st.info("No similar reviews found. Try importing more data to Qdrant.")
                        
            except Exception as e:
                st.error(f"Error analyzing sentiment: {str(e)}")
        elif not user_input:
            st.warning("Please enter a review to analyze.")
        else:
            st.warning(f"No trained model available for {st.session_state.selected_category}. Please go to the Model Training tab to train a model first.")

# Tab 2: Data Explorer
with tab2:
    st.header("Explore Amazon Review Data")
    
    # Function to sample data from Dropbox
    @st.cache_data(ttl=3600, show_spinner=True)
    def load_sample_data(category, sample_size=1000, product_id=None):
        """Load a sample of reviews directly from Dropbox."""
        # Define Dropbox URLs
        dropbox_url = {
            "Books": "https://www.dropbox.com/s/312wv7jtm1tpxeo/Books.jsonl.gz?dl=1",
            "Beauty_and_Personal_Care": "https://www.dropbox.com/s/w2bg91ewpziaaa3/Beauty_and_Personal_Care.jsonl.gz?dl=1",
            "Electronics": "https://www.dropbox.com/s/st07mgrwzazitru/Electronics.jsonl.gz?dl=1",
            "Home_and_Kitchen": "https://www.dropbox.com/s/oxn45ntlkxo8ju5/Home_and_Kitchen.jsonl.gz?dl=1"
        }
        
        url = dropbox_url.get(category)
        if not url:
            st.error(f"No URL found for category: {category}")
            return None
        
        try:
            # Stream the data to avoid downloading the entire file
            response = requests.get(url, stream=True)
            if response.status_code != 200:
                st.error(f"Failed to download data: HTTP {response.status_code}")
                return None
            
            # Process the gzipped JSONL file
            reviews = []
            with gzip.GzipFile(fileobj=BytesIO(response.content)) as f:
                for i, line in enumerate(f):
                    if i >= sample_size and not product_id:
                        break
                    try:
                        review = json.loads(line)
                        
                        # Filter by product_id if specified
                        if product_id and review.get('asin') != product_id:
                            continue
                            
                        # Add sentiment based on rating
                        rating = review.get('overall', 0)
                        if rating >= 4:
                            sentiment = 'positive'
                        elif rating <= 2:
                            sentiment = 'negative'
                        else:
                            sentiment = 'neutral'
                        review['sentiment'] = sentiment
                        reviews.append(review)
                        
                        # Break if we found enough reviews for the specific product
                        if product_id and len(reviews) >= sample_size:
                            break
                    except json.JSONDecodeError:
                        continue
            
            # Convert to DataFrame
            df = pd.DataFrame(reviews)
            return df
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None
    
    # Load button with progress indicator
    col1, col2 = st.columns([1, 3])
    with col1:
        sample_size = st.number_input("Sample Size", min_value=100, max_value=10000, value=1000, step=100)
    with col2:
        load_button = st.button("Load Sample Data")
    
    if load_button:
        with st.spinner(f"Loading sample data for {selected_category}..."):
            df = load_sample_data(selected_category, sample_size, st.session_state.selected_product)
            if df is not None:
                st.session_state.sample_df = df
                
                # Store in Qdrant if connected
                if st.session_state.qdrant_service.is_connected and st.session_state.embedding_generator.model_loaded:
                    with st.spinner("Generating embeddings and storing in Qdrant..."):
                        processed_data = st.session_state.embedding_generator.process_reviews_dataframe(
                            df, text_column='reviewText', category=selected_category
                        )
                        
                        # Store embeddings in Qdrant
                        success = st.session_state.qdrant_service.store_embeddings(
                            processed_data["embeddings"],
                            processed_data["payloads"]
                        )
                        
                        if success:
                            st.success("Data stored in vector database for faster searching")
                
                st.success(f"Loaded {len(df)} reviews from {selected_category}")
    
    # Display sample data
    if st.session_state.sample_df is not None:
        df = st.session_state.sample_df
        
        # Show basic statistics
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Reviews", len(df))
        with col2:
            avg_rating = df['overall'].mean() if 'overall' in df.columns else 0
            st.metric("Average Rating", f"{avg_rating:.2f}")
        with col3:
            sentiment_counts = df['sentiment'].value_counts()
            pos_pct = sentiment_counts.get('positive', 0) / len(df) * 100
            st.metric("Positive Reviews", f"{pos_pct:.1f}%")
        
        # Show data preview
        with st.expander("Data Preview"):
            cols_to_show = ['reviewerID', 'asin', 'reviewText', 'overall', 'summary', 'sentiment']
            cols_available = [col for col in cols_to_show if col in df.columns]
            st.dataframe(df[cols_available].head(10))
        
        # Rating distribution
        if 'overall' in df.columns:
            st.subheader("Rating Distribution")
            fig = px.histogram(df, x='overall', nbins=5, 
                              labels={'overall': 'Rating', 'count': 'Number of Reviews'},
                              title="Distribution of Ratings")
            st.plotly_chart(fig)
        
        # Sentiment distribution
        st.subheader("Sentiment Distribution")
        sentiment_counts = df['sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        fig = px.pie(sentiment_counts, values='Count', names='Sentiment', 
                    title="Sentiment Distribution")
        st.plotly_chart(fig)
        
        # Word cloud
        if 'reviewText' in df.columns:
            st.subheader("Word Cloud of Reviews")
            
            # Filter by sentiment
            sentiment_filter = st.radio("Filter by sentiment", ['All', 'Positive', 'Negative', 'Neutral'], horizontal=True)
            
            if sentiment_filter != 'All':
                filtered_df = df[df['sentiment'] == sentiment_filter.lower()]
            else:
                filtered_df = df
            
            if not filtered_df.empty:
                # Combine all review text
                all_text = ' '.join(filtered_df['reviewText'].dropna().astype(str))
                
                # Generate word cloud
                try:
                    # Download stopwords if needed
                    nltk.download('stopwords', quiet=True)
                    stopwords_set = set(stopwords.words('english'))
                    
                    wordcloud = WordCloud(width=800, height=400, 
                                          background_color='white',
                                          stopwords=stopwords_set,
                                          max_words=100).generate(all_text)
                    
                    # Display the word cloud
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    st.pyplot(plt)
                except Exception as e:
                    st.error(f"Error generating word cloud: {str(e)}")
            else:
                st.warning(f"No reviews found with {sentiment_filter} sentiment.")

# Tab 3: Model Training
with tab3:
    st.header("Train Sentiment Analysis Models")
    
    st.write("""
    This tab allows you to train sentiment analysis models using the data from Dropbox.
    The process includes:
    1. Downloading data from Dropbox
    2. Processing the reviews
    3. Splitting the data into train/validation/test sets
    4. Training the selected model
    """)
    
    # Training parameters
    train_category = st.selectbox("Select Category for Training", categories)
    train_model_type = st.selectbox("Select Model Type for Training", ["logistic", "random_forest", "svm"])
    
    max_reviews = st.slider("Maximum Reviews to Process", 
                           min_value=1000, max_value=100000, value=10000, step=1000)
    
    # Training button
    if st.button("Start Training"):
        st.info("Training process started. This might take a while...")
        
        # Create progress bars
        prog1 = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Download data
            status_text.text("Step 1/4: Downloading data from Dropbox...")
            prog1.progress(10)
            
            download_from_dropbox(train_category, "data/raw")
            
            # Step 2: Process reviews
            status_text.text(f"Step 2/4: Processing {max_reviews} reviews...")
            prog1.progress(30)
            
            df = process_reviews(train_category, max_reviews=max_reviews)
            
            # Step 3: Split dataset
            status_text.text("Step 3/4: Splitting dataset into train/val/test...")
            prog1.progress(60)
            
            split_dataset(train_category)
            
            # Step 4: Train model (import here to avoid circular imports)
            status_text.text(f"Step 4/4: Training {train_model_type} model...")
            prog1.progress(80)
            
            from src.train_model import train_model
            model, metadata = train_model(train_category, model_type=train_model_type)
            
            # Completed
            prog1.progress(100)
            status_text.text("Training completed!")
            
            # Show results
            st.success(f"Model trained successfully with accuracy: {metadata['accuracy']:.4f}")
            
            # Display metadata
            with st.expander("Model Details"):
                st.json(metadata)
            
            # Refresh the predictor to load the new model
            st.session_state.predictor = load_predictor()
            
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            prog1.progress(0)
            status_text.text("Training process failed.")

# Tab 4: Vector Search
with tab4:
    st.header("Semantic Search with Qdrant")
    
    if st.session_state.qdrant_service.is_connected and st.session_state.embedding_generator.model_loaded:
        st.write("""
        This tab allows you to search for semantically similar reviews using vector embeddings.
        The search is powered by Qdrant vector database and uses semantic similarity rather than
        keyword matching.
        """)
        
        # Search input
        search_query = st.text_area("Enter search query", height=100)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            search_limit = st.number_input("Number of results", min_value=1, max_value=50, value=10)
        
        with col2:
            search_sentiment = st.selectbox("Filter by sentiment", ["All", "Positive", "Negative", "Neutral"])
        
        with col3:
            search_button = st.button("Search Reviews")
        
        if search_button and search_query:
            with st.spinner("Searching for similar reviews..."):
                # Generate embedding for search query
                query_embedding = st.session_state.embedding_generator.generate_embedding(search_query)
                
                # Prepare sentiment filter
                sentiment_filter = None if search_sentiment == "All" else search_sentiment.lower()
                
                # Search for similar reviews
                similar_reviews = st.session_state.qdrant_service.search_similar(
                    query_embedding, 
                    category=selected_category,
                    product_id=st.session_state.selected_product,
                    sentiment=sentiment_filter,
                    limit=search_limit
                )
                
                if similar_reviews:
                    st.subheader(f"Found {len(similar_reviews)} similar reviews")
                    
                    # Create a dataframe for easier display
                    results_data = []
                    for review in similar_reviews:
                        payload = review["payload"]
                        results_data.append({
                            "Product": payload.get("product_title", "Unknown"),
                            "Review": payload.get("text", "No text"),
                            "Rating": payload.get("rating", "N/A"),
                            "Sentiment": payload.get("sentiment", "Unknown"),
                            "Similarity": review["score"]
                        })
                    
                    results_df = pd.DataFrame(results_data)
                    
                    # Display as table
                    st.dataframe(results_df)
                    
                    # Show detailed view
                    st.subheader("Detailed Reviews")
                    for i, review in enumerate(similar_reviews):
                        payload = review["payload"]
                        with st.expander(f"Review {i+1}: {payload.get('product_title', 'Product')[:50]}..."):
                            st.write(f"**Text:** {payload.get('text', 'No text')}")
                            st.write(f"**Rating:** {payload.get('rating', 'N/A')}")
                            st.write(f"**Product ID:** {payload.get('product_id', 'N/A')}")
                            st.write(f"**Similarity Score:** {review['score']:.4f}")
                else:
                    st.info("No similar reviews found. Try importing more data to Qdrant or using different search terms.")
        
        # Data import section
        st.subheader("Import Data to Vector Database")
        st.write("""
        To make the semantic search more effective, you can import more review data into the vector database.
        This will create embeddings for the reviews and store them for faster searching.
        """)
        
        col1, col2 = st.columns([1, 3])
        with col1:
            import_size = st.number_input("Import Size", min_value=100, max_value=10000, value=1000, step=100)
        with col2:
            import_button = st.button("Import Reviews to Qdrant")
        
        if import_button:
            with st.spinner(f"Importing {import_size} reviews to vector database..."):
                # Load data
                df = load_sample_data(selected_category, import_size, st.session_state.selected_product)
                
                if df is not None and not df.empty:
                    # Generate embeddings and store in Qdrant
                    processed_data = st.session_state.embedding_generator.process_reviews_dataframe(
                        df, text_column='reviewText', category=selected_category
                    )
                    
                    # Store embeddings in Qdrant
                    success = st.session_state.qdrant_service.store_embeddings(
                        processed_data["embeddings"],
                        processed_data["payloads"]
                    )
                    
                    if success:
                        st.success(f"Successfully imported {len(df)} reviews to vector database")
                    else:
                        st.error("Failed to import reviews to vector database")
                else:
                    st.error("Failed to load review data")
    else:
        st.warning("""
        Vector search requires connection to Qdrant and a working embedding model.
        
        Please check:
        1. Qdrant connection in your Streamlit secrets
        2. Embedding model installation
        
        Then restart the app to enable vector search functionality.
        """)

# Tab 5: About
with tab5:
    st.header("About This Project")
    
    st.write("""
    ## Amazon Reviews Sentiment Analysis
    
    This application analyzes sentiment in Amazon product reviews across multiple categories including Electronics, Books, Beauty & Personal Care products, and Home & Kitchen products.
    
    ### Features:
    
    - **Sentiment Analysis**: Analyze the sentiment of any product review using trained machine learning models
    - **Data Explorer**: Explore sample data from Amazon reviews, visualize ratings and sentiment distributions
    - **Model Training**: Train custom sentiment analysis models on Amazon review data
    - **Vector Search**: Semantically search for similar reviews using vector embeddings (powered by Qdrant)
    - **Product Filtering**: Filter reviews by specific products within each category
    
    ### How It Works:
    
    1. The application reads Amazon review data directly from Dropbox
    2. Data is processed and cleaned for sentiment analysis
    3. Machine learning models are trained to predict sentiment (positive, negative, or neutral)
    4. Review text is converted to vector embeddings for semantic search
    5. The Streamlit interface provides an easy way to interact with the models and data
    
    ### Technical Details:
    
    - **Data Source**: Amazon Product Reviews Dataset
    - **Models**: Logistic Regression, Random Forest, and SVM classifiers
    - **Features**: TF-IDF vectorization of review text
    - **Preprocessing**: Text cleaning, stopword removal, and lemmatization
    - **Vector Database**: Qdrant for efficient semantic search
    - **Embeddings**: Sentence transformers for generating text embeddings
    
    ### GitHub Repository:
    
    [https://github.com/nessaleal4/AmazonReviews](https://github.com/nessaleal4/AmazonReviews)
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Developed with Streamlit and scikit-learn")
