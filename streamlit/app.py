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
    from src.preprocess import download_from_dropbox
except ImportError as e:
    st.error(f"Error importing project modules: {str(e)}")
    # Alternative import approach for Streamlit Cloud
    try:
        sys.path.append('/mount/src/amazonreviews')
        from src.predict import SentimentPredictor
        from src.preprocess import download_from_dropbox
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
    st.session_state.predictor = SentimentPredictor(model_dir="pretrained")
if 'selected_category' not in st.session_state:
    st.session_state.selected_category = None
if 'sample_df' not in st.session_state:
    st.session_state.sample_df = None
if 'product_list' not in st.session_state:
    st.session_state.product_list = []
if 'selected_product' not in st.session_state:
    st.session_state.selected_product = None

# Title
st.title("📊 Amazon Reviews Sentiment Analysis")

# Sidebar
st.sidebar.header("Options")

# Get available categories from pre-trained models
available_categories = st.session_state.predictor.get_available_categories()
if not available_categories:
    available_categories = ["Electronics", "Books", "Beauty_and_Personal_Care", "Home_and_Kitchen"]
    st.sidebar.warning("No pre-trained models found. Some features may be limited.")

# Category selection
selected_category = st.sidebar.selectbox("Select Product Category", available_categories)

# Product selection
if st.session_state.product_list:
    product_options = ["All Products"] + [p[:50] + "..." if len(p) > 50 else p 
                                        for p in st.session_state.product_list]
    product_index = st.sidebar.selectbox("Select Product", range(len(product_options)), 
                                        format_func=lambda x: product_options[x])
    
    if product_index > 0:
        st.session_state.selected_product = st.session_state.product_list[product_index-1]
    else:
        st.session_state.selected_product = None
else:
    st.sidebar.text("No products available. Import data first.")
    st.session_state.selected_product = None

# Update session state
st.session_state.selected_category = selected_category

# Add tabs
tab1, tab2, tab3 = st.tabs(["Sentiment Analysis", "Data Explorer", "About"])

# Tab 1: Sentiment Analysis
with tab1:
    st.header("Review Sentiment Analyzer")
    
    # Input for sentiment analysis
    user_input = st.text_area("Enter a product review to analyze", height=150)
    
    if st.button("Analyze Sentiment"):
        if user_input and selected_category in st.session_state.predictor.get_available_categories():
            try:
                result = st.session_state.predictor.predict(
                    user_input, 
                    selected_category
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
                        
            except Exception as e:
                st.error(f"Error analyzing sentiment: {str(e)}")
        elif not user_input:
            st.warning("Please enter a review to analyze.")
        else:
            st.warning(f"No trained model available for {selected_category}. Pre-trained models are needed.")

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
            product_ids = set()
            
            with gzip.GzipFile(fileobj=BytesIO(response.content)) as f:
                for i, line in enumerate(f):
                    if i >= sample_size and not product_id:
                        break
                    try:
                        review = json.loads(line)
                        
                        # Collect product IDs
                        prod_id = review.get('asin', '')
                        if prod_id:
                            product_ids.add(prod_id)
                        
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
            
            # Update product list in session state
            if not st.session_state.product_list or st.session_state.selected_category != category:
                st.session_state.product_list = sorted(list(product_ids))
            
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

# Tab 3: About
with tab3:
    st.header("About This Project")
    
    st.write("""
    ## Amazon Reviews Sentiment Analysis
    
    This application analyzes sentiment in Amazon product reviews across multiple categories including Electronics, Books, Beauty & Personal Care products, and Home & Kitchen products.
    
    ### Features:
    
    - **Sentiment Analysis**: Analyze the sentiment of any product review using our pre-trained MiniLM model
    - **Data Explorer**: Explore sample data from Amazon reviews, visualize ratings and sentiment distributions
    - **Product Filtering**: Browse specific products and their reviews within each category
    
    ### How It Works:
    
    1. The application uses fine-tuned MiniLM models pre-trained on Amazon reviews
    2. Each product category has its own dedicated model fine-tuned on relevant reviews
    3. The models classify sentiment as positive, negative, or neutral based on review text
    4. The Streamlit interface provides an easy way to interact with the models and data
    
    ### Technical Details:
    
    - **Model**: Microsoft MiniLM, a compact language model that offers excellent performance for sentiment analysis
    - **Fine-tuning**: Each model is fine-tuned on 10,000 product reviews from its respective category
    - **Vectorization**: The model converts text to contextual embeddings that capture semantic meaning
    - **Classification**: Three-way classification (positive, negative, neutral) based on review content
    
    ### GitHub Repository:
    
    [https://github.com/nessaleal4/AmazonReviews](https://github.com/nessaleal4/AmazonReviews)
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Developed with Streamlit and Transformers")
