import pandas as pd
import numpy as np
from collections import Counter
import json
import gzip
from io import BytesIO
import requests
from tqdm import tqdm

# Optional import for summarization
try:
    from transformers import pipeline
    SUMMARIZATION_AVAILABLE = True
except ImportError:
    SUMMARIZATION_AVAILABLE = False
    print("Transformers library not available. Summarization will be disabled.")

# Define Dropbox URLs (same as in your other modules)
DROPBOX_LINKS = {
    "Books": "https://www.dropbox.com/s/312wv7jtm1tpxeo/Books.jsonl.gz?dl=1",
    "Beauty_and_Personal_Care": "https://www.dropbox.com/s/w2bg91ewpziaaa3/Beauty_and_Personal_Care.jsonl.gz?dl=1",
    "Electronics": "https://www.dropbox.com/s/st07mgrwzazitru/Electronics.jsonl.gz?dl=1",
    "Home_and_Kitchen": "https://www.dropbox.com/s/oxn45ntlkxo8ju5/Home_and_Kitchen.jsonl.gz?dl=1"
}

def find_top_products(category, num_products=10, sample_size=50000):
    """Find the top most reviewed products in a category."""
    url = DROPBOX_LINKS.get(category)
    if not url:
        print(f"No URL found for category: {category}")
        return None
    
    try:
        print(f"Analyzing products in {category}...")
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            print(f"Failed to download data: HTTP {response.status_code}")
            return None
        
        # Count product occurrences
        product_counter = Counter()
        product_titles = {}
        
        with gzip.GzipFile(fileobj=BytesIO(response.content)) as f:
            for i in tqdm(range(sample_size), desc="Processing reviews"):
                try:
                    line_content = f.readline()
                    if not line_content:
                        break
                    
                    review = json.loads(line_content)
                    product_id = review.get('asin')
                    
                    if product_id:
                        product_counter[product_id] += 1
                        
                        # Store product title if available
                        if 'title' in review and product_id not in product_titles:
                            product_titles[product_id] = review.get('title', 'Unknown Product')
                        
                except (json.JSONDecodeError, Exception):
                    continue
        
        # Get top products
        top_products = []
        for product_id, count in product_counter.most_common(num_products):
            top_products.append({
                'product_id': product_id,
                'title': product_titles.get(product_id, 'Unknown Product'),
                'review_count': count
            })
        
        return top_products
    
    except Exception as e:
        print(f"Error finding top products: {str(e)}")
        return None

def gather_product_reviews(category, product_id, max_reviews=100):
    """Gather reviews for a specific product."""
    url = DROPBOX_LINKS.get(category)
    if not url:
        print(f"No URL found for category: {category}")
        return None
    
    try:
        print(f"Gathering reviews for product {product_id}...")
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            print(f"Failed to download data: HTTP {response.status_code}")
            return None
        
        reviews = []
        with gzip.GzipFile(fileobj=BytesIO(response.content)) as f:
            for line in f:
                try:
                    review = json.loads(line)
                    
                    if review.get('asin') == product_id:
                        # Extract relevant information
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
                            'summary': review.get('summary', ''),
                            'sentiment': sentiment,
                            'rating': rating,
                            'helpful_votes': review.get('helpful', [0, 0])[0] if isinstance(review.get('helpful'), list) else 0,
                            'reviewer_id': review.get('reviewerID', '')
                        })
                        
                        if len(reviews) >= max_reviews:
                            break
                            
                except json.JSONDecodeError:
                    continue
        
        return reviews
    
    except Exception as e:
        print(f"Error gathering product reviews: {str(e)}")
        return None

def generate_product_summary(reviews, perform_summarization=False):
    """Generate a summary of reviews for a product."""
    if not reviews:
        return {
            "overall_sentiment": "Unknown",
            "avg_rating": 0,
            "review_count": 0,
            "sentiment_distribution": {},
            "summary": "No reviews available",
            "sample_positive": [],
            "sample_negative": []
        }
    
    # Calculate metrics
    ratings = [r.get('rating', 0) for r in reviews]
    sentiments = [r.get('sentiment', 'neutral') for r in reviews]
    
    avg_rating = sum(ratings) / len(ratings) if ratings else 0
    sentiment_counts = Counter(sentiments)
    
    # Determine overall sentiment
    if sentiment_counts.get('positive', 0) > len(reviews) * 0.6:
        overall_sentiment = "Mostly Positive"
    elif sentiment_counts.get('negative', 0) > len(reviews) * 0.6:
        overall_sentiment = "Mostly Negative"
    elif sentiment_counts.get('positive', 0) > sentiment_counts.get('negative', 0):
        overall_sentiment = "Somewhat Positive"
    elif sentiment_counts.get('negative', 0) > sentiment_counts.get('positive', 0):
        overall_sentiment = "Somewhat Negative"
    else:
        overall_sentiment = "Mixed"
    
    # Prepare key points from positive and negative reviews
    positive_reviews = [r.get('text', '') for r in reviews if r.get('sentiment') == 'positive']
    negative_reviews = [r.get('text', '') for r in reviews if r.get('sentiment') == 'negative']
    
    # Use a summarization model (optional)
    summary_text = ""
    
    if perform_summarization and SUMMARIZATION_AVAILABLE:
        try:
            # Only attempt summarization if we have a meaningful number of reviews
            if len(reviews) >= 5:
                # Concatenate some reviews for a sample to summarize
                sample_text = " ".join([r.get('text', '')[:200] for r in reviews[:10]])
                
                # Initialize the summarizer
                summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
                
                # Generate summary (max 100 words)
                summary = summarizer(sample_text, max_length=100, min_length=30, do_sample=False)
                summary_text = summary[0]['summary_text']
            else:
                summary_text = "Not enough reviews for meaningful summarization."
        except Exception as e:
            print(f"Summarization error: {str(e)}")
            summary_text = "Summarization unavailable."
    else:
        # Alternative simple summary without ML model
        pos_count = sentiment_counts.get('positive', 0)
        neg_count = sentiment_counts.get('negative', 0)
        neutral_count = sentiment_counts.get('neutral', 0)
        total = len(reviews)
        
        summary_text = f"Based on {total} reviews analyzed, {pos_count} ({pos_count/total*100:.1f}%) were positive, "
        summary_text += f"{neg_count} ({neg_count/total*100:.1f}%) were negative, and {neutral_count} "
        summary_text += f"({neutral_count/total*100:.1f}%) were neutral. The average rating was {avg_rating:.1f}/5."
    
    # Create response
    result = {
        "overall_sentiment": overall_sentiment,
        "avg_rating": round(avg_rating, 1),
        "review_count": len(reviews),
        "sentiment_distribution": {
            "positive": sentiment_counts.get('positive', 0),
            "neutral": sentiment_counts.get('neutral', 0),
            "negative": sentiment_counts.get('negative', 0)
        },
        "summary": summary_text,
        "sample_positive": positive_reviews[:3] if positive_reviews else [],
        "sample_negative": negative_reviews[:3] if negative_reviews else []
    }
    
    return result

def analyze_top_products(category, num_products=10, reviews_per_product=50, perform_summarization=False):
    """Analyze the top products in a category."""
    # Find top products
    top_products = find_top_products(category, num_products)
    if not top_products:
        return {"error": f"Could not find top products for {category}"}
    
    # Analyze each product
    product_summaries = []
    for product in top_products:
        product_id = product['product_id']
        
        # Gather reviews for this product
        reviews = gather_product_reviews(category, product_id, reviews_per_product)
        
        # Generate summary
        if reviews:
            summary = generate_product_summary(reviews, perform_summarization)
            
            # Add product info
            summary.update({
                "product_id": product_id,
                "product_title": product['title'],
                "total_reviews": product['review_count']
            })
            
            product_summaries.append(summary)
    
    return {
        "category": category,
        "num_products_analyzed": len(product_summaries),
        "products": product_summaries
    }

if __name__ == "__main__":
    # Test functionality
    category = "Electronics"
    results = analyze_top_products(category, num_products=3, reviews_per_product=20)
    
    if "error" in results:
        print(results["error"])
    else:
        print(f"Analyzed {results['num_products_analyzed']} products in {category}")
        for product in results["products"]:
            print(f"\nProduct: {product['product_title']}")
            print(f"Average Rating: {product['avg_rating']}")
            print(f"Overall Sentiment: {product['overall_sentiment']}")
            print(f"Review Count: {product['review_count']}")
            print(f"Summary: {product['summary']}")
