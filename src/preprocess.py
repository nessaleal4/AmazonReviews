import os
import json
import gzip
import pandas as pd
import argparse
from tqdm import tqdm
import requests

# Dropbox file links
DROPBOX_LINKS = {
    # Review files
    "Books": "https://www.dropbox.com/s/312wv7jtm1tpxeo/Books.jsonl.gz?dl=1",
    "Beauty_and_Personal_Care": "https://www.dropbox.com/s/w2bg91ewpziaaa3/Beauty_and_Personal_Care.jsonl.gz?dl=1",
    "Electronics": "https://www.dropbox.com/s/st07mgrwzazitru/Electronics.jsonl.gz?dl=1",
    "Home_and_Kitchen": "https://www.dropbox.com/s/oxn45ntlkxo8ju5/Home_and_Kitchen.jsonl.gz?dl=1",
    
    # Metadata files
    "meta_Books": "https://www.dropbox.com/s/k6gjgea3sn68xq0/meta_Books.jsonl.gz?dl=1",
    "meta_Beauty_and_Personal_Care": "https://www.dropbox.com/s/ghblw1oskik0bui/meta_Beauty_and_Personal_Care.jsonl.gz?dl=1",
    "meta_Electronics": "https://www.dropbox.com/s/v9rso2vqr3qyxf4/meta_Electronics.jsonl.gz?dl=1",
    "meta_Home_and_Kitchen": "https://www.dropbox.com/s/h9zdk9841dgp26s/meta_Home_and_Kitchen.jsonl.gz?dl=1",
}

def download_from_dropbox(category, data_dir="data/raw"):
    """Downloads a dataset from Dropbox if it doesn't already exist."""
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, f"{category}.jsonl.gz")
    
    if not os.path.exists(file_path):
        print(f"Downloading {category} dataset from Dropbox...")
        url = DROPBOX_LINKS[category]
        response = requests.get(url, stream=True)
        
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print(f"Saved to {file_path}")
    else:
        print(f"{category} dataset already exists. Skipping download.")
    
    return file_path

def process_reviews(category, input_dir="data/raw", output_dir="data/processed", max_reviews=10000):
    """Processes Amazon reviews."""
    # Ensure data is available
    download_from_dropbox(category, input_dir)
    
    # Also download metadata if available
    meta_category = f"meta_{category}"
    if meta_category in DROPBOX_LINKS:
        download_from_dropbox(meta_category, input_dir)
    
    input_path = os.path.join(input_dir, f"{category}.jsonl.gz")
    output_path = os.path.join(output_dir, f"{category}_sample.parquet")
    
    if not os.path.exists(input_path):
        print(f"File does not exist: {input_path}")
        return
    
    processed_reviews = []
    review_count = 0
    
    # Process reviews
    with gzip.open(input_path, 'rt', encoding='utf-8') as file:
        for line in tqdm(file, desc=f"Processing {category} reviews"):
            try:
                if review_count >= max_reviews:
                    break
                
                review = json.loads(line.strip())
                review['processed_text'] = review.get('text', '').lower().strip()
                processed_reviews.append(review)
                review_count += 1
            except json.JSONDecodeError:
                continue
    
    df = pd.DataFrame(processed_reviews)
    os.makedirs(output_dir, exist_ok=True)
    df.to_parquet(output_path)
    print(f"Processed {review_count} reviews from {category} and saved to {output_path}")
    
    # Process metadata if available
    meta_input_path = os.path.join(input_dir, f"{meta_category}.jsonl.gz")
    meta_output_path = os.path.join(output_dir, f"{meta_category}_sample.parquet")
    
    if os.path.exists(meta_input_path):
        meta_count = 0
        processed_meta = []
        
        with gzip.open(meta_input_path, 'rt', encoding='utf-8') as file:
            for line in tqdm(file, desc=f"Processing {meta_category} metadata"):
                try:
                    # Limit metadata to a reasonable sample size
                    if meta_count >= max_reviews * 2:  # Metadata often has more entries than reviews
                        break
                    
                    metadata = json.loads(line.strip())
                    processed_meta.append(metadata)
                    meta_count += 1
                except json.JSONDecodeError:
                    continue
        
        if processed_meta:
            meta_df = pd.DataFrame(processed_meta)
            meta_df.to_parquet(meta_output_path)
            print(f"Processed {meta_count} metadata entries from {meta_category} and saved to {meta_output_path}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Preprocess Amazon review data")
    parser.add_argument("--categories", nargs="+", 
                       default=["Electronics", "Books", "Beauty_and_Personal_Care", "Home_and_Kitchen"])
    parser.add_argument("--max_reviews", type=int, default=10000, help="Maximum number of reviews to process per category")
    args = parser.parse_args()
    
    for category in args.categories:
        process_reviews(category, max_reviews=args.max_reviews)

if __name__ == "__main__":
    main()
