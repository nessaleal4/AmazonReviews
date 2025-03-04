import os
import json
import gzip
import pandas as pd
import argparse
from tqdm import tqdm
import requests

# Dropbox file links
DROPBOX_LINKS = {
    "Books": "https://www.dropbox.com/s/312wv7jtm1tpxeo/Books.jsonl.gz?dl=1",
    "Beauty_and_Personal_Care": "https://www.dropbox.com/s/w2bg91ewpziaaa3/Beauty_and_Personal_Care.jsonl.gz?dl=1",
    "Electronics": "https://www.dropbox.com/s/st07mgrwzazitru/Electronics.jsonl.gz?dl=1",
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

def process_reviews(category, input_dir="data/raw", output_dir="data/processed", max_reviews=10000):
    """Processes Amazon reviews."""
    download_from_dropbox(category, input_dir)  # Ensure data is available
    
    input_path = os.path.join(input_dir, f"{category}.jsonl.gz")
    output_path = os.path.join(output_dir, f"{category}_sample.parquet")
    
    if not os.path.exists(input_path):
        print(f"File does not exist: {input_path}")
        return
    
    processed_reviews = []
    review_count = 0

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
    return df

def main():
    parser = argparse.ArgumentParser(description="Preprocess Amazon review data")
    parser.add_argument("--categories", nargs="+", default=["Electronics", "Books", "Beauty_and_Personal_Care"])
    args = parser.parse_args()

    for category in args.categories:
        process_reviews(category)

if __name__ == "__main__":
    main()
