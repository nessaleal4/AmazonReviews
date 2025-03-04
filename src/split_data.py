import os
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

def split_dataset(category, input_dir="data/processed", output_dir="data/sampled", test_size=0.2, val_size=0.1):
    """Splits processed dataset into train/val/test sets."""
    input_path = os.path.join(input_dir, f"{category}_sample.parquet")

    if not os.path.exists(input_path):
        print(f"File does not exist: {input_path}")
        return

    df = pd.read_parquet(input_path)

    # Split into train (80%), temp (20%)
    train_df, temp_df = train_test_split(df, test_size=test_size, random_state=42)

    # Split temp into validation (10%) and test (10%)
    val_size_adjusted = val_size / (test_size)  # Adjust val size relative to temp
    val_df, test_df = train_test_split(temp_df, test_size=val_size_adjusted, random_state=42)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save splits
    train_df.to_parquet(os.path.join(output_dir, f"{category}_train.parquet"))
    val_df.to_parquet(os.path.join(output_dir, f"{category}_val.parquet"))
    test_df.to_parquet(os.path.join(output_dir, f"{category}_test.parquet"))

    print(f"Split {category}: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples saved.")

def main():
    parser = argparse.ArgumentParser(description="Split processed data into train/val/test")
    parser.add_argument("--categories", nargs="+", default=["Electronics", "Books", "Beauty_and_Personal_Care, "Home_and_Kitchen"])
    args = parser.parse_args()

    for category in args.categories:
        split_dataset(category)

if __name__ == "__main__":
    main()
