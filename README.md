# Amazon Reviews Sentiment Analysis

A Streamlit application for analyzing sentiment in Amazon product reviews across multiple categories.

## Project Structure

```
AmazonReviews/
├── data/                  # Data directory
│   ├── raw/               # Raw data from Dropbox
│   ├── processed/         # Processed data
│   └── sampled/           # Train/val/test splits
├── models/                # Trained models
├── src/                   # Source code
│   ├── preprocess.py      # Data preprocessing
│   ├── split_data.py      # Data splitting
│   ├── train_model.py     # Model training
│   └── predict.py         # Prediction functionality
├── streamlit/             # Streamlit app
│   ├── app.py             # Main app
│   └── .streamlit/        # Streamlit configuration
├── .gitignore             # Git ignore file
├── requirements.txt       # Full development requirements
└── requirements-streamlit.txt  # Streamlit Cloud requirements
```

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

4. Run the Streamlit app:
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

5. Deploy!

## Working with the Data

This project handles large Amazon review datasets (22GB+) by:

1. Storing data in Dropbox instead of GitHub
2. Reading data directly from Dropbox URLs
3. Processing only samples of data for demonstration purposes
4. Providing functionality to train models on full datasets when needed

## Models

The application supports three types of sentiment analysis models:

1. Logistic Regression (default, fast and effective)
2. Random Forest (more complex, potentially more accurate)
3. SVM (good for text classification tasks)

Models are trained on specific product categories and can be selected in the UI.
