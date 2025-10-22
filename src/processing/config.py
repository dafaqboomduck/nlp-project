# src/processing/config.py

import gensim.downloader as api
from pathlib import Path  # Import Path
from src.config import DATA_DIR, ARTIFACTS_DIR

# Gensim model loading remains the same
GENSIM_MODEL = api.load("word2vec-google-news-300")

YELP_REVIEWS_PATH = DATA_DIR / "Yelp Restaurant Reviews.csv"
FEATURE_OUTPUT_PATH = DATA_DIR / "NLP_features.csv"

MODEL_DIR = ARTIFACTS_DIR / "custom_w2v_model_files"
MODEL_PATH = MODEL_DIR / "w2v_model"
