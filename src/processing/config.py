import gensim.downloader as api

GENSIM_MODEL = api.load("word2vec-google-news-300")
YELP_REVIEWS_PATH = r'Data\CSV\Yelp Restaurant Reviews.csv'
FEATURE_OUTPUT_PATH = r'Data\CSV\NLP_features.csv'
MODEL_DIR = r"custom_w2v_model_files"
MODEL_PATH = r"w2v_model"