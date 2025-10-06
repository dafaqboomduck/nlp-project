import gensim.downloader as api

MODEL = api.load("word2vec-google-news-300")
YELP_REVIEWS_PATH = r'Data\CSV\Yelp Restaurant Reviews.csv'
FEATURE_OUTPUT_PATH = r'Data\CSV\NLP_features.csv'