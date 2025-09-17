import nltk
from nltk.tag import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('universal_tagset')
nltk.download('punkt')
nltk.download('stopwords')

import re
import numpy as np
import pandas as pd

from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer

from config import TRANSRIPT_PATH
from processing.config import MODEL, YELP_REVIEWS_PATH
from processing import Word2VecHelper

class FeatureEngine:

    def __init__(self, transcript_path = TRANSRIPT_PATH, custom_embeddings_path = YELP_REVIEWS_PATH):
        self.transcript_path = transcript_path
        self.custom_embeddings_path = custom_embeddings_path
    
    def _read_csv(self, path):
        df = pd.read_csv(path)
        return df
    
    def _preprocess_text(self, text):
        """
        Cleans and standardizes a single text string.
        """
        text = text.lower()
        
        # Remove URLs, email addresses, special characters
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _tokenize_corpus(self, corpus, min_length):
        """
        Tokenizes a corpus of documents into a list of sentences.
        """
        sentences = []
        
        for document in corpus:
            # Clean and split into sentences
            clean_doc = self._preprocess_text(document)
            doc_sentences = sent_tokenize(clean_doc)
            
            for sentence in doc_sentences:
                # Tokenize words and remove short sentences
                words = word_tokenize(sentence)
                if len(words) >= min_length:
                    sentences.append(words)
        
        return sentences
    
    def _POS_tagging(self, transcript_df):
        transcript_df['POS_tags'] = transcript_df['Sentence'].apply(lambda x: pos_tag(word_tokenize(x), tagset='universal'))
        return transcript_df

    def _sentiment_score(self, transcript_df):
        transcript_df['Sentiment'] = transcript_df['Sentence'].apply(lambda x: TextBlob(x).sentiment.polarity)
        return transcript_df
    
    def _tfidf_vectorization(self, transcript_df):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(transcript_df['Sentence'])

        # Convert sparse matrix to dense and then to list of arrays
        tfidf_dense = tfidf_matrix.toarray()

        # Add the TF-IDF vectors as a new column
        transcript_df['TF-IDF'] = [row for row in tfidf_dense]
        return transcript_df
    
    def _word2vec_embedding(self, transcript_df, model = MODEL):
        vectors = []
        all_missing_words = []
        
        for sentence in transcript_df['Sentence']:
            words = sentence.lower().split()
            word_vectors = []
            missing_words = []
            
            for word in words:
                try:
                    word_vectors.append(model[word])
                except KeyError:
                    missing_words.append(word)
            
            if word_vectors:
                # Average all word vectors in the sentence
                sentence_vector = np.mean(word_vectors, axis=0)
            else:
                # If no words found, return zero vector
                sentence_vector = np.zeros(300)
            
            vectors.append(sentence_vector)
            all_missing_words.extend(missing_words)

            transcript_df["word2vec_embedding"] = vectors

        return transcript_df, all_missing_words
    
    def _get_yelp_corpus(self, reviews):
        reviews = reviews.drop(columns=['Yelp URL','Rating','Date'])
        raw_corpus = reviews['Review Text'].tolist()
        return raw_corpus
    
    def main(self):
        transcript_df = self._read_csv(self.transcript_path)
        transcript_df = self._POS_tagging(transcript_df)
        transcript_df = self._sentiment_score(transcript_df)
        transcript_df = self._tfidf_vectorization(transcript_df)
        transcript_df, _ = self._word2vec_embedding(MODEL, transcript_df)

        reviews_df = self._read_csv(self.custom_embeddings_path)
        raw_corpus = self._get_yelp_corpus(reviews_df)
        processed_sentences = self._tokenize_corpus(raw_corpus)

        w2v = Word2VecHelper(vector_size=300, window=5, min_count=5, sg=1, epochs=30, alpha=0.025, negative=20)
        word2vec_model, vector_size = w2v.train_model(processed_sentences)

        transcript_df, _,  = w2v.create_sentence_embeddings(
        sentences=processed_sentences, 
        model=word2vec_model, 
        vector_size=vector_size,
        text_column="Sentence")

    


    

