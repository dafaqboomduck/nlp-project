import nltk
nltk.download('universal_tagset', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

from nltk.tag import pos_tag
from nltk.tokenize import sent_tokenize, word_tokenize

import re
import os
import numpy as np
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

from src.config import TRANSRIPT_PATH, ARTIFACTS_DIR
from src.processing.config import (GENSIM_MODEL, YELP_REVIEWS_PATH,
                                   MODEL_DIR, MODEL_PATH)
from src.processing import Word2VecHelper  
from src.processing import BertEmbeddingsHelper
from src.helpers import CSVHandler

logger = logging.getLogger(__name__)

class FeatureEngine:
    def __init__(self, transcript_input=TRANSRIPT_PATH, custom_embeddings_path=YELP_REVIEWS_PATH):
        self.transcript_input = transcript_input
        self.custom_embeddings_path = custom_embeddings_path
        self.transcript_df_initial = transcript_input if isinstance(transcript_input, pd.DataFrame) else None

    def _preprocess_text(self, text):
        if not isinstance(text, str):
            text = "" if pd.isna(text) else str(text)
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        return text

    def _tokenize_corpus(self, corpus, min_length=0):
        if not isinstance(corpus, (list, tuple)):
            raise TypeError("corpus must be a list of documents (strings).")
        sentences = []
        for document in corpus:
            if not isinstance(document, str):
                document = "" if pd.isna(document) else str(document)
            clean_doc = self._preprocess_text(document)
            if not clean_doc:
                continue
            doc_sentences = sent_tokenize(clean_doc)
            for sentence in doc_sentences:
                words = word_tokenize(sentence)
                if len(words) >= min_length:
                    sentences.append(words)
        if not sentences:
            logger.warning("Tokenization produced zero sentences from the provided corpus.")
        return sentences

    def _POS_tagging(self, transcript_df, text_col='Sentence'):
        if not isinstance(transcript_df, pd.DataFrame):
            raise TypeError("transcript_df must be a pandas DataFrame.")
        if text_col not in transcript_df.columns:
            raise KeyError(f"'{text_col}' column is required for POS tagging.")
        transcript_df['POS_tags'] = transcript_df[text_col].apply(
            lambda x: pos_tag(word_tokenize(str(x)), tagset='universal')
        )
        return transcript_df

    def _sentiment_score(self, transcript_df, text_col='Sentence'):
        if not isinstance(transcript_df, pd.DataFrame):
            raise TypeError("transcript_df must be a pandas DataFrame.")
        if text_col not in transcript_df.columns:
            raise KeyError(f"'{text_col}' column is required for sentiment scoring.")
        transcript_df['Sentiment'] = transcript_df[text_col].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity
        )
        return transcript_df

    def _tfidf_vectorization(self, transcript_df, text_col='Sentence', embedding_column='TF-IDF'):
        if not isinstance(transcript_df, pd.DataFrame):
            raise TypeError("transcript_df must be a pandas DataFrame.")
        if text_col not in transcript_df.columns:
            raise KeyError(f"'{text_col}' column is required for TF-IDF vectorization.")
        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(transcript_df[text_col].fillna("").astype(str))
            tfidf_dense = tfidf_matrix.toarray()
            transcript_df[embedding_column] = [row for row in tfidf_dense]
            return transcript_df
        except Exception:
            logger.exception("TF-IDF vectorization failed.")
            raise

    def _word2vec_embedding(self, transcript_df, model=GENSIM_MODEL, text_col='Sentence',
                            embedding_column='word2vec_embedding', vector_size=300):
        if not isinstance(transcript_df, pd.DataFrame):
            raise TypeError("transcript_df must be a pandas DataFrame.")
        if text_col not in transcript_df.columns:
            raise KeyError(f"'{text_col}' column is required for word2vec embedding.")
        vectors = []
        all_missing_words = []

        for sentence in transcript_df[text_col].fillna("").astype(str):
            words = sentence.lower().split()
            word_vectors = []
            missing_words = []
            for word in words:
                try:
                    word_vectors.append(model[word])
                except KeyError:
                    missing_words.append(word)
                except Exception:
                    logger.debug("Unexpected exception while retrieving vector for word '%s'", word, exc_info=True)
                    missing_words.append(word)
            if word_vectors:
                sentence_vector = np.mean(word_vectors, axis=0)
            else:
                sentence_vector = np.zeros(vector_size)
            vectors.append(sentence_vector)
            all_missing_words.extend(missing_words)

        transcript_df[embedding_column] = vectors
        return transcript_df, all_missing_words

    def _get_yelp_corpus(self, reviews):
        if not isinstance(reviews, pd.DataFrame):
            raise TypeError("reviews must be a pandas DataFrame.")
        expected_cols = {'Review Text'}
        if not expected_cols.issubset(set(reviews.columns)):
            raise KeyError(f"Expected columns {expected_cols} in reviews DataFrame. Found {list(reviews.columns)}")
        to_drop = [c for c in ['Yelp URL', 'Rating', 'Date'] if c in reviews.columns]
        if to_drop:
            reviews = reviews.drop(columns=to_drop)
        raw_corpus = reviews['Review Text'].astype(str).tolist()
        return raw_corpus

    def create_features(self, transcript_df_input: pd.DataFrame = None, output_path: str = None, text_col: str = 'Sentence'):
        """
        Main pipeline orchestration. Returns DataFrame with appended features.
        """
        w2v = Word2VecHelper(vector_size=300, window=5, min_count=5, sg=1, epochs=30, alpha=0.025, negative=20)
        bert = BertEmbeddingsHelper()
        csv_handler = CSVHandler()
        custom_w2v_path = os.path.join(ARTIFACTS_DIR, MODEL_DIR, MODEL_PATH)

        # Resolve transcript input
        if transcript_df_input is not None and isinstance(transcript_df_input, pd.DataFrame):
            transcript_df = transcript_df_input.copy()
        elif self.transcript_df_initial is not None:
            transcript_df = self.transcript_df_initial.copy()
        elif isinstance(self.transcript_input, str):
            transcript_df = csv_handler.read_csv(self.transcript_input)
        else:
            raise ValueError("No valid transcript path or DataFrame provided.")

        # Validate that text_col exists
        if text_col not in transcript_df.columns:
            raise KeyError(f"'{text_col}' column is required in the transcript DataFrame.")

        # 2. Apply feature transformations
        transcript_df = self._POS_tagging(transcript_df, text_col=text_col)
        transcript_df = self._sentiment_score(transcript_df, text_col=text_col)
        transcript_df = self._tfidf_vectorization(transcript_df, text_col=text_col)

        # Word2Vec (global)
        try:
            transcript_df, missing_words = self._word2vec_embedding(transcript_df, model=GENSIM_MODEL, text_col=text_col)
            if missing_words:
                logger.info("Word2Vec missing words (sample): %s", list(set(missing_words))[:10])
        except Exception:
            logger.exception("Word2Vec embedding step failed.")
            raise

        # Custom embeddings (Yelp)
        try:
            reviews_df = csv_handler.read_csv(self.custom_embeddings_path)
            raw_corpus = self._get_yelp_corpus(reviews_df)
            processed_sentences = self._tokenize_corpus(raw_corpus)
            if not processed_sentences:
                logger.warning("Processed sentences for Word2Vec training are empty; skipping training.")
            else:
                w2v.create_model(processed_sentences, model_path=custom_w2v_path)
                transcript_df, missing_custom_words = w2v.create_sentence_embeddings(
                    sentences=transcript_df, text_column=text_col
                )
                if missing_custom_words:
                    logger.info("Custom Word2Vec missing words (sample): %s", missing_custom_words[:10])
        except Exception:
            logger.exception("Failed during custom (Yelp) embeddings handling.")
            raise

        # BERT embeddings
        try:
            transcript_df = bert.create_bert_embeddings(transcript_df, text_col)
        except Exception:
            logger.exception("Failed to create BERT embeddings.")
            raise

        # Optional output
        if output_path:
            try:
                transcript_df.to_csv(output_path, index=False, sep=';')
                logger.info("The NLP Features were saved at %s", output_path)
            except Exception:
                logger.exception("Failed to save features to %s", output_path)
                raise

        return transcript_df
