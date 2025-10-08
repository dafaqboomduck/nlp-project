import multiprocessing
import numpy as np
import os
from gensim.models import Word2Vec, KeyedVectors
import logging
import types

logger = logging.getLogger(__name__)

class Word2VecHelper:
    """
    A helper class to train, load, and create sentence embeddings
    from a text corpus using Word2Vec.
    """

    def __init__(self, vector_size: int, window: int, min_count: int, sg: int, epochs: int, alpha: float, negative: int, workers: int = None):
        try:
            self.vector_size = int(vector_size)
            self.window = int(window)
            self.min_count = int(min_count)
            self.sg = int(sg)
            self.epochs = int(epochs)
            self.alpha = float(alpha)
            self.negative = int(negative)
        except (TypeError, ValueError):
            raise ValueError("Invalid hyperparameter type passed to Word2VecHelper.")
        
        self.workers = multiprocessing.cpu_count() if workers is None else max(1, int(workers))
        self.model = None

    # ----------------------------------------------------------------------
    # NEW: Unified train-or-load logic
    # ----------------------------------------------------------------------
    def fit(self, sentences, model_path: str):
        """
        Trains a Word2Vec model or loads it from disk if it already exists.

        Parameters
        ----------
        sentences : list[list[str]]
            Tokenized text corpus.
        model_path : str
            Path where the model should be saved or loaded from.

        Returns
        -------
        gensim.models.Word2Vec
            The trained or loaded model.
        """
        if not isinstance(model_path, str):
            raise TypeError("model_path must be a valid string path.")

        # Check for an existing saved model
        if os.path.exists(model_path):
            try:
                logger.info(f"Found existing Word2Vec model at {model_path}. Loading it...")
                self.model = Word2Vec.load(model_path)
                logger.info("Model successfully loaded from disk.")
                return self.model
            except Exception as e:
                logger.warning(f"Failed to load existing model at {model_path}. Retraining. Reason: {e}")

        # If no model found or load failed, train a new one
        self.train_model(sentences)

        # Attempt to save the trained model
        try:
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            self.model.save(model_path)
            logger.info(f"Trained Word2Vec model saved to {model_path}")
        except Exception as e:
            logger.warning(f"Failed to save model to {model_path}. Reason: {e}")

        return self.model

    # ----------------------------------------------------------------------
    def train_model(self, sentences):
        """
        Trains a new Word2Vec model using the provided tokenized sentences.
        """
        if not isinstance(sentences, (list, tuple)):
            raise TypeError("sentences must be a list of tokenized sentences.")
        if len(sentences) == 0:
            raise ValueError("sentences is empty; nothing to train on.")
        if not all(isinstance(s, (list, tuple)) for s in sentences):
            raise TypeError("Each element of sentences must be a list/tuple of tokens (strings).")

        logger.info("Training Word2Vec model with %d sentences...", len(sentences))
        try:
            self.model = Word2Vec(
                sentences=sentences,
                vector_size=self.vector_size,
                window=self.window,
                min_count=self.min_count,
                workers=self.workers,
                sg=self.sg,
                epochs=self.epochs,
                alpha=self.alpha,
                negative=self.negative
            )
            logger.info("Word2Vec training complete.")
        except Exception as e:
            logger.exception("Failed to train Word2Vec model.")
            raise RuntimeError("Word2Vec training failed.") from e

    # ----------------------------------------------------------------------
    def load_pretrained(self, keyed_vectors_or_path):
        """
        Load pretrained keyed vectors or a model path. Accepts either:
        - a gensim KeyedVectors object
        - path to a KeyedVectors / .kv file
        """
        if isinstance(keyed_vectors_or_path, KeyedVectors):
            self.model = types.SimpleNamespace(wv=keyed_vectors_or_path)
            return

        if not isinstance(keyed_vectors_or_path, str):
            raise TypeError("keyed_vectors_or_path must be a KeyedVectors instance or a file path string.")

        try:
            kv = KeyedVectors.load(keyed_vectors_or_path, mmap='r')
            self.model = types.SimpleNamespace(wv=kv)
            logger.info(f"Loaded pretrained keyed vectors from {keyed_vectors_or_path}")
        except Exception as e:
            logger.exception("Failed to load pretrained keyed vectors.")
            raise RuntimeError(f"Could not load pretrained keyed vectors from {keyed_vectors_or_path}") from e

    # ----------------------------------------------------------------------
    def create_sentence_embeddings(self, sentences, text_column, embedding_column="custom_word2vec_embedding"):
        """
        Creates sentence embeddings by averaging the word vectors.
        Returns (DataFrame with embeddings, list of missing words)
        """
        import pandas as pd

        if self.model is None:
            raise RuntimeError("Model not trained or loaded. Call train_or_load_model() or load_pretrained() first.")
        if not isinstance(sentences, pd.DataFrame):
            raise TypeError("sentences must be a pandas DataFrame.")
        if text_column not in sentences.columns:
            raise KeyError(f"text_column '{text_column}' not found in DataFrame.")

        vectors, missing_words = [], []
        for idx, text in enumerate(sentences[text_column].fillna("").astype(str)):
            words = text.lower().split()
            word_vectors = []
            for word in words:
                try:
                    word_vectors.append(self.model.wv[word])
                except KeyError:
                    missing_words.append(word)

            sentence_vector = np.mean(word_vectors, axis=0) if word_vectors else np.zeros(self.vector_size)
            vectors.append(sentence_vector)

        sentences[embedding_column] = vectors
        return sentences, list(set(missing_words))
