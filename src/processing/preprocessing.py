import os
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Optional
from src.processing import FeatureEngine
from src.config import TRANSRIPT_PATH

class TranscriptPreprocessor:
    def __init__(self, transcript_path: str = TRANSRIPT_PATH, max_seq_len: int = 30):
        """
        Initialize the TranscriptPreprocessor.
        
        Args:
            transcript_path (str): Path to the transcript CSV file.
            max_seq_len (int): Maximum length for padded sequences.
        """
        self.transcript_path = transcript_path
        self.max_seq_len = max_seq_len
        
        # Load data
        self.df_concat = pd.read_csv(self.transcript_path)
        
        # Initialize feature engine
        self.features = FeatureEngine()
        
        # Placeholders
        self.preprocessed_data: Optional[pd.DataFrame] = None
        self.tokenizer: Optional[tf.keras.preprocessing.text.Tokenizer] = None
        self.data_toks: Optional[np.ndarray] = None
        self.data_features: Optional[np.ndarray] = None

    def _preprocess_features(self):
        """
        Preprocess the transcript data using FeatureEngine and drop unnecessary columns.
        """
        self.preprocessed_data = self.features.create_features(transcript_df_input=self.df_concat, output_path=None)
        columns_to_drop = ['POS_tags', 'TF-IDF', 'Unnamed: 0']
        self.preprocessed_data = self.preprocessed_data.drop(columns=columns_to_drop)

    def _tokenize_sentences(self):
        """
        Tokenize the 'Sentence' column and pad sequences.
        """
        if self.preprocessed_data is None:
            raise ValueError("Data must be preprocessed first. Call preprocess_features() before tokenizing.")
        
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        self.tokenizer.fit_on_texts(self.preprocessed_data['Sentence'])
        
        toks = self.tokenizer.texts_to_sequences(self.preprocessed_data['Sentence'])
        pads = tf.keras.preprocessing.sequence.pad_sequences(toks, padding='post', maxlen=self.max_seq_len)
        
        self.preprocessed_data['Sentence_Tok'] = pads
        self.preprocessed_data.drop(columns='Sentence', inplace=True)
        self.data_toks = np.array(self.preprocessed_data['Sentence_Tok'].tolist())

    def _assemble_features(self):
        """
        Combine tokenized sequences with embeddings to create final feature array.
        """
        if self.preprocessed_data is None or self.data_toks is None:
            raise ValueError("Ensure features are preprocessed and sentences tokenized before assembling features.")
        
        self.data_features = np.hstack([
            np.stack(self.preprocessed_data["word2vec_embedding"].to_numpy()),
            np.stack(self.preprocessed_data["custom_word2vec_embedding"].to_numpy())
        ])
    
    def preprocess_data(self):
        """
        Execute the full preprocessing pipeline.
        """
        self._preprocess_features()
        self._tokenize_sentences()
        self._assemble_features()
        return self.data_toks, self.data_features, self.tokenizer


