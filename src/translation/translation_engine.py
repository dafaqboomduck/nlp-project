# src/translation/translation_engine.py

import logging
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
import torch

from src.translation.config import MODEL_EN_NL_PATH, MODEL_NL_EN_PATH

logger = logging.getLogger(__name__)


class TranslationEngine:
    """
    Handles round-translation (EN→NL→EN) using saved pretrained MarianMT models.
    """
    
    def __init__(self, model_en_nl_path=MODEL_EN_NL_PATH, model_nl_en_path=MODEL_NL_EN_PATH):
        """
        Initialize translation engine.
        
        Args:
            model_en_nl_path: Path to saved EN→NL model
            model_nl_en_path: Path to saved NL→EN model
        """
        self.model_en_nl_path = model_en_nl_path
        self.model_nl_en_path = model_nl_en_path
        
        self.model_en_nl = None
        self.tokenizer_en_nl = None
        self.model_nl_en = None
        self.tokenizer_nl_en = None
        
        # Check device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"TranslationEngine initialized - using device: {self.device}")
    
    def _load_single_model(self, path, direction_name):
        logger.info(f"Loading {direction_name} model from {path}...")
        tokenizer = MarianTokenizer.from_pretrained(path)
        model = MarianMTModel.from_pretrained(path)
        model.to(self.device)
        model.eval()
        return tokenizer, model

    def load_models(self):
        """Load saved pretrained models and tokenizers"""
        try:
            # Load EN->NL
            try:
                self.tokenizer_en_nl, self.model_en_nl = self._load_single_model(
                    self.model_en_nl_path, "EN→NL"
                )
            except Exception as e:
                logger.exception(f"Failed to load EN→NL model. Error:{e}")
                raise

            # Load NL->EN
            try:
                self.tokenizer_nl_en, self.model_nl_en = self._load_single_model(
                    self.model_nl_en_path, "NL→EN"
                )
            except Exception as e:
                logger.exception(f"Failed to load NL→EN model. Error:{e}")
                raise
                
            logger.info("✓ Pretrained translation models loaded successfully")
            
        except Exception: # Outer catch block handles all raised exceptions
            logger.exception("Failed to load translation models")
            raise

            
    def _translate(self, text, model, tokenizer, max_length=128, num_beams=5):
        """
        Translate a single text using the specified model.
        
        Args:
            text: Text to translate
            model: MarianMT model
            tokenizer: Corresponding tokenizer
            max_length: Maximum output length
            num_beams: Beam search width
        
        Returns:
            Translated text
        """
        if not isinstance(text, str):
            text = str(text)
        
        if not text.strip():
            return ""
        
        try:
            # Tokenize input
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=max_length
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate translation with beam search
            with torch.no_grad():
                translated = model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True
                )
            
            # Decode output
            result = tokenizer.decode(translated[0], skip_special_tokens=True)
            return result
            
        except Exception as e:
            logger.warning(f"Translation failed for text: '{text[:50]}...' Error: {e}")
            return ""
    
    def translate_en_to_nl(self, text):
        """Translate English to Dutch"""
        return self._translate(text, self.model_en_nl, self.tokenizer_en_nl)
    
    def translate_nl_to_en(self, text):
        """Translate Dutch to English"""
        return self._translate(text, self.model_nl_en, self.tokenizer_nl_en)
    
    def round_translate(self, transcript_df, text_column='Sentence', batch_size=32):
        """
        Apply round-translation to transcript DataFrame.
        
        Args:
            transcript_df: DataFrame with transcribed text
            text_column: Column containing text to translate
            batch_size: Number of sentences to process at once (for progress tracking)
        
        Returns:
            DataFrame with added 'Translation' and 'English_Translation' columns
        """
        if not isinstance(transcript_df, pd.DataFrame):
            raise TypeError("transcript_df must be a pandas DataFrame")
        
        if text_column not in transcript_df.columns:
            raise KeyError(f"'{text_column}' column not found in DataFrame")
        
        # Make a copy to avoid modifying original
        result_df = transcript_df.copy()
        
        logger.info(f"Starting round-translation on {len(result_df)} sentences")
        
        translations_nl = []
        translations_en = []
        
        total = len(result_df)
        
        # Process sentences
        for i in range(0, total, batch_size):
            batch_end = min(i + batch_size, total)
            batch_texts = result_df[text_column].iloc[i:batch_end].tolist()
            
            # EN → NL
            batch_nl = [self.translate_en_to_nl(text) for text in batch_texts]
            
            # NL → EN
            batch_en = [self.translate_nl_to_en(nl_text) for nl_text in batch_nl]
            
            translations_nl.extend(batch_nl)
            translations_en.extend(batch_en)
            
            # Progress logging
            if batch_end % 100 == 0 or batch_end >= total:
                logger.info(f"Progress: {batch_end}/{total} sentences translated")
        
        # Add new columns
        result_df['Translation'] = translations_nl
        result_df['English_Translation'] = translations_en
        
        logger.info("✓ Round-translation complete")
        
        return result_df