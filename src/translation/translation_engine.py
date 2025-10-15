# src/translation/translation_engine.py

import logging
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
import torch
from typing import List, Tuple, Optional

from src.translation.config import MODEL_EN_NL_PATH, MODEL_NL_EN_PATH

logger = logging.getLogger(__name__)


class TranslationEngine:
    """
    Handles round-translation (EN→NL→EN) using saved pretrained MarianMT models.
    """
    
    def __init__(self, model_en_nl_path: str = MODEL_EN_NL_PATH, model_nl_en_path: str = MODEL_NL_EN_PATH):
        """
        Initialize translation engine.
        
        Args:
            model_en_nl_path: Path to saved EN→NL model
            model_nl_en_path: Path to saved NL→EN model
        """
        self.model_en_nl_path = model_en_nl_path
        self.model_nl_en_path = model_nl_en_path
        
        self.model_en_nl: Optional[MarianMTModel] = None
        self.tokenizer_en_nl: Optional[MarianTokenizer] = None
        self.model_nl_en: Optional[MarianMTModel] = None
        self.tokenizer_nl_en: Optional[MarianTokenizer] = None
        
        # Check device
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"TranslationEngine initialized - using device: {self.device}")
    
    def _load_single_model(self, path: str, direction_name: str) -> Tuple[MarianTokenizer, MarianMTModel]:
        logger.info(f"Loading {direction_name} model from {path}...")
        tokenizer = MarianTokenizer.from_pretrained(path)
        model = MarianMTModel.from_pretrained(path)
        model.to(self.device)
        model.eval()
        return tokenizer, model

    def load_models(self) -> None:
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

    def _check_models_loaded(self):
        """Helper to ensure models are loaded before translation is attempted."""
        if self.model_en_nl is None or self.tokenizer_en_nl is None or \
           self.model_nl_en is None or self.tokenizer_nl_en is None:
            raise RuntimeError("Translation models not loaded. Call load_models() first.")
            
    def _translate_batch(self, texts: List[str], model: MarianMTModel, tokenizer: MarianTokenizer, max_length: int = 128, num_beams: int = 5) -> List[str]:
        """
        Translate a list of texts using the specified model efficiently.
        
        Args:
            texts: List of strings to translate.
            model: MarianMT model.
            tokenizer: Corresponding tokenizer.
            max_length: Maximum output length.
            num_beams: Beam search width.
        
        Returns:
            List of translated texts. Returns empty string for input that failed/was empty.
        """
        if model is None or tokenizer is None:
             raise RuntimeError("Model or Tokenizer not provided for batch translation.")
        
        # Ensure all texts are strings and handle non-string/None inputs safely
        text_inputs = [str(t) if t is not None else "" for t in texts]
        
        try:
            # Tokenize input (handles batching naturally)
            inputs = tokenizer(
                text_inputs, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=max_length
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                translated_ids = model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True
                )
            
            # Decode output
            results = tokenizer.batch_decode(translated_ids, skip_special_tokens=True)
            return results
            
        except Exception as e:
            logger.warning(f"Batch translation failed for a batch of size {len(texts)}. Error: {e}")
            # Return a list of empty strings for the entire batch upon failure
            return [""] * len(texts)

    # NOTE: The _translate method is no longer used by round_translate but is kept for single-text convenience/legacy.
    # It has been modified to use the batch translation logic internally for consistency/safety.
    def _translate(self, text: str, model: Optional[MarianMTModel], tokenizer: Optional[MarianTokenizer], max_length: int = 128, num_beams: int = 5) -> str:
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
        if model is None or tokenizer is None:
             logger.warning("Attempted single translation with unloaded models.")
             return ""

        if not isinstance(text, str):
            text = str(text)
        
        if not text.strip():
            return ""
        
        # Use the batch function for the single input for consistency
        result_list = self._translate_batch(
            [text], model, tokenizer, max_length=max_length, num_beams=num_beams
        )
        return result_list[0] if result_list else ""
    
    def translate_en_to_nl(self, text: str) -> str:
        """Translate English to Dutch"""
        self._check_models_loaded()
        return self._translate(text, self.model_en_nl, self.tokenizer_en_nl)
    
    def translate_nl_to_en(self, text: str) -> str:
        """Translate Dutch to English"""
        self._check_models_loaded()
        return self._translate(text, self.model_nl_en, self.tokenizer_nl_en)
    
    def round_translate(self, transcript_df: pd.DataFrame, text_column: str = 'Sentence', batch_size: int = 32) -> pd.DataFrame:
        """
        Apply round-translation to transcript DataFrame.
        
        Args:
            transcript_df: DataFrame with transcribed text
            text_column: Column containing text to translate
            batch_size: Number of sentences to process at once
        
        Returns:
            DataFrame with added 'Translation' and 'English_Translation' columns
        """
        self._check_models_loaded() # Ensure models are ready
        
        if not isinstance(transcript_df, pd.DataFrame):
            raise TypeError("transcript_df must be a pandas DataFrame")
        
        if text_column not in transcript_df.columns:
            raise KeyError(f"'{text_column}' column not found in DataFrame")
        
        # Make a copy to avoid modifying original
        result_df = transcript_df.copy()
        
        logger.info(f"Starting round-translation on {len(result_df)} sentences")
        
        translations_nl: List[str] = []
        translations_en: List[str] = []
        total = len(result_df)
        
        # Process sentences efficiently in batches using _translate_batch
        for i in range(0, total, batch_size):
            batch_end = min(i + batch_size, total)
            batch_texts = result_df[text_column].iloc[i:batch_end].tolist()
            
            # EN → NL (Efficient batch call)
            batch_nl = self._translate_batch(
                batch_texts, self.model_en_nl, self.tokenizer_en_nl
            )
            
            # NL → EN (Efficient batch call)
            batch_en = self._translate_batch(
                batch_nl, self.model_nl_en, self.tokenizer_nl_en
            )
            
            translations_nl.extend(batch_nl)
            translations_en.extend(batch_en)
            
            # Progress logging: Log every 100 sentences or when finished
            if batch_end % 100 == 0 or batch_end >= total:
                logger.info(f"Progress: {batch_end}/{total} sentences translated")
        
        # Add new columns
        result_df['Translation'] = translations_nl
        result_df['English_Translation'] = translations_en
        
        logger.info("✓ Round-translation complete")
        
        return result_df