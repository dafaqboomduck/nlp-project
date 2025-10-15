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
    Handles direct and round-trip translations between English (en) and Dutch (nl)
    using pretrained MarianMT models.
    """
    
    def __init__(self, 
                 model_en_nl_path: str = MODEL_EN_NL_PATH, 
                 model_nl_en_path: str = MODEL_NL_EN_PATH):
        """
        Initializes the translation engine with model paths. Models are loaded on-demand.
        """
        self.model_paths = {'en-nl': model_en_nl_path, 'nl-en': model_nl_en_path}
        self.models = {}  # Cache for loaded models
        self.tokenizers = {}  # Cache for loaded tokenizers
        
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"TranslationEngine initialized - using device: {self.device}")
    
    def load_model(self, direction: str) -> None:
        """Loads a model and tokenizer for a specific translation direction (e.g., 'en-nl')."""
        if direction in self.models:
            return  # Already loaded

        path = self.model_paths.get(direction)
        if not path:
            raise ValueError(f"Model path for direction '{direction}' not configured.")

        try:
            logger.info(f"Loading {direction.upper()} model from {path}...")
            tokenizer = MarianTokenizer.from_pretrained(path)
            model = MarianMTModel.from_pretrained(path)
            model.to(self.device)
            model.eval()
            self.models[direction] = model
            self.tokenizers[direction] = tokenizer
            logger.info(f"✓ {direction.upper()} model loaded successfully.")
        except Exception:
            logger.exception(f"Failed to load model for direction '{direction}' from {path}.")
            raise
            
    def _validate_languages(self, input_language: str, output_language: str, is_round_trip: bool = False):
        """Validates that the provided language codes are supported."""
        supported_langs = {'en', 'nl'}
        if input_language not in supported_langs or output_language not in supported_langs:
            raise ValueError(f"Language not available. Supported languages are: {list(supported_langs)}")
        
        if is_round_trip:
            if input_language != output_language:
                raise ValueError("For round-translation, input and output languages must be the same (e.g., 'en' to 'en').")
        else:
            if input_language == output_language:
                raise ValueError("Input and output languages cannot be the same for a direct translation.")

    def _translate_batch(self, texts: List[str], model: MarianMTModel, tokenizer: MarianTokenizer) -> List[str]:
        """Translates a list of texts using a specified model efficiently."""
        text_inputs = [str(t) if t is not None else "" for t in texts]
        
        try:
            inputs = tokenizer(text_inputs, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                translated_ids = model.generate(**inputs, max_length=128, num_beams=5, early_stopping=True)
            
            return tokenizer.batch_decode(translated_ids, skip_special_tokens=True)
        except Exception:
            logger.warning(f"Batch translation failed for a batch of size {len(texts)}.", exc_info=True)
            return [""] * len(texts)

    def translate(self, transcript_df: pd.DataFrame, text_column: str, input_language: str, output_language: str, batch_size: int = 32) -> pd.DataFrame:
        """
        Performs a direct translation on a DataFrame column from an input language to an output language.

        Args:
            transcript_df: DataFrame containing the text to translate.
            text_column: The name of the column with the text.
            input_language: The source language ('en' or 'nl').
            output_language: The target language ('en' or 'nl').
            batch_size: Number of sentences to process at once.

        Returns:
            A new DataFrame with an added translation column.
        """
        self._validate_languages(input_language, output_language)
        direction = f"{input_language}-{output_language}"
        self.load_model(direction)
        
        model = self.models[direction]
        tokenizer = self.tokenizers[direction]
        
        result_df = transcript_df.copy()
        logger.info(f"Starting '{direction.upper()}' translation for {len(result_df)} sentences...")
        
        translations = []
        total = len(result_df)
        
        for i in range(0, total, batch_size):
            batch_texts = result_df[text_column].iloc[i:min(i + batch_size, total)].tolist()
            translated_batch = self._translate_batch(batch_texts, model, tokenizer)
            translations.extend(translated_batch)
        
        result_df[f'Translation_{output_language.upper()}'] = translations
        logger.info(f"✓ Direct translation complete.")
        return result_df

    def round_translate(self, transcript_df: pd.DataFrame, text_column: str, input_language: str, output_language: str, batch_size: int = 32) -> pd.DataFrame:
        """
        Performs a round-trip translation (e.g., EN -> NL -> EN) on a DataFrame column.

        Args:
            transcript_df: DataFrame containing the text to translate.
            text_column: The name of the column with the text.
            input_language: The starting language ('en' or 'nl').
            output_language: The ending language (must be same as input_language).
            batch_size: Number of sentences to process at once.

        Returns:
            A new DataFrame with columns for the intermediate and final translations.
        """
        self._validate_languages(input_language, output_language, is_round_trip=True)
        
        # Determine directions
        intermediate_lang = 'nl' if input_language == 'en' else 'en'
        direction1 = f"{input_language}-{intermediate_lang}"
        direction2 = f"{intermediate_lang}-{output_language}"
        
        # Load models
        self.load_model(direction1)
        self.load_model(direction2)
        
        model1, tokenizer1 = self.models[direction1], self.tokenizers[direction1]
        model2, tokenizer2 = self.models[direction2], self.tokenizers[direction2]
        
        result_df = transcript_df.copy()
        logger.info(f"Starting round-trip translation ({input_language.upper()}→{intermediate_lang.upper()}→{output_language.upper()}) for {len(result_df)} sentences...")
        
        translations1, translations2 = [], []
        total = len(result_df)
        
        for i in range(0, total, batch_size):
            batch_texts = result_df[text_column].iloc[i:min(i + batch_size, total)].tolist()
            
            # First leg
            intermediate_batch = self._translate_batch(batch_texts, model1, tokenizer1)
            translations1.extend(intermediate_batch)
            
            # Second leg
            final_batch = self._translate_batch(intermediate_batch, model2, tokenizer2)
            translations2.extend(final_batch)

        result_df[f'Intermediate_Translation_{intermediate_lang.upper()}'] = translations1
        result_df[f'Round_Trip_Translation_{output_language.upper()}'] = translations2
        logger.info("✓ Round-trip translation complete.")
        return result_df

