# src/translation/translation_engine.py

import logging
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
import torch
from typing import List, Tuple, Optional
import os

# Internal project imports to align with project structure
from src.helpers import CSVHandler
from src.config import TRANSRIPT_PATH, ARTIFACTS_DIR
from src.translation.config import MODEL_EN_NL_PATH, MODEL_NL_EN_PATH

logger = logging.getLogger(__name__)


class TranslationEngine:
    """
    Handles the end-to-end round-translation pipeline (EN→NL→EN).
    It loads a transcript, applies translation using pretrained MarianMT models,
    and saves the output, all through a single API call.
    """
    
    def __init__(self, 
                 transcript_input_path: str = TRANSRIPT_PATH,
                 model_en_nl_path: str = MODEL_EN_NL_PATH, 
                 model_nl_en_path: str = MODEL_NL_EN_PATH):
        """
        Initializes the translation engine with necessary paths.
        
        Args:
            transcript_input_path: Path to the transcript CSV to be translated.
            model_en_nl_path: Path to the saved EN→NL model.
            model_nl_en_path: Path to the saved NL→EN model.
        """
        self.transcript_input_path = transcript_input_path
        self.model_en_nl_path = model_en_nl_path
        self.model_nl_en_path = model_nl_en_path
        
        self.model_en_nl: Optional[MarianMTModel] = None
        self.tokenizer_en_nl: Optional[MarianTokenizer] = None
        self.model_nl_en: Optional[MarianMTModel] = None
        self.tokenizer_nl_en: Optional[MarianTokenizer] = None
        
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"TranslationEngine initialized - using device: {self.device}")
    
    def _load_models(self) -> None:
        """Loads all required translation models and tokenizers internally."""
        try:
            logger.info("Loading translation models...")
            self.tokenizer_en_nl, self.model_en_nl = self._load_single_model(
                self.model_en_nl_path, "EN→NL"
            )
            self.tokenizer_nl_en, self.model_nl_en = self._load_single_model(
                self.model_nl_en_path, "NL→EN"
            )
            logger.info("✓ Pretrained translation models loaded successfully")
        except Exception:
            logger.exception("A critical error occurred while loading translation models.")
            raise

    def _load_single_model(self, path: str, direction_name: str) -> Tuple[MarianTokenizer, MarianMTModel]:
        """Helper to load one model-tokenizer pair."""
        logger.info(f"Loading {direction_name} model from {path}...")
        tokenizer = MarianTokenizer.from_pretrained(path)
        model = MarianMTModel.from_pretrained(path)
        model.to(self.device)
        model.eval()
        return tokenizer, model

    def _check_models_loaded(self):
        """Ensures models are loaded before translation is attempted."""
        if not all([self.model_en_nl, self.tokenizer_en_nl, self.model_nl_en, self.tokenizer_nl_en]):
            raise RuntimeError("Translation models are not loaded. The pipeline should call _load_models() first.")
            
    def _translate_batch(self, texts: List[str], model: MarianMTModel, tokenizer: MarianTokenizer) -> List[str]:
        """Translates a list of texts using a specified model efficiently."""
        text_inputs = [str(t) if t is not None else "" for t in texts]
        
        try:
            inputs = tokenizer(
                text_inputs, return_tensors="pt", padding=True, truncation=True, max_length=128
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                translated_ids = model.generate(
                    **inputs, max_length=128, num_beams=5, early_stopping=True
                )
            
            return tokenizer.batch_decode(translated_ids, skip_special_tokens=True)
        except Exception:
            logger.warning(f"Batch translation failed for a batch of size {len(texts)}.", exc_info=True)
            return [""] * len(texts)

    def _round_translate(self, transcript_df: pd.DataFrame, text_column: str = 'Sentence', batch_size: int = 32) -> pd.DataFrame:
        """Internal method to apply round-translation to a DataFrame."""
        self._check_models_loaded()
        if text_column not in transcript_df.columns:
            raise KeyError(f"'{text_column}' column not found in DataFrame")
        
        result_df = transcript_df.copy()
        logger.info(f"Starting round-translation on {len(result_df)} sentences...")
        
        translations_nl, translations_en = [], []
        total = len(result_df)
        
        for i in range(0, total, batch_size):
            batch_end = min(i + batch_size, total)
            batch_texts = result_df[text_column].iloc[i:batch_end].tolist()
            
            batch_nl = self._translate_batch(batch_texts, self.model_en_nl, self.tokenizer_en_nl)
            batch_en = self._translate_batch(batch_nl, self.model_nl_en, self.tokenizer_nl_en)
            
            translations_nl.extend(batch_nl)
            translations_en.extend(batch_en)
            
            if (i // batch_size + 1) % 5 == 0 or batch_end >= total:
                 logger.info(f"Progress: {batch_end}/{total} sentences translated")

        result_df['Dutch_Translation'] = translations_nl
        result_df['English_Translation'] = translations_en
        logger.info("✓ Round-translation complete")
        return result_df

    def run_translation_pipeline(self, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Orchestrates the entire translation pipeline: loads data, loads models,
        performs round-translation, formats the output, and saves the result.
        
        Args:
            output_path: Optional. Path to save the final CSV. If None, a default path is used.
            
        Returns:
            A pandas DataFrame with the translation results.
        """
        csv_handler = CSVHandler()
        
        # 1. Load Data from configured path
        logger.info(f"Loading transcript from {self.transcript_input_path}...")
        try:
            transcript_df = csv_handler.read_csv(self.transcript_input_path)
        except Exception:
            logger.exception("Failed to load transcript data.")
            raise
            
        # 2. Load Models (Internal Step)
        self._load_models()

        # 3. Perform Round-Translation
        translated_df = self._round_translate(transcript_df, text_column='Sentence')
        
        # 4. Format Final Output
        final_df = translated_df[['Sentence', 'Dutch_Translation', 'English_Translation']].copy()
        final_df.rename(columns={'Sentence': 'Original_Sentence'}, inplace=True)
        
        # 5. Save Output to CSV
        if output_path is None:
            output_dir = os.path.join(ARTIFACTS_DIR, 'translations')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 'round_translations.csv')
            
        try:
            final_df.to_csv(output_path, index=False, sep=';')
            logger.info(f"✓ Translation results saved to {output_path}")
        except Exception:
            logger.exception(f"Failed to save translation results to {output_path}")
            raise
            
        return final_df