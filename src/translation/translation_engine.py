# src/translation/translation_engine.py

import logging
import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
import torch
from typing import List, Tuple, Optional, Literal
import os

# Internal project imports to align with project structure
from src.helpers import CSVHandler
from src.config import TRANSRIPT_PATH
from src.translation.config import MODEL_EN_NL_PATH, MODEL_NL_EN_PATH, TRANSLATED_OUTPUT_PATH

logger = logging.getLogger(__name__)


class TranslationEngine:
    """
    Handles direct and round-trip translations between English (en) and Dutch (nl)
    using pretrained MarianMT models.
    """
    
    # Supported languages for this engine
    SUPPORTED_LANGS = {'en', 'nl'}
    
    def __init__(self,
                 transcript_input: str | pd.DataFrame = TRANSRIPT_PATH, 
                 model_en_nl_path: str = MODEL_EN_NL_PATH, 
                 model_nl_en_path: str = MODEL_NL_EN_PATH,
                 output_path: str = TRANSLATED_OUTPUT_PATH):
        """
        Initializes the translation engine with necessary paths.
        
        Args:
            transcript_input: Path to the transcript CSV or a ready-to-use DataFrame.
            model_en_nl_path: Path to the saved EN→NL model.
            model_nl_en_path: Path to the saved NL→EN model.
            output_path: Path where the final translated CSV will be saved.
        """
        self.transcript_input: str | pd.DataFrame = transcript_input
        # Use an explicit variable for the initial DataFrame if one was provided
        self.transcript_df_initial: Optional[pd.DataFrame] = transcript_input if isinstance(transcript_input, pd.DataFrame) else None
        
        self.model_paths = {'en-nl': model_en_nl_path, 'nl-en': model_nl_en_path}
        self.output_path: str = output_path
        self.models: dict = {}  # Cache for loaded models
        self.tokenizers: dict = {}  # Cache for loaded tokenizers
        
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"TranslationEngine initialized - using device: {self.device}")
    
    def load_model(self, direction: Literal['en-nl', 'nl-en']) -> None:
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
            
    def _validate_languages(self, input_lang: str, output_lang: str, intermediate_lang: Optional[str] = None):
        """
        Validates that the provided language codes are supported and checks for
        logical consistency based on translation type.
        """
        
        all_langs = {input_lang, output_lang}
        if intermediate_lang:
            all_langs.add(intermediate_lang)
            
        if not all_langs.issubset(self.SUPPORTED_LANGS):
            unsupported = all_langs.difference(self.SUPPORTED_LANGS)
            raise ValueError(f"Unsupported language(s) {unsupported}. Supported languages are: {list(self.SUPPORTED_LANGS)}")
        
        if intermediate_lang: # Round-trip validation
            # REDUNDANT if input_lang != output_lang:
            # REDUNDANT    raise ValueError("For round-translation, the starting (input) and ending (output) languages must be the same.")
            if input_lang == intermediate_lang or output_lang == intermediate_lang:
                raise ValueError("The intermediate language must be different from the input/output language.")
        else: # Simple translation validation
            if input_lang == output_lang:
                raise ValueError("Input and output languages cannot be the same for a direct translation.")

    def _translate_batch(self, texts: List[str], model: MarianMTModel, tokenizer: MarianTokenizer) -> List[str]:
        """Translates a list of texts using a specified model efficiently."""
        
        # Ensure all texts are strings for tokenization
        text_inputs = [str(t) if t is not None else "" for t in texts]
        
        try:
            inputs = tokenizer(text_inputs, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                # Use greedy decoding for translation, adjust parameters for quality if needed
                translated_ids = model.generate(**inputs, max_length=128, num_beams=5, early_stopping=True)
            
            return tokenizer.batch_decode(translated_ids, skip_special_tokens=True)
        except Exception:
            # Revert to a safe failure, returning empty strings for the entire batch
            logger.warning(f"Batch translation failed for a batch of size {len(texts)}.", exc_info=True)
            return [""] * len(texts)

    def translate(self, transcript_df: pd.DataFrame, text_column: str, input_lang: Literal['en', 'nl'], output_lang: Literal['en', 'nl'], batch_size: int = 32) -> pd.DataFrame:
        """
        Performs a direct translation on a DataFrame column from an input language to an output language.

        Args:
            transcript_df: DataFrame containing the text to translate.
            text_column: The name of the column with the text.
            input_lang: The source language ('en' or 'nl').
            output_lang: The target language ('en' or 'nl').
            batch_size: Number of sentences to process at once.

        Returns:
            A new DataFrame with an added translation column.
        """
        self._validate_languages(input_lang, output_lang)
        direction = f"{input_lang}-{output_lang}"
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
        
        result_df[f'Translation_{output_lang.upper()}'] = translations
        logger.info(f"✓ Direct translation complete.")
        return result_df

    def round_translate(self, transcript_df: pd.DataFrame, text_column: str, input_lang: Literal['en', 'nl'], intermediate_lang: Literal['en', 'nl'], batch_size: int = 32) -> pd.DataFrame:
        """
        Performs a round-trip translation (e.g., EN -> NL -> EN) on a DataFrame column.

        Args:
            transcript_df: DataFrame containing the text to translate.
            text_column: The name of the column with the text.
            input_lang: The starting and ending language ('en' or 'nl').
            intermediate_lang: The language used for the intermediate step (e.g., 'nl' if input is 'en').
            batch_size: Number of sentences to process at once.

        Returns:
            A new DataFrame with columns for the intermediate and final translations.
        """
        output_lang = input_lang # For round trip, output must equal input
        self._validate_languages(input_lang, output_lang, intermediate_lang)
        
        # Determine directions
        direction1 = f"{input_lang}-{intermediate_lang}"
        direction2 = f"{intermediate_lang}-{output_lang}"
        
        # Load models
        self.load_model(direction1)
        self.load_model(direction2)
        
        model1, tokenizer1 = self.models[direction1], self.tokenizers[direction1]
        model2, tokenizer2 = self.models[direction2], self.tokenizers[direction2]
        
        result_df = transcript_df.copy()
        logger.info(f"Starting round-trip translation ({input_lang.upper()}→{intermediate_lang.upper()}→{output_lang.upper()}) for {len(result_df)} sentences...")
        
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
        result_df[f'Round_Trip_Translation_{output_lang.upper()}'] = translations2
        logger.info("✓ Round-trip translation complete.")
        return result_df
    
    def run_pipeline(self, translation_type: Literal['simple', 'round'], input_lang: Literal['en', 'nl'], output_lang: Literal['en', 'nl'], transcript_df_input: Optional[pd.DataFrame] = None, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Orchestrates the entire translation pipeline: loads data, loads models,
        performs translation, formats the output, and saves the result.
        """

        csv_handler = CSVHandler()

        # 1. Resolve transcript input
        logger.info(f"Resolving the Input Path type...")
        if transcript_df_input is not None and isinstance(transcript_df_input, pd.DataFrame):
            transcript_df = transcript_df_input.copy()
            logger.info(f"Loaded the transcript_df variable successfully from the argument 'transcript_df_input' provided in the function's initialization.")
        elif self.transcript_df_initial is not None:
            transcript_df = self.transcript_df_initial.copy()
            logger.info(f"Loaded the transcript_df variable successfully from the argument 'transcript_df_initial' provided in the class's initialization.")
        elif isinstance(self.transcript_input, str):
            transcript_df = csv_handler.read_csv(self.transcript_input)
            logger.info(f"Loaded the transcript_df variable successfully {self.transcript_input} using the CSVHandler.")
        else:
            raise ValueError("No valid transcript path or DataFrame provided.")

        # Validate that 'Sentence' column exists
        if 'Sentence' not in transcript_df.columns:
            raise KeyError("'Sentence' column is required in the transcript DataFrame.")

        # 2. Determine intermediate language for round-trip (if applicable)
        intermediate_lang = None
        if translation_type == 'round':
            # Determine the intermediate language: if input is 'en', intermediate is 'nl', else 'en'
            intermediate_lang = 'nl' if input_lang == 'en' else 'en'
            if input_lang != output_lang:
                 raise ValueError("For round-trip translation, the `input_lang` and `output_lang` must be the same.")


        # 3. Perform Translation
        try:
            if translation_type == 'simple':
                translated_df = self.translate(transcript_df, text_column = 'Sentence', input_lang = input_lang, output_lang = output_lang)
            elif translation_type == 'round':
                translated_df = self.round_translate(transcript_df, text_column = 'Sentence', input_lang = input_lang, intermediate_lang = intermediate_lang)
            logger.info(f"The {translation_type} translation was completed successfully.")
        except Exception as e:
            logger.exception(f"Failed to complete the {translation_type} translation. Reason: {e}")
            raise
        
        # 4. Format Final Output
        if translation_type == 'simple':
            translation_col = f'Translation_{output_lang.upper()}'
            final_df = translated_df[['Sentence', translation_col]].copy()
        elif translation_type == 'round':
            intermediate_col = f'Intermediate_Translation_{intermediate_lang.upper()}'
            round_trip_col = f'Round_Trip_Translation_{output_lang.upper()}'
            final_df = translated_df[['Sentence', intermediate_col, round_trip_col]].copy()

        final_df.rename(columns={'Sentence': 'Original_Sentence'}, inplace=True)
        
        # 5. Save Output to CSV
        if output_path is None:
            output_path = self.output_path
            
        try:
            final_df.to_csv(output_path, index=False, sep=';')
            logger.info(f"✓ Translation results saved successfully to {output_path}")
        except Exception:
            logger.exception(f"Failed to save translation results to {output_path}")
            raise
            
        return final_df