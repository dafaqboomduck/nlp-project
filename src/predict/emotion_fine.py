import os
import pandas as pd
import torch
from transformers import pipeline
from typing import List, Optional, Dict, Tuple
import logging

from src.predict.config import FINE_EMOTION_MAP
from src.predict.preprocessing import InferencePreprocessor

logger = logging.getLogger(__name__)

class EmotionFinePredictor:
    """
    A class for emotion intensity classification using a zero-shot model,
    designed for efficient batch prediction via the Hugging Face pipeline.

    The primary method, :meth:`predict`, handles internal
    grouping of texts by their known 'Core Emotion' (e.g., 'happiness', 'sadness')
    to efficiently batch predictions for the smaller, relevant set of fine-grained 
    candidate labels for that emotion group.
    """
    def __init__(self, 
                 checkpoint: str = 'facebook/bart-large-mnli', 
                 device: str = None, 
                 batch_size: int = 128,
                 labels_map: dict | list = FINE_EMOTION_MAP):
        """
        Initializes the Zero-Shot Classifier for fine grained prediction.

        :param model_name: The name or path of the zero-shot classification model (e.g., 'facebook/bart-large-mnli').
        :type model_name: str
        :param device: The device to run inference on (e.g., 'cuda', 'cuda:0', 'cpu').
                       If None, automatically selects GPU if available, otherwise CPU.
        :type device: str, optional
        :param batch_size: The batch size for pipeline inference.
        :type batch_size: int
        """
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.labels_map = labels_map
        
        try:
            # --- Device Setup ---
            if device is None:
                is_cuda = torch.cuda.is_available()
                self.device = 'cuda' if is_cuda else 'cpu'
                if not is_cuda:
                    logger.warning("No CUDA device found. Falling back to CPU.")
            elif 'cuda' in device.lower() and not torch.cuda.is_available():
                logger.error(f"Requested device '{device}', but no CUDA device is available.")
                raise RuntimeError("CUDA device requested but not available.")
            else:
                self.device = device

            logger.debug(f"Initializing {self.__class__.__name__} with model: {self.checkpoint}")
            logger.debug(f"Device selected for inference: {self.device}")

            # Initialize Preprocessor (Handles its own exceptions)
            self.preprocessor = InferencePreprocessor(self.checkpoint, max_length=128) 

            # --- Model Loading ---
            logger.debug(f"Loading model from checkpoint '{self.checkpoint}'...")
            try:
                # Initialize the zero-shot-classification pipeline from Hugging Face
                self.classifier = pipeline(
                'zero-shot-classification',
                model=checkpoint,
                device=device, # This maps to the device index
                batch_size=batch_size
                )
                logger.info("Model successfully loaded.")
            except Exception as e:
                msg = f"Failed to load model from checkpoint {self.checkpoint}. See details: {e}"
                logger.error(msg)
                raise ImportError(msg)
            
            # Define the fixed candidate labels for fine emotions
            self.fine_emotions = self.labels_map
            logger.debug(f"Fine grained emotions candidate labels: {self.fine_emotions}")

        except RuntimeError:
            raise
        except Exception as e:
            msg = f"Failed to initialize {self.__class__.__name__} due to model/library error. Details: {str(e)}"
            logger.error(msg, exc_info=True)
            raise RuntimeError(msg) from e

    def predict(self, df: pd.DataFrame, column: str = 'Sentence') -> List[Optional[str]]:
        """
        Classifies the fine-grained emotion for a batch of sentences, conditional
        on their previously predicted core emotion.

        The method groups sentences by their core emotion to optimize zero-shot 
        classification by using a minimal, relevant set of candidate labels per batch.

        :param df: The input data for prediction. Must contain both the text column
                (default 'Sentence') and a 'Core Emotion' column.
        :type df: pandas.DataFrame
        :param column: The name of the column in the DataFrame containing the input text.
                    Defaults to 'Sentence'.
        :type column: str, optional
        :return: A list of predicted fine-grained emotions. Returns None where prediction failed.
        :rtype: List[Optional[str]]
        """
        # Validate input first before logging sentence count
        if df.empty or column not in df.columns or 'Core Emotion' not in df.columns:
            logger.warning("Input DataFrame is empty or missing required columns.")
            return []

        sentences = df[column].astype(str).tolist()
        core_emotions = df['Core Emotion'].astype(str).str.lower().tolist()
        results: List[Optional[str]] = [None] * len(sentences)

        logger.info(f"Starting {self.__class__.__name__} inference for {len(sentences)} sentences.")

        # Group by core emotion
        grouped = {}
        for i, (text, core) in enumerate(zip(sentences, core_emotions)):
            if core in self.fine_emotions:
                grouped.setdefault(core, []).append((text, i))
            else:
                results[i] = 'N/A'

        # Batch classify per emotion
        for emotion, items in grouped.items():
            texts, idxs = zip(*items)
            labels = self.fine_emotions[emotion]

            try:
                preds = self.classifier(list(texts), labels, multi_label=False)
                if not isinstance(preds, list):
                    preds = [preds]

                for res, i in zip(preds, idxs):
                    results[i] = res['labels'][0] if 'labels' in res else None

            except Exception as e:
                logger.error(f"Error processing '{emotion}' batch: {e}")
                for i in idxs:
                    results[i] = None
                # Do NOT raise here — continue with other batches
                # Raising would abort the whole batch processing prematurely
                continue

        logger.info(f"{self.__class__.__name__} inference completed successfully.")
        return results