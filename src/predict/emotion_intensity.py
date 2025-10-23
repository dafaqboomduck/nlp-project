import os
import pandas as pd
import torch
from transformers import pipeline
from typing import List, Optional
import logging

from src.predict.config import INTENSITY_LABELS
from src.predict.preprocessing import InferencePreprocessor

logger = logging.getLogger(__name__)

class EmotionIntensityPredictor:
    """
    A class for emotion intensity classification using a zero-shot model,
    designed for efficient batch prediction via the Hugging Face pipeline.

    It classifies input text into a fixed set of intensity labels: 'weak', 
    'moderate', or 'intense'.
    """
    def __init__(self, 
                 checkpoint: str = 'facebook/bart-large-mnli', 
                 device: str = None, 
                 batch_size: int = 128,
                 labels_map: dict | list = INTENSITY_LABELS):
        """
        Initializes the Zero-Shot Classifier for intensity prediction.

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
            
            # Define the fixed candidate labels for intensity
            self.intensity_labels = self.labels_map
            logger.debug(f"Intensity candidate labels: {self.intensity_labels}")

        except RuntimeError:
            raise
        except Exception as e:
            msg = f"Failed to initialize {self.__class__.__name__} due to model/library error. Details: {str(e)}"
            logger.error(msg, exc_info=True)
            raise RuntimeError(msg) from e


    def predict(self, df: pd.DataFrame, column: str = 'Sentence') -> List[Optional[str]]:
        """
        Classifies the intensity for a batch of sentences using the underlying
        Hugging Face pipeline's batch functionality.

        :param df: The input data for prediction.
        :type df: :class:`pandas.DataFrame`
        :param column: The name of the column in the DataFrame containing the input text, defaults to 'Sentence'.
        :type column: str, optional
        :return: A list of predicted intensities. Returns None for texts where prediction failed.
        :rtype: :class:`List[Optional[str]]`
        """
        results: List[Optional[str]] = []
        
        if df.empty or column not in df.columns:
            logger.warning("Input DataFrame is empty or missing the target column.")
            return []

        # Step 1: Extract sentences from the DataFrame column
        sentences = df[column].astype(str).tolist()
        if not sentences:
            return []

        try:
            logger.info(f"Starting {self.__class__.__name__} inference for {len(sentences)} sentences.")
            
            # Step 2: Run zero-shot classification on the entire batch
            # self.classifier is the Hugging Face pipeline object
            intensity_results = self.classifier(
                sentences,
                self.intensity_labels, # The fixed candidate labels: ['weak', 'moderate', 'intense']
                multi_label=False # We expect a single top label
            )

            # Step 3: Extract the top label
            # The top label is the first item in the 'labels' list for each result dictionary.
            results = [res['labels'][0] for res in intensity_results]
            logger.info(f"{self.__class__.__name__} Inference completed successfully.")
            
        except Exception as e:
            msg = f'Error processing intensity batch: {str(e)}'
            logger.error(msg)
            # In case of an error, return None for all sentences in the batch
            results = [None] * len(sentences)
            # Raise an error to match the exception handling pattern of the refactored code
            raise RuntimeError(msg) from e 

        return results