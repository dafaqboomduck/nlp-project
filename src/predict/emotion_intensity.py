import os
import pandas as pd
from transformers import pipeline
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class EmotionIntensityPredictor:
    """
    A class for emotion intensity classification using a zero-shot model,
    designed for efficient batch prediction via the Hugging Face pipeline.

    It classifies input text into a fixed set of intensity labels: 'weak', 
    'moderate', or 'intense'.
    """
    def __init__(self, model_name: str = 'facebook/bart-large-mnli', device: int = 0, batch_size: int = 128):
        """
        Initializes the Zero-Shot Classifier for intensity prediction.

        :param model_name: The name or path of the zero-shot classification model (e.g., 'facebook/bart-large-mnli').
        :type model_name: str
        :param device: The device index to run the pipeline on (0 for cuda:0, -1 for CPU).
        :type device: int
        :param batch_size: The batch size for pipeline inference.
        :type batch_size: int
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        
        device_str = f"cuda:{device}" if device >= 0 else "cpu"
        logger.info(f"Initializing Intensity Zero-Shot Pipeline on {device_str} using {model_name}.")

        # Initialize the zero-shot-classification pipeline from Hugging Face
        self.classifier = pipeline(
            'zero-shot-classification',
            model=model_name,
            device=device, # This maps to the device index
            batch_size=batch_size
        )

        # Define the fixed candidate labels for intensity
        self.intensity_labels = ['weak', 'moderate', 'intense']
        logger.info(f"Intensity candidate labels: {self.intensity_labels}")


    def predict_intensity_batch(self, sentences: List[str]) -> List[Optional[str]]:
        """
        Classifies the intensity for a batch of sentences using the underlying
        Hugging Face pipeline's batch functionality.

        :param sentences: The texts to classify.
        :type sentences: :class:`List[str]`
        :return: A list of predicted intensities. Returns None for texts where prediction failed.
        :rtype: :class:`List[Optional[str]]`
        """
        results: List[Optional[str]] = []
        if not sentences:
            return []
            
        try:
            # Run zero-shot classification on the entire batch
            intensity_results = self.classifier(
                sentences,
                self.intensity_labels,
                multi_label=False # We expect a single top label
            )

            # Ensure the output is a list of results (for consistency)
            if isinstance(intensity_results, dict):
                intensity_results = [intensity_results]
            
            # Extract the top label
            # Note: No ' emotion' suffix cleanup is typically needed for these labels
            results = [res['labels'][0] for res in intensity_results]

        except Exception as e:
            logger.error(f'Error processing intensity batch: {str(e)}')
            # In case of an error, return None for all sentences in the batch
            results = [None] * len(sentences)

        return results