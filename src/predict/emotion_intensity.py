import os
import pandas as pd
from transformers import pipeline
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class EmotionIntensityPredictor:
    """
    A class for emotion intensity classification using a zero-shot model,
    designed for efficient batch prediction.
    """
    def __init__(self, model_name: str = 'facebook/bart-large-mnli', device: int = 0, batch_size: int = 128):
        """
        Initializes the Zero-Shot Classifier for intensity.
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        
        device_str = f"cuda:{device}" if device >= 0 else "cpu"
        logger.info(f"Initializing Intensity Zero-Shot Pipeline on {device_str} using {model_name}.")

        self.classifier = pipeline(
            'zero-shot-classification',
            model=model_name,
            device=device,
            batch_size=batch_size
        )

        # Intensity labels from the notebook
        self.intensity_labels = ['weak', 'moderate', 'intense']

    def predict_intensity_batch(self, sentences: List[str]) -> List[Optional[str]]:
        """
        Classifies the intensity for a batch of sentences using the underlying
        Hugging Face pipeline's batch functionality.

        Args:
            sentences (List[str]): The texts to classify.

        Returns:
            List[Optional[str]]: A list of predicted intensities.
        """
        results = []
        try:
            # Pass all sentences as a batch
            intensity_results = self.classifier(
                sentences,
                self.intensity_labels,
                multi_label=False
            )

            # Ensure the output is a list of results
            if isinstance(intensity_results, dict):
                intensity_results = [intensity_results]
            
            # Extract the top label and clean it up (as done in the notebook's approach)
            results = [res['labels'][0].replace(' emotion', '') for res in intensity_results]

        except Exception as e:
            logger.error(f'Error processing intensity batch: {str(e)}')
            results = [None] * len(sentences)

        return results