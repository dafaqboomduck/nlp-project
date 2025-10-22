import os
import pandas as pd
from transformers import pipeline
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class EmotionIntensityPredictor:
    """
    A class for emotion intensity classification using a zero-shot model.
    """
    def __init__(self, model_name: str = 'facebook/bart-large-mnli', device: int = 0, batch_size: int = 128):
        """
        Initializes the Zero-Shot Classifier for intensity.

        Args:
            model_name (str): The name of the zero-shot classification model.
            device (int): The device to run the model on (-1 for CPU, >=0 for GPU).
            batch_size (int): The batch size for classification.
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.classifier = pipeline(
            'zero-shot-classification',
            model=model_name,
            device=device,
            batch_size=batch_size
        )

        # Intensity labels from the notebook
        self.intensity_labels = ['weak', 'moderate', 'intense']

    def predict_intensity(self, sentence: str) -> Optional[str]:
        """
        Classifies the intensity (weak, moderate, intense) for a single sentence.

        Args:
            sentence (str): The text to classify.

        Returns:
            Optional[str]: The predicted intensity, or None on error.
        """
        try:
            # The classifier naturally handles single sentence input
            intensity_result = self.classifier(
                sentence,
                self.intensity_labels,
                multi_label=False
            )
            
            # The notebook output showed a .replace(' emotion', '') might be needed,
            # but standard zero-shot output should just be the label name if no template is used.
            # Assuming the output is one of the labels directly for simplicity, but keeping the cleanup in mind.
            intensity = intensity_result['labels'][0]
            if ' emotion' in intensity:
                intensity = intensity.replace(' emotion', '')

            return intensity
        
        except Exception as e:
            logger.error(f'Error processing intensity for: {sentence[:50]}... - {str(e)}')
            return None

    def predict_intensity_batch(self, sentences: List[str]) -> List[Optional[str]]:
        """
        Classifies the intensity for a batch of sentences.

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
            
            # Note: The notebook example had a `.replace(' emotion', '')` which is included here.
            results = [res['labels'][0].replace(' emotion', '') for res in intensity_results]

        except Exception as e:
            logger.error(f'Error processing intensity batch: {str(e)}')
            results = [None] * len(sentences)

        return results