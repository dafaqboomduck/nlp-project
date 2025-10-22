import os
import pandas as pd
from transformers import pipeline
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class EmotionFinePredictor:
    """
    A class for fine-grained emotion classification using a zero-shot model.
    """
    def __init__(self, model_name: str = 'facebook/bart-large-mnli', device: int = 0, batch_size: int = 128):
        """
        Initializes the Zero-Shot Classifier for fine-grained emotions.

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
        
        # Emotion mappings from the notebook
        self.fine_emotion_map = {
            'happiness': ['anticipation', 'satisfaction', 'gratitude', 'admiration'],
            'sadness': ['disappointment', 'resignation', 'frustration', 'regret'],
            'fear': ['concern', 'uncertainty', 'apprehension', 'urgency'],
            'anger': ['frustration', 'resentment', 'rejection', 'impatience'],
            'surprise': ['confusion', 'amazement', 'curiosity', 'astonishment'],
            'disgust': ['contempt', 'disappointment', 'disapproval', 'dismissiveness'],
            'neutral': ['curiosity', 'neutrality', 'acceptance', 'uncertainty']
        }

    def predict_fine_emotion(self, sentence: str, general_emotion: str) -> Optional[str]:
        """
        Classifies the fine-grained emotion for a single sentence given its general emotion.

        Args:
            sentence (str): The text to classify.
            general_emotion (str): The general emotion category (e.g., 'fear', 'sadness').

        Returns:
            Optional[str]: The predicted fine-grained emotion, or None if labels are missing.
        """
        try:
            emotion_lower = general_emotion.lower()
            fine_labels = self.fine_emotion_map.get(emotion_lower, [])
            
            if not fine_labels:
                return None
            
            # The classifier naturally handles single sentence input
            fine_result = self.classifier(
                sentence, 
                fine_labels,
                multi_label=False
            )
            
            return fine_result['labels'][0]
        
        except Exception as e:
            logger.error(f'Error processing fine-emotion for: {sentence[:50]}... - {str(e)}')
            return None

    def predict_fine_emotion_batch(self, sentences: List[str], general_emotion: str) -> List[Optional[str]]:
        """
        Classifies the fine-grained emotion for a batch of sentences given a single general emotion.

        Args:
            sentences (List[str]): The texts to classify.
            general_emotion (str): The general emotion category (e.g., 'fear', 'sadness').

        Returns:
            List[Optional[str]]: A list of predicted fine-grained emotions.
        """
        emotion_lower = general_emotion.lower()
        fine_labels = self.fine_emotion_map.get(emotion_lower, [])
        
        if not fine_labels:
            return [None] * len(sentences)

        results = []
        try:
            # Pass all sentences as a batch
            fine_results = self.classifier(
                sentences,
                fine_labels,
                multi_label=False
            )

            # Ensure the output is a list of results
            if isinstance(fine_results, dict):
                fine_results = [fine_results]
            
            results = [res['labels'][0] for res in fine_results]

        except Exception as e:
            logger.error(f'Error processing fine-emotion batch for {general_emotion}: {str(e)}')
            results = [None] * len(sentences)

        return results