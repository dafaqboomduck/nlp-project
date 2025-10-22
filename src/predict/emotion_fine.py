import os
import pandas as pd
from transformers import pipeline
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class EmotionFinePredictor:
    """
    A class for fine-grained emotion classification using a zero-shot model.
    The predict_fine_emotion_batch method handles internal grouping by core emotion
    to efficiently batch predictions for the same set of candidate labels.
    """
    def __init__(self, model_name: str = 'facebook/bart-large-mnli', device: int = 0, batch_size: int = 128):
        """
        Initializes the Zero-Shot Classifier for fine-grained emotions.
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        
        device_str = f"cuda:{device}" if device >= 0 else "cpu"
        logger.info(f"Initializing Fine Emotion Zero-Shot Pipeline on {device_str} using {model_name}.")

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

    def predict_fine_emotion_batch(self, sentences: List[str], general_emotions: List[str]) -> List[Optional[str]]:
        """
        Classifies the fine-grained emotion for a batch of sentences.
        """
        if len(sentences) != len(general_emotions):
            raise ValueError("Sentences and general_emotions lists must have the same length.")

        results = [None] * len(sentences)
        grouped_data = {}
        
        # 1. Group sentences by their Core Emotion
        for i, (sentence, emotion) in enumerate(zip(sentences, general_emotions)):
            emotion_lower = emotion.lower()
            if emotion_lower in self.fine_emotion_map:
                if emotion_lower not in grouped_data:
                    grouped_data[emotion_lower] = []
                
                # Store (sentence, original_index) tuple
                grouped_data[emotion_lower].append((sentence, i))
            else:
                results[i] = 'N/A'
        
        # 2. Process each group in batches
        for emotion, items in grouped_data.items():
            texts = [item[0] for item in items]
            original_indices = [item[1] for item in items]
            fine_labels = self.fine_emotion_map.get(emotion, [])

            if not fine_labels:
                continue

            try:
                # Classify the batch of sentences for this emotion group
                fine_results = self.classifier(
                    texts,
                    fine_labels,
                    multi_label=False
                )

                # Ensure results is a list
                if not isinstance(fine_results, list):
                    fine_results = [fine_results]
                
                # Fill the results list in the correct original position
                for i, res in enumerate(fine_results):
                    original_pos = original_indices[i]
                    results[original_pos] = res['labels'][0]

            except Exception as e:
                logger.error(f'Error processing fine-emotion batch for {emotion}: {str(e)}')
                # If a batch fails, mark all in that batch as None
                for original_pos in original_indices:
                    results[original_pos] = None
            
        return results