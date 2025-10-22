import os
import pandas as pd
from transformers import pipeline
from typing import List, Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class EmotionFinePredictor:
    """
    A class for fine-grained emotion classification using a zero-shot model.

    The primary method, :meth:`predict_fine_emotion_batch`, handles internal
    grouping of texts by their known 'Core Emotion' (e.g., 'happiness', 'sadness')
    to efficiently batch predictions for the smaller, relevant set of fine-grained 
    candidate labels for that emotion group.
    """
    def __init__(self, model_name: str = 'facebook/bart-large-mnli', device: int = 0, batch_size: int = 128):
        """
        Initializes the Zero-Shot Classifier for fine-grained emotions.

        :param model_name: The name or path of the zero-shot classification model.
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
        logger.info(f"Initializing Fine Emotion Zero-Shot Pipeline on {device_str} using {model_name}.")

        # Initialize the zero-shot-classification pipeline
        self.classifier = pipeline(
            'zero-shot-classification',
            model=model_name,
            device=device,
            batch_size=batch_size # The pipeline handles batching automatically
        )
        
        # Define the mapping from Core Emotion (key) to its fine-grained candidate labels (value)
        # This mapping dictates which labels the zero-shot model considers for a given core emotion.
        self.fine_emotion_map: Dict[str, List[str]] = {
            'happiness': ['anticipation', 'satisfaction', 'gratitude', 'admiration'],
            'sadness': ['disappointment', 'resignation', 'frustration', 'regret'],
            'anger': ['contempt', 'hostility', 'fury', 'irritation'],
            'fear': ['anxiety', 'horror', 'nervousness', 'apprehension'],
            'disgust': ['revulsion', 'aversion', 'loathing', 'repulsion'],
            'surprise': ['amazement', 'shock', 'wonder', 'confusion'],
            'neutral': ['calmness', 'boredom', 'tiredness', 'confusion'] 
        }
        logger.info(f"Fine emotion maps initialized for {len(self.fine_emotion_map)} core emotions.")


    def predict_fine_emotion_batch(self, sentences: List[str], core_emotions: List[str]) -> List[Optional[str]]:
        """
        Classifies the fine-grained emotion for a batch of sentences, conditional
        on their previously predicted core emotion.

        The method groups sentences by their core emotion to optimize zero-shot 
        classification by using a minimal, relevant set of candidate labels per batch.

        :param sentences: The list of texts to classify.
        :type sentences: :class:`List[str]`
        :param core_emotions: The corresponding list of core emotion labels (e.g., 'happiness') 
                              for each sentence. Must be the same length as ``sentences``.
        :type core_emotions: :class:`List[str]`
        :return: A list of predicted fine-grained emotions. Returns 'N/A' or None if prediction fails.
        :rtype: :class:`List[Optional[str]]`
        """
        if len(sentences) != len(core_emotions):
            raise ValueError("Input lists 'sentences' and 'core_emotions' must have the same length.")
        if not sentences:
            return []
            
        # 1. Group input data by core emotion
        # Grouped data: {'happiness': [(sentence, original_index), ...]}
        grouped_data: Dict[str, List[Tuple[str, int]]] = {}
        
        # Initialize results list to be filled in the correct original order
        results: List[Optional[str]] = [None] * len(sentences)

        for i, (sentence, emotion) in enumerate(zip(sentences, core_emotions)):
            emotion_lower = emotion.lower()
            
            # Only process if the core emotion has a corresponding fine-grained map
            if emotion_lower in self.fine_emotion_map:
                if emotion_lower not in grouped_data:
                    grouped_data[emotion_lower] = []
                # Store the sentence and its original index
                grouped_data[emotion_lower].append((sentence, i))
            else:
                # If the core emotion is unknown/unmapped, mark it as not applicable
                results[i] = 'N/A'

        # 2. Process each emotion group in batches
        for emotion, items in grouped_data.items():
            texts = [item[0] for item in items]
            original_indices = [item[1] for item in items]
            
            # Retrieve the specific candidate labels for this core emotion
            fine_labels = self.fine_emotion_map.get(emotion, [])

            if not fine_labels:
                # Should not happen if data is grouped correctly, but serves as a safeguard
                continue

            try:
                logger.debug(f"Classifying {len(texts)} texts for core emotion '{emotion}' with {len(fine_labels)} candidate labels.")
                
                # Run zero-shot classification on the batch using the group's specific labels
                fine_results = self.classifier(
                    texts,
                    fine_labels,
                    multi_label=False # We expect a single top label
                )

                # Ensure results is a list (necessary if the batch size is 1)
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