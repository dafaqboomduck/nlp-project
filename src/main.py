# main.py

import logging
import os
from typing import Dict
from tqdm import tqdm # FIXED: Correctly import the tqdm function
import numpy as np 

# Import the configuration function
from src.helpers.logging_config import configure_logging

# --- CALL THE CONFIGURATION FUNCTION FIRST ---
configure_logging() 
# ---------------------------------------------

# Once the configuration is set, all the loggers 
# will inherit the root logger's handlers.
logger = logging.getLogger(__name__) 

from src.processing import FeatureEngine
from src.transcript import TranscriptEngine
from src.translation import TranslationEngine
from src.predict import EmotionAnalysisManager

from src.config import AUDIO_PATH, TRANSCRIPT_PATH, PREDS_PATH
from src.processing.config import FEATURE_OUTPUT_PATH as FEATURES
from src.translation.config import TRANSLATED_OUTPUT_PATH as TRANSLATIONS

def main():

    # 1. Transcription Step
    # Create an instance of the class
    transcript_engine = TranscriptEngine()
    # Call the transcribe method on the instance
    transcript_engine.transcribe(mp3_file=AUDIO_PATH, output_path=TRANSCRIPT_PATH, post_process='simple')

    # 2. Translation Step
    # Create an instance of the class
    translation_engine = TranslationEngine()
    # Call the create_features method on the instance
    translation_engine.translate(translation_type='round', input_lang='en', interm_lang='nl', output_path=TRANSLATIONS)

    # 2. Emotion Analysis Step
    # Create an instance of the class
    emotion_manager = EmotionAnalysisManager()
    # Call the create_features method on the instance
    emotion_manager.run_emotion_classification()


    # TEMPORARILY DISABLED
    # # Create an instance of the class
    # feature_engine = FeatureEngine()
    # # Call the create_features method on the instance
    # feature_engine.create_features(output_path=FEATURES) 

if __name__ == '__main__':
    main()