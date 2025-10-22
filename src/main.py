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
from src.predict.predict_engine import PredictEngine 
import pandas as pd 
# NEW IMPORTS
from src.predict.emotion_fine import EmotionFinePredictor 
from src.predict.emotion_intensity import EmotionIntensityPredictor

from src.config import AUDIO_PATH, TRANSRIPT_PATH, ARTIFACTS_DIR, PREDS_PATH
from src.processing.config import FEATURE_OUTPUT_PATH as FEATURES
from src.translation.config import TRANSLATED_OUTPUT_PATH as TRANSLATIONS


# Define the emotion mapping
EMOTION_MAP: Dict[int, str] = {
    0: "neutral", 1: "anger", 2: "disgust", 3: "fear",
    4: "happiness", 5: "sadness", 6: "surprise"
}

# Define the zero-shot model checkpoint used for fine-grained classification
ZERO_SHOT_CHECKPOINT = 'facebook/bart-large-mnli'
# Define the device (0 for GPU, -1 for CPU)
DEVICE_ID = 0 
BATCH_SIZE = 128


os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def main():

    # 1. Transcription Step
    # Create an instance of the class
    transcript_engine = TranscriptEngine()
    # Call the transcribe method on the instance
    transcript_engine.transcribe(mp3_file=AUDIO_PATH, output_path=TRANSRIPT_PATH, post_process='simple')

    # 2. Translation Step
    # Create an instance of the class
    translation_engine = TranslationEngine()
    # Call the create_features method on the instance
    translation_engine.translate(translation_type='round', input_lang='en', interm_lang='nl', output_path=TRANSLATIONS)

    # 3. Prediction Step   
    # Define paths and columns
    checkpoint = r"artifacts/bert-base" 
    data_path = TRANSLATIONS
    INPUT_COLUMN = "Round_Trip_Translation_EN"

    # Read the translated data
    try:
        df = pd.read_csv(data_path, quotechar='"', sep=';')
    except Exception as e:
        logger.error(f"Failed to read translated data from {data_path}. Details: {e}")
        return

    # --- Step 3a: Core Emotion Prediction ---
    logger.info("Starting Core Emotion prediction (using PredictEngine)...")
    engine = PredictEngine(
        checkpoint=checkpoint,
        data_source=df,
        column=INPUT_COLUMN
    )
    predictions = engine.predict()
    
    # Convert numerical predictions to emotion labels
    emotions = [EMOTION_MAP.get(pred, "unknown") for pred in predictions]
    df['Core Emotion'] = emotions
    logger.info("Core Emotion prediction complete.")

    # --- Step 3b: Fine-Grained Emotion and Intensity Prediction (Batch Processing) ---
    logger.info("Starting Fine-Grained Emotion and Intensity prediction (Batch Mode)...")

    # Initialize the new predictors (using zero-shot model and GPU if available)
    FINE_PREDICTOR = EmotionFinePredictor(
        model_name=ZERO_SHOT_CHECKPOINT, 
        device=DEVICE_ID, 
        batch_size=BATCH_SIZE
    )
    INTENSITY_PREDICTOR = EmotionIntensityPredictor(
        model_name=ZERO_SHOT_CHECKPOINT, 
        device=DEVICE_ID, 
        batch_size=BATCH_SIZE
    )
    
    sentences = df[INPUT_COLUMN].tolist()
    core_emotions = df['Core Emotion'].tolist()
    
    # 1. Predict Intensity (Efficient Batch Call)
    logger.info("Predicting Intensity in batch...")
    intensities = INTENSITY_PREDICTOR.predict_intensity_batch(sentences)
    df['Intensity'] = intensities
    
    # 2. Predict Fine Emotion (Efficient Batch Call)
    # This predictor groups sentences by core emotion internally for optimization
    logger.info("Predicting Fine Emotion in batch...")
    fine_emotions = FINE_PREDICTOR.predict_fine_emotion_batch(sentences, core_emotions)
    df['Fine_Emotion'] = fine_emotions
    
    logger.info("Fine-Grained Emotion and Intensity prediction complete.")


    # 4. Post-processing and Saving Results
    
    df.to_csv(PREDS_PATH, index=False)
    logger.info(f"Results saved to: {PREDS_PATH}")

    # TEMPORARILY DISABLED
    # # Create an instance of the class
    # feature_engine = FeatureEngine()
    # # Call the create_features method on the instance
    # feature_engine.create_features(output_path=FEATURES) 

if __name__ == '__main__':
    main()