# main.py

import logging
import os
from typing import Dict

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
from src.predict.predict_engine import PredictEngine # New Import
import pandas as pd # New Import

from src.config import AUDIO_PATH, TRANSRIPT_PATH, ARTIFACTS_DIR, PREDS_PATH
from src.processing.config import FEATURE_OUTPUT_PATH as FEATURES
from src.translation.config import TRANSLATED_OUTPUT_PATH as TRANSLATIONS


# Define the emotion mapping
EMOTION_MAP: Dict[int, str] = {
    0: "neutral", 1: "anger", 2: "disgust", 3: "fear",
    4: "happiness", 5: "sadness", 6: "surprise"
}

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

    # Initialize and Run PredictEngine
    engine = PredictEngine(
        checkpoint=checkpoint,
        data_source=df,
        column=INPUT_COLUMN
    )
    predictions = engine.predict()
    
    # 4. Post-processing and Saving Results (NEW)
    
    # Convert numerical predictions to emotion labels
    emotions = [EMOTION_MAP.get(pred, "unknown") for pred in predictions]

    df['Core Emotion'] = emotions

    df.to_csv(PREDS_PATH, index=False)
    logger.info(f"Results saved to: {PREDS_PATH}")

    # TEMPORARILY DISABLED
    # # Create an instance of the class
    # feature_engine = FeatureEngine()
    # # Call the create_features method on the instance
    # feature_engine.create_features(output_path=FEATURES) 

if __name__ == '__main__':
    main()