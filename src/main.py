# main.py

import logging
import os
from typing import Dict
import tqdm

# Import the configuration function
from src.helpers.logging_config import configure_logging

configure_logging() 
logger = logging.getLogger(__name__) 

from src.processing import FeatureEngine
from src.transcript import TranscriptEngine
from src.translation import TranslationEngine
from src.predict.predict_engine import PredictEngine
from src.predict.emotion_fine import EmotionFinePredictor 
from src.predict.emotion_intensity import EmotionIntensityPredictor
import pandas as pd

from src.config import AUDIO_PATH, TRANSRIPT_PATH, ARTIFACTS_DIR, PREDS_PATH
from src.processing.config import FEATURE_OUTPUT_PATH as FEATURES
from src.translation.config import TRANSLATED_OUTPUT_PATH as TRANSLATIONS


# Define the emotion mapping
EMOTION_MAP: Dict[int, str] = {
    0: "neutral", 1: "anger", 2: "disgust", 3: "fear",
    4: "happiness", 5: "sadness", 6: "surprise"
}

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Define the zero-shot model checkpoint used for fine-grained classification
ZERO_SHOT_CHECKPOINT = 'facebook/bart-large-mnli'

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

    # 3. Core Prediction Step   
    checkpoint = r"artifacts/bert-base" 
    data_path = TRANSLATIONS
    INPUT_COLUMN = "Round_Trip_Translation_EN"

    # Read the translated data
    try:
        # Assuming you've already run transcription and translation steps
        # For a full run, ensure TRANSLATIONS path exists.
        df = pd.read_csv(data_path, quotechar='\"', sep=';')
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

    # --- Step 3b: Fine-Grained Emotion and Intensity Prediction ---
    logger.info("Starting Fine-Grained Emotion and Intensity prediction...")

    # Initialize the new predictors (using a zero-shot model)
    # Note: Setting device=0 for GPU, change to device=-1 for CPU
    FINE_PREDICTOR = EmotionFinePredictor(model_name=ZERO_SHOT_CHECKPOINT, device=0)
    INTENSITY_PREDICTOR = EmotionIntensityPredictor(model_name=ZERO_SHOT_CHECKPOINT, device=0)
    
    # Prepare lists for new columns
    fine_emotions = []
    intensities = []

    # Iterate through the DataFrame to apply classification
    # This loop processes row-by-row, which is simpler but less performant than batching.
    # The dedicated batch methods in the predictor classes are better for large datasets.
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Predicting Fine Emotion & Intensity"):
        sentence = row[INPUT_COLUMN]
        core_emotion = row['Core Emotion']
        
        # 1. Predict Intensity (Independent of Core Emotion)
        intensity = INTENSITY_PREDICTOR.predict_intensity(sentence)
        intensities.append(intensity)
        
        # 2. Predict Fine Emotion (Dependent on Core Emotion)
        # We only predict fine emotion if a known core emotion is present
        if core_emotion.lower() in FINE_PREDICTOR.fine_emotion_map:
            fine_emotion = FINE_PREDICTOR.predict_fine_emotion(sentence, core_emotion)
        else:
            fine_emotion = 'N/A' # Or other placeholder if Core Emotion is 'unknown'
            
        fine_emotions.append(fine_emotion)

    # Assign new predictions to the DataFrame
    df['Intensity'] = intensities
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