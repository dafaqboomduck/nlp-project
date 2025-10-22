from tqdm import tqdm
import pandas as pd # Ensure pandas is imported if used outside of method signatures

from src.predict.predict_engine import PredictEngine
from src.predict.emotion_fine import EmotionFinePredictor 
from src.predict.emotion_intensity import EmotionIntensityPredictor
# Assuming config file exists and provides constants
from src.predict.config import CORE_CHECKPOINT, FINE_CHECKPOINT, DATA_PATH, INPUT_COLUMN, EMOTION_MAP 
from src.helpers import CSVHandler
from src.config import PREDS_PATH

import logging
logger = logging.getLogger(__name__) 

class EmotionAnalysisManager:
    # Assuming standard model device and batch sizes for zero-shot predictors
    DEVICE_ID = 0 
    BATCH_SIZE = 128

    def __init__(self, input_col = INPUT_COLUMN, emotion_map = EMOTION_MAP, input_path = DATA_PATH, output_path = PREDS_PATH):
        self.input_col = input_col
        self.emotion_map = emotion_map
        self.input_path = input_path
        self.output_path = output_path

    def _get_data(self, data_path):
        # Read the translated data
        try:
            csv_handler = CSVHandler()
            df = csv_handler.read_csv(path=data_path, delimiter=";")
        except Exception as e:
            logger.error(f"Failed to read translated data from {data_path}. Details: {e}")
            return pd.DataFrame() # Return empty DataFrame on failure

        return df

    def _predict_core(self, df: pd.DataFrame, checkpoint: str, col: str) -> pd.DataFrame:
        # --- Step 3a: Core Emotion Prediction ---
        logger.info("Starting Core Emotion prediction (using PredictEngine)...")
        engine = PredictEngine(
            checkpoint=checkpoint,
            data_source=df,
            column=col
        )
        predictions = engine.predict()

        # Convert numerical predictions to emotion labels
        emotions = [self.emotion_map.get(pred, "unknown") for pred in predictions]
        df['Core Emotion'] = emotions
        logger.info("Core Emotion prediction complete.")

        return df

    def _predict_fine(self, df: pd.DataFrame, checkpoint: str, col: str) -> pd.DataFrame:
        """
        Performs fine-grained emotion and intensity prediction using efficient batch calls.
        """
        logger.info("Starting Fine-Grained Emotion and Intensity prediction (Batch Mode)...")

        # Initialize the new predictors (using a zero-shot model)
        # Note: The checkpoint for fine-grained prediction is the zero-shot model name (e.g., 'facebook/bart-large-mnli')
        fine_predictor = EmotionFinePredictor(
            model_name=checkpoint, 
            device=self.DEVICE_ID, 
            batch_size=self.BATCH_SIZE
        )
        intensity_predictor = EmotionIntensityPredictor(
            model_name=checkpoint, 
            device=self.DEVICE_ID, 
            batch_size=self.BATCH_SIZE
        )

        # 1. Prepare data for batching
        sentences = df[col].tolist()
        core_emotions = df['Core Emotion'].tolist()
        
        # 2. Predict Intensity (Efficient Batch Call)
        # Fix: Call the correct method, predict_intensity_batch, which takes a list.
        logger.info("Predicting Intensity in batch...")
        intensities = intensity_predictor.predict_intensity_batch(sentences)
        
        # 3. Predict Fine Emotion (Efficient Batch Call)
        # Fix: Call the correct method, predict_fine_emotion_batch, which takes two lists.
        logger.info("Predicting Fine Emotion in batch...")
        fine_emotions = fine_predictor.predict_fine_emotion_batch(sentences, core_emotions)

        # Assign new predictions to the DataFrame
        df['Intensity'] = intensities
        df['Fine_Emotion'] = fine_emotions
        logger.info("Fine-Grained Emotion and Intensity prediction complete.")

        return df

    def _save_preds(self, df, output_path):
        # 4. Post-processing and Saving Results
        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to: {output_path}")

    def run_emotion_classification(self, col = INPUT_COLUMN, input_path = str(DATA_PATH), output_path = str(PREDS_PATH), core_checkpoint = CORE_CHECKPOINT, fine_checkpoint = FINE_CHECKPOINT):

        df = self._get_data(input_path)
        if df.empty:
            logger.warning("Aborting emotion classification due to empty data.")
            return

        df = self._predict_core(df, checkpoint=core_checkpoint, col=col)
        df = self._predict_fine(df, checkpoint=fine_checkpoint, col=col)

        self._save_preds(df, output_path)