from tqdm import tqdm
import pandas as pd
from typing import Dict, Any

from src.predict.emotion_core import PredictEngine
from src.predict.emotion_fine import EmotionFinePredictor 
from src.predict.emotion_intensity import EmotionIntensityPredictor
# Assuming config file exists and provides constants
from src.predict.config import CORE_CHECKPOINT, FINE_CHECKPOINT, DATA_PATH, INPUT_COLUMN, EMOTION_MAP 
# Assuming CSVHandler is a necessary external utility for data loading
from src.helpers import CSVHandler 
from src.config import PREDS_PATH

import logging
logger = logging.getLogger(__name__) 

class EmotionAnalysisManager:
    """
    Orchestrates the multi-stage emotion classification pipeline:
    1. Core Emotion (fine-tuned BERT/Transformer)
    2. Fine-Grained Emotion (Zero-shot classification)
    3. Intensity (Zero-shot classification)
    """
    # Use class-level constants for shared predictor parameters
    DEVICE_ID = 0 
    BATCH_SIZE = 128
    
    # New: Inject dependencies or config constants here
    def __init__(self, emotion_map: Dict[int, str] = EMOTION_MAP, device_id: int = DEVICE_ID, batch_size: int = BATCH_SIZE):
        """
        Initializes the manager with core configuration.
        """
        self.emotion_map = emotion_map
        self.device_id = device_id
        self.batch_size = batch_size
        self.csv_handler = CSVHandler() # Explicitly initialize dependency

    def _get_data(self, data_path: str) -> pd.DataFrame:
        """
        Reads the input data from a path.
        """
        try:
            # Assumes CSVHandler.read_csv is the correct interface
            df = self.csv_handler.read_csv(path=data_path, delimiter=";")
        except Exception as e:
            logger.error(f"Failed to read data from {data_path}. Details: {e}")
            return pd.DataFrame() 

        return df

    def _predict_core(self, df: pd.DataFrame, checkpoint: str, col: str) -> pd.DataFrame:
        """
        Performs Core Emotion prediction using PredictEngine.
        """
        logger.info("Starting Core Emotion prediction (using PredictEngine)...")
        # PredictEngine now only takes checkpoint and device in __init__
        engine = PredictEngine(
            checkpoint=checkpoint,
            device=self.device_id if self.device_id >= 0 else 'cpu' 
        )
        
        # PredictEngine.predict now takes the DataFrame and column
        predictions = engine.predict(df=df, column=col)

        # Convert numerical predictions to emotion labels
        emotions = [self.emotion_map.get(pred, "unknown") for pred in predictions]
        # It's cleaner to return a new DataFrame/Series and let the caller assign it.
        # But for state management in the pipeline, we keep the original assignment.
        df['Core Emotion'] = emotions
        logger.info("Core Emotion prediction complete.")

        return df

    def _predict_fine(self, df: pd.DataFrame, fine_checkpoint: str, col: str) -> pd.DataFrame:
        """
        Performs fine-grained emotion and intensity prediction using zero-shot models.
        """
        logger.info("Starting Fine-Grained Emotion and Intensity prediction (Batch Mode)...")

        # Initialize the zero-shot predictors
        fine_predictor = EmotionFinePredictor(
            model_name=fine_checkpoint, 
            device=self.device_id, 
            batch_size=self.batch_size
        )
        intensity_predictor = EmotionIntensityPredictor(
            model_name=fine_checkpoint, 
            device=self.device_id, 
            batch_size=self.batch_size
        )

        # 1. Prepare data for batching
        # Check if 'Core Emotion' exists from the previous step
        if 'Core Emotion' not in df.columns:
            logger.error("Core Emotion column missing. Cannot proceed with fine-grained analysis.")
            return df
            
        sentences = df[col].tolist()
        core_emotions = df['Core Emotion'].tolist()
        
        # 2. Predict Intensity
        logger.info("Predicting Intensity in batch...")
        intensities = intensity_predictor.predict_intensity_batch(sentences)
        
        # 3. Predict Fine Emotion
        logger.info("Predicting Fine Emotion in batch...")
        fine_emotions = fine_predictor.predict_fine_emotion_batch(sentences, core_emotions)

        # Assign new predictions to the DataFrame
        df['Intensity'] = intensities
        df['Fine_Emotion'] = fine_emotions
        logger.info("Fine-Grained Emotion and Intensity prediction complete.")

        return df

    def _save_preds(self, df: pd.DataFrame, output_path: str):
        """
        Saves the results DataFrame to a CSV file.
        """
        try:
            # Ensure index=False is used to prevent writing row numbers
            df.to_csv(output_path, index=False)
            logger.info(f"Results saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save results to {output_path}. Details: {e}")


    def run_emotion_classification(self, 
                                   input_path: str = str(DATA_PATH), 
                                   output_path: str = str(PREDS_PATH), 
                                   input_column: str = INPUT_COLUMN, 
                                   core_checkpoint: str = CORE_CHECKPOINT, 
                                   fine_checkpoint: str = FINE_CHECKPOINT):
        """
        Main method to execute the entire emotion classification pipeline.
        This method uses the default constants from config.py if no arguments are provided.
        """
        logger.info(f"--- Starting Emotion Classification Pipeline for {input_path} ---")

        # 1. Load Data
        df = self._get_data(input_path)
        if df.empty:
            logger.warning("Aborting emotion classification due to empty data or failed load.")
            return

        # 2. Predict Core Emotion
        df = self._predict_core(df, checkpoint=core_checkpoint, col=input_column)
        
        # 3. Predict Fine Emotion and Intensity
        df = self._predict_fine(df, fine_checkpoint=fine_checkpoint, col=input_column)

        # 4. Save Results
        self._save_preds(df, output_path)
        
        logger.info("--- Emotion Classification Pipeline Completed ---")