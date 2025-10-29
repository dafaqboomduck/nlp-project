from tqdm import tqdm
import pandas as pd
from typing import Dict, Any
import torch

from src.predict.emotion_core import EmotionCorePredictor
from src.predict.emotion_fine import EmotionFinePredictor 
from src.predict.emotion_intensity import EmotionIntensityPredictor
from src.predict.config import CORE_CHECKPOINT, FINE_CHECKPOINT, DATA_PATH
# Assuming CSVHandler is an essential external utility for data loading
from src.helpers import CSVHandler 
from src.config import PREDS_PATH

import logging
logger = logging.getLogger(__name__) 

class PredictEngine:
    """
    Orchestrates the multi-stage emotion classification pipeline:

    1. Core Emotion Classification (Custom fine-tuned Transformer via :class:`EmotionCorePredictor`)
    2. Fine-Grained Emotion Classification (Zero-shot via :class:`EmotionFinePredictor`)
    3. Intensity Classification (Zero-shot via :class:`EmotionIntensityPredictor`)

    The manager is designed to be configured and run via its single public method,
    :meth:`run_emotion_classification`, using config constants as defaults.
    """
    
    def __init__(self, device_id: str = None, batch_size: int = 128):
        """
        Initializes the manager with core configuration settings.

        :param device_id: The device to run inference on (e.g., 'cuda', 'cuda:0', 'cpu').
                       If None, automatically selects GPU if available, otherwise CPU.
        :type device_id: str, optional
        :param batch_size: The batch size for zero-shot predictors.
        :type batch_size: int
        """
        self.batch_size = batch_size
        self.csv_handler = CSVHandler() 

        # --- Device Setup ---
        if device_id is None:
            is_cuda = torch.cuda.is_available()
            self.device_id = 'cuda' if is_cuda else 'cpu'
            if not is_cuda:
                logger.warning("No CUDA device found. Falling back to CPU.")
        elif 'cuda' in device_id.lower() and not torch.cuda.is_available():
            logger.error(f"Requested device '{device_id}', but no CUDA device is available.")
            raise RuntimeError("CUDA device requested but not available.")
        else:
            self.device_id = device_id

    def _get_data(self, data_path: str) -> pd.DataFrame:
        """
        Reads the input data from a path using the internal CSVHandler dependency.

        :param data_path: The file path to the input CSV file.
        :type data_path: str
        :return: The loaded DataFrame, or an empty DataFrame if loading fails.
        :rtype: :class:`pandas.DataFrame`
        """
        logger.debug(f"Attempting to read data from: {data_path}")
        try:
            # Assumes CSVHandler.read_csv is the correct interface
            df = self.csv_handler.read_csv(path=data_path, delimiter=";")
            logger.debug(f"Data loaded successfully. Rows: {len(df)}")
        except Exception as e:
            logger.error(f"Failed to read data from {data_path}. Details: {e}")
            return pd.DataFrame() 

        return df

    def _predict_core(self, df: pd.DataFrame, checkpoint: str, col: str) -> pd.DataFrame:
        """
        Performs Core Emotion prediction using the fine-tuned Transformer model.

        :param df: The input DataFrame containing the text to classify.
        :type df: :class:`pandas.DataFrame`
        :param checkpoint: The model checkpoint/path for the Core Emotion model.
        :type checkpoint: str
        :param col: The name of the column containing the input text.
        :type col: str
        :return: The DataFrame with the new 'Core Emotion' column added.
        :rtype: :class:`pandas.DataFrame`
        """
        logger.info(f"Starting Core Emotion prediction with checkpoint: {checkpoint}")
        
        # 1. Initialize the PredictEngine
        engine = EmotionCorePredictor(
            checkpoint=checkpoint,
            device=self.device_id,
            batch_size=self.batch_size
        )
        
        # 2. Run prediction
        predictions = engine.predict(df=df, column=col)

        # 3. Post-process and assign results
        # Convert numerical predictions (0-6) to emotion labels ('neutral', 'anger', etc.)
        emotions = [engine.core_emotions.get(pred, "unknown") for pred in predictions]
        df['Core Emotion'] = emotions
        logger.info("Core Emotion prediction complete.")

        return df

    def _predict_fine(self, df: pd.DataFrame, fine_checkpoint: str, col: str) -> pd.DataFrame:
        """
        Performs fine-grained emotion and intensity prediction using zero-shot models.

        This process utilizes the efficient batch methods of the zero-shot predictors.

        :param df: The DataFrame (must contain the 'Core Emotion' column).
        :type df: :class:`pandas.DataFrame`
        :param fine_checkpoint: The model checkpoint for the zero-shot models.
        :type fine_checkpoint: str
        :param col: The name of the column containing the input text.
        :type col: str
        :return: The DataFrame with the new 'Intensity' and 'Fine_Emotion' columns added.
        :rtype: :class:`pandas.DataFrame`
        """
        logger.info(f"Starting Fine-Grained/Intensity prediction with checkpoint: {fine_checkpoint}")

        # Check dependency on the previous step
        if 'Core Emotion' not in df.columns:
            logger.error("Core Emotion column missing. Cannot proceed with fine-grained analysis.")
            return df
        
        try:    
            # Initialize the zero-shot predictors with common settings
            fine_predictor = EmotionFinePredictor(
                checkpoint=fine_checkpoint, 
                device=self.device_id, 
                batch_size=self.batch_size
            )
            intensity_predictor = EmotionIntensityPredictor(
                checkpoint=fine_checkpoint, 
                device=self.device_id, 
                batch_size=self.batch_size
            )
        except Exception as e:
            msg = f"Error initializing Fine Emotion or Intensity Predictors. Details: {e}"
            logger.error(msg, exc_info=True)
            raise RuntimeError(msg) from e
        
        try:
            # 1. Predict Intensity (Independent zero-shot classification)
            logger.info("Predicting Intensity in batch...")
            intensities = intensity_predictor.predict(df, col)
            
            # 2. Predict Fine Emotion (Zero-shot classification conditional on Core Emotion)
            logger.info("Predicting Fine Emotion in batch...")
            fine_emotions = fine_predictor.predict(df, col)
        except Exception as e:
            msg = f"Error predicting Fine Emotion or Intensity. Details: {e}"
            logger.error(msg, exc_info=True)
            raise RuntimeError(msg) from e
        
        # Assign new predictions to the DataFrame
        df['Intensity'] = intensities
        df['Fine_Emotion'] = fine_emotions
        logger.info("Fine-Grained Emotion and Intensity prediction complete.")

        return df

    def _save_preds(self, df: pd.DataFrame, output_path: str):
        """
        Saves the results DataFrame to a CSV file.

        :param df: The final DataFrame containing all prediction results.
        :type df: :class:`pandas.DataFrame`
        :param output_path: The file path where the results should be saved.
        :type output_path: str
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
                                   input_column: str = "Round_Trip_Translation_EN", 
                                   core_checkpoint: str = CORE_CHECKPOINT, 
                                   fine_checkpoint: str = FINE_CHECKPOINT):
        """
        Main public method to execute the entire emotion classification pipeline.

        This method orchestrates data loading, the two-stage prediction process, and saving.
        It uses default constants from config.py unless explicit arguments are provided.

        :param input_path: File path to the input data (CSV).
        :type input_path: str
        :param output_path: File path to save the final results (CSV).
        :type output_path: str
        :param input_column: The column name in the input data that contains the text.
        :type input_column: str
        :param core_checkpoint: Checkpoint for the core emotion (fine-tuned) model.
        :type core_checkpoint: str
        :param fine_checkpoint: Checkpoint for the zero-shot (fine-grained and intensity) model.
        :type fine_checkpoint: str
        """
        logger.info(f"--- Starting Emotion Classification Pipeline for {input_path} ---")

        # 1. Load Data
        df = self._get_data(input_path)
        if df.empty:
            logger.warning("Aborting emotion classification due to empty data or failed load.")
            return

        # 2. Predict Core Emotion (Stage 1)
        df = self._predict_core(df, checkpoint=core_checkpoint, col=input_column)
        
        # 3. Predict Fine Emotion and Intensity (Stage 2)
        df = self._predict_fine(df, fine_checkpoint=fine_checkpoint, col=input_column)

        # 4. Save Results
        self._save_preds(df, output_path)
        
        logger.info("--- Emotion Classification Pipeline Completed ---")