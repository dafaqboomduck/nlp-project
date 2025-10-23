from src.predict.preprocessing import InferencePreprocessor
from src.predict.config import EMOTION_MAP
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import pandas as pd
import torch
import logging

logger = logging.getLogger(__name__)


class EmotionCorePredictor:
    """
    A class responsible for handling model loading, preprocessing orchestration,
    and performing inference using Hugging Face's :class:`~transformers.Trainer` API.

    This engine is designed for efficient batch inference with transformer-based
    sequence classification models. It requires a model checkpoint and
    a pre-loaded :class:`pandas.DataFrame` for prediction.
    """

    def __init__(self, 
                 checkpoint: str, 
                 device: str = None,
                 batch_size: int = 128,
                 labels_map: dict | list = EMOTION_MAP):
        """
        Initializes the prediction engine by loading the model and tokenizer.

        :param checkpoint: The path or Hugging Face model hub identifier of the pretrained model checkpoint.
        :type checkpoint: str
        :param device: The device to run inference on (e.g., 'cuda', 'cuda:0', 'cpu').
                       If None, automatically selects GPU if available, otherwise CPU.
        :type device: str, optional
        :raises RuntimeError: If model loading or Trainer initialization fails.
        """
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.labels_map = labels_map

        try:
            # --- Device Setup ---
            if device is None:
                is_cuda = torch.cuda.is_available()
                self.device = 'cuda' if is_cuda else 'cpu'
                if not is_cuda:
                    logger.warning("No CUDA device found. Falling back to CPU.")
            elif 'cuda' in device.lower() and not torch.cuda.is_available():
                logger.error(f"Requested device '{device}', but no CUDA device is available.")
                raise RuntimeError("CUDA device requested but not available.")
            else:
                self.device = device

            logger.info(f"Initializing {self.__class__.__name__} with model: {self.checkpoint}")
            logger.info(f"Device selected for inference: {self.device}")

            # Initialize Preprocessor (Handles its own exceptions)
            self.preprocessor = InferencePreprocessor(self.checkpoint, max_length=128) 

            # --- Model Loading ---
            logger.info(f"Loading model from checkpoint '{self.checkpoint}'...")
            try:
                self.model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint).to(self.device)
                logger.info("Model successfully loaded.")
            except Exception as e:
                msg = f"Failed to load model from checkpoint {self.checkpoint}. See details: {e}"
                logger.error(msg)
                raise ImportError(msg)
            
            # Define the fixed candidate labels for core emotions
            self.core_emotions = self.labels_map
            logger.debug(f"Core emotions candidate labels: {self.core_emotions}")

            # Set model to evaluation mode
            self.model.eval() 

            # --- Trainer Setup ---
            training_args = TrainingArguments(
                output_dir="./tmp_trainer_output",
                per_device_eval_batch_size=batch_size,
                dataloader_drop_last=False,
                no_cuda=self.device == 'cpu',
                report_to="none",
                disable_tqdm=True 
            )
            
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                processing_class=self.preprocessor.get_tokenizer(), 
                data_collator=self.preprocessor.data_collator, 
            )
            logger.info("Hugging Face Trainer initialized successfully.")

        except RuntimeError:
            raise
        except Exception as e:
            msg = f"Failed to initialize {self.__class__.__name__} due to model/library error. Details: {str(e)}"
            logger.error(msg, exc_info=True)
            raise RuntimeError(msg) from e
        
    def _postprocess(self, predictions):
        """
        Internal method to convert raw model output logits into discrete class predictions
        using argmax.

        :param predictions: The output object from :meth:`~transformers.Trainer.predict`.
        :type predictions: :class:`~transformers.PredictionOutput`
        :return: A NumPy array containing the predicted class indices.
        :rtype: :class:`numpy.ndarray`
        """
        try:
            logger.info("Postprocessing predictions...")
            # Apply argmax across the last dimension (the logits for each class)
            preprocessed_preds = np.argmax(predictions.predictions, axis=-1)
            logger.info("Predictions successfully postprocessed.")
            return preprocessed_preds

        except Exception as e:
            msg = f"Error during postprocessing: {str(e)}"
            logger.error(msg)
            raise RuntimeError(msg) from e

    def predict(self, df: pd.DataFrame, column: str = 'Sentence'):
        """
        Executes the full inference pipeline for a given DataFrame.

        Steps:
        1. Preprocesses the input data (DataFrame) using :class:`InferencePreprocessor`.
        2. Performs batched prediction using the internal :class:`~transformers.Trainer`.
        3. Postprocesses the logits into final class predictions.

        :param df: The input data for prediction.
        :type df: :class:`pandas.DataFrame`
        :param column: The name of the column in the DataFrame containing the input text, defaults to 'Sentence'.
        :type column: str, optional
        :raises RuntimeError: If any step of the prediction pipeline fails.
        :return: A NumPy array containing the predicted class indices for each input sample.
        :rtype: :class:`numpy.ndarray`
        """
        try:
            logger.info(f"Starting {self.__class__.__name__} Inference...")

            # Step 1: Preprocess data
            # The preprocessor handles conversion and tokenization from DataFrame to Dataset.
            logger.debug("Starting data preprocessing...")
            tokenized_data, _ = self.preprocessor.preprocess(df, column) 
            logger.debug("Data successfully preprocessed.")

            # Step 2: Run prediction
            logger.debug("Performing batch inference using Trainer...")
            # Prediction uses the configured Trainer, which handles batching and device placement.
            predictions = self.trainer.predict(tokenized_data)

            # Step 3: Postprocess results
            final_predictions = self._postprocess(predictions)
            logger.info(f"{self.__class__.__name__} Inference completed successfully.")

            return final_predictions

        except Exception as e:
            msg = f"{self.__class__.__name__} Inference failed: {str(e)}"
            logger.error(msg)
            raise RuntimeError(msg) from e