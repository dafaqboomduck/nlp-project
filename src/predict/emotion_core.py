from src.predict.preprocessing import InferencePreprocessor
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import pandas as pd
import torch
import logging

logger = logging.getLogger(__name__)


class PredictEngine:
    """
    A class responsible for handling model loading, preprocessing orchestration,
    and performing inference using Hugging Face's :class:`~transformers.Trainer` API.

    This engine is designed for efficient batch inference with transformer-based
    sequence classification models. It requires a model checkpoint and
    a pre-loaded :class:`pandas.DataFrame` for prediction.
    """

    def __init__(self, 
                 checkpoint: str, 
                 device: str = None):
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

        try:
            # Determine device: Use provided device, or auto-select
            if device is None:
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                self.device = device

            logger.info(f"Initializing PredictEngine with model: {self.checkpoint}")
            logger.info(f"Device selected for inference: {self.device}")

            # Initialize Preprocessor: Handles tokenization logic
            self.preprocessor = InferencePreprocessor(self.checkpoint, max_length=128)

            # Load Model and move it to the selected device
            logger.info("Loading model from checkpoint...")
            self.model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint).to(self.device)
            self.model.eval() # Set model to evaluation mode
            logger.info("Model successfully loaded.")

            # Configure TrainingArguments for inference only
            training_args = TrainingArguments(
                output_dir="./tmp_trainer_output",
                per_device_eval_batch_size=32,
                dataloader_drop_last=False,
                no_cuda=self.device == 'cpu',
                logging_dir="./logs",
                report_to="none",
                disable_tqdm=True # Disable TQDM for clean log output
            )
            
            # Initialize Trainer: The main batch prediction utility
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                tokenizer=self.preprocessor.get_tokenizer(),
                data_collator=self.preprocessor.data_collator,
            )
            logger.info("Trainer initialized successfully.")

        except Exception as e:
            msg = f"Failed to initialize PredictEngine. Details: {str(e)}"
            logger.error(msg)
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
            logger.info("Starting prediction pipeline...")

            # Step 1: Preprocess data
            # The preprocessor handles conversion and tokenization from DataFrame to Dataset.
            logger.info("Starting data preprocessing...")
            tokenized_data, data_collator = self.preprocessor.preprocess(df, column) 
            logger.info("Data successfully preprocessed.")

            # Step 2: Run prediction
            logger.info("Performing batch inference using Trainer...")
            # Prediction uses the configured Trainer, which handles batching and device placement.
            predictions = self.trainer.predict(tokenized_data)
            logger.info("Model inference completed successfully.")

            # Step 3: Postprocess results
            final_predictions = self._postprocess(predictions)
            logger.info("Prediction pipeline completed successfully.")

            return final_predictions

        except Exception as e:
            msg = f"Prediction pipeline failed: {str(e)}"
            logger.error(msg)
            raise RuntimeError(msg) from e