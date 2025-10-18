from src.predict.preprocessing import InferencePreprocessor
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import pandas as pd
import torch  # For device handling
import logging

logger = logging.getLogger(__name__)


class PredictEngine:
    """
    A class responsible for handling model loading, data preprocessing,
    and performing inference using Hugging Face's `Trainer` API.

    This engine is designed for efficient batch inference with transformer-based
    sequence classification models. It supports automatic device selection,
    customizable preprocessing, and returns class predictions as NumPy arrays.
    """

    def __init__(self, 
                 checkpoint: str, 
                 data_source: str | pd.DataFrame, 
                 column: str = 'Sentence', 
                 device: str = None,
                 output_path: str = None):
        """
        Initializes the prediction engine.

        Parameters
        ----------
        checkpoint : str
            The path or Hugging Face model hub identifier of the pretrained model checkpoint.
        data_source : str | pandas.DataFrame
            The input data for prediction. This can be either:
            - A path to a file (e.g., CSV) supported by the preprocessor.
            - A pandas DataFrame containing the data directly.
        column : str, optional
            The name of the column in the dataset containing the input text.
            Defaults to 'Sentence'.
        device : str, optional
            The device to run inference on (e.g., 'cuda', 'cuda:0', 'cpu').
            If not provided, automatically selects GPU if available, otherwise CPU.

        Notes
        -----
        - The `InferencePreprocessor` is expected to handle tokenizer initialization.
        - The Hugging Face `Trainer` is configured for efficient batch inference.
        """
        self.checkpoint = checkpoint
        self.data_source = data_source
        self.column = column

        try:
            # Determine device
            if device is None:
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                self.device = device

            logger.info(f"Initializing PredictEngine with model: {self.checkpoint}")
            logger.info(f"Device selected for inference: {self.device}")

            # Initialize Preprocessor
            self.preprocessor = InferencePreprocessor(self.checkpoint, max_length=128)

            # Load Model
            logger.info("Loading model from checkpoint...")
            self.model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint).to(self.device)
            logger.info("Model successfully loaded.")

            # Initialize Trainer
            self.trainer = Trainer(
                model=self.model,
                args=TrainingArguments(
                    output_dir="./tmp_trainer_output",
                    per_device_eval_batch_size=32,
                    dataloader_drop_last=False,
                    no_cuda=self.device == 'cpu',
                    logging_dir="./logs",
                    report_to="none",
                ),
            )
            logger.info("Trainer initialized successfully.")

        except Exception as e:
            msg = f"Failed to initialize PredictEngine. Details: {str(e)}"
            logger.error(msg)
            raise RuntimeError(msg) from e
        
    def _get_data(self):
        """
        Preprocesses the input data into a format suitable for the Trainer.

        Returns
        -------
        datasets.Dataset
            A Hugging Face `Dataset` object containing tokenized and preprocessed inputs.

        Notes
        -----
        This method delegates preprocessing to the `InferencePreprocessor`,
        which handles tokenization, truncation, and dataset conversion.
        """
        try:
            logger.info("Starting data preprocessing...")

            # Validate data source
            if isinstance(self.data_source, pd.DataFrame):
                if self.column not in self.data_source.columns:
                    msg = f"'{self.column}' column is required in the input DataFrame."
                    logger.error(msg)
                    raise KeyError(msg)
            elif isinstance(self.data_source, str):
                logger.info(f"Data source provided as file path: {self.data_source}")
            else:
                msg = f"Unsupported data source type: {type(self.data_source)}. Must be str or pandas.DataFrame."
                logger.error(msg)
                raise TypeError(msg)

            data, data_collator = self.preprocessor.preprocess(self.data_source, self.column)
            logger.info("Data successfully preprocessed.")

            return data

        except Exception as e:
            msg = f"Error during data preprocessing: {str(e)}"
            logger.error(msg)
            raise RuntimeError(msg) from e


    def _postprocess(self, predictions):
        """
        Converts raw model output logits into discrete class predictions.

        Parameters
        ----------
        predictions : transformers.trainer_utils.PredictionOutput
            The prediction output from the Hugging Face `Trainer.predict()` method.
            Contains `predictions` (logits), `label_ids`, and optionally `metrics`.

        Returns
        -------
        numpy.ndarray
            An array of predicted class indices corresponding to each input example.

        Notes
        -----
        The method applies an argmax operation along the last axis to convert logits
        into integer class labels.
        """
        try:
            logger.info("Postprocessing predictions...")
            preprocessed_preds = np.argmax(predictions.predictions, axis=-1)
            logger.info("Predictions successfully postprocessed.")
            return preprocessed_preds

        except Exception as e:
            msg = f"Error during postprocessing: {str(e)}"
            logger.error(msg)
            raise RuntimeError(msg) from e

    def predict(self):
        """
        Executes the full inference pipeline:
        - Preprocesses the input data.
        - Performs batched prediction using the `Trainer`.
        - Postprocesses the logits into final class predictions.

        Returns
        -------
        numpy.ndarray
            A NumPy array containing the predicted class indices for each input sample.

        Notes
        -----
        This method does not return model confidence scores or probabilities.
        To obtain probabilities, modify `_postprocess()` to apply a softmax
        transformation instead of `argmax`.
        """
        try:
            logger.info("Starting prediction pipeline...")

            # Step 1: Preprocess data
            data = self._get_data()

            # Step 2: Run prediction
            logger.info("Performing batch inference using Trainer...")
            predictions = self.trainer.predict(data)
            logger.info("Model inference completed successfully.")

            # Step 3: Postprocess results
            final_predictions = self._postprocess(predictions)
            logger.info("Prediction pipeline completed successfully.")

            return final_predictions

        except Exception as e:
            msg = f"Prediction pipeline failed: {str(e)}"
            logger.error(msg)
            raise RuntimeError(msg) from e
