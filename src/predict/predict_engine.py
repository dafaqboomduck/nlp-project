from src.predict.preprocessing import InferencePreprocessor
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import torch # For device handling
import logging 

logger = logging.getLogger(__name__)

class PredictEngine:
    """
    Handles model loading, data preprocessing, and prediction 
    using the Hugging Face Trainer for efficiency.
    """
    def __init__(self, checkpoint: str, data_source: str, device: str = None):
        # 1. Store Preprocessor and Model (Load once in __init__)
        self.checkpoint = checkpoint
        self.data_source = data_source
        
        # Determine device (e.g., 'cuda:0', 'cpu')
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        logger.info(f"Loading model and running on: {self.device}")
        
        # Initialize Preprocessor (assuming it handles the tokenizer load)
        self.preprocessor = InferencePreprocessor(self.checkpoint, max_length=128)
        
        # Load Model and move it to the correct device
        self.model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint).to(self.device)
        
        # Initialize Trainer (Best practice for efficient batch inference)
        # We need minimal TrainingArguments for inference
        self.trainer = Trainer(
            model=self.model,
            args=TrainingArguments(
                output_dir="./tmp_trainer_output",  # Required, but won't be used for prediction
                per_device_eval_batch_size=32,      # Efficient batch size
                dataloader_drop_last=False,
                no_cuda=self.device == 'cpu',
            )
        )

    def _get_data(self):
        """Preprocesses the data source into a Dataset object for the Trainer."""
        # Use the initialized preprocessor
        data = self.preprocessor.preprocess(self.data_source)
        # Data is returned, not stored in self.data (cleaner)
        return data
    
    def _postprocess(self, predictions):
        """Converts raw prediction logits into class labels."""
        # `predictions.predictions` contains the logits/scores
        preprocessed_preds = np.argmax(predictions.predictions, axis=-1)
        return preprocessed_preds
      
    def predict(self):
        # 1. Get the preprocessed data (as a Hugging Face Dataset)
        data = self._get_data()
        
        # 2. Use the Trainer for efficient prediction
        # The result includes: predictions (logits), label_ids, and metrics
        predictions = self.trainer.predict(data)
        
        # 3. Postprocess and return the final predictions
        final_predictions = self._postprocess(predictions)
        
        return final_predictions