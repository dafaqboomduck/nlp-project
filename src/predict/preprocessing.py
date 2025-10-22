import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
# from src.helpers import CSVHandler  # Removed non-local dependency

class InferencePreprocessor:
    """
    Helper class for preprocessing text data for Transformer model inference.

    This class handles tokenizing text using a specified pretrained checkpoint, 
    and returning a ready-to-use Hugging Face Dataset and DataCollator.
    It is designed to take a pandas DataFrame directly for cleaner separation
    of concerns from data loading.
    """

    def __init__(self, checkpoint: str, max_length: int = 128):
        """
        Initialize the preprocessor with a tokenizer from the given checkpoint.

        Parameters
        ----------
        checkpoint : str
            Model checkpoint name or path for the tokenizer.
        max_length : int
            Maximum token length for each sentence.
        """
        self.checkpoint = checkpoint
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        # Use return_tensors='pt' (PyTorch) for compatibility with Trainer/DataLoader
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors='pt')

    def _tokenize_function(self, example):
        """
        Internal method for tokenizing a batch of examples.
        """
        return self.tokenizer(
            example["text"],
            truncation=True,
            padding='max_length',
            max_length=self.max_length
        )

    def preprocess(self, df: pd.DataFrame, column: str = 'Sentence'):
        """
        Preprocess a DataFrame containing sentences for model inference.

        Parameters
        ----------
        df : pandas.DataFrame
            The input DataFrame. The data loading logic should be external.
        column : str, optional
            The name of the column in the dataset containing the input text.
            Defaults to 'Sentence'.
            
        Returns
        -------
        tokenized_dataset : datasets.Dataset
            Tokenized dataset ready for inference.
        data_collator : transformers.DataCollatorWithPadding
            Data collator suitable for batching.

        Raises
        ------
        ValueError
            If the DataFrame is empty, the column is missing, contains only nulls, 
            or non-string values.
        """
        # Validate data source
        if df.empty:
            raise ValueError("The input DataFrame is empty.")

        if column not in df.columns:
            raise ValueError(f"The DataFrame must contain a column named {column}.")
        
        # Check for non-string or null values only in the target column
        if df[column].isnull().all():
            raise ValueError(f"The {column} column contains only null or empty values.")
        
        # Filter out rows where the text is not a string (or is null) before conversion
        df_clean = df.dropna(subset=[column])
        if not df_clean[column].apply(lambda x: isinstance(x, str)).all():
            raise ValueError(f"All non-null entries in the {column} column must be strings.")

        # Convert clean DataFrame to Hugging Face Dataset
        # The Trainer expects the input features for sequence classification to be named 'input_ids', 'attention_mask', etc.,
        # which are produced by the tokenizer from the 'text' column.
        dataset = Dataset.from_pandas(df_clean[[column]].rename(columns={column: 'text'}))

        # Apply tokenization
        # Only keep necessary columns for the model ('input_ids', 'attention_mask', etc.)
        tokenized_dataset = dataset.map(
            self._tokenize_function, 
            batched=True,
            remove_columns=[col for col in dataset.column_names if col != 'text']
        )
        # The 'text' column is also not needed for the Trainer
        tokenized_dataset = tokenized_dataset.remove_columns(['text'])

        return tokenized_dataset, self.data_collator
    
    def get_tokenizer(self):
        """
        Returns the tokenizer instance used by this preprocessor.
        """
        return self.tokenizer