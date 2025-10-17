import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from src.helpers import CSVHandler


class InferencePreprocessor:
    """
    Helper class for preprocessing text data for Transformer model inference.

    This class handles loading a CSV of sentences, tokenizing them using a specified
    pretrained checkpoint, and returning a ready-to-use Hugging Face Dataset and DataCollator.
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
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

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

    def preprocess(self, data_source: str | pd.DataFrame, column: str = 'Sentence'):
        """
        Load and preprocess a CSV file containing sentences for model inference.

        Parameters
        ----------
        data_source : str | pandas.DataFrame
            The input data for prediction. This can be either:
            - A path to a file (e.g., CSV).
            - A pandas DataFrame.
        column : str, optional
            The name of the column in the dataset containing the input text.
            Defaults to 'Sentence'.
            
        Returns
        -------
        tokenized_dataset : datasets.Dataset
            Tokenized dataset ready for inference.
        data_collator : transformers.DataCollatorWithPadding
            Data collator suitable for batching.
        """
        if isinstance(data_source, pd.DataFrame):
            # If dataframe use it  
            df = data_source
        elif isinstance(data_source, str):
            # If string, assume it is a path and try loadding it 
            # Load CSV
            csv_path = data_source
            csv_handler = CSVHandler
            df = csv_handler.read_csv(csv_path)
        
        if df.empty:
            raise ValueError(f"The CSV file '{csv_path}' is empty.")

        if column not in df.columns:
            raise ValueError(f"The CSV file must contain a column named {column}.")
        
        if df[column].isnull().all():
            raise ValueError(f"The {column} column contains only null or empty values.")

        if not df[column].apply(lambda x: isinstance(x, str)).all():
            raise ValueError(f"All entries in the {column} column must be strings.")

        # Convert DataFrame to Hugging Face Dataset
        dataset = Dataset.from_pandas(df[[column]].rename(columns={column: 'text'}))

        # Apply tokenization
        tokenized_dataset = dataset.map(self._tokenize_function, batched=True)

        return tokenized_dataset, self.data_collator

    def get_tokenizer(self):
        """
        Returns the tokenizer instance used by this preprocessor.
        """
        return self.tokenizer
