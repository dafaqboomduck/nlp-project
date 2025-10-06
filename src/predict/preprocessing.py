import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding


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

    def preprocess(self, csv_path: str):
        """
        Load and preprocess a CSV file containing sentences for model inference.

        Parameters
        ----------
        csv_path : str
            Path to the CSV file. The file must contain a column named 'Sentence'.

        Returns
        -------
        tokenized_dataset : datasets.Dataset
            Tokenized dataset ready for inference.
        data_collator : transformers.DataCollatorWithPadding
            Data collator suitable for batching.
        """
        # Load CSV
        df = pd.read_csv(csv_path)

        if 'Sentence' not in df.columns:
            raise ValueError("The CSV file must contain a column named 'Sentence'.")

        # Convert DataFrame to Hugging Face Dataset
        dataset = Dataset.from_pandas(df[['Sentence']].rename(columns={'Sentence': 'text'}))

        # Apply tokenization
        tokenized_dataset = dataset.map(self._tokenize_function, batched=True)

        return tokenized_dataset, self.data_collator

    def get_tokenizer(self):
        """
        Returns the tokenizer instance used by this preprocessor.
        """
        return self.tokenizer
