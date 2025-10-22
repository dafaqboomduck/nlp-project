import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
# from src.helpers import CSVHandler  # Removed dependency on CSVHandler

class InferencePreprocessor:
    """
    Helper class for preprocessing text data for Transformer model inference.

    This class handles:
    1. Initializing a tokenizer from a specified pretrained checkpoint.
    2. Converting an input :class:`pandas.DataFrame` column into a tokenized
       :class:`datasets.Dataset` object ready for the Hugging Face Trainer.
    3. Providing a configured :class:`transformers.DataCollatorWithPadding`.
    
    The preprocessor is designed to be **decoupled from all file I/O**.
    """

    def __init__(self, checkpoint: str, max_length: int = 128):
        """
        Initialize the preprocessor with a tokenizer from the given checkpoint.

        :param checkpoint: Model checkpoint name or path for the tokenizer (e.g., 'bert-base-uncased').
        :type checkpoint: str
        :param max_length: Maximum token length for each sentence, defaults to 128.
        :type max_length: int
        """
        self.checkpoint = checkpoint
        self.max_length = max_length
        # Initialize the tokenizer from the specified checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        # Initialize the DataCollator, which dynamically pads batches to the max length in that batch.
        # Use return_tensors='pt' (PyTorch) for compatibility with Trainer/DataLoader
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, return_tensors='pt')

    def _tokenize_function(self, example):
        """
        Internal method for tokenizing a batch of examples via map().

        :param example: A dictionary containing the 'text' key.
        :type example: dict
        :return: A dictionary containing 'input_ids', 'attention_mask', etc.
        :rtype: dict
        """
        # The tokenizer is called with truncation and max_length set during initialization.
        return self.tokenizer(
            example["text"],
            truncation=True,
            padding='max_length',
            max_length=self.max_length
        )

    def preprocess(self, df: pd.DataFrame, column: str = 'Sentence'):
        """
        Preprocess a DataFrame containing sentences for model inference.

        Performs data validation, converts the DataFrame to a Hugging Face Dataset,
        and applies the tokenization function.

        :param df: The input DataFrame. Must contain the text column specified by ``column``.
        :type df: :class:`pandas.DataFrame`
        :param column: The name of the column in the DataFrame containing the input text, defaults to 'Sentence'.
        :type column: str
        :raises ValueError: If the DataFrame is empty, the column is missing, or contains non-string data.
        :return: A tuple containing the tokenized dataset and the data collator.
        :rtype: tuple[:class:`datasets.Dataset`, :class:`transformers.DataCollatorWithPadding`]
        """
        # --- Data Validation ---
        if df.empty:
            raise ValueError("The input DataFrame is empty.")

        if column not in df.columns:
            raise ValueError(f"The DataFrame must contain a column named {column}.")
        
        # Check for non-string or null values in the target column
        df_clean = df.dropna(subset=[column])
        if df_clean.empty:
            raise ValueError(f"The {column} column contains only null or empty values after dropna.")
            
        if not df_clean[column].apply(lambda x: isinstance(x, str)).all():
            raise ValueError(f"All non-null entries in the {column} column must be strings.")

        # --- Dataset Conversion and Tokenization ---
        # 1. Convert clean DataFrame to Hugging Face Dataset.
        # The tokenizer expects the input text feature to be named 'text'.
        dataset = Dataset.from_pandas(df_clean[[column]].rename(columns={column: 'text'}))

        # 2. Apply tokenization in a batched manner.
        tokenized_dataset = dataset.map(
            self._tokenize_function, 
            batched=True,
            # Remove all original columns except the tokenized ones
            remove_columns=[col for col in dataset.column_names]
        )
        
        return tokenized_dataset, self.data_collator
    
    def get_tokenizer(self):
        """
        Returns the tokenizer instance used by this preprocessor.

        :return: The AutoTokenizer instance.
        :rtype: :class:`transformers.AutoTokenizer`
        """
        return self.tokenizer