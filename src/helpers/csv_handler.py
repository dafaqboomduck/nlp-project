import pandas as pd
import os
import csv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CSVHandler:
    def __init__(self, input_path: str = None, output_path: str = None):
        # Validate input_path only if it is provided (not None)
        if input_path is not None:
            if not isinstance(input_path, str):
                raise TypeError("input_path must be a string to CSV file.")
            if not os.path.exists(input_path):
                raise ValueError ("input_path doesn't exist.")
            if not input_path.lower().endswith(".csv"):
                raise ValueError("input_path must point to a .csv file.")
        self.input_path = input_path

        # Validate output_path only if it is provided (not None)
        if output_path is not None:
            if not isinstance(output_path, str):
                raise TypeError("output_path must be a string to CSV file.")
        
        self.output_path = output_path
        
    def read_csv(self) -> pd.DataFrame:
        """
        Reads the input CSV into a pandas DataFrame.
        Returns:
            pd.DataFrame: The loaded dataset.
        Raises:
            FileNotFoundError, pd.errors.EmptyDataError, RuntimeError
        """
        if self.input_path == None:
            return
        try:
            df = pd.read_csv(self.input_path)
            return df
        except FileNotFoundError as e:
            logger.exception("CSV file not found: %s", self.input_path)
            raise
        except pd.errors.EmptyDataError as e:
            logger.exception("CSV file is empty: %s", self.input_path)
            raise
        except Exception as e:
            logger.exception("Failed to read CSV at %s", self.input_path)
            raise
    
    def save_to_csv(self, sentences):
        """
        Saves a list of sentences or a DataFrame column to the specified CSV file.
        """
        if self.output_path == None:
            return
        try:
            if isinstance(sentences, pd.DataFrame):
                if 'Sentence' not in sentences.columns:
                    raise KeyError("DataFrame must contain a 'Sentence' column.")
                sentences_to_write = sentences['Sentence'].astype(str).tolist()
            elif isinstance(sentences, (list, tuple)):
                sentences_to_write = [str(s) for s in sentences]
            else:
                raise TypeError("sentences must be a list, tuple, or pandas DataFrame.")
            
            with open(self.output_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Sentence"])
                for sentence in sentences_to_write:
                    writer.writerow([sentence])
            logger.info(f"Successfully wrote {len(sentences_to_write)} sentences to {self.output_path}")

        except Exception as e:
            logger.exception("Failed to save sentences to CSV: %s", self.output_path)
            raise RuntimeError("Failed to write sentences to CSV.") from e


    
