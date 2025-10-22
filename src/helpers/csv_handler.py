import pandas as pd
import os
import csv
import logging
from typing import Union, List, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CSVHandler:
    def __init__(self, input_path: str = None, output_path: str = None):
        if input_path is not None:
            self.input_path = self._validate_input_path(input_path)
        else:
            self.input_path = None

        if output_path is not None:
            if not isinstance(output_path, str):
                raise TypeError("output_path must be a string to CSV file.")
        self.output_path = output_path
        
    def _validate_input_path(self, path: str) -> str:
        """Helper method to validate input file paths."""
        if not isinstance(path, str):
            raise TypeError("Path must be a string to CSV file.")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path doesn't exist: {path}")
        if not path.lower().endswith(".csv"):
            raise ValueError("Path must point to a .csv file.")
        return path

    def read_csv(self, path: str = None, delimiter: str = ',') -> pd.DataFrame:
        """
        Reads the input CSV into a pandas DataFrame.
        Args:
            path (str, optional): Overrides the instance's input_path for this read.
        Returns:
            pd.DataFrame: The loaded dataset.
        """
        path_to_read = path if path is not None else self.input_path
        
        if path_to_read is None:
            raise ValueError("No input_path was provided during initialization or method call.")
        
        # Validate path even if it's the class attribute (e.g., if set without check)
        self._validate_input_path(path_to_read)

        try:
            df = pd.read_csv(path_to_read, delimiter=delimiter)
            return df
        except FileNotFoundError:
            # logger.exception logs the full traceback
            logger.exception("CSV file not found: %s", path_to_read)
            raise
        except pd.errors.EmptyDataError:
            logger.exception("CSV file is empty: %s", path_to_read)
            raise
        except Exception:
            logger.exception("Failed to read CSV at %s", path_to_read)
            raise
    
    def save_to_csv(self, data: Union[pd.DataFrame, List[str], Tuple[str, ...]], path: str = None):
        """
        Saves a list of strings/tuples or a DataFrame column to the specified CSV file.
        Args:
            data: The data to save (DataFrame, list, or tuple).
            path (str, optional): Overrides the instance's output_path for this save.
        """
        path_to_write = path if path is not None else self.output_path
        
        if path_to_write is None:
            raise ValueError("No output_path was provided during initialization or method call.")
            
        if not isinstance(path_to_write, str):
             raise TypeError("Output path must be a string.")
        
        if not path_to_write.lower().endswith(".csv"):
             logger.warning("Output path does not end with .csv, appending it.")
             path_to_write += ".csv"

        try:
            if isinstance(data, pd.DataFrame):
                if 'Sentence' not in data.columns:
                    raise KeyError("DataFrame must contain a 'Sentence' column to save.")
                sentences_to_write = data['Sentence'].astype(str).tolist()
            elif isinstance(data, (list, tuple)):
                sentences_to_write = [str(s) for s in data]
            else:
                raise TypeError("Data must be a list, tuple, or pandas DataFrame with a 'Sentence' column.")
            
            # Ensure the directory exists before writing
            os.makedirs(os.path.dirname(path_to_write) or '.', exist_ok=True)
            
            with open(path_to_write, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Sentence"]) # Write header
                writer.writerows([[sentence] for sentence in sentences_to_write]) # Write all rows
            
            logger.info(f"Successfully wrote {len(sentences_to_write)} sentences to {path_to_write}")

        except Exception as e:
            logger.exception("Failed to save data to CSV: %s", path_to_write)
            # Raise a custom, more informative exception
            raise RuntimeError(f"Failed to write data to CSV at {path_to_write}.") from e