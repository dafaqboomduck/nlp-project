import pandas as pd
import csv
import logging
from typing import Union, List, Tuple
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the type alias for paths
PathLike = Union[str, Path]

class CSVHandler:
    def __init__(self, input_path: PathLike = None, output_path: PathLike = None):
        # Convert all incoming paths (str or Path) to Path objects immediately
        # Store as Path or None
        self.input_path: Path | None = Path(input_path) if input_path is not None else None
        self.output_path: Path | None = Path(output_path) if output_path is not None else None

        if self.input_path is not None:
            # We call validate on the Path object
            self.input_path = self._validate_input_path(self.input_path)
        
        # Note: Output path validation is now simplified in __init__ and handled before writing
        if self.output_path is not None and not isinstance(self.output_path, Path):
             raise TypeError("output_path must be a string or a pathlib.Path object.")
            
    def _validate_input_path(self, path: Path) -> Path:
        """Helper method to validate input file paths using Path methods."""
        if not isinstance(path, Path):
             # This check handles cases where the method might be called directly with wrong type
             raise TypeError("Path must be a pathlib.Path object.")

        if not path.exists():
            raise FileNotFoundError(f"Path doesn't exist: {path}")
        # Use Path.suffix instead of string methods
        if path.suffix.lower() != ".csv":
            raise ValueError("Path must point to a .csv file.")
        return path

    def read_csv(self, path: PathLike = None, delimiter: str = ',') -> pd.DataFrame:
        """
        Reads the input CSV into a pandas DataFrame.
        Args:
            path (PathLike, optional): Overrides the instance's input_path for this read.
        Returns:
            pd.DataFrame: The loaded dataset.
        """
        path_to_read = self.input_path
        if path is not None:
            # Ensure the override path is converted to a Path object
            path_to_read = Path(path)
        
        if path_to_read is None:
            raise ValueError("No input_path was provided during initialization or method call.")
        
        # Validate path
        self._validate_input_path(path_to_read)

        try:
            # pandas read_csv handles Path objects natively
            df = pd.read_csv(path_to_read, delimiter=delimiter)
            return df
        except FileNotFoundError:
            logger.exception("CSV file not found: %s", path_to_read)
            raise
        except pd.errors.EmptyDataError:
            logger.exception("CSV file is empty: %s", path_to_read)
            raise
        except Exception:
            logger.exception("Failed to read CSV at %s", path_to_read)
            raise
    
    def save_to_csv(self, data: Union[pd.DataFrame, List[str], Tuple[str, ...]], path: PathLike = None):
        """
        Saves a list of strings/tuples or a DataFrame column to the specified CSV file.
        Args:
            data: The data to save (DataFrame, list, or tuple).
            path (PathLike, optional): Overrides the instance's output_path for this save.
        """
        path_to_write = self.output_path
        if path is not None:
            # Ensure the override path is converted to a Path object
            path_to_write = Path(path)

        if path_to_write is None:
            raise ValueError("No output_path was provided during initialization or method call.")
            
        if not isinstance(path_to_write, Path):
             # Should be a Path object due to __init__ and Path(path) conversion above
             raise TypeError("Internal error: Path to write is not a Path object.")

        # Check and enforce .csv extension using Path methods
        if path_to_write.suffix.lower() != ".csv":
            logger.warning("Output path does not end with .csv, appending it.")
            # Use .with_suffix to correctly append/change the extension
            path_to_write = path_to_write.with_suffix('.csv')

        try:
            if isinstance(data, pd.DataFrame):
                if 'Sentence' not in data.columns:
                    raise KeyError("DataFrame must contain a 'Sentence' column to save.")
                sentences_to_write = data['Sentence'].astype(str).tolist()
            elif isinstance(data, (list, tuple)):
                sentences_to_write = [str(s) for s in data]
            else:
                raise TypeError("Data must be a list, tuple, or pandas DataFrame with a 'Sentence' column.")
            
            # Ensure the directory exists before writing using Path.parent and mkdir
            path_to_write.parent.mkdir(parents=True, exist_ok=True)
            
            # Use the Path object directly in the open function
            with open(path_to_write, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["Sentence"]) # Write header
                writer.writerows([[sentence] for sentence in sentences_to_write]) # Write all rows
            
            logger.info(f"Successfully wrote {len(sentences_to_write)} sentences to {path_to_write}")

        except Exception as e:
            logger.exception("Failed to save data to CSV: %s", path_to_write)
            raise RuntimeError(f"Failed to write data to CSV at {path_to_write}.") from e