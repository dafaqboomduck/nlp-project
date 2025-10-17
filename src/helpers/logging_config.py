# helpers/logging_config.py

import logging

# Define the log file path
LOG_FILE_PATH = 'application.log'

def configure_logging():
    """
    Configures the root logger with a FileHandler (for all messages)
    and a StreamHandler (for WARNING and above).
    """

    # 1. Get the root logger
    root_logger = logging.getLogger()
    
    # Remove any existing handlers to prevent duplicate logs on re-runs
    # This is a good practice, especially in environments like Jupyter or tests.
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set the overall lowest level to capture all messages the application generates
    root_logger.setLevel(logging.DEBUG)  

    # 2. Define the formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 3. --- Console Handler (for WARNING and above) ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING) # WARNING, ERROR, CRITICAL to console
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 4. --- File Handler (for DEBUG and above) ---
    # Use 'a' for append, or 'w' for overwrite on each run
    file_handler = logging.FileHandler(LOG_FILE_PATH, mode='a') 
    file_handler.setLevel(logging.DEBUG) # All messages to file
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)