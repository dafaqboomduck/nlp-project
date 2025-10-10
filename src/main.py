# main.py (or your original script)

import logging
import os

# Import the configuration function
from src.logging_config import configure_logging

# --- CALL THE CONFIGURATION FUNCTION FIRST ---
configure_logging() 
# ---------------------------------------------

# Once the configuration is set, you can get loggers anywhere
# and they will inherit the root logger's handlers.
logger = logging.getLogger(__name__) 

from src.processing import FeatureEngine
from src.transcript import TranscriptEngine

from src.config import AUDIO_PATH, TRANSRIPT_PATH, ARTIFACTS_DIR
from src.processing.config import FEATURE_OUTPUT_PATH as FEATURES

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def main():

    # Create an instance of the class
    transcript_engine = TranscriptEngine()
    # Call the transcribe method on the instance
    transcript_engine.transcribe(mp3_file=AUDIO_PATH, output_path=TRANSRIPT_PATH, post_process='simple')

    # Create an instance of the class
    feature_engine = FeatureEngine()
    # Call the create_features method on the instance
    feature_engine.create_features(output_path=FEATURES)

if __name__ == '__main__':
    main()