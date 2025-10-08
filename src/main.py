import logging
import os

# Central configuration
# This configures the root logger and creates a default StreamHandler (to console)
# Set the level for all loggers UNLESS a specific logger overrides it.
logging.basicConfig(level=logging.INFO) 
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