from src.processing import FeatureEngine
from src.transcript import TranscriptEngine

from src.config import AUDIO_PATH, TRANSRIPT_PATH
from src.processing.config import FEATURE_OUTPUT_PATH as FEATURES

def main():

    # Create an instance of the class
    transcript_engine = TranscriptEngine()
    # Call the transcribe method on the instance
    transcript_engine.transcribe(mp3_file=AUDIO_PATH, output_path=TRANSRIPT_PATH, post_process='simple')

    feature_engine = FeatureEngine()
    feature_engine.create_features(output_path=FEATURES)

if __name__ == '__main__':
    main()