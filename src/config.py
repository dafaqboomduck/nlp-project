from pathlib import Path

# Define base directories
BASE_DIR = Path.cwd()
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "Data" / "CSV"
AUDIO_DIR = BASE_DIR / "Data" / "Recordings"

# Define file paths
AUDIO_PATH = AUDIO_DIR / "recording.mp3"
TRANSCRIPT_PATH = DATA_DIR / "audio_transcript.csv"
PREDS_PATH = DATA_DIR / "predictions.csv"

# Create necessary directories
for directory in [ARTIFACTS_DIR, DATA_DIR, AUDIO_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
