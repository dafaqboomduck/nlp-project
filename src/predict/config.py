from typing import Dict
from pathlib import Path  # Import Path
from src.translation.config import TRANSLATED_OUTPUT_PATH as TRANSLATIONS
from src.config import ARTIFACTS_DIR

# Define the emotion mapping
EMOTION_MAP: Dict[int, str] = {
    0: "neutral", 1: "anger", 2: "disgust", 3: "fear",
    4: "happiness", 5: "sadness", 6: "surprise"
}

CORE_CHECKPOINT = ARTIFACTS_DIR / 'bert-base'
FINE_CHECKPOINT = 'facebook/bart-large-mnli'
DATA_PATH = TRANSLATIONS
INPUT_COLUMN = "Round_Trip_Translation_EN"