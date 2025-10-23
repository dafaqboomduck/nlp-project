# src/translation/config.py

from pathlib import Path
from src.config import DATA_DIR, ARTIFACTS_DIR 

# Translation directory
# Use the '/' operator for path joining with Path objects
TRANSLATION_DIR = ARTIFACTS_DIR / 'translation'

# Create the directory using the Path object's method
TRANSLATION_DIR.mkdir(exist_ok=True) 

# Paths to saved pretrained models
MODEL_EN_NL_PATH = TRANSLATION_DIR / 'pretrained_en_nl'
MODEL_NL_EN_PATH = TRANSLATION_DIR / 'pretrained_nl_en'

# Output path for translated transcript
TRANSLATED_OUTPUT_PATH = DATA_DIR / 'transcript_translated.csv'