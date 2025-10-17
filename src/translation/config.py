# src/translation/config.py

import os
from src.config import DATA_DIR, ARTIFACTS_DIR

# Translation directory
TRANSLATION_DIR = os.path.join(ARTIFACTS_DIR, 'translation')
os.makedirs(TRANSLATION_DIR, exist_ok=True)

# Paths to saved pretrained models
MODEL_EN_NL_PATH = os.path.join(TRANSLATION_DIR, 'pretrained_en_nl')
MODEL_NL_EN_PATH = os.path.join(TRANSLATION_DIR, 'pretrained_nl_en')

# Output path for translated transcript
TRANSLATED_OUTPUT_PATH = os.path.join(DATA_DIR, 'transcript_translated.csv')