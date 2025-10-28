# src/transcript/assemblyai_client.py
"""
Dedicated client for interacting with the AssemblyAI API.
Handles API key setup, configuration, and the actual transcription process.
"""

import assemblyai as aai
import logging
from src.transcript.config import HEADERS # Assuming config is accessible

logger = logging.getLogger(__name__)


class AssemblyAITranscriber:
    def __init__(self):
        """Initializes the AssemblyAI SDK with the API key."""
        api_key = HEADERS.get("authorization")
        if not api_key:
            raise ValueError("AssemblyAI API key not found in HEADERS configuration.")
            
        # Set the global API key for the SDK
        aai.settings.api_key = api_key

    def _create_transcription_config(self):
        """Create transcription configuration with optimal settings."""
        config = aai.TranscriptionConfig(
            speaker_labels=True,
            format_text=True,
            punctuate=True,
            speech_model=aai.SpeechModel.slam_1,
            language_code="en_us",
        )
        # Using the same hardcoded keyterms for consistency
        config.keyterms_prompt = [
            "Restaurant",
            "Cooking",
            "Chef",
            "Cook",
            "Food",
            "British",
            "Bad",
            "Decisions",
        ]
        return config

    def transcribe_audio(self, source_path_or_url: str) -> str:
        """
        Transcribe an audio source (local file path or URL) using AssemblyAI SDK.

        Args:
            source_path_or_url (str): Path to local file or a public media URL.

        Returns:
            str: The raw, transcribed text.
        """
        config = self._create_transcription_config()
        logger.info(f"Starting AAI transcription for source: {source_path_or_url}...")

        # The AAI SDK handles internal file upload if a local path is provided
        transcriber = aai.Transcriber(config=config)
        transcript = transcriber.transcribe(source_path_or_url)

        if transcript.status == aai.TranscriptStatus.error:
            raise RuntimeError(f"Transcription failed: {transcript.error}")

        return transcript.text