# src/transcript/transcript_engine.py
"""
Speech to Text Converter Engine
Orchestrates media conversion, transcription (via AssemblyAITranscriber), and post-processing.
"""

import re
# Removed requests and assemblyai imports as they are now in the client
import os
from typing import Literal
import logging

# Assuming MediaConverter and YouTubeDownloader are available
from src.transcript.config import BASE_URL, HEADERS
from src.transcript.post_process import TranscriptionPostProcessor
from src.helpers import CSVHandler

# Importing the new utility classes
from .media_converter import MediaConverter
from .youtube_downloader import YouTubeDownloader
# IMPORTANT: Import the new decoupled client
from .assemblyAI_client import AssemblyAITranscriber 

logger = logging.getLogger(__name__)


class TranscriptEngine:
    """
    The TranscriptEngine class provides a high-level interface for performing speech-to-text
    transcription from various input sources, including local media files and YouTube URLs.

    Attributes:
        base_url (str): The API base URL for AssemblyAI.
        headers (dict): HTTP headers used for API communication.
        media_converter (MediaConverter): Utility for converting videos to audio.
        youtube_downloader (YouTubeDownloader): Utility for downloading YouTube videos.
        transcriber_client (AssemblyAITranscriber): Client for handling transcription tasks.
    """

    def __init__(self, base_url = BASE_URL, headers = HEADERS):
        self.base_url = base_url
        self.headers = headers
        self.media_converter = MediaConverter()
        self.youtube_downloader = YouTubeDownloader()
        
        # Initialize the decoupled AssemblyAI Transcriber Client
        self.transcriber_client = AssemblyAITranscriber()

    def _split_into_sentences(self, text):
        """Split text into sentences using simple regex"""
        # Split on sentence-ending punctuation followed by space and capital letter
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())

        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 0:  # Filter very short sentences
                cleaned_sentences.append(sentence)

        return cleaned_sentences

    def transcribe(self, input_source: str, output_path: str, post_process: Literal['simple', 'transformer_based', 'none'] = 'simple'):
        """
        Main transcription pipeline. Converts the provided media (local or online) into text and saves
        it as a CSV file of sentences.

        Steps:
            1. Identify and prepare the input media (local file or YouTube URL)
            2. Convert to audio format if required
            3. Send audio to the transcription client (AssemblyAI)
            4. Optionally post-process the text
            5. Save the final sentences into a CSV file
            6. Clean up temporary files

        Args:
            input_source (str): Path to a local file (audio/video) or a YouTube URL.
            output_path (str): Path where the CSV output will be saved.
            post_process (Literal): Type of post-processing to apply:
                - 'simple': Rule-based sentence splitting.
                - 'transformer_based': Transformer model for punctuation and segmentation.
                - 'none': No post-processing; saves the raw transcription as a single entry.

        Returns:
            str: The absolute path to the generated CSV file.

        Raises:
            ValueError: If the input source is not a valid file path or YouTube link.
        """
        # 1. Setup and Pre-check (API key check is now handled in client initialization)
        
        # Determine the file path/URL that will be passed to transcription
        temp_audio_file = None # To track files created for cleanup
        transcription_source = None
        input_source = str(input_source)

        # 2. Handle YouTube URL - FORCE LOCAL DOWNLOAD AND CONVERSION
        try:
            if self.youtube_downloader._is_youtube_url(input_source):
                logger.info(f"Input is a YouTube URL. Downloading and converting to local audio file...")
                
                # Use a temporary filename base for the downloaded file
                temp_base_filename = "temp_youtube_audio"
                
                # The downloader returns the final path to the MP3 file
                local_mp3_path = self.youtube_downloader.download_audio_from_youtube(
                    url=input_source, 
                    output_path=temp_base_filename
                )
                
                # Update the source to the local file path and set the cleanup file
                transcription_source = local_mp3_path
                temp_audio_file = local_mp3_path
            
            # 3. Handle Local Video File (check common video extensions)
            elif os.path.exists(input_source) and input_source.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                logger.info(f"Input is a local video file. Converting to audio...")
                # Use a temporary filename that the conversion utility saves to
                temp_audio_file_base = "temp_converted_audio.mp3"
                
                local_mp3_path = self.media_converter.convert_video_to_audio(
                    video_path=input_source, 
                    output_audio_path=temp_audio_file_base
                )
                
                # Update the source to the local file path and set the cleanup file
                transcription_source = local_mp3_path
                temp_audio_file = local_mp3_path
            elif os.path.exists(input_source) and input_source.lower().endswith(('.mp3', 'wav')):
                transcription_source = input_source
        except:
            msg = f"Cannot use {input_source} as an input for the pipeline. Please provide an audio file, a video file, or a YouTube Video link."
            logger.error(msg)
            raise ValueError(msg)

        # 4. Transcription Process (uses the decoupled client)
        try:
            # Call the decoupled transcriber client method
            transcript_text = self.transcriber_client.transcribe_audio(transcription_source)

            # 5. Post-Processing
            if post_process == 'simple':
                sentences = self._split_into_sentences(transcript_text)
            elif post_process == 'transformer_based':
                # Note: The punctuation_model_name might need to be configurable
                processor = TranscriptionPostProcessor(punctuation_model_name='HuggingFaceH4/zephyr-7b-beta')
                sentences = processor.process(transcript_text)
            else:
                sentences = [transcript_text] # Ensure it's iterable for CSVHandler
            
            # 6. Save to CSV
            csv_handler = CSVHandler(output_path=output_path)
            csv_handler.save_to_csv(sentences)

            logger.info(f"Transcribed {len(sentences)} sentences to {output_path}")
            return str(output_path)

        finally:
            # 7. Cleanup temporary files
            if temp_audio_file and os.path.exists(temp_audio_file):
                logger.info(f"Cleaning up temporary audio file: {temp_audio_file}")
                os.remove(temp_audio_file)