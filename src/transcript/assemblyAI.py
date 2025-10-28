# src/transcript/assemblyAI
"""
Speech to Text Converter using AssemblyAI
Converts various media (YouTube links, video files, audio files) to sentences and saves to CSV
"""

import re
import requests
import assemblyai as aai
import os # Added for path manipulation and cleanup
from typing import Literal
import logging

# Assuming MediaConverter and YouTubeDownloader are available at this path level
from src.transcript.config import BASE_URL, HEADERS
from src.transcript.post_process import TranscriptionPostProcessor
from src.helpers import CSVHandler
# Importing the new utility classes
from .media_converter import MediaConverter
from .youtube_downloader import YouTubeDownloader

logger = logging.getLogger(__name__)


class TranscriptEngine:
    def __init__(self, base_url = BASE_URL, headers = HEADERS):
        self.base_url = base_url,
        self.headers = headers
        self.media_converter = MediaConverter()
        self.youtube_downloader = YouTubeDownloader()


    def _create_transcription_config(self):
        """Create transcription configuration with optimal settings"""
        config = aai.TranscriptionConfig(
            speaker_labels=True,
            format_text=True,
            punctuate=True,
            speech_model=aai.SpeechModel.slam_1,
            language_code="en_us",
        )
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


    def _upload_audio(self, file_path):
        """Upload audio file to AssemblyAI and return the upload URL"""
        with open(file_path, "rb") as f:
            # Note: The original code was using a requests call to /v2/upload.
            # While the SDK can handle local files directly, keeping the requests call for upload 
            # as it was defined in the original `assemblyAI.py` structure.
            response = requests.post(BASE_URL + "/v2/upload", headers=HEADERS, data=f)

        if response.status_code != 200:
            raise RuntimeError(f"Upload failed: {response.text}")

        return response.json()["upload_url"]


    def _transcribe_audio(self, file_path, config):
        """Transcribe audio file using AssemblyAI SDK"""
        # The transcriber expects an accessible path or a URL
        transcriber = aai.Transcriber(config=config)
        # Note: If file_path is a URL, this works. If it's a local file, it also works 
        # (the SDK handles local file upload internally if a local path is provided).
        transcript = transcriber.transcribe(file_path)

        if transcript.status == aai.TranscriptStatus.error:
            raise RuntimeError(f"Transcription failed: {transcript.error}")

        return transcript.text


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
        Main function: convert various media to sentences CSV using AssemblyAI.
        Handles local files (audio/video) and publicly accessible URLs (like YouTube).

        Args:
            input_source (str): Path to local file (MP3, MP4, etc.) OR a YouTube URL.
            output_path (CSV file): An output path where the pipeline will save the 
                                    final processed sentences as a CSV.
            post_process (Literal['simple', 'transformer_based', 'none']): The type of 
                                                                          post-processing to apply.

        Returns:
            str: Path to saved CSV file
        """
        # 1. Setup and Pre-check
        if not HEADERS["authorization"]:
            raise ValueError("Please set your AssemblyAI API key in the HEADERS variable")

        aai.settings.api_key = HEADERS["authorization"]
        
        # Determine the file path/URL that will be passed to transcription
        temp_audio_file = None # To track files created for cleanup

        # 2. Handle YouTube URL - FORCE LOCAL DOWNLOAD AND CONVERSION (FIX for text/html error)
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
            temp_audio_file = "temp_converted_audio.mp3"
            
            local_mp3_path = self.media_converter.convert_video_to_audio(
                video_path=input_source, 
                output_audio_path=temp_audio_file
            )
            
            # Update the source to the local file path and set the cleanup file
            transcription_source = local_mp3_path
            temp_audio_file = local_mp3_path
        elif os.path.exists(input_source) and input_source.lower().endswith(('.mp3', 'wav')):
            transcription_source = input_source
        else:
            msg = f"Cannot use {input_source} as an input for the pipeline. Please provide an audio file, a video file, or a YouTube Video link."
            logger.error(msg)
            raise ValueError(msg)

        # 4. Transcription Process (works for local audio file, converted local video file, or public URL)
        try:
            config = self._create_transcription_config()

            # Now, `transcription_source` is either a local file path (audio/converted video) or a public media URL
            logger.info(f"Starting transcription for source: {transcription_source}...")
            # We call _transcribe_audio, which uses the AAI SDK and handles local files/URLs appropriately.
            transcript_text = self._transcribe_audio(transcription_source, config)

            # 5. Post-Processing
            if post_process == 'simple':
                sentences = self._split_into_sentences(transcript_text)
            elif post_process == 'transformer_based':
                processor = TranscriptionPostProcessor(punctuation_model_name='HuggingFaceH4/zephyr-7b-beta')
                sentences = processor.process(transcript_text)
            else:
                sentences = transcript_text
            
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
