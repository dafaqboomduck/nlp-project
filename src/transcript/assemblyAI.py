# src/transcript/assemblyAI
"""
Speech to Text Converter using AssemblyAI
Converts MP3 audio files to sentences and saves to CSV
"""

import csv
import re
import requests
import assemblyai as aai
from typing import Literal

from src.transcript.config import BASE_URL, HEADERS
from src.transcript.post_process import TranscriptionPostProcessor


class TranscriptEngine:
    def __init__(self, base_url = BASE_URL, headers = HEADERS):
        self.base_url = base_url,
        self.headers = headers
        

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
            response = requests.post(BASE_URL + "/v2/upload", headers=HEADERS, data=f)

        if response.status_code != 200:
            raise RuntimeError(f"Upload failed: {response.text}")

        return response.json()["upload_url"]


    def _transcribe_audio(self, file_path, config):
        """Transcribe audio file using AssemblyAI SDK"""
        transcriber = aai.Transcriber(config=config)
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


    def _save_to_csv(self, sentences, output_path):
        """Save sentences to CSV file"""
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Sentence"])  # Header

            for sentence in sentences:
                writer.writerow([sentence])


    def transcribe(self, mp3_file, output_path, post_process: Literal['simple', 'transformer_based', 'none'] = 'simple'):
        """
        Main function: convert MP3 to sentences CSV using AssemblyAI

        Args:
            mp3_file (str): Path to MP3 file
            output_path (CSV file): An output path where the pipeline will save the 
            post_process (Literal['simple', 'transformer_based', 'none']): The type of post_processing the pipeline will apply to the raw transcript

        Returns:
            str: Path to saved CSV file
        """
        # Check if API key is set
        if not HEADERS["authorization"]:
            raise ValueError("Please set your AssemblyAI API key in the HEADERS variable")

        aai.settings.api_key = HEADERS["authorization"]

        config = self._create_transcription_config()

        print(f"Uploading {mp3_file}...")
        audio_url = self._upload_audio(mp3_file)

        print("Starting transcription...")
        transcript_text = self._transcribe_audio(audio_url, config)

        if post_process == 'simple':
            sentences = self._split_into_sentences(transcript_text)
        elif post_process == 'transformer_based':
            processor = TranscriptionPostProcessor(punctuation_model_name='HuggingFaceH4/zephyr-7b-beta')
            sentences = processor.process(transcript_text)
        else:
            # Otherwise use the raw transcript text as output 
            sentences = transcript_text
        # Save to CSV
        self._save_to_csv(sentences, output_path)

        print(f"Transcribed {len(sentences)} sentences to {output_path}")
        return str(output_path)


