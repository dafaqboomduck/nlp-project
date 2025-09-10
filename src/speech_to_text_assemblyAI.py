# REDACTED
"""
Speech to Text Converter using AssemblyAI
Converts MP3 audio files to sentences and saves to CSV
"""

import requests
import time
import csv
import re
from pathlib import Path
import assemblyai as aai


# AssemblyAI configuration
BASE_URL = "https://api.assemblyai.com"
HEADERS = {
    "authorization": "REDACTED"  #  API key
}


def create_transcription_config():
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


def upload_audio(file_path):
    """Upload audio file to AssemblyAI and return the upload URL"""
    with open(file_path, "rb") as f:
        response = requests.post(BASE_URL + "/v2/upload", headers=HEADERS, data=f)

    if response.status_code != 200:
        raise RuntimeError(f"Upload failed: {response.text}")

    return response.json()["upload_url"]


def transcribe_audio(file_path, config):
    """Transcribe audio file using AssemblyAI SDK"""
    transcriber = aai.Transcriber(config=config)
    transcript = transcriber.transcribe(file_path)

    if transcript.status == aai.TranscriptStatus.error:
        raise RuntimeError(f"Transcription failed: {transcript.error}")

    return transcript.text


def split_into_sentences(text):
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


def save_to_csv(sentences, output_path):
    """Save sentences to CSV file"""
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Sentence"])  # Header

        for sentence in sentences:
            writer.writerow([sentence])


def main(mp3_file):
    """
    Main function: convert MP3 to sentences CSV using AssemblyAI

    Args:
        mp3_file (str): Path to MP3 file

    Returns:
        str: Path to saved CSV file
    """
    # Check if API key is set
    if not HEADERS["authorization"]:
        raise ValueError("Please set your AssemblyAI API key in the HEADERS variable")

    aai.settings.api_key = HEADERS["authorization"]

    config = create_transcription_config()

    print(f"Uploading {mp3_file}...")
    audio_url = upload_audio(mp3_file)

    print("Starting transcription...")
    transcript_text = transcribe_audio(audio_url, config)

    # Split transcription into sentences
    sentences = split_into_sentences(transcript_text)

    # Create output filename
    input_path = Path(mp3_file)
    output_path = input_path.parent / f"{input_path.stem}_sentences_AssemlyAI.csv"

    # Save to CSV
    save_to_csv(sentences, output_path)

    print(f"Transcribed {len(sentences)} sentences to {output_path}")
    return str(output_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python speech_to_text_assemblyAI.py <mp3_file>")
        sys.exit(1)

    mp3_file = sys.argv[1]
    main(mp3_file)
