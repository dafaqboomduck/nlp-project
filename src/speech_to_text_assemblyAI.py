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

# AssemblyAI configuration
BASE_URL = "https://api.assemblyai.com"
HEADERS = {
    "authorization": "REDACTED"  #  API key
}


def upload_audio(file_path):
    """Upload audio file to AssemblyAI and return the upload URL"""
    with open(file_path, "rb") as f:
        response = requests.post(BASE_URL + "/v2/upload", headers=HEADERS, data=f)

    if response.status_code != 200:
        raise RuntimeError(f"Upload failed: {response.text}")

    return response.json()["upload_url"]


def transcribe_audio(audio_url):
    """Start transcription and wait for completion"""
    data = {"audio_url": audio_url, "speech_model": "universal"}

    # Start transcription
    response = requests.post(BASE_URL + "/v2/transcript", json=data, headers=HEADERS)

    if response.status_code != 200:
        raise RuntimeError(f"Transcription request failed: {response.text}")

    transcript_id = response.json()["id"]
    polling_endpoint = BASE_URL + "/v2/transcript/" + transcript_id

    # Poll for completion
    while True:
        transcription_result = requests.get(polling_endpoint, headers=HEADERS).json()

        if transcription_result["status"] == "completed":
            return transcription_result["text"]

        elif transcription_result["status"] == "error":
            raise RuntimeError(f"Transcription failed: {transcription_result['error']}")

        else:
            time.sleep(3)


def split_into_sentences(text):
    """Split text into sentences using simple regex"""
    # Split on sentence-ending punctuation followed by space and capital letter
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())

    # Clean and filter sentences
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 5:  # Filter very short sentences
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

    print(f"Uploading {mp3_file}...")
    audio_url = upload_audio(mp3_file)

    print("Starting transcription...")
    transcript_text = transcribe_audio(audio_url)

    # Split transcription into sentences
    sentences = split_into_sentences(transcript_text)

    # Create output filename
    input_path = Path(mp3_file)
    output_path = input_path.parent / f"{input_path.stem}_sentences_AssemlyAI.csv"

    # Save to CSV
    save_to_csv(sentences, output_path)

    print(f"Transcribed {len(sentences)} sentences to {output_path}")
    return str(output_path)
