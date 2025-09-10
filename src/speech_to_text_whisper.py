"""
Speech to Text Converter using Whisper
Converts MP3 audio files to sentences and saves to CSV
"""

import whisper
import csv
import re
from pathlib import Path


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
    Main function: convert MP3 to sentences CSV

    Args:
        mp3_file (str): Path to MP3 file

    Returns:
        str: Path to saved CSV file
    """
    # Load Whisper model and transcribe
    model = whisper.load_model("base")
    result = model.transcribe(mp3_file)

    # Split transcription into sentences
    sentences = split_into_sentences(result["text"])

    # Create output filename
    input_path = Path(mp3_file)
    output_path = input_path.parent / f"{input_path.stem}_sentences.csv"

    # Save to CSV
    save_to_csv(sentences, output_path)

    print(f"Transcribed {len(sentences)} sentences to {output_path}")
    return str(output_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python speech_to_text_whisper.py <mp3_file>")
        sys.exit(1)

    mp3_file = sys.argv[1]
    main(mp3_file)
