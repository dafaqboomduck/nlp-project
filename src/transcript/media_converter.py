import os
import logging
from moviepy import VideoFileClip

logger = logging.getLogger(__name__)

class MediaConverter:
    """
    Utility class for converting local media files (like videos) to audio format.
    
    NOTE: This class requires the 'moviepy' library to be installed (pip install moviepy), 
    and also requires the 'ffmpeg' system dependency for video processing.
    """

    def convert_video_to_audio(self, video_path: str, output_audio_path: str) -> str:
        """
        Converts a video file (e.g., MP4, MOV) to an audio file (e.g., MP3).

        Args:
            video_path (str): The local path to the input video file (e.g., 'video.mp4').
            output_audio_path (str): The local path to save the output audio file (e.g., 'audio.mp3').
                                     If the path's extension is missing or non-audio, it defaults to '.mp3'.

        Returns:
            str: The path to the generated audio file.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at: {video_path}")

        # Ensure the output path has a common audio extension, default to .mp3
        if not any(output_audio_path.lower().endswith(ext) for ext in ['.mp3', '.wav', '.flac']):
            base, _ = os.path.splitext(output_audio_path)
            output_audio_path = base + '.mp3'
            logger.warning(f"Output path extension set to default: {output_audio_path}")

        try:
            logger.info(f"Loading video file: {video_path}")
            clip = VideoFileClip(video_path)

            logger.info(f"Writing audio to: {output_audio_path}")
            # Use appropriate codec for MP3 output
            clip.audio.write_audiofile(output_audio_path, codec='mp3')

            clip.close()
            logger.info(f"Successfully converted video to audio: {output_audio_path}")
            return output_audio_path
        except Exception as e:
            logger.error(f"Error during video to audio conversion: {e}")
            # Reraise a clean RuntimeError for clarity
            raise RuntimeError(f"Conversion failed. Ensure 'moviepy' and 'ffmpeg' are installed: {e}")