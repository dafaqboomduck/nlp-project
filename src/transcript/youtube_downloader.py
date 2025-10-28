import os
import logging
import yt_dlp
import re

logger = logging.getLogger(__name__)

class YouTubeDownloader:
    """
    Utility class for downloading audio from YouTube videos.
    
    NOTE: This class requires the 'yt-dlp' library (pip install yt-dlp) 
    and 'ffmpeg' for the audio conversion post-processor.
    """
    
    def _is_youtube_url(self, url: str) -> bool:
        """Checks if the given URL is a likely YouTube URL."""
        youtube_regex = (
            r"(?:https?:\/\/)?(?:www\.)?"
            r"(?:youtube\.com\/(?:watch\?v=|embed\/|shorts\/)|youtu\.be\/)"
            r"([A-Za-z0-9_-]{11})"
        )
        return re.match(youtube_regex, url) is not None

    def download_audio_from_youtube(self, url: str, output_path: str) -> str:
        """
        Downloads a YouTube video and extracts the audio stream as an MP3 file.

        Args:
            url (str): The YouTube video URL.
            output_path (str): The base path/filename to save the output MP3 file 
                               (e.g., 'downloads/video_audio'). The '.mp3' extension 
                               will be automatically appended.

        Returns:
            str: The final, expected path to the saved MP3 file.
        """
        if not self._is_youtube_url(url):
             raise ValueError(f"Invalid YouTube URL provided: {url}")
             
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # yt-dlp options for audio extraction and MP3 conversion
        ydl_opts = {
            'format': 'bestaudio/best', # Select the best audio stream
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192', # High quality MP3
            }],
            # Define output template: {output_path} will be the filename, %(ext)s will be .mp3
            'outtmpl': f'{output_path}', 
            'noprogress': True,
            'verbose': False,
            'logger': logger,
        }

        # Determine the final expected file path 
        final_file_path = f"{output_path}.mp3"

        try:
            logger.info(f"Starting audio download from YouTube URL: {url}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # The download process extracts the audio and saves it as MP3
                ydl.download([url])

                logger.info(f"Download complete. Audio saved to: {final_file_path}")
                return final_file_path

        except yt_dlp.utils.DownloadError as e:
            logger.error(f"YouTube Download Error: {e}")
            raise RuntimeError(f"Failed to download from YouTube: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise RuntimeError(f"An unexpected error occurred during download: {e}")
