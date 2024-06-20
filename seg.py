import os
import subprocess
from pydub import AudioSegment, silence
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
import logging
import moviepy.config as mp_config
import whisper

# Setup logging
logging.basicConfig(level=logging.INFO)

# Set the path to the ImageMagick binary
mp_config.IMAGEMAGICK_BINARY = r'C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe'  # Update this path to where ImageMagick is installed

def extract_audio_from_video(video_path, audio_path):
    """Extracts audio from the given video file and saves it to audio_path."""
    try:
        subprocess.run(['ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', audio_path, '-y'], check=True)
        logging.info(f"Extracted audio to {audio_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error extracting audio: {e}")
        raise

def detect_silence(audio_path, silence_thresh=-40, min_silence_len=500):
    """Detects silence in the audio file and returns intervals of silence."""
    audio = AudioSegment.from_file(audio_path)
    silent_intervals = silence.detect_silence(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)
    silent_intervals = [(start / 1000, stop / 1000) for start, stop in silent_intervals]
    return silent_intervals

def generate_split_intervals(silent_intervals, audio_duration):
    """Generates intervals to split the audio based on detected silence."""
    intervals = []
    prev_end = 0
    for start, end in silent_intervals:
        if prev_end < start:
            intervals.append((prev_end, start))
        prev_end = end
    if prev_end < audio_duration:
        intervals.append((prev_end, audio_duration))
    return intervals

def split_audio(audio, intervals, output_folder):
    """Splits the audio into segments based on the given intervals and saves them."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    audio_segments = []
    for i, (start, end) in enumerate(intervals):
        segment = audio[start * 1000:end * 1000]
        segment_path = os.path.join(output_folder, f'segment_{i + 1}.wav')
        segment.export(segment_path, format="wav")
        audio_segments.append(segment_path)
    return audio_segments

def transcribe_audio_segment_whisper(model, audio_path):
    """Transcribes the given audio segment using the Whisper model."""
    try:
        result = model.transcribe(audio_path, fp16=False)
        text = result['text'].strip()
        if not text:
            text = "No transcription available"
        logging.info(f"Transcribed text: {text}")
        return text
    except Exception as e:
        logging.error(f"Error transcribing audio: {e}")
        return "No transcription available"

def split_video(video_path, intervals, output_folder, texts, gif_output_folder):
    """Splits the video into segments based on the given intervals and adds transcribed text."""
    video = VideoFileClip(video_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(gif_output_folder):
        os.makedirs(gif_output_folder)

    for i, (start, end) in enumerate(intervals):
        clip = video.subclip(start, end)
        text = texts[i] if texts[i] else "No transcription available"
        logging.info(f"Adding text to video segment: '{text}'")
        clip = add_text_to_clip(clip, text)
        clip_path = os.path.join(output_folder, f'word_{i + 1}.mp4')
        clip.write_videofile(clip_path, codec='libx264')
        logging.info(f"Saved video segment: {clip_path}")
        gif_filename = f'word_{i + 1}.gif'
        convert_to_gif(clip, gif_filename, gif_output_folder)

def add_text_to_clip(clip, text):
    """Adds text to the given video clip."""
    try:
        txt_clip = TextClip(text, fontsize=120, color='red', font='Arial-Bold')  # Adjust fontsize for readability
        txt_clip = txt_clip.set_pos(('center', 'bottom')).set_duration(clip.duration)
        return CompositeVideoClip([clip, txt_clip])
    except Exception as e:
        logging.error(f"Error creating TextClip with text '{text}': {e}")
        raise

def convert_to_gif(clip, gif_filename, gif_output_folder):
    """Converts a video clip to GIF and saves it."""
    try:
        gif_path = os.path.join(gif_output_folder, gif_filename)
        clip.write_gif(gif_path, fps=15, program='imageio', opt='nq', fuzz=10)
        logging.info(f"Saved GIF: {gif_path}")
    except Exception as e:
        logging.error(f"Error converting video to GIF: {e}")
        raise


def main():
    video_path = 'video.avi'
    audio_path = 'audio.wav'
    output_folder = 'output_segments'
    gif_output_folder = 'gif_segments'  # New folder for saving GIFs

    try:
        # Step 1: Extract audio from video
        extract_audio_from_video(video_path, audio_path)

        # Step 2: Detect silence in the audio
        silent_intervals = detect_silence(audio_path)

        # Step 3: Generate split intervals
        audio = AudioSegment.from_file(audio_path)
        audio_duration = len(audio) / 1000
        split_intervals = generate_split_intervals(silent_intervals, audio_duration)

        # Step 4: Split audio into segments
        audio_segments = split_audio(audio, split_intervals, output_folder)

        # Step 5: Load Whisper model once
        model = whisper.load_model("base")

        # Step 6: Transcribe each audio segment using Whisper
        texts = [transcribe_audio_segment_whisper(model, segment) for segment in audio_segments]

        # Step 7: Split the video based on intervals and convert to GIF with text
        split_video(video_path, split_intervals, output_folder, texts, gif_output_folder)

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
