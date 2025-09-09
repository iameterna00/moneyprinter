import os
import uuid

import requests
import srt_equalizer
import assemblyai as aai
import whisper
from typing import List
from moviepy.editor import *
from termcolor import colored
from dotenv import load_dotenv
from datetime import timedelta
from moviepy.video.fx.all import crop
from moviepy.video.tools.subtitles import SubtitlesClip
from moviepy.config import change_settings
change_settings({"IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe"})

load_dotenv("../.env")

ASSEMBLY_AI_API_KEY = os.getenv("ASSEMBLY_AI_API_KEY")


def save_video(video_url: str, directory: str = "../temp") -> str:
    """
    Saves a video from a given URL and returns the path to the video.

    Args:
        video_url (str): The URL of the video to save.
        directory (str): The path of the temporary directory to save the video to

    Returns:
        str: The path to the saved video.
    """
    video_id = uuid.uuid4()
    video_path = f"{directory}/{video_id}.mp4"
    with open(video_path, "wb") as f:
        f.write(requests.get(video_url).content)

    return video_path


def __generate_subtitles_whisper(audio_path: str, model_size: str = "base") -> str:
    """
    Generates subtitles from a given audio file using local Whisper.
    """
    def convert_to_srt_time_format(total_seconds):
        h, remainder = divmod(int(total_seconds), 3600)
        m, s = divmod(remainder, 60)
        ms = int((total_seconds - int(total_seconds)) * 1000)
        return f"{h:01}:{m:02}:{s:02},{ms:03}"

    print(colored(f"[+] Transcribing locally with Whisper ({model_size})...", "blue"))
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, word_timestamps=False, fp16=False)

    subtitles = []
    for i, seg in enumerate(result["segments"], start=1):
        start = convert_to_srt_time_format(seg["start"])
        end = convert_to_srt_time_format(seg["end"])
        text = seg["text"].strip()
        subtitles.append(f"{i}\n{start} --> {end}\n{text}\n")

    return "\n".join(subtitles)


def generate_subtitles(audio_path: str, model_size: str = "base") -> str:
    """
    Generates subtitles from an audio file using Whisper locally.
    """
    def equalize_subtitles(srt_path: str, max_chars: int = 10):
        srt_equalizer.equalize_srt_file(srt_path, srt_path, max_chars)

    subtitles_path = f"../subtitles/{uuid.uuid4()}.srt"

    subtitles = __generate_subtitles_whisper(audio_path, model_size=model_size)

    with open(subtitles_path, "w", encoding="utf-8") as f:
        f.write(subtitles)

    equalize_subtitles(subtitles_path)
    print(colored("[+] Subtitles generated.", "green"))

    return subtitles_path


def combine_videos(video_paths: List[str], max_duration: int, max_clip_duration: int, threads: int) -> str:
    """
    Combines a list of videos into one video and returns the path to the combined video.

    Args:
        video_paths (List): A list of paths to the videos to combine.
        max_duration (int): The maximum duration of the combined video.
        max_clip_duration (int): The maximum duration of each clip.
        threads (int): The number of threads to use for the video processing.

    Returns:
        str: The path to the combined video.
    """
    video_id = uuid.uuid4()
    combined_video_path = f"../temp/{video_id}.mp4"
    
    # Required duration of each clip
    req_dur = max_duration / len(video_paths)

    print(colored("[+] Combining videos...", "blue"))
    print(colored(f"[+] Each clip will be maximum {req_dur} seconds long.", "blue"))

    clips = []
    tot_dur = 0
    # Add downloaded clips over and over until the duration of the audio (max_duration) has been reached
    while tot_dur < max_duration:
        for video_path in video_paths:
            clip = VideoFileClip(video_path)
            clip = clip.without_audio()
            # Check if clip is longer than the remaining audio
            if (max_duration - tot_dur) < clip.duration:
                clip = clip.subclip(0, (max_duration - tot_dur))
            # Only shorten clips if the calculated clip length (req_dur) is shorter than the actual clip to prevent still image
            elif req_dur < clip.duration:
                clip = clip.subclip(0, req_dur)
            clip = clip.set_fps(30)

            # Not all videos are same size,
            # so we need to resize them
            if round((clip.w/clip.h), 4) < 0.5625:
                clip = crop(clip, width=clip.w, height=round(clip.w/0.5625), \
                            x_center=clip.w / 2, \
                            y_center=clip.h / 2)
            else:
                clip = crop(clip, width=round(0.5625*clip.h), height=clip.h, \
                            x_center=clip.w / 2, \
                            y_center=clip.h / 2)
            clip = clip.resize((720, 1280))  #1080x1920

            if clip.duration > max_clip_duration:
                clip = clip.subclip(0, max_clip_duration)

            clips.append(clip)
            tot_dur += clip.duration

    final_clip = concatenate_videoclips(clips)
    final_clip = final_clip.set_fps(30)
    final_clip.write_videofile(combined_video_path, threads=threads)

    return combined_video_path

def generate_video(
    combined_video_path: str,
    tts_path: str,
    subtitles_path: str,
    threads: int,
    subtitles_position: str,
    text_color: str
) -> str:
    """
    This function creates the final video, with subtitles and audio.

    Returns:
        str: The path to the final video inside ../Generated_Video/
    """
    # Ensure Generated_Video folder exists
    output_dir = "../Generated_Video"
    os.makedirs(output_dir, exist_ok=True)

    # Create a unique filename for each generated video
    video_name = f"{uuid.uuid4()}.mp4"
    final_video_path = os.path.join(output_dir, video_name)

    # Make a generator that returns a TextClip when called with consecutive
    generator = lambda txt: TextClip(
        txt,
        font="../fonts/bold_font.ttf",
        fontsize=100,
        color=text_color,
        stroke_color="black",
        stroke_width=5,
    )

    # Split the subtitles position into horizontal and vertical
    horizontal_subtitles_position, vertical_subtitles_position = subtitles_position.split(",")

    # Burn the subtitles into the video
    subtitles = SubtitlesClip(subtitles_path, generator)
    result = CompositeVideoClip([
        VideoFileClip(combined_video_path),
        subtitles.set_pos((horizontal_subtitles_position, vertical_subtitles_position))
    ])

    # Add the audio
    audio = AudioFileClip(tts_path)
    result = result.set_audio(audio)

    # Save in Generated_Video folder
    result.write_videofile(final_video_path, threads=threads or 2)

    print(colored(f"[+] Final video saved as {final_video_path}", "green"))
    return final_video_path
if __name__ == "__main__":
    # Test input files
    video_files = [
        "../temp/a7a9b011-088c-4c00-8ed8-5d9ae9a6ce5e.mp4",
        "../temp/82fb6aa6-40e6-4fe3-8ffa-7c8c317f3c87.mp4"
    ]
    tts_file = "../temp/460edbfa-8796-4912-88cb-72da783f3589.mp3"
    sentences = ["Hello world", "This is a test"]
    voice = "en"
    max_duration = 60
    max_clip_duration = 30
    threads = 4
    subtitles_position = "center,bottom"
    text_color = "#FFFFFF"

    print(colored("[*] Combining videos...", "blue"))
    combined_video = combine_videos(video_files, max_duration, max_clip_duration, threads)

    print(colored("[*] Generating subtitles...", "blue"))
    # Create dummy AudioFileClip list for local subtitle generation
    audio_clips = [AudioFileClip(tts_file).subclip(i, i+1) for i in range(len(sentences))]  
    subtitles_path = generate_subtitles(tts_file, sentences, audio_clips, voice)

    print(colored("[*] Creating final video with subtitles and audio...", "blue"))
    final_video = generate_video(combined_video, tts_file, subtitles_path, threads, subtitles_position, text_color)

    print(colored(f"[+] Test finished. Final video: {final_video}", "green"))
