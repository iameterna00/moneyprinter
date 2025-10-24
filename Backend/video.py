import os
import uuid
import random
import numpy as np
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
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.editor import CompositeAudioClip
from moviepy.editor import TextClip, CompositeVideoClip
from moviepy.config import change_settings
from video_effect.popuptext import create_pop_text_clip
from video_effect.videomoment import add_shaky_effect, add_subtle_zoom_movement, create_video_from_images

# Configure ImageMagick path
change_settings({"IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-7.1.2-Q16\magick.exe"})

load_dotenv("../.env")

ASSEMBLY_AI_API_KEY = os.getenv("ASSEMBLY_AI_API_KEY")


def save_video(video_url: str, directory: str = "../temp") -> str:
    """
    Saves a video from a given URL and returns the path to the video.
    """
    video_id = uuid.uuid4()
    video_path = f"{directory}/{video_id}.mp4"
    with open(video_path, "wb") as f:
        f.write(requests.get(video_url).content)
    return video_path

def cleanup_images(image_paths: List[str]):
    """Clean up temporary image files."""
    for path in image_paths:
        try:
            os.unlink(path)
        except:
            pass


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
    result = model.transcribe(audio_path, word_timestamps=True, fp16=False)

    subtitles = []
    for i, seg in enumerate(result["segments"], start=1):
        start = convert_to_srt_time_format(seg["start"])
        end = convert_to_srt_time_format(seg["end"])
        text = seg["text"].strip()
        subtitles.append(f"{i}\n{start} --> {end}\n{text}\n")

    return "\n".join(subtitles)


def generate_subtitles(audio_path: str, model_size: str = "base") -> str:
    """
    Generates subtitles from an audio file using Whisper locally, with debug prints.
    """
    def equalize_subtitles(srt_path: str, max_chars: int = 10):
        print(colored(f"[DEBUG] Equalizing subtitles with max_chars={max_chars}: {srt_path}", "yellow"))
        srt_equalizer.equalize_srt_file(srt_path, srt_path, max_chars)
        print(colored("[DEBUG] Subtitle equalization complete.", "yellow"))

    subtitles_path = f"../subtitles/{uuid.uuid4()}.srt"
    print(colored(f"[DEBUG] Subtitles will be saved to: {subtitles_path}", "yellow"))
    print(colored(f"[DEBUG] Using audio file: {audio_path}", "yellow"))
    print(colored(f"[DEBUG] Using Whisper model: {model_size}", "yellow"))

    try:
        subtitles = __generate_subtitles_whisper(audio_path, model_size=model_size)
        print(colored(f"[DEBUG] Raw subtitles generated:\n{subtitles[:500]}...", "yellow"))  # show first 500 chars
    except Exception as e:
        print(colored(f"[ERROR] Whisper transcription failed: {e}", "red"))
        raise

    try:
        with open(subtitles_path, "w", encoding="utf-8") as f:
            f.write(subtitles)
        print(colored(f"[DEBUG] Subtitles written to file: {subtitles_path}", "yellow"))
    except Exception as e:
        print(colored(f"[ERROR] Failed to write subtitles file: {e}", "red"))
        raise

    try:
        equalize_subtitles(subtitles_path)
    except Exception as e:
        print(colored(f"[ERROR] Subtitle equalization failed: {e}", "red"))
        raise

    print(colored("[+] Subtitles generated successfully.", "green"))
    return subtitles_path



def combine_videos(video_paths: List[str], max_duration: int, max_clip_duration: int, threads: int) -> str:
    """
    Combines a list of videos into one video.
    """
    video_id = uuid.uuid4()
    combined_video_path = f"../temp/{video_id}.mp4"
    
    # Filter out non-video files
    valid_video_paths = []
    for video_path in video_paths:
        if video_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv')):
            valid_video_paths.append(video_path)
        else:
            print(colored(f"[!] Skipping non-video file: {video_path}", "yellow"))
    
    if not valid_video_paths:
        raise ValueError("No valid video files found to combine")
    
    # Required duration of each clip
    req_dur = max_duration / len(valid_video_paths)

    print(colored("[+] Combining videos...", "blue"))
    print(colored(f"[+] Each clip will be maximum {req_dur} seconds long.", "blue"))

    clips = []
    tot_dur = 0
    # Add downloaded clips over and over until the duration of the audio (max_duration) has been reached
    while tot_dur < max_duration:
        for video_path in valid_video_paths:
            try:
                clip = VideoFileClip(video_path)
                clip = clip.without_audio()
                
                # Check if clip is longer than the remaining audio
                if (max_duration - tot_dur) < clip.duration:
                    clip = clip.subclip(0, (max_duration - tot_dur))
                # Only shorten clips if the calculated clip length (req_dur) is shorter than the actual clip to prevent still image
                elif req_dur < clip.duration:
                    clip = clip.subclip(0, req_dur)
                clip = clip.set_fps(30)

                # Not all videos are same size, so we need to resize them
                if round((clip.w/clip.h), 4) < 0.5625:
                    clip = crop(clip, width=clip.w, height=round(clip.w/0.5625), 
                                x_center=clip.w / 2, y_center=clip.h / 2)
                else:
                    clip = crop(clip, width=round(0.5625*clip.h), height=clip.h,
                                x_center=clip.w / 2, y_center=clip.h / 2)
                clip = clip.resize((720, 1280))

                if clip.duration > max_clip_duration:
                    clip = clip.subclip(0, max_clip_duration)

                clips.append(clip)
                tot_dur += clip.duration
                
                if tot_dur >= max_duration:
                    break
                    
            except Exception as e:
                print(colored(f"[!] Error processing {video_path}: {e}", "red"))
                continue

    final_clip = concatenate_videoclips(clips)
    final_clip = final_clip.set_fps(30)
    final_clip.write_videofile(combined_video_path, threads=threads, verbose=False, logger=None)

    return combined_video_path



def add_background_music(video_clip, music_path, volume=0.3, loop=True):
    """
    Add background music to a video clip.
    """
    # Load the background music
    bg_music = AudioFileClip(music_path)
    
    # Adjust volume
    bg_music = bg_music.volumex(volume)
    
    # Loop the music if needed
    if loop and bg_music.duration < video_clip.duration:
        # Calculate how many times we need to loop
        num_loops = int(np.ceil(video_clip.duration / bg_music.duration))
        bg_music = concatenate_audioclips([bg_music] * num_loops)
    
    # Trim the music to match video duration
    bg_music = bg_music.subclip(0, video_clip.duration)
    
    # Mix the background music with the original audio
    if video_clip.audio is not None:
        # Combine original audio with background music
        composite_audio = CompositeAudioClip([video_clip.audio, bg_music])
    else:
        # If no original audio, just use the background music
        composite_audio = bg_music
    
    # Set the composite audio to the video clip
    return video_clip.set_audio(composite_audio)

def generate_video(
    combined_video_path: str,
    tts_path: str,
    subtitles_path: str,
    threads: int,
    subtitles_position: str,
    text_color: str,
    bg_music_path: str = None,
    bg_music_volume: float = 0.3,
    shaky_effect: bool = True,
    shake_intensity: int = 8,  
    shake_frequency: int = 20,  
    zoom_effect: bool = True,  
    max_zoom: float = 1.13,  # Changed from zoom_range to max_zoom (3% zoom)
    movement_range: float = 50,  
    zoom_change_interval: float = 3.0 
) -> str:
    """
    This function creates the final video, with subtitles and audio.
    """
    # Ensure Generated_Video folder exists
    output_dir = "../Generated_Video"
    os.makedirs(output_dir, exist_ok=True)

    # Create a unique filename for each generated video
    video_name = f"{uuid.uuid4()}.mp4"
    final_video_path = os.path.join(output_dir, video_name)

    # Load the video clip
    video_clip = VideoFileClip(combined_video_path)

    # Read subtitles from file
    with open(subtitles_path, 'r', encoding='utf-8') as f:
        subtitle_content = f.read()
    
    # Parse subtitles manually to create animated text clips
    subtitle_clips = []
    subtitle_blocks = subtitle_content.strip().split('\n\n')
    
    for block in subtitle_blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
            
        # Parse timecodes
        timecode_line = lines[1]
        start_time, end_time = timecode_line.split(' --> ')
        
        # Convert timecode to seconds
        def timecode_to_seconds(timecode):
            h, m, s_ms = timecode.split(':')
            s, ms = s_ms.split(',')
            return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000.0
        
        start_seconds = timecode_to_seconds(start_time)
        end_seconds = timecode_to_seconds(end_time)
        duration = end_seconds - start_seconds
        
        text = ' '.join(lines[2:])
        
        text_clip = create_pop_text_clip(
            text,
            duration=duration,
            font="../fonts/luck.ttf",
            fontsize=50,
            color=text_color,
            stroke_color="black",
            stroke_width=1,
            pop_duration=0.2
        )
        
        horizontal_subtitles_position, vertical_subtitles_position = subtitles_position.split(",")
        text_clip = text_clip.set_position((horizontal_subtitles_position, vertical_subtitles_position))
        text_clip = text_clip.set_start(start_seconds)
        
        subtitle_clips.append(text_clip)

    result = CompositeVideoClip([video_clip] + subtitle_clips)

    audio = AudioFileClip(tts_path)
    result = result.set_audio(audio)
    
    # Add background music if provided
    if bg_music_path and os.path.exists(bg_music_path):
        print(colored("[+] Adding background music...", "blue"))
        result = add_background_music(result, bg_music_path, bg_music_volume)

    # Apply effects with proper error handling
    try:
        if zoom_effect:
            print(colored("[+] Applying zoom-in only effect...", "blue"))
            result = add_subtle_zoom_movement(
                result, 
                min_zoom=1.0,
                max_zoom=max_zoom,  
                horizontal_range=movement_range,
                cycles=zoom_change_interval
            )
    except Exception as e:
        print(colored(f"[WARNING] Zoom effect failed: {e}, continuing without it", "yellow"))
    
    try:
        if shaky_effect:
            print(colored("[+] Applying continuous smooth shaky effect to final video...", "blue"))
            # Use the MoviePy version (more reliable)
            result = add_shaky_effect(result, shake_intensity, shake_frequency)
    except Exception as e:
        print(colored(f"[WARNING] Shaky effect failed: {e}, trying OpenCV version", "yellow"))
        try:
            result = add_shaky_effect(result, shake_intensity, shake_frequency)
        except Exception as e2:
            print(colored(f"[WARNING] Both shaky effect methods failed: {e2}, continuing without shaky effect", "yellow"))

    # Ensure duration is set before writing
    if not hasattr(result, 'duration') or result.duration is None:
        print(colored("[WARNING] Result clip has no duration, setting to audio duration", "yellow"))
        result = result.set_duration(audio.duration)

    result.write_videofile(
        final_video_path, 
        threads=threads or 2,
        fps=24, 
        codec='libx264',
        preset='medium', 
        ffmpeg_params=['-crf', '23', '-pix_fmt', 'yuv420p'],
        verbose=False,
        logger=None
    )

    print(colored(f"[+] Final video saved as {final_video_path}", "green"))
    return final_video_path


def main():
    """
    Main function to create a video from images with continuous shaky effect
    """
    # Configuration
    image_files = [
        "../test/9fbd030a-1864-48c7-a980-4483019fef75.png",
        "../test/42cf679c-0a76-436d-9a0b-ddfdf39c61d5.png",
        "../test/73ee5776-1ba7-4049-9d3e-203f3aaffead.png",
        "../test/0936ea2f-40f7-4f6d-90ad-59609640c994.png",
        "../test/3730d9c3-7719-4dac-a49c-6f92438d7e16.png",
        "../test/b3db93c6-9658-4d82-811a-28ec03dfc8d0.png",
        "../test/d5316f38-7a36-4405-b8ea-7bf5153772a0.png",
        "../test/d5316f38-7a36-4405-b8ea-7bf5153772a0.png"
    ]
    
    image_prompts_with_timing = [
        {"start": 0, "end": 3},
        {"start": 3, "end": 6},
        {"start": 6, "end": 9},
        {"start": 9, "end": 12},
        {"start": 12, "end": 15},
        {"start": 15, "end": 18},
        {"start": 18, "end": 21},
        {"start": 21, "end": 24}
    ]
    
    tts_file = "../test/91317266-5a1b-4d18-b3fd-7b92cb92df8eTEMP_MPY_wvf_snd.mp3"
    audio_duration = 15 
    threads = 4
    subtitles_position = "center,center"
    text_color = "#FFFFFF"
    shaky_effect = True 
    shake_intensity = 10
    shake_frequency = 150
    whisper_model_size = "base" 
    bg_music_path = "../Songs/dark.mp3" 
    bg_music_volume = 0.3 
    transition_duration = 0.5

    print(colored("[*] Starting video creation process...", "blue"))
    
    # Step 1: Create video from images (without shaky effect)
    print(colored("[*] Creating video from images...", "blue"))
    try:
        combined_video = create_video_from_images(
            image_files, 
            image_prompts_with_timing, 
            audio_duration,
            transition_duration
        )
        print(colored(f"[+] Video created: {combined_video}", "green"))
    except Exception as e:
        print(colored(f"[!] Error creating video from images: {e}", "red"))
        return

    print(colored("[*] Generating subtitles...", "blue"))
    try:
        subtitles_path = generate_subtitles(tts_file, model_size=whisper_model_size)
        print(colored(f"[+] Subtitles generated: {subtitles_path}", "green"))
    except Exception as e:
        print(colored(f"[!] Error generating subtitles: {e}", "red"))
        return

    print(colored("[*] Creating final video with continuous shaky effect...", "blue"))
    try:
        final_video = generate_video(
            combined_video, 
            tts_file, 
            subtitles_path, 
            threads, 
            subtitles_position, 
            text_color,
            bg_music_path,
            bg_music_volume,
            shaky_effect,
            shake_intensity,
            shake_frequency
        )
        print(colored(f"[+] Final video created: {final_video}", "green"))
    except Exception as e:
        print(colored(f"[!] Error creating final video: {e}", "red"))
        return

    print(colored("[+] Video creation process completed successfully!", "green"))


if __name__ == "__main__":
    main()