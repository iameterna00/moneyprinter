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
from moviepy.audio.fx.volumex import volumex
from moviepy.video.tools.subtitles import SubtitlesClip
from moviepy.config import change_settings
import cv2

# Configure ImageMagick path
change_settings({"IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe"})

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


def add_shaky_effect(clip, intensity=3, frequency=5):
    """
    Adds a smooth continuous movement effect to a video clip.
    """
    print(colored(f"[DEBUG] Applying continuous shaky effect: intensity={intensity}, frequency={frequency}", "yellow"))
    
    # Initialize smooth movement variables
    current_offset_x, current_offset_y = 0, 0
    target_offset_x, target_offset_y = 0, 0
    last_change_time = 0
    
    def smooth_movement(get_frame, t):
        nonlocal current_offset_x, current_offset_y, target_offset_x, target_offset_y, last_change_time
        
        # Change target position based on frequency
        if t - last_change_time >= 1.0 / frequency:
            target_offset_x = random.uniform(-intensity, intensity)
            target_offset_y = random.uniform(-intensity, intensity)
            last_change_time = t
        
        # Smoothly interpolate toward target position
        current_offset_x += (target_offset_x - current_offset_x) * 0.2
        current_offset_y += (target_offset_y - current_offset_y) * 0.2
        
        # Get the original frame
        frame = get_frame(t)
        height, width = frame.shape[:2]
        
        # Create transformation matrix for smooth movement
        M = np.float32([[1, 0, current_offset_x], [0, 1, current_offset_y]])
        
        # Apply the smooth transformation using warpAffine
        smooth_frame = cv2.warpAffine(frame, M, (width, height), borderMode=cv2.BORDER_REFLECT)
        
        return smooth_frame
    
    # Apply the effect to the clip
    return clip.fl(smooth_movement)

def create_video_from_images(image_paths, image_prompts_with_timing, audio_duration, transition_duration=0.5):
    """
    Create a video from images with precise timing
    """
    from moviepy.editor import ImageClip, ColorClip, CompositeVideoClip
    
    # Validate and clean image prompts
    valid_prompts = []
    for prompt_data in image_prompts_with_timing:
        try:
            # Check if it's a dictionary with the expected structure
            if (isinstance(prompt_data, dict) and 
                "start" in prompt_data and "end" in prompt_data and
                isinstance(prompt_data["start"], (int, float)) and 
                isinstance(prompt_data["end"], (int, float)) and
                prompt_data["end"] > prompt_data["start"]):
                valid_prompts.append(prompt_data)
        except (KeyError, TypeError, AttributeError):
            # Skip invalid entries
            continue
    
    # If no valid prompts, create default timing
    if not valid_prompts:
        print(colored("[!] No valid timing data found, using default timing", "yellow"))
        segment_duration = audio_duration / len(image_paths)
        for i in range(len(image_paths)):
            valid_prompts.append({
                "start": i * segment_duration,
                "end": (i + 1) * segment_duration
            })
    
    # Make sure we have the same number of images as prompts
    if len(image_paths) != len(valid_prompts):
        print(colored(f"[!] Mismatch: {len(image_paths)} images vs {len(valid_prompts)} prompts. Adjusting...", "yellow"))
        # Use the minimum of both to avoid index errors
        min_length = min(len(image_paths), len(valid_prompts))
        image_paths = image_paths[:min_length]
        valid_prompts = valid_prompts[:min_length]
    
    # Create a list to hold all clips with their precise timing
    all_clips = []
    
    # Process each image segment
    for i in range(len(valid_prompts)):
        try:
            prompt_data = valid_prompts[i]
            image_path = image_paths[i]
            
            # Check if image file exists
            if not os.path.exists(image_path):
                print(colored(f"[!] Image file not found: {image_path}", "red"))
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            segment_duration = prompt_data["end"] - prompt_data["start"]
            
            # Create image clip for this segment
            clip = ImageClip(image_path)
            clip = clip.set_duration(segment_duration)
            clip = clip.set_start(prompt_data["start"])  # CRITICAL: Set exact start time
            clip = clip.resize(height=1280)
            
            # Center the image
            if clip.w > 720:
                clip = clip.crop(x_center=clip.w/2, y_center=clip.h/2, width=720, height=1280)
            else:
                # Create a black background with the image centered
                bg_clip = ColorClip(size=(720, 1280), color=(0, 0, 0), duration=segment_duration)
                clip = CompositeVideoClip([bg_clip, clip.set_position('center')])
            
            all_clips.append(clip)
            
        except Exception as e:
            print(colored(f"[!] Error processing image {image_paths[i] if i < len(image_paths) else 'unknown'}: {e}", "red"))
            # Add a black frame as fallback
            segment_duration = valid_prompts[i]["end"] - valid_prompts[i]["start"]
            fallback_clip = ColorClip(size=(720, 1280), color=(0, 0, 0), duration=segment_duration)
            fallback_clip = fallback_clip.set_start(valid_prompts[i]["start"])
            all_clips.append(fallback_clip)
    
    # Create the final composite video with all clips at their precise timings
    if not all_clips:
        print(colored("[!] No clips to composite", "red"))
        # Create a blank video as fallback
        blank_clip = ColorClip(size=(720, 1280), color=(0, 0, 0), duration=audio_duration)
        final_clip = blank_clip
    else:
        final_clip = CompositeVideoClip(all_clips)
    
    final_clip = final_clip.set_duration(audio_duration)
    
    # Save the video
    output_path = f"../temp/{uuid.uuid4()}.mp4"
    final_clip.write_videofile(output_path, fps=24, threads=2, verbose=False, logger=None)
    
    return output_path

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
    Generates subtitles from an audio file using Whisper locally.
    """
    def equalize_subtitles(srt_path: str, max_chars: int = 20):
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


def create_pop_text_clip(txt, duration=5, font="../fonts/luck.ttf", fontsize=70, color="#FFFFFF", 
                        stroke_color="black", stroke_width=5, pop_duration=0.3):
    """
    Creates a text clip with a pop animation effect.
    """
    # Create the base text clip
    txt_clip = TextClip(
        txt,
        font=font,
        fontsize=fontsize,
        color=color,
        stroke_color=stroke_color,
        stroke_width=stroke_width,
        size=(700, None),
        method="caption",
        align="center"
    )
    
    # Set the duration
    txt_clip = txt_clip.set_duration(duration)
    
    # Add pop animation (scale up then back to normal)
    def pop_effect(get_frame, t):
        frame = get_frame(t)
        
        # Apply pop effect in the first 30% of the clip duration
        if t < pop_duration:
            # Scale up during pop-in
            scale = 1.0 + (0.2 * (t / pop_duration))
            height, width = frame.shape[:2]
            
            # Create a new frame with the same dimensions
            new_frame = np.zeros_like(frame)
            
            # Calculate scaled dimensions
            new_height = int(height * scale)
            new_width = int(width * scale)
            
            # Resize the frame
            from PIL import Image
            pil_img = Image.fromarray(frame)
            pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
            scaled_frame = np.array(pil_img)
            
            # Center the scaled frame
            y_offset = (height - new_height) // 2
            x_offset = (width - new_width) // 2
            
            # Copy the scaled frame to the center
            if y_offset >= 0 and x_offset >= 0 and y_offset + new_height <= height and x_offset + new_width <= width:
                new_frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = scaled_frame
            else:
                new_frame = frame
                
            return new_frame
        
        return frame
    
    # Apply the pop effect
    txt_clip = txt_clip.fl(pop_effect)
    
    return txt_clip


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
    shake_frequency: int = 5
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
            fontsize=80,
            color=text_color,
            stroke_color="black",
            stroke_width=1,
            pop_duration=0.5 
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

    # Apply continuous shaky effect to the final composite video
    if shaky_effect:
        print(colored("[+] Applying continuous smooth shaky effect to final video...", "blue"))
        result = add_shaky_effect(result, shake_intensity, shake_frequency)

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

    print(colored(f"[+] Final video with continuous shaky effect saved as {final_video_path}", "green"))
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