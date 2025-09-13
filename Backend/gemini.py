import os
import tempfile
from uuid import uuid4
from typing import List
from PIL import Image

if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.Resampling.LANCZOS
from termcolor import colored
from moviepy.editor import ImageClip, concatenate_videoclips
from huggingface_hub import InferenceClient

# Load HF token from environment
HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize Hugging Face client
client = InferenceClient(
    provider="hf-inference",  
    api_key='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',
)

def generate_hf_images(prompt: str, model: str = "black-forest-labs/FLUX.1-schnell") -> str:
    output_dir = os.path.join(os.path.dirname(__file__), "../temp")
    os.makedirs(output_dir, exist_ok=True)

    try:
        image = client.text_to_image(prompt, model=model)
        file_path = os.path.join(output_dir, f"{uuid4()}.png")
        image.save(file_path, "PNG")
        print(colored(f"[+] Saved image -> {file_path}", "green"))
        return file_path
    except Exception as e:
        print(colored(f"[-] Error generating image: {e}", "red"))
        return None


# def create_video_from_images(image_paths, image_prompts_with_timing, audio_duration):
#     """
#     Create a video from images with precise timing based on image prompts with timing
#     """
#     from moviepy.editor import ImageClip, concatenate_videoclips, ColorClip, CompositeVideoClip
    
#     clips = []
    
#     # If we have fewer images than segments, we need to reuse some images
#     if len(image_paths) < len(image_prompts_with_timing):
#         # Repeat images to match the number of segments
#         repeated_images = []
#         for i in range(len(image_prompts_with_timing)):
#             repeated_images.append(image_paths[i % len(image_paths)])
#         image_paths = repeated_images
    
#     # Create clips for each segment
#     for i, prompt_data in enumerate(image_prompts_with_timing):
#         if i >= len(image_paths):
#             break
            
#         segment_duration = prompt_data["end"] - prompt_data["start"]
        
#         # Create image clip for this segment
#         clip = ImageClip(image_paths[i])
#         clip = clip.set_duration(segment_duration)
#         clip = clip.resize(height=1280)  # Resize to match vertical format
        
#         # Center the image
#         if clip.w > 720:
#             clip = clip.crop(x_center=clip.w/2, y_center=clip.h/2, width=720, height=1280)
#         else:
#             # Center the image on a black background
#             clip = clip.set_position(('center', 'center'))
#             clip = CompositeVideoClip([ColorClip(size=(720, 1280), color=(0, 0, 0)), clip])
        
#         clips.append(clip)
    
#     # Concatenate all clips
#     final_clip = concatenate_videoclips(clips)
    
#     # If the total duration is less than audio, extend the last frame
#     if final_clip.duration < audio_duration:
#         last_clip = ImageClip(image_paths[-1])
#         last_clip = last_clip.set_duration(audio_duration - final_clip.duration)
#         last_clip = last_clip.resize(height=1280)
        
#         if last_clip.w > 720:
#             last_clip = last_clip.crop(x_center=last_clip.w/2, y_center=last_clip.h/2, width=720, height=1280)
#         else:
#             last_clip = last_clip.set_position(('center', 'center'))
#             last_clip = CompositeVideoClip([ColorClip(size=(720, 1280), color=(0, 0, 0)), last_clip])
        
#         final_clip = concatenate_videoclips([final_clip, last_clip])
    
#     # Save the video
#     output_path = f"../temp/{uuid4()}.mp4"
#     final_clip.write_videofile(output_path, fps=24, threads=2)
    
#     return output_path

def cleanup_images(image_paths: List[str]):
    """Clean up temporary image files."""
    for path in image_paths:
        try:
            os.unlink(path)
        except:
            pass


