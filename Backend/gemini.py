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


def create_video_from_images(image_paths: List[str], duration: float) -> str:
    """Create a video from generated images."""
    if not image_paths:
        raise ValueError("No images provided for video creation")

    clip_duration = max(3.0, duration / len(image_paths))
    clips = []

    for path in image_paths:
        try:
            clip = ImageClip(path).set_duration(clip_duration)
            clips.append(clip)
        except Exception as e:
            print(colored(f"[-] Error processing image {path}: {e}", "red"))

    if not clips:
        raise ValueError("No valid clips created from images")

    final_clip = concatenate_videoclips(clips, method="compose")
    output_path = f"../temp/{uuid4()}.mp4"
    final_clip.write_videofile(output_path, fps=24, codec="libx264", audio_codec="aac")
    print(colored(f"[+] Video created from {len(clips)} images -> {output_path}", "green"))
    return output_path


def cleanup_images(image_paths: List[str]):
    """Clean up temporary image files."""
    for path in image_paths:
        try:
            os.unlink(path)
        except:
            pass


