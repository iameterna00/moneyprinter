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

def generate_hf_images(prompt: str, amount: int = 2, model: str = "black-forest-labs/FLUX.1-schnell") -> List[str]:
    """Generate images using Hugging Face Inference API."""
    image_paths = []

    for i in range(amount):
        print(colored(f"[+] Generating image {i+1}/{amount} for: {prompt}", "blue"))

        try:
            # output is a PIL.Image object
            image = client.text_to_image(prompt, model=model)

            # Save image to temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            image.save(temp_file.name, "PNG")
            image_paths.append(temp_file.name)
            print(colored(f"[+] Saved image {i+1} -> {temp_file.name}", "green"))

        except Exception as e:
            print(colored(f"[-] Error generating image {i+1}: {e}", "red"))

    return image_paths


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


