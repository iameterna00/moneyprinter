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

def generate_hf_images(prompt: str, contentType: str, model: str = "black-forest-labs/FLUX.1-schnell") -> str:
    import os
    from uuid import uuid4
    output_dir = os.path.join(os.path.dirname(__file__), "../temp")
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Fix comparison and colons
        if contentType == 'cartoon':
            newprompt = f'{prompt} night dark, cinematic cartoon style, with heavy shadows, dramatic lighting, night-time setting, detailed linework, and an eerie, surreal atmosphere. Inspired by adult animated shows and noir comics. Subtle neon glow, slightly distorted facial expressions, thick outlines, VHS effect, muted colors, and vintage textures. Stylized background with twilight skies, mysterious environments, and emotional tension. 4K, highly detailed, digital painting. Aspect ratio 9:16'
        else:
            newprompt = prompt

        # Generate the image
        image = client.text_to_image(newprompt, model=model)

        file_path = os.path.join(output_dir, f"{uuid4()}.png")
        image.save(file_path, "PNG")
        print(colored(f"[+] Saved image -> {file_path}", "green"))
        return file_path

    except Exception as e:
        print(colored(f"[-] Error generating image: {e}", "red"))
        return None


def cleanup_images(image_paths: List[str]):
    """Clean up temporary image files."""
    for path in image_paths:
        try:
            os.unlink(path)
        except:
            pass


