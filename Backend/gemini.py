import os
import tempfile
from uuid import uuid4
from typing import List
from PIL import Image

# Compatibility fix for Pillow
if not hasattr(Image, "ANTIALIAS"):
    Image.ANTIALIAS = Image.Resampling.LANCZOS

from termcolor import colored
from moviepy.editor import ImageClip, concatenate_videoclips
from huggingface_hub import InferenceClient
from dotenv import load_dotenv


# ============================
# Load environment variables
# ============================
load_dotenv()  # loads .env file if available

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("[-] HF_TOKEN not found in environment variables or .env file.")

print(colored(f"[+] HF_TOKEN loaded", "green"))

# Initialize Hugging Face client
client = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN,
)


# ============================
# Image Generation Function
# ============================
def generate_hf_images(
    prompt: str,
    contentType: str,
    model: str = "black-forest-labs/FLUX.1-schnell",
) -> str:
    output_dir = os.path.join(os.path.dirname(__file__), "../temp")
    os.makedirs(output_dir, exist_ok=True)

    try:
        if contentType == "cartoon":
            newprompt = (
                f"{prompt} night dark, old cartoon style, with heavy shadows, "
                f"dramatic lighting, night-time setting, detailed linework, and an eerie, surreal atmosphere. "
                f"Inspired by adult animated shows and noir comics. Subtle neon glow, slightly distorted "
                f"facial expressions, thick outlines, VHS effect, muted colors, and vintage textures. "
                f"Stylized background with twilight skies, mysterious environments, and emotional tension. "
                f"4K, highly detailed, digital painting. Aspect ratio 9:16"
            )
        
        elif contentType == "silhouette":
            newprompt = (
                f"{prompt}. The scene is rendered in a dark, atmospheric style with a sharp silhouette of the character "
                f"standing out against a shadowy, abstract background. An eerie, glowing aura radiates from the figure, "
                f"casting high-contrast light and subtle highlights. Swirling tendrils of energy or smoke surround the character, "
                f"giving a sense of power and mystery. Intricate, faintly glowing symbols or patterns are subtly integrated into the composition, "
                f"adding layers of psychic or supernatural meaning. The overall mood is surreal, intense, and moody, "
                f"evoking a sense of control, dominance, or hidden power. High detail, digital art, dramatic lighting, cinematic 4K, vertical composition (9:16)."
            )

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


# ============================
# Cleanup Function
# ============================
def cleanup_images(image_paths: List[str]):
    """Clean up temporary image files."""
    for path in image_paths:
        try:
            os.unlink(path)
            print(colored(f"[+] Deleted -> {path}", "yellow"))
        except Exception as e:
            print(colored(f"[-] Could not delete {path}: {e}", "red"))


# ============================
# Main Entry Point
# ============================
def main():
    prompt = "A futuristic cityscape with neon lights"
    print(colored(f"[+] Generating image for prompt: {prompt}", "cyan"))

    image_path = generate_hf_images(prompt, contentType="cartoon")
    if image_path:
        print(colored(f"[+] Image available at: {image_path}", "green"))


if __name__ == "__main__":
    main()
