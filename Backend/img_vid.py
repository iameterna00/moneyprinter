import os
import uuid
from termcolor import colored
from moviepy.editor import VideoFileClip, ColorClip, CompositeVideoClip, ImageClip
from moviepy.video.fx.all import resize
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv("../.env")

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("[-] HF_TOKEN not found in environment variables or .env file.")

# Hugging Face client
client = InferenceClient(provider="fal-ai", api_key=HF_TOKEN)

# TikTok video dimensions
TIKTOK_WIDTH = 720
TIKTOK_HEIGHT = 1280

def generate_video_from_image(image_path, prompt, duration=4):
    """
    Generate a short video from an image using LTX-Video.
    Returns path to the generated video. If generation fails, creates fallback video from image.
    """
    output_path = f"../temp/{uuid.uuid4()}.mp4"
    try:
        with open(image_path, "rb") as f:
            img_bytes = f.read()

        response = client.image_to_video(img_bytes, prompt=prompt, model="Lightricks/LTX-Video")

        if isinstance(response, dict) and "video" in response:
            video_url = response["video"]["url"]
            video_bytes = client._fetch_video(video_url)
            with open(output_path, "wb") as f:
                f.write(video_bytes)
        elif isinstance(response, bytes):
            with open(output_path, "wb") as f:
                f.write(response)
        else:
            raise ValueError("No video returned from API")

        return output_path

    except Exception as e:
        print(colored(f"❌ Failed to generate video for {image_path}: {e}", "red"))
        print(colored(f"⚠️ Using fallback video from image", "yellow"))

        # Fallback: create video from static image
        fallback_clip = ImageClip(image_path).set_duration(duration)
        fallback_clip = resize(fallback_clip, height=TIKTOK_HEIGHT)

        # Pad width if smaller than TikTok width
        if fallback_clip.w < TIKTOK_WIDTH:
            bg_clip = ColorClip(size=(TIKTOK_WIDTH, TIKTOK_HEIGHT), color=(0,0,0), duration=duration)
            fallback_clip = CompositeVideoClip([bg_clip, fallback_clip.set_position("center")])

        fallback_clip.write_videofile(output_path, fps=24, threads=2)
        return output_path

def create_video_from_images_with_ltx(image_paths, image_prompts_with_timing, audio_duration):
    """
    Combine multiple images into a TikTok video using LTX-Video with fallback handling.
    """
    video_clips = []

    # Ensure prompt list matches image list
    min_len = min(len(image_paths), len(image_prompts_with_timing))
    image_paths = image_paths[:min_len]
    valid_prompts = image_prompts_with_timing[:min_len]

    for i, img_path in enumerate(image_paths):
        prompt = valid_prompts[i].get("Img prompt", "Cinematic transformation")
        start_time = valid_prompts[i].get("start", 0)
        end_time = valid_prompts[i].get("end", start_time + 4)
        duration = end_time - start_time

        print(colored(f"🎬 Generating video from image {img_path} ({duration}s) with prompt: {prompt[:50]}...", "blue"))
        video_path = generate_video_from_image(img_path, prompt, duration)

        # Load clip and apply TikTok format adjustments
        if video_path and os.path.exists(video_path):
            clip = VideoFileClip(video_path).set_start(start_time).set_duration(duration)
            clip = resize(clip, height=TIKTOK_HEIGHT)

            # Crop or pad width to TikTok width
            if clip.w > TIKTOK_WIDTH:
                clip = clip.crop(x_center=clip.w/2, width=TIKTOK_WIDTH)
            elif clip.w < TIKTOK_WIDTH:
                bg_clip = ColorClip(size=(TIKTOK_WIDTH, TIKTOK_HEIGHT), color=(0,0,0), duration=duration)
                clip = CompositeVideoClip([bg_clip, clip.set_position("center")])

            video_clips.append(clip)
        else:
            # Extra safety fallback (black screen)
            fallback_clip = ColorClip(size=(TIKTOK_WIDTH, TIKTOK_HEIGHT), color=(0,0,0), duration=duration).set_start(start_time)
            video_clips.append(fallback_clip)

    # Combine all clips
    if not video_clips:
        print(colored("❌ No clips generated. Creating blank video.", "red"))
        final_clip = ColorClip(size=(TIKTOK_WIDTH, TIKTOK_HEIGHT), color=(0,0,0), duration=audio_duration)
    else:
        final_clip = CompositeVideoClip(video_clips).set_duration(audio_duration)

    # Save final video
    output_file = f"../temp/final_combined_video_{uuid.uuid4()}.mp4"
    final_clip.write_videofile(output_file, fps=24, threads=2)
    return output_file

def main():
    # Example images and prompts
    image_paths = [
        "../test/9fbd030a-1864-48c7-a980-4483019fef75.png",
        "../test/42cf679c-0a76-436d-9a0b-ddfdf39c61d5.png",
        "../test/73ee5776-1ba7-4049-9d3e-203f3aaffead.png",
        "../test/0936ea2f-40f7-4f6d-90ad-59609640c994.png",
    ]

    image_prompts_with_timing = [
        {"Img prompt": "Extreme close-up on a porcelain doll's face, cracked eye oozing black fluid", "start": 0.0, "end": 4.0},
        {"Img prompt": "The porcelain doll is held by a trembling young girl in a dark attic", "start": 4.0, "end": 8.0},
        {"Img prompt": "The girl is lying motionless, doll on her chest, moonlight highlights dust", "start": 8.0, "end": 12.0},
    ]

    audio_duration = 12.0

    print(colored("[+] Creating final TikTok video from images using LTX-Video...", "green"))
    final_video_path = create_video_from_images_with_ltx(image_paths, image_prompts_with_timing, audio_duration)
    print(colored(f"✅ Final TikTok video saved at: {final_video_path}", "green"))

if __name__ == "__main__":
    main()
