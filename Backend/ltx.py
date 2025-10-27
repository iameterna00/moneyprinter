import os
import uuid
import shutil
from termcolor import colored
from moviepy.editor import VideoFileClip, ColorClip, CompositeVideoClip, ImageClip
from moviepy.video.fx.all import resize
from gradio_client import Client, handle_file
from dotenv import load_dotenv

# Load environment variables
load_dotenv("../.env")

# TikTok video dimensions
TIKTOK_WIDTH = 720
TIKTOK_HEIGHT = 1280

# Initialize Gradio client for local LTX
client = Client("http://localhost:7860/")

def safe_predict(step_name, **kwargs):
    """Safe wrapper for Gradio client predictions"""
    try:
        result = client.predict(**kwargs)
        print(f"[SUCCESS] {step_name}")
        return result
    except Exception as e:
        print(f"[FAIL] {step_name} -> {e}")
        return None

def copy_video_from_gradio_temp(gradio_temp_path: str) -> str:
    """
    Copy the generated video from Gradio's temp directory to our temp folder
    """
    output_dir = os.path.join(os.path.dirname(__file__), "../temp")
    os.makedirs(output_dir, exist_ok=True)
    
    file_extension = os.path.splitext(gradio_temp_path)[1] or '.mp4'
    local_file_path = os.path.join(output_dir, f"{uuid.uuid4()}{file_extension}")
    
    try:
        shutil.copy2(gradio_temp_path, local_file_path)
        print(f"[+] Copied video from Gradio temp -> {local_file_path}")
        return local_file_path
    except Exception as e:
        print(f"[-] Error copying video: {e}")
        return None

def generate_video_from_image_local_ltx(image_path, prompt, duration=4, contentType=None):
    """
    Generate a short video from an image using local LTX-Video.
    Returns path to the generated video. If generation fails, creates fallback video from image.
    """
    output_path = f"../temp/{uuid.uuid4()}.mp4"
    
    try:
        print(colored(f"🎬 Starting local LTX video generation for: {os.path.basename(image_path)}", "blue"))
        print(colored(f"📝 Prompt: {prompt}", "cyan"))
        
        # Apply content type modifications to prompt if needed
        if contentType == "cartoon":
            newprompt = (
                f"{prompt} night dark, old cartoon style, with heavy shadows, "
                f"dramatic lighting, night-time setting, detailed linework, and an eerie, surreal atmosphere. "
                f"Inspired by adult animated shows and noir comics. Subtle neon glow, slightly distorted "
                f"facial expressions, thick outlines, VHS effect, muted colors, and vintage textures. "
                f"Stylized background with twilight skies, mysterious environments, and emotional tension. "
                f"4K, highly detailed, digital painting."
            )
        elif contentType == "silhouette":
            newprompt = (
                f"{prompt}. The scene is rendered in a dark, atmospheric style with a sharp silhouette of the character "
                f"standing out against a shadowy, abstract background. An eerie, glowing aura radiates from the figure, "
                f"casting high-contrast light and subtle highlights. Swirling tendrils of energy or smoke surround the character, "
                f"giving a sense of power and mystery. Intricate, faintly glowing symbols or patterns are subtly integrated into the composition, "
                f"adding layers of psychic or supernatural meaning. The overall mood is surreal, intense, and moody, "
                f"evoking a sense of control, dominance, or hidden power. High detail, digital art, dramatic lighting, cinematic 4K."
            )
        else:
            newprompt = prompt

        # Store the final video path returned by Gradio
        gradio_video_path = None

        # --- Step 1: Change model family ---

        safe_predict("Change model family", current_model_family="ltxv", api_name="/change_model_family")
        safe_predict(
         "restart",
        api_name="/preload_model_when_switching"
        )
        safe_predict("Change model", model_choice='ltxv_distilled', api_name="/change_model")

        # --- Step 2: Refresh image prompt type ---
        safe_predict("Refresh image prompt type", image_prompt_type='', image_prompt_type_radio='', api_name="/refresh_image_prompt_type_radio")

        # --- Step 3: Upload and process the input image ---
        safe_predict("Upload image", value=[{"image": handle_file(image_path), "caption": None}], api_name="/_on_upload")
        safe_predict("Gallery change", value=[{"image": handle_file(image_path), "caption": None}], api_name="/_on_gallery_change")

        # --- Step 4: Initialize generation ---
        safe_predict("Init generate", input_file_list=None, last_choice=0, api_name="/init_generate")

        # --- Step 5: Validate wizard prompt ---
        safe_predict(
            "Validate wizard prompt",
            wizard_prompt_activated=None,
            wizard_variables_names="on",
            prompt="",
            wizard_prompt=newprompt,
            param_5="make it walk perfectly",
            param_6="",
            param_7="",
            param_8="",
            param_9="",
            param_10="",
            param_11="",
            param_12="",
            param_13="",
            param_14="",
            api_name="/validate_wizard_prompt_11"
        )

        video_length = int(duration * 30)  

        # --- Step 6: Save inputs with image reference ---
        safe_predict(
            "Save inputs for video generation",
            target="state",
            image_mask_guide={"background": None, "layers": [], "composite": None},
            lset_name="",
            image_mode=0,
            prompt=newprompt,
            negative_prompt="",
            resolution="480x832",
            video_length=video_length,
            batch_size=1,
            seed=-1,
            force_fps="",
            num_inference_steps=6,
            guidance_scale=5,
            guidance2_scale=5,
            guidance3_scale=5,
            switch_threshold=0,
            switch_threshold2=0,
            guidance_phases=1,
            model_switch_phase=1,
            audio_guidance_scale=4,
            flow_shift=5,
            sample_solver="",
            embedded_guidance_scale=6,
            repeat_generation=1,
            multi_prompts_gen_type=0,
            multi_images_gen_type=0,
            skip_steps_cache_type="",
            skip_steps_multiplier=1.5,
            skip_steps_start_step_perc=20,
            loras_choices=[],
            loras_multipliers="",
            image_prompt_type="S",  # Single image mode
            image_start=[{"image": handle_file(image_path), "caption": None}],  # Your input image
            image_end=[],
            model_mode=None,
            video_source=None,
            keep_frames_video_source="",
            video_guide_outpainting="#",
            video_prompt_type="",
            image_refs=[],
            frames_positions="",
            video_guide=None,
            image_guide=None,
            keep_frames_video_guide="",
            denoising_strength=0.5,
            video_mask=None,
            image_mask=None,
            control_net_weight=1,
            control_net_weight2=1,
            control_net_weight_alt=1,
            mask_expand=0,
            audio_guide=None,
            audio_guide2=None,
            audio_source=None,
            audio_prompt_type="V",
            speakers_locations="0:45 55:100",
            sliding_window_size=129,
            sliding_window_overlap=9,
            sliding_window_color_correction_strength=0,
            sliding_window_overlap_noise=20,
            sliding_window_discard_last_frames=0,
            image_refs_relative_size=50,
            remove_background_images_ref=0,
            temporal_upsampling="",
            spatial_upsampling="",
            film_grain_intensity=0,
            film_grain_saturation=0.5,
            MMAudio_setting=0,
            MMAudio_prompt="",
            MMAudio_neg_prompt="",
            RIFLEx_setting=0,
            NAG_scale=1,
            NAG_tau=3.5,
            NAG_alpha=0.5,
            slg_switch=0,
            slg_layers=[9],
            slg_start_perc=10,
            slg_end_perc=90,
            apg_switch=0,
            cfg_star_switch=0,
            cfg_zero_step=-1,
            prompt_enhancer="",
            min_frames_if_references=1,
            override_profile=-1,
            mode="",
            api_name="/save_inputs_11"
        )

        # --- Step 7: Process prompt and add tasks ---
        safe_predict("Process prompt and add tasks", model_choice="ltxv_distilled", api_name="/process_prompt_and_add_tasks")

        # --- Step 8: Prepare video generation ---
        safe_predict("Prepare video generation", api_name="/prepare_generate_video")
        safe_predict("Activate status", api_name="/activate_status")

        # --- Step 9: Process tasks (this is where video generation happens) ---
        result = safe_predict("Process tasks", api_name="/process_tasks")
        
        # Check if result contains the video path
        if result and isinstance(result, str) and os.path.exists(result):
            gradio_video_path = result
            print(colored(f"✅ Video generated at: {gradio_video_path}", "green"))

        # --- Step 10: Refresh status and galleries ---
        safe_predict("Refresh status", api_name="/refresh_status_async")

        # Refresh gallery multiple times to catch the video
        for i in range(5):
            gallery_result = safe_predict(f"Refresh gallery {i+1}", api_name="/refresh_gallery")
            
            # Gallery might return the latest video path
            if gallery_result and isinstance(gallery_result, str) and os.path.exists(gallery_result):
                gradio_video_path = gallery_result
                print(colored(f"✅ Found video in gallery: {gradio_video_path}", "green"))

        # Refresh preview multiple times
        for i in range(10):
            safe_predict(f"Refresh preview {i+1}", api_name="/refresh_preview")

        # --- Step 11: Finalize generation ---
        final_result = safe_predict("Finalize generation", api_name="/finalize_generation")
        if final_result and isinstance(final_result, str) and os.path.exists(final_result):
            gradio_video_path = final_result

        # Refresh gallery one more time
        safe_predict("Final gallery refresh", api_name="/refresh_gallery")
        safe_predict("Final preview refresh", api_name="/refresh_preview")

        # --- Step 12: Copy the video to our temp folder ---
        local_video_path = None
        if gradio_video_path and os.path.exists(gradio_video_path):
            local_video_path = copy_video_from_gradio_temp(gradio_video_path)
        else:
            print(colored("⚠️ No video path found in results, searching Gradio temp directory...", "yellow"))
            
            # Try to find the latest video in Gradio temp directory
            gradio_temp_dir = r"C:\Users\Anuj\AppData\Local\Temp\gradio"
            if os.path.exists(gradio_temp_dir):
                video_files = []
                for root, dirs, files in os.walk(gradio_temp_dir):
                    for file in files:
                        if file.lower().endswith(('.mp4', '.avi', '.mov', '.webm')):
                            full_path = os.path.join(root, file)
                            video_files.append((full_path, os.path.getctime(full_path)))
                
                if video_files:
                    latest_video = max(video_files, key=lambda x: x[1])
                    gradio_video_path = latest_video[0]
                    print(colored(f"✅ Found latest video: {gradio_video_path}", "green"))
                    local_video_path = copy_video_from_gradio_temp(gradio_video_path)

        # --- Step 13: Cleanup ---
        safe_predict("Unload model", api_name="/unload_model_if_needed")

        if local_video_path:
            return local_video_path
        else:
            raise Exception("Local LTX video generation failed")

    except Exception as e:
        print(colored(f"❌ Failed to generate video for {image_path} using local LTX: {e}", "red"))
        print(colored(f"⚠️ Using fallback video from image", "yellow"))

        # Fallback: create video from static image (same as before)
        fallback_clip = ImageClip(image_path).set_duration(duration)
        fallback_clip = resize(fallback_clip, height=TIKTOK_HEIGHT)

        # Pad width if smaller than TikTok width
        if fallback_clip.w < TIKTOK_WIDTH:
            bg_clip = ColorClip(size=(TIKTOK_WIDTH, TIKTOK_HEIGHT), color=(0,0,0), duration=duration)
            fallback_clip = CompositeVideoClip([bg_clip, fallback_clip.set_position("center")])

        fallback_clip.write_videofile(output_path, fps=30, threads=1)
        return output_path

def create_video_from_images_with_local_ltx(image_paths, image_prompts_with_timing, audio_duration, contentType=None):
    """
    Combine multiple images into a TikTok video using local LTX-Video with fallback handling.
    """
    print("image prompt============timing",image_prompts_with_timing)
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
        
        # Use local LTX instead of Hugging Face API
        video_path = generate_video_from_image_local_ltx(img_path, prompt, duration, contentType)

        # Load clip and apply TikTok format adjustments (same as before)
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
    final_clip.write_videofile(output_file, fps=30, threads=1)
    print(colored("🧹 Unloading model after full video generation...", "yellow"))
    safe_predict("Unload model", api_name="/unload_model_if_needed")
    return output_file

def main():
    # Example images and prompts (same as before)
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

    print(colored("[+] Creating final TikTok video from images using Local LTX-Video...", "green"))
    final_video_path = create_video_from_images_with_local_ltx(image_paths, image_prompts_with_timing, audio_duration)
    print(colored(f"✅ Final TikTok video saved at: {final_video_path}", "green"))

if __name__ == "__main__":
    main()