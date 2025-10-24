import os
import shutil
import requests
from uuid import uuid4
from gradio_client import Client

# Initialize client
client = Client("http://localhost:7860/")

def safe_predict(step_name, **kwargs):
    try:
        result = client.predict(**kwargs)
        print(f"[SUCCESS] {step_name}")
        return result
    except Exception as e:
        print(f"[FAIL] {step_name} -> {e}")
        return None

def copy_image_from_gradio_temp(gradio_temp_path: str) -> str:
    """
    Copy the generated image from Gradio's temp directory to our temp folder
    """
    # Create our output directory
    output_dir = os.path.join(os.path.dirname(__file__), "../temp")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate unique filename
    file_extension = os.path.splitext(gradio_temp_path)[1] or '.jpg'
    local_file_path = os.path.join(output_dir, f"{uuid4()}{file_extension}")
    
    try:
        # Copy the file from Gradio temp to our temp
        shutil.copy2(gradio_temp_path, local_file_path)
        print(f"[+] Copied image from Gradio temp -> {local_file_path}")
        return local_file_path
    except Exception as e:
        print(f"[-] Error copying image: {e}")
        return None

def generate_flux_image(prompt: str, contentType: str):
    """
    Full Gradio pipeline for generating an image using the provided prompt.
    Returns the file path of the saved image in our temp folder.
    """
    if contentType == "cartoon":
        newprompt = (
            f"{prompt} night dark, old cartoon style, with heavy shadows, "
            f"dramatic lighting, night-time setting, detailed linework, and an eerie, surreal atmosphere. "
            f"Inspired by adult animated shows and noir comics. Subtle neon glow, slightly distorted "
            f"facial expressions, thick outlines, VHS effect, muted colors, and vintage textures. "
            f"Stylized background with twilight skies, mysterious environments, and emotional tension. "
            f"4K, highly detailed. Aspect ratio 9:16"
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

    # Store the final image path returned by Gradio
    gradio_image_path = None

    # --- Step 1: Change model family base ---
    safe_predict(
        "Change model family base",
        current_model_family="flux",
        api_name="/change_model_family"
    )

    safe_predict(
         "restart",
        api_name="/preload_model_when_switching"
    )
    safe_predict(
         "Change model",
        model_choice="flux_schnell",
        api_name="/change_model"
    )

    # --- Step 2: Validate wizard prompt ---
    safe_predict(
        "Validate wizard prompt",
        wizard_prompt_activated="off",
        wizard_variables_names="",
        prompt=newprompt,
        wizard_prompt="Hello!!",
        param_5="Hello!!",
        param_6="Hello!!",
        param_7="Hello!!",
        param_8="Hello!!",
        param_9="Hello!!",
        param_10="Hello!!",
        param_11="Hello!!",
        param_12="Hello!!",
        param_13="Hello!!",
        param_14="Hello!!",
        api_name="/validate_wizard_prompt_11"
    )

    # --- Step 3: Save inputs ---
    safe_predict(
        "Save inputs",
        target="state",
        image_mask_guide={"background": None, "layers": [], "composite": None},
        lset_name='',
        image_mode=1,
        prompt=newprompt,
        negative_prompt="",
        resolution="480x832",
        video_length=81,
        batch_size=1,
        seed=-1,
        force_fps="",
        num_inference_steps=10,
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
        image_prompt_type="",
        image_start=[],
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
        sliding_window_overlap=5,
        sliding_window_color_correction_strength=0,
        sliding_window_overlap_noise=0,
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
    


    # --- Step 5: Preload / fill inputs ---
    safe_predict("Fill inputs", api_name="/fill_inputs_1")
    safe_predict("Preload model when switching", api_name="/preload_model_when_switching")

    # --- Step 6: Process prompt & tasks ---
    safe_predict("Process prompt & add tasks", api_name="/process_prompt_and_add_tasks",model_choice="flux_schnell")
    safe_predict("Prepare video generation", api_name="/prepare_generate_video")
    safe_predict("Activate status", api_name="/activate_status")
    
    # This is where the image generation happens
    result = safe_predict("Process tasks", api_name="/process_tasks")
    
    # Check if result contains the image path
    if result and isinstance(result, str) and os.path.exists(result):
        gradio_image_path = result
        print(f"[+] Image generated at: {gradio_image_path}")

    # --- Step 7: Refresh galleries to see the result ---
    for i in range(3):  # Fewer iterations should be enough
        gallery_result = safe_predict(f"Refresh gallery {i+1}", api_name="/refresh_gallery")
        
        # Gallery might return the latest image path
        if gallery_result and isinstance(gallery_result, str) and os.path.exists(gallery_result):
            gradio_image_path = gallery_result
            print(f"[+] Found image in gallery: {gradio_image_path}")

    # --- Step 8: Finalize generation ---
    final_result = safe_predict("Finalize generation", api_name="/finalize_generation")
    if final_result and isinstance(final_result, str) and os.path.exists(final_result):
        gradio_image_path = final_result

    # --- Step 9: Copy the image to our temp folder ---
    local_image_path = None
    if gradio_image_path and os.path.exists(gradio_image_path):
        local_image_path = copy_image_from_gradio_temp(gradio_image_path)
    else:
        print("[-] No image path found in results")
        gradio_temp_dir = r"C:\Users\Anuj\AppData\Local\Temp\gradio"
        if os.path.exists(gradio_temp_dir):
            image_files = []
            for root, dirs, files in os.walk(gradio_temp_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        full_path = os.path.join(root, file)
                        image_files.append((full_path, os.path.getctime(full_path)))
            
            if image_files:
                # Get the most recently created image
                latest_image = max(image_files, key=lambda x: x[1])
                gradio_image_path = latest_image[0]
                print(f"[+] Found latest image: {gradio_image_path}")
                local_image_path = copy_image_from_gradio_temp(gradio_image_path)

    # --- Step 10: Cleanup ---
    safe_predict("Unload model", api_name="/unload_model_if_needed")

    return local_image_path


# ============================
# Cleanup Function
# ============================
def cleanup_images(image_paths: list):
    """Clean up temporary image files."""
    for path in image_paths:
        try:
            os.unlink(path)
            print(f"[+] Deleted -> {path}")
        except Exception as e:
            print(f"[-] Could not delete {path}: {e}")


# ============================
# Main Entry Point
# ============================
def main():
    prompt = "A futuristic cityscape with neon lights"
    print(f"[+] Generating image for prompt: {prompt}")

    image_path = generate_flux_image(prompt, contentType="cartoon")
    if image_path:
        print(f"[+] Image available at: {image_path}")
    else:
        print("[-] Failed to generate image")


if __name__ == "__main__":
    main()