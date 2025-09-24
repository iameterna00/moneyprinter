import random
import numpy as np
from moviepy.editor import VideoFileClip
from termcolor import colored
import uuid
from PIL import Image
import os



def add_subtle_zoom_movement(clip, min_zoom=1.0, max_zoom=1.05, horizontal_range=50, cycles=1):
    """
    Smooth zoom in/out with **visible left-right movement**.
    - min_zoom -> max_zoom -> min_zoom
    - Horizontal movement goes left -> right -> left (oscillates)
    - 'horizontal_range' is max pixels moved left or right
    - 'cycles' = number of horizontal swings during clip
    """
    print(colored(f"[DEBUG] Applying Ken Burns effect: min_zoom={min_zoom}, max_zoom={max_zoom}, horizontal_range={horizontal_range}, cycles={cycles}", "yellow"))

    duration = clip.duration if getattr(clip, "duration", None) else 1.0
    duration = max(duration, 0.0001)

    def normalize_frame(frame):
        if frame is None or frame.size == 0:
            return np.zeros((720, 1280, 3), dtype=np.uint8)
        if frame.ndim == 2:
            frame = np.stack([frame]*3, axis=-1)
        if frame.dtype in (np.float32, np.float64):
            frame = np.clip(frame*255, 0, 255).astype(np.uint8)
        else:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if frame.shape[2] > 3:
            frame = frame[:, :, :3]
        return frame

    def apply_effect(get_frame, t):
        frame = normalize_frame(get_frame(t))
        h, w = frame.shape[:2]

        progress = t / duration  # 0 -> 1

        # Smooth zoom oscillation
        zoom_factor = min_zoom + (max_zoom - min_zoom) * 0.5 * (1 + np.sin(progress * 2 * np.pi))

        # Horizontal left-right movement (sine wave)
        offset_x = int(horizontal_range * np.sin(progress * 2 * np.pi * cycles))
        offset_y = 0  # optional vertical wiggle

        # Resize frame
        zoomed_w, zoomed_h = int(w * zoom_factor), int(h * zoom_factor)
        pil_img = Image.fromarray(frame).resize((zoomed_w, zoomed_h), Image.LANCZOS)
        resized_frame = np.array(pil_img)

        # Crop back to original size with offsets
        crop_x = (zoomed_w - w) // 2 + offset_x
        crop_y = (zoomed_h - h) // 2 + offset_y
        crop_x = np.clip(crop_x, 0, zoomed_w - w)
        crop_y = np.clip(crop_y, 0, zoomed_h - h)

        cropped_frame = resized_frame[crop_y:crop_y+h, crop_x:crop_x+w]

        # Safety: pad if mismatch
        if cropped_frame.shape[0] != h or cropped_frame.shape[1] != w:
            final = np.zeros((h, w, 3), dtype=np.uint8)
            y0 = max(0, (h - cropped_frame.shape[0]) // 2)
            x0 = max(0, (w - cropped_frame.shape[1]) // 2)
            final[y0:y0+cropped_frame.shape[0], x0:x0+cropped_frame.shape[1]] = cropped_frame
            return final

        return cropped_frame

    return clip.fl(apply_effect).set_duration(duration)



def add_shaky_effect(clip, intensity=10, frequency=20):
    """
    Adds a smooth vertical-only shaky effect (up and down) to a clip.
    
    Parameters:
    - intensity: maximum vertical displacement in pixels
    - frequency: how many shakes per second
    """
    print(colored(f"[DEBUG] Applying vertical-only shaky effect: intensity={intensity}, frequency={frequency}", "yellow"))
    
    original_duration = clip.duration
    
    # Initialize vertical movement variables
    current_offset_y = 0
    target_offset_y = 0
    last_change_time = 0
    
    def apply_shake(get_frame, t):
        nonlocal current_offset_y, target_offset_y, last_change_time
        
        # Update target vertical position based on frequency
        if t - last_change_time >= 1.0 / frequency:
            target_offset_y = random.uniform(-intensity, intensity)
            last_change_time = t
        
        # Smoothly interpolate towards target
        current_offset_y += (target_offset_y - current_offset_y) * 0.2
        
        # Get the original frame
        frame = get_frame(t)
        if frame is None or frame.size == 0:
            height, width = 720, 1280
            return np.zeros((height, width, 3), dtype=np.uint8)
        
        # Convert frame to uint8
        if frame.dtype != np.uint8:
            if frame.dtype in (np.float32, np.float64):
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
        
        height, width = frame.shape[:2]
        new_frame = np.zeros_like(frame)
        
        offset_y_int = int(current_offset_y)
        
        # Source and destination coordinates for vertical-only shift
        src_y_start = max(0, -offset_y_int)
        src_y_end = min(height, height - offset_y_int)
        dest_y_start = max(0, offset_y_int)
        dest_y_end = min(height, height + offset_y_int)
        
        # Copy shifted vertical portion
        if src_y_end > src_y_start and dest_y_end > dest_y_start:
            new_frame[dest_y_start:dest_y_end, :, :] = frame[src_y_start:src_y_end, :, :]
        else:
            return frame
        
        return new_frame
    
    # Apply effect
    modified_clip = clip.fl(apply_shake)
    return modified_clip.set_duration(original_duration)

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
            clip = clip.set_start(prompt_data["start"])
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
