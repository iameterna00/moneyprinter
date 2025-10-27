import os
import json
import re
from uuid import uuid4
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ltx import create_video_from_images_with_local_ltx
from autotts import tts_hf
from search import search_for_stock_videos
from termcolor import colored
from moviepy.config import change_settings
from moviepy.editor import AudioFileClip
from dotenv import load_dotenv
from gemini import generate_flux_image
from utils import clean_dir, check_env_vars
from gpt import generate_script, generate_metadata, get_image_search_terms, get_search_terms
from video import combine_videos, generate_subtitles, generate_video, save_video
from youtube import upload_video
from apiclient.errors import HttpError
import threading
load_dotenv("../.env")
check_env_vars()
SESSION_ID = os.getenv("TIKTOK_SESSION_ID")
change_settings({"IMAGEMAGICK_BINARY": os.getenv("IMAGEMAGICK_BINARY")})

app = Flask(__name__)
CORS(
    app,
    origins=[
        "https://nepwoop.com",  
        "http://localhost:5173" 
    ],
    methods=["GET", "POST", "OPTIONS", "DELETE"],
    allow_headers=["Content-Type", "Authorization"],
    supports_credentials=True
)
HOST = "0.0.0.0"
PORT = 8000

AMOUNT_OF_STOCK_VIDEOS = 8
GENERATING = False

GENERATED_VIDEOS_DIR = os.path.abspath("../Generated_Video")
os.makedirs(GENERATED_VIDEOS_DIR, exist_ok=True)

SONGS_DIR = os.path.abspath("../Songs")
os.makedirs(SONGS_DIR, exist_ok=True)

VOICE_DIR = os.path.abspath("../voice")
os.makedirs(VOICE_DIR, exist_ok=True)


# ============================
# Helper: Safe JSON parsing
# ============================
def safe_parse_json(response: str):
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r'\[.*\]', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return []
        return []

active_tasks = {}

def update_task_progress(task_id, status, progress=None, current_video=None, total_videos=None, message=None):
    """Update task progress in the shared dictionary"""
    if task_id not in active_tasks:
        active_tasks[task_id] = {}
    
    active_tasks[task_id]["status"] = status
    if progress is not None:
        active_tasks[task_id]["progress"] = progress
    if current_video is not None:
        active_tasks[task_id]["current_video"] = current_video
    if total_videos is not None:
        active_tasks[task_id]["total_videos"] = total_videos
    if message is not None:
        active_tasks[task_id]["message"] = message

# ===========================================
# Video generation endpoint - IMMEDIATE RESPONSE
# ============================================
@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.get_json()
    task_id = str(uuid4())
    
    # Initialize task
    update_task_progress(task_id, "processing", progress=0, current_video=0, message="Starting video generation...")
    
    # Start processing in background thread
    thread = threading.Thread(target=background_generation, args=(task_id, data))
    thread.daemon = True
    thread.start()
    
    # Return immediate response to prevent timeout
    return jsonify({
        "status": "processing", 
        "message": "Video generation has started. Check back in 5–10 minutes for results.",
        "task_id": task_id
    })

def background_generation(task_id, data):
    """Background task that contains your original generation logic"""
    global GENERATING
    try:
        GENERATING = True
        
        # =================================================
        # YOUR ORIGINAL GENERATION CODE STARTS HERE
        # =================================================
        custom_prompts = data.get('customPrompts', [])
        custom_prompts = [p for p in custom_prompts if p and p.strip()]
        use_custom_prompts = len(custom_prompts) > 0
        
        if use_custom_prompts:
            amountofshorts = len(custom_prompts)
        else:
            amountofshorts = int(data.get('threads', 1))
        
        paragraph_number = int(data.get('paragraphNumber', 1))
        ai_model = data.get('aiModel')
        subtitles_position = data.get('subtitlesPosition')
        text_color = data.get('color')
        use_music = data.get('useMusic', False)
        automate_youtube_upload = data.get('automateYoutubeUpload', False)
        contentType = data.get('contentType', "stock")
        songsName = data.get('songsName')
        voice = data.get("voiceName")
        video_subject = data.get('videoSubject', '')
        
        print(colored(f"[SELECTED SONG: {songsName}]", "blue"))
        print(colored(f"[Videos to be generated: {amountofshorts}]", "blue"))
        print(colored(f"   Subject: {video_subject}", "blue"))
        print(colored(f"   AI Model: {ai_model}", "blue"))
        print(colored(f"   Using custom prompts: {use_custom_prompts}", "blue"))
        if use_custom_prompts:
            print(colored(f"   Custom prompts count: {len(custom_prompts)}", "blue"))

        generated_video_paths = []
        
        # ============================
        # LOOP GENERATION
        # ============================
        for video_index in range(amountofshorts):
            # Check for cancellation at the start of each video
            if not GENERATING:
                update_task_progress(task_id, "cancelled", message="Video generation was cancelled.")
                break
                
            print(colored(f"\n[+] Generating video {video_index + 1} of {amountofshorts}", "green"))
            
            # Update progress for current video
            progress = 10 + (video_index / amountofshorts) * 80
            update_task_progress(
                task_id, 
                "processing", 
                progress=progress,
                current_video=video_index + 1,
                total_videos=amountofshorts,
                message=f"Generating video {video_index + 1} of {amountofshorts}"
            )
            
            # Clean temp directories for each video
            clean_dir("../temp/")
            clean_dir("../subtitles/")

            # ============================
            # Generate script
            # ============================
            update_task_progress(task_id, "processing", message="Generating script...")
            
            if use_custom_prompts:
                script = custom_prompts[video_index]  # use specific custom prompt
                print(colored(f"   Using custom prompt: {script[:100]}...", "blue"))
            else:
                script = generate_script(
                    video_subject, 
                    paragraph_number, 
                    ai_model, 
                    None  # no custom prompt in traditional flow
                )
                print(colored(f"   Generated script: {script[:100]}...", "blue"))

            # ============================
            # Generate TTS Audio
            # ============================
            update_task_progress(task_id, "processing", message="Generating audio...")
            
            # Save the full script as one TTS clip
            tts_path = f"../temp/{uuid4()}.mp3"
            voice_path = f"../voice/{voice}" if voice else "../voice/Michel.mp3"
            tts_hf(script, output_file=tts_path, audio_prompt=voice_path)
            final_audio = AudioFileClip(tts_path)

            # ===================================
            # Generate subtitles With Time Stamp
            # ==================================
            update_task_progress(task_id, "processing", message="Generating subtitles...")
            
            try:
                subtitles_path = generate_subtitles(
                    audio_path=tts_path,
                )
            except Exception as e:
                print(colored(f"[-] Error generating subtitles: {e}", "red"))
                subtitles_path = None

            # ============================
            # Fetch media based on contentType
            # ============================
            update_task_progress(task_id, "processing", message="Fetching media content...")
            
            media_paths = []
            image_prompts = []
            
            if contentType == "stock":
                search_terms = get_search_terms(video_subject, AMOUNT_OF_STOCK_VIDEOS, script, ai_model)
                video_urls = []
                for term in search_terms:
                    found = search_for_stock_videos(term, os.getenv("PEXELS_API_KEY"), it=15, min_dur=10)
                    for url in found:
                        if url not in video_urls:
                            video_urls.append(url)
                            break
                if not video_urls:
                    print(colored("[-] No stock videos found for this video.", "red"))
                    continue
                media_paths = [save_video(url) for url in video_urls]
            else:
                # Generative content flow
                image_prompts = get_image_search_terms(video_subject, AMOUNT_OF_STOCK_VIDEOS, subtitles_path, ai_model)
                for term_data in image_prompts:
                    prompt = term_data["Img prompt"] if isinstance(term_data, dict) else term_data
                    try:
                        generated = generate_flux_image(prompt, contentType)
                        if generated:  # make sure it's not None
                            media_paths.append(generated)
                        if len(media_paths) >= AMOUNT_OF_STOCK_VIDEOS:
                            break
                    except Exception as e:
                        print(f"Could not generate image: {e}")

                if not media_paths:
                    print(colored("[-] No images generated for this video.", "red"))
                    continue
                print(colored(f"[+] {len(media_paths)} images generated!", "green"))

            print(colored("[+] Script generated!\n", "green"))

            # ============================
            # Create video
            # ============================
            update_task_progress(task_id, "processing", message="Creating video...")
            
            n_threads = 1
            
            if contentType == "stock":
                # Stock videos: combine downloaded video clips
                temp_audio = AudioFileClip(tts_path)
                combined_video_path = combine_videos(media_paths, temp_audio.duration, 3, n_threads)
                temp_audio.close()
            else:
                if len(image_prompts) < len(media_paths):
                    last_prompt = image_prompts[-1] if image_prompts else {"Img prompt": "Abstract technology background"}
                    while len(image_prompts) < len(media_paths):
                        image_prompts.append(last_prompt)
                
                combined_video_path = create_video_from_images_with_local_ltx(media_paths, image_prompts, final_audio.duration)

            # Save final video with unique name
            final_filename = f"output_{uuid4().hex[:8]}.mp4"
            final_video_path = os.path.join(GENERATED_VIDEOS_DIR, final_filename)
            
            bg_music_path = f"../Songs/{songsName}" if songsName else "../Songs/shadow.mp3"
            bg_music_volume = 0.3
            
            # Generate the final video
            update_task_progress(task_id, "processing", message="Finalizing video...")
            
            generate_video(
                combined_video_path, tts_path, subtitles_path,
                n_threads, subtitles_position, text_color or "#FFFF00", 
                bg_music_path, bg_music_volume, final_video_path
            )
            
            # Close audio clips to free resources
            final_audio.close()
            
            # Add to list of generated videos
            generated_video_paths.append(final_video_path)
            
            # ============================
            # Generate metadata
            # ============================
            update_task_progress(task_id, "processing", message="Generating metadata...")
            
            title, description, keywords = generate_metadata(video_subject, script, ai_model)

            # ============================
            # Optional YouTube upload
            # ============================
            if automate_youtube_upload:
                update_task_progress(task_id, "processing", message="Uploading to YouTube...")
                
                client_secrets_file = os.path.abspath("./client_secret.json")
                if os.path.exists(client_secrets_file):
                    video_metadata = {
                        'video_path': os.path.abspath(final_video_path),
                        'title': title,
                        'description': description,
                        'category': "28",  # Science & Technology
                        'keywords': ",".join(keywords),
                        'privacyStatus': "private",
                    }
                    try:
                        upload_video(**video_metadata)
                    except HttpError as e:
                        print(f"An HTTP error {e.resp.status} occurred:\n{e.content}")

            print(colored(f"[+] Video {video_index + 1} generated: {final_video_path}!", "green"))

        # After loop completion, determine final status
        if not GENERATING:
            # Generation was cancelled during the process
            update_task_progress(task_id, "cancelled", message="Video generation was cancelled.")
        elif generated_video_paths:
            # Generation completed successfully
            GENERATING = False
            video_filenames = [os.path.basename(path) for path in generated_video_paths]
            update_task_progress(
                task_id, 
                "success", 
                progress=100,
                message=f"{len(generated_video_paths)} videos generated!",
                data=video_filenames
            )
        else:
            # Generation completed but no videos were created
            GENERATING = False
            update_task_progress(task_id, "error", message="No videos were generated.")

    except Exception as err:
        print(colored(f"[-] Error: {err}", "red"))
        GENERATING = False
        update_task_progress(task_id, "error", message=str(err))
# ===========================================
# Check generation status
# ===========================================
@app.route("/api/generate/status/<task_id>")
def check_status(task_id):
    task = active_tasks.get(task_id)
    if not task:
        return jsonify({"status": "not_found", "message": "Task not found"})
    
    return jsonify(task)

# ============================
# Cancel generation
# ============================
@app.route("/api/cancel", methods=["POST"])
def cancel():
    global GENERATING
    GENERATING = False
    print(colored("[!] Received cancellation request...", "yellow"))
    return jsonify({"status": "success", "message": "Cancelled video generation."})

# ============================
# List and serve generated videos
# ============================

@app.route("/api/videos")
def list_videos():
    if not os.path.exists(GENERATED_VIDEOS_DIR):
        return jsonify({"status": "success", "videos": []})
    
    files = [f for f in os.listdir(GENERATED_VIDEOS_DIR) if f.endswith(".mp4")]
    return jsonify({"status": "success", "videos": files})


@app.route("/api/video/<filename>")
def serve_video(filename):
    if os.path.exists(os.path.join(GENERATED_VIDEOS_DIR, filename)):
        return send_from_directory(GENERATED_VIDEOS_DIR, filename)
    else:
        return jsonify({"status": "error", "message": "Video not found"}), 404
    

@app.route("/api/songs")
def list_songs(): 
    if not os.path.exists(SONGS_DIR):
        return jsonify({"status": "success", "songs": []})
    
    files = [f for f in os.listdir(SONGS_DIR) if f.endswith(".mp3")]
    return jsonify({"status": "success", "songs": files})


@app.route("/api/songs/<path:filename>")
def get_song(filename):  
    return send_from_directory(SONGS_DIR, filename)

@app.route("/api/voice")
def list_voice(): 
    if not os.path.exists(VOICE_DIR):
        return jsonify({"status": "success", "voice": []})
    
    files = [f for f in os.listdir(VOICE_DIR) if f.endswith(".mp3")]
    return jsonify({"status": "success", "voice": files})


@app.route("/api/voice/<path:filename>")
def get_voice(filename):  
    return send_from_directory(VOICE_DIR, filename)


# ============================
# Run server
# ============================
if __name__ == "__main__":
    print(colored(f"[INFO] Server is running on http://{HOST}:{PORT}", "green"))
    app.run(debug=True, host=HOST, port=PORT)