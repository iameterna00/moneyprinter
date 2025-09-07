import os
import json
import re
from uuid import uuid4
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from eleven import ttselv
from search import search_for_stock_videos
from termcolor import colored
from moviepy.config import change_settings
from moviepy.editor import AudioFileClip, concatenate_audioclips
from dotenv import load_dotenv

# Custom imports
from gemini import create_video_from_images, generate_hf_images
from utils import clean_dir, check_env_vars, fetch_songs
from gpt import generate_response, generate_script, generate_metadata, get_image_search_terms, get_search_terms
from tiktokvoice import tts
from video import combine_videos, generate_subtitles, generate_video, save_video
from youtube import upload_video
from apiclient.errors import HttpError

# Load environment variables
load_dotenv("../.env")
check_env_vars()
SESSION_ID = os.getenv("TIKTOK_SESSION_ID")
change_settings({"IMAGEMAGICK_BINARY": os.getenv("IMAGEMAGICK_BINARY")})

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Constants
HOST = "0.0.0.0"
PORT = 8080
AMOUNT_OF_STOCK_VIDEOS = 1
GENERATING = False

# Directory for storing all generated videos
GENERATED_VIDEOS_DIR = os.path.abspath("../Generated_Video")
os.makedirs(GENERATED_VIDEOS_DIR, exist_ok=True)

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

# ============================
# Video generation endpoint
# ============================
@app.route("/api/generate", methods=["POST"])
def generate():
    global GENERATING
    try:
        GENERATING = True

        # Clean temp directories
        clean_dir("../temp/")
        clean_dir("../subtitles/")

        # Parse request data
        data = request.get_json()
        paragraph_number = int(data.get('paragraphNumber', 1))
        ai_model = data.get('aiModel')
        n_threads = data.get('threads')
        subtitles_position = data.get('subtitlesPosition')
        text_color = data.get('color')
        use_music = data.get('useMusic', False)
        automate_youtube_upload = data.get('automateYoutubeUpload', False)
        contentType = data.get('contentType', "stock")
        songs_zip_url = data.get('zipUrl')
        voice = data.get("voice", "en_us_001")
        voice_prefix = voice[:2]

        # Download music if needed
        if use_music:
            fetch_songs(songs_zip_url or
                        "https://filebin.net/2avx134kdibc4c3q/drive-download-20240209T180019Z-001.zip")

        print(colored("[Video to be generated]", "blue"))
        print(colored(f"   Subject: {data['videoSubject']}", "blue"))
        print(colored(f"   AI Model: {ai_model}", "blue"))
        print(colored(f"   Custom Prompt: {data['customPrompt']}", "blue"))

        if not GENERATING:
            return jsonify({"status": "error", "message": "Video generation was cancelled.", "data": []})

        # ============================
        # Generate script
        # ============================
        script = generate_script(data["videoSubject"], paragraph_number, ai_model, voice, data["customPrompt"])

        # ============================
        # Fetch media based on contentType
        # ============================
        if contentType == "stock":
            # Stock videos flow
            search_terms = get_search_terms(data["videoSubject"], AMOUNT_OF_STOCK_VIDEOS, script, ai_model)
            video_urls = []
            for term in search_terms:
                found = search_for_stock_videos(term, os.getenv("PEXELS_API_KEY"), it=15, min_dur=10)
                for url in found:
                    if url not in video_urls:
                        video_urls.append(url)
                        break
            if not video_urls:
                return jsonify({"status": "error", "message": "No stock videos found.", "data": []})
            media_paths = [save_video(url) for url in video_urls]
        else:
            # Generative content flow
            image_prompts = get_image_search_terms(data["videoSubject"], AMOUNT_OF_STOCK_VIDEOS, script, ai_model)
            media_paths = []
            for prompt in image_prompts:
                try:
                    generated = generate_hf_images(prompt, amount=2)
                    media_paths.extend(generated)
                    if len(media_paths) >= AMOUNT_OF_STOCK_VIDEOS:
                        break
                except Exception as e:
                    print(f"Could not generate image: {e}")
            if not media_paths:
                return jsonify({"status": "error", "message": "No images generated.", "data": []})
            print(colored(f"[+] {len(media_paths)} images generated!", "green"))

        print(colored("[+] Script generated!\n", "green"))

        # ============================
        # Generate TTS
        # ============================
        sentences = [s for s in script.split(". ") if s]
        audio_clips = []
        for sentence in sentences:
            current_tts_path = f"../temp/{uuid4()}.mp3"
            ttselv(sentence, filename=current_tts_path)
            audio_clips.append(AudioFileClip(current_tts_path))

        final_audio = concatenate_audioclips(audio_clips)
        tts_path = f"../temp/{uuid4()}.mp3"
        final_audio.write_audiofile(tts_path)

        # ============================
        # Generate subtitles
        # ============================
        try:
            subtitles_path = generate_subtitles(audio_path=tts_path, sentences=sentences, audio_clips=audio_clips, voice=voice_prefix)
        except Exception as e:
            print(colored(f"[-] Error generating subtitles: {e}", "red"))
            subtitles_path = None

        # ============================
        # Create video
        # ============================
        if contentType == "stock":
            # Stock videos: combine downloaded video clips
            temp_audio = AudioFileClip(tts_path)
            combined_video_path = combine_videos(media_paths, temp_audio.duration, 3, n_threads or 2)
        else:
            # Generative: create video from images
            combined_video_path = create_video_from_images(media_paths, final_audio.duration)

        # Save final video inside GENERATED_VIDEOS_DIR
        final_video_path = generate_video(
            combined_video_path, tts_path, subtitles_path,
            n_threads or 2, subtitles_position, text_color or "#FFFF00"
        )


        # ============================
        # Generate metadata
        # ============================
        title, description, keywords = generate_metadata(data["videoSubject"], script, ai_model)

        # ============================
        # Optional YouTube upload
        # ============================
        if automate_youtube_upload:
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

        print(colored(f"[+] Video generated: {final_video_path}!", "green"))
        GENERATING = False
        return jsonify({"status": "success", "message": "Video generated!", "data": final_video_path})

    except Exception as err:
        print(colored(f"[-] Error: {err}", "red"))
        GENERATING = False
        return jsonify({"status": "error", "message": str(err), "data": []})

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
@app.route("/videos")
def list_videos():
    if not os.path.exists(GENERATED_VIDEOS_DIR):
        return jsonify({"status": "success", "videos": []})
    
    files = [f for f in os.listdir(GENERATED_VIDEOS_DIR) if f.endswith(".mp4")]
    return jsonify({"status": "success", "videos": files})

@app.route("/video/<filename>")
def serve_video(filename):
    if os.path.exists(os.path.join(GENERATED_VIDEOS_DIR, filename)):
        return send_from_directory(GENERATED_VIDEOS_DIR, filename)
    else:
        return jsonify({"status": "error", "message": "Video not found"}), 404

# ============================
# Run server
# ============================
if __name__ == "__main__":
    app.run(debug=True, host=HOST, port=PORT)
