# author: GiorDior aka Giorgio
# modified for safe long text handling

import base64
import requests
import threading
from typing import List
from termcolor import colored
from playsound import playsound
from pathlib import Path

VOICES = [
    "en_us_ghostface", "en_us_chewbacca", "en_us_c3po", "en_us_stitch", "en_us_stormtrooper", "en_us_rocket",
    "en_au_001", "en_au_002", "en_uk_001", "en_uk_003", "en_us_001", "en_us_002", "en_us_006", "en_us_007",
    "en_us_009", "en_us_010", "fr_001", "fr_002", "de_001", "de_002", "es_002", "es_mx_002", "br_001",
    "br_003", "br_004", "br_005", "id_001", "jp_001", "jp_003", "jp_005", "jp_006", "kr_002", "kr_003",
    "kr_004", "en_female_f08_salut_damour", "en_male_m03_lobby", "en_female_f08_warmy_breeze",
    "en_male_m03_sunshine_soon", "en_male_narration", "en_male_funny", "en_female_emotional"
]

ENDPOINTS = [
    "https://tiktok-tts.weilnet.workers.dev/api/generation",
    "https://tiktoktts.com/api/tiktok-tts",
]
current_endpoint = 0
TEXT_BYTE_LIMIT = 300  # max characters per request

# Split text into chunks of <= chunk_size characters
def split_text(text: str, chunk_size: int = 300) -> List[str]:
    words = text.split()
    chunks, current = [], ""
    for word in words:
        if len(current) + len(word) + 1 <= chunk_size:
            current += " " + word
        else:
            if current:
                chunks.append(current.strip())
            current = word
    if current:
        chunks.append(current.strip())
    return chunks

# Check if TTS endpoint is alive
def is_endpoint_alive() -> bool:
    global current_endpoint
    try:
        url = ENDPOINTS[current_endpoint].split("/a")[0]
        return requests.get(url).status_code == 200
    except:
        return False

# Convert base64 to audio file
def save_audio(base64_data: str, filename: str) -> None:
    audio_bytes = base64.b64decode(base64_data)
    with open(filename, "wb") as f:
        f.write(audio_bytes)

# Send POST request to get audio base64
def fetch_audio_base64(text: str, voice: str) -> str:
    url = ENDPOINTS[current_endpoint]
    headers = {"Content-Type": "application/json"}
    data = {"text": text, "voice": voice}
    resp = requests.post(url, headers=headers, json=data)
    resp_str = resp.content.decode("utf-8")
    if current_endpoint == 0:
        return resp_str.split('"')[5]
    else:
        return resp_str.split('"')[3].split(",")[1]

# Main TTS function
def tts(text: str, voice: str, filename: str = "output.mp3", play_sound: bool = False) -> None:
    global current_endpoint
    if not voice or voice not in VOICES:
        print(colored("[-] Invalid or missing voice", "red"))
        return
    if not text:
        print(colored("[-] No text provided", "red"))
        return
    if not is_endpoint_alive():
        current_endpoint = (current_endpoint + 1) % 2
        if not is_endpoint_alive():
            print(colored("[-] TTS service unavailable", "red"))
            return
    print(colored("[+] TTS service is available", "green"))

    # Split text if longer than limit
    text_chunks = split_text(text, TEXT_BYTE_LIMIT)
    audio_chunks = [None] * len(text_chunks)

    def worker(chunk, idx):
        try:
            audio_chunks[idx] = fetch_audio_base64(chunk, voice)
        except Exception as e:
            print(colored(f"[-] Error fetching chunk {idx}: {e}", "red"))

    threads = []
    for i, chunk in enumerate(text_chunks):
        t = threading.Thread(target=worker, args=(chunk, i))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    if None in audio_chunks:
        print(colored("[-] Failed to generate some chunks", "red"))
        return

    # Combine all audio chunks and save
    full_base64 = "".join(audio_chunks)
    save_audio(full_base64, filename)
    print(colored(f"[+] Audio saved as '{filename}'", "green"))
    if play_sound:
        playsound(filename)

# Example usage
if __name__ == "__main__":
    long_text = "Your very long text goes here. It can be much longer than 300 characters, " \
                "and this script will split it, generate TTS for each part, and combine the audio."
    tts(long_text, voice="en_us_001", filename="output.mp3", play_sound=True)
