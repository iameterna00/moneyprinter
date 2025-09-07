
# Author: Anuj
# Topic: ElevenLabs Voice TTS (SDK version, working)
# Version: 2.3

import os
from uuid import uuid4
from typing import List
from elevenlabs import ElevenLabs
from termcolor import colored
from moviepy.editor import AudioFileClip, concatenate_audioclips

# -----------------------------
# Configuration
# -----------------------------
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY") or "sk_89c8c8b41426f1d0c38e92de7f0dedc9a70f38819f0b05bc"

client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
MODEL_ID = "eleven_multilingual_v2"
TEXT_BYTE_LIMIT = 300  # Characters per request

# -----------------------------
# Helper Functions
# -----------------------------
def split_string(text: str, chunk_size: int = TEXT_BYTE_LIMIT) -> List[str]:
    words = text.split()
    result, current_chunk = [], ""
    for word in words:
        if len(current_chunk) + len(word) + 1 <= chunk_size:
            current_chunk += f" {word}"
        else:
            if current_chunk:
                result.append(current_chunk.strip())
            current_chunk = word
    if current_chunk:
        result.append(current_chunk.strip())
    return result


def save_audio_from_generator(gen, filename: str):
    """Save audio bytes from ElevenLabs generator to a file."""
    with open(filename, "wb") as f:
        for chunk in gen:
            f.write(chunk)
    print(colored(f"[+] Audio file saved successfully as '{filename}'", "green"))

# -----------------------------
# TTS Function
# -----------------------------
def ttselv(text: str, filename: str = "output.mp3"):
    if not text:
        print(colored("[-] Please specify text", "red"))
        return

    try:
        if len(text) <= TEXT_BYTE_LIMIT:
            # Single request
            gen = client.text_to_speech.convert(
                voice_id=VOICE_ID,
                model_id=MODEL_ID,
                text=text,
                output_format="mp3_44100_128",
            )
            save_audio_from_generator(gen, filename)

        else:
            # Long text: split + concatenate
            text_parts = split_string(text, TEXT_BYTE_LIMIT)
            audio_clips = []

            for part in text_parts:
                temp_path = f"../temp/{uuid4()}.mp3"
                gen = client.text_to_speech.convert(
                    voice_id=VOICE_ID,
                    model_id=MODEL_ID,
                    text=part,
                    output_format="mp3_44100_128",
                )
                save_audio_from_generator(gen, temp_path)
                audio_clips.append(AudioFileClip(temp_path))

            final_audio = concatenate_audioclips(audio_clips)
            final_audio.write_audiofile(filename)
            print(colored(f"[+] Final concatenated audio saved as '{filename}'", "green"))

    except Exception as e:
        print(colored(f"[-] TTS generation failed: {e}", "red"))


# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    sample_text = (
        "The cow walked through the gate into the new field. "
        "The grass was tall and green. She lowered her head to take the first bite."
    )
    ttselv(sample_text, filename="../temp/sample.mp3")
