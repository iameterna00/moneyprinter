from gradio_client import Client, handle_file
from pathlib import Path
from uuid import uuid4
import soundfile as sf
import numpy as np
import os
import re
import requests
import time
from typing import List, Tuple, Optional

# client pointing to local Gradio server
GRADIO_URL = "http://127.0.0.1:8080"
client = Client(GRADIO_URL)

DOWNLOAD_DIR = Path.cwd() / "downloads"
DOWNLOAD_DIR.mkdir(exist_ok=True)

def split_text_into_chunks(text: str, max_chars: int = 300) -> List[str]:
    """Split text into chunks of <= max_chars, preferring sentence boundaries."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""
    for s in sentences:
        candidate = s if not current else f"{current} {s}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            # if a single sentence longer than max_chars, hard-split it
            if len(s) > max_chars:
                for i in range(0, len(s), max_chars):
                    chunks.append(s[i:i+max_chars].strip())
                current = ""
            else:
                current = s
    if current:
        chunks.append(current.strip())
    return chunks

def _normalize_audio(a: np.ndarray) -> np.ndarray:
    """Simple peak-normalize if abs peak > 1.0."""
    peak = np.max(np.abs(a)) if a.size else 1.0
    if peak > 1.0:
        return a / peak
    return a

def _parse_gradio_result(result) -> Tuple[int, np.ndarray]:
    """
    Parse the return value from client.predict(..., api_name="/generate_tts_audio").
    Accepts common shapes:
      - [sr, audio_array] or (sr, audio_array)
      - dict with 'data' key where first element may be [sr, audio]
      - path string to a file (local path)
    Returns: (sr, audio_numpy_array)
    """
    # direct tuple/list
    if isinstance(result, (list, tuple)) and len(result) == 2:
        sr, wav = result
        wav = np.asarray(wav)
        return int(sr), wav

    # sometimes gradio returns dict like {'data': [...]} or nested list
    if isinstance(result, dict):
        data = result.get("data") or result.get("label") or result.get("output")
        if data:
            # if data is list containing [sr, wav] or file path
            first = data[0]
            # if first itself is a two-item list
            if isinstance(first, (list, tuple)) and len(first) == 2:
                sr, wav = first
                return int(sr), np.asarray(wav)
            # else maybe data is [sr, wav]
            if isinstance(data, (list, tuple)) and len(data) == 2:
                sr, wav = data
                return int(sr), np.asarray(wav)

    # if result is a string path (pointing to a temporary file)
    if isinstance(result, str) and Path(result).exists():
        wav, sr = sf.read(result)
        # sf.read returns (data, samplerate) only with return order swapped depending on call
        # ensure sr is int
        return int(sr), np.asarray(wav)

    # fallback: try numpy conversion if nested lists
    try:
        arr = np.asarray(result)
        if arr.ndim >= 1:
            # assume default sample rate 22050 if unknown (unsafe fallback)
            return 22050, arr
    except Exception:
        pass

    raise RuntimeError("Unable to parse Gradio response format. Inspect `result` value.")

def shutdown_server():
    """Shutdown the Gradio server after audio generation is complete"""
    print("🎯 Audio generation complete. Initiating server shutdown...")
    
    try:
        # Send shutdown request to Gradio server
        response = requests.post(f"{GRADIO_URL}/shutdown", timeout=10)
        if response.status_code == 200:
            print("✅ Shutdown command sent successfully")
        else:
            print(f"⚠️  Shutdown request failed with status: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("✅ Server is already shutting down or not reachable (expected)")
    except requests.exceptions.RequestException as e:
        print(f"⚠️  Error sending shutdown request: {e}")
    
    # Small delay to ensure the message is sent before process termination
    time.sleep(1)

def tts_hf(
    script: str,
    output_file: Optional[str] = None,
    audio_prompt: Optional[str] = None
) -> str:
    """
    Generate TTS by calling your local Gradio API (/generate_tts_audio).
    Splits long text into chunks, generates each chunk, concatenates outputs,
    and saves final WAV to downloads/. Automatically shuts down server after completion.

    Args:
        script: text to synthesize.
        output_file: optional output filepath. If omitted, a random file in downloads/ is used.
        audio_prompt: optional URL or local path for reference audio.

    Returns:
        Path to saved WAV file as string.
    """
    if not script or not script.strip():
        raise ValueError("Script is empty.")

    chunks = split_text_into_chunks(script, max_chars=300)
    if not chunks:
        raise RuntimeError("Failed to split script into chunks.")

    # Prepare ref audio handle if provided
    ref_handle = None
    if audio_prompt:
        ref_handle = handle_file(audio_prompt)

    # prepare output filename
    if output_file is None:
        out_path = DOWNLOAD_DIR / f"{uuid4()}.wav"
    else:
        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    sr_final = None
    parts = []

    try:
        for i, chunk in enumerate(chunks, start=1):
            print(f"[tts_hf] Generating chunk {i}/{len(chunks)} ({len(chunk)} chars)...")
            result = client.predict(
                text_input=chunk,
                audio_prompt_path_input=ref_handle,
                exaggeration_input=0.5,
                temperature_input=0.8,
                seed_num_input=0,
                language_id="en",
                cfgw_input=0.5,
                api_name="/generate_tts_audio"
            )

            # parse result robustly
            sr, wav_np = _parse_gradio_result(result)
            wav_np = np.asarray(wav_np)

            # If mono with extra dims, squeeze
            if wav_np.ndim > 1 and wav_np.shape[0] == 1:
                wav_np = wav_np.squeeze(0)
            if wav_np.ndim > 1 and wav_np.shape[1] == 1:
                wav_np = wav_np.squeeze(1)

            if sr_final is None:
                sr_final = sr
            elif sr_final != sr:
                raise RuntimeError(f"Sample rate mismatch: {sr_final} != {sr}")

            parts.append(wav_np)

        # concatenate parts
        full = np.concatenate(parts, axis=0)
        full = _normalize_audio(full)

        # save
        sf.write(str(out_path), full, sr_final or 22050)
        print(f"[tts_hf] Saved {out_path} (sr={sr_final})")
        
        # ALWAYS shutdown server after successful generation
        print("✅ Audio generation completed successfully. Shutting down server...")
        shutdown_server()
        
        return str(out_path)
        
    except Exception as e:
        print(f"❌ Error during TTS generation: {e}")
        # Shutdown even on error
        print("🛑 Error occurred - initiating shutdown...")
        shutdown_server()
        raise

if __name__ == "__main__":
    sample_text = (
        "The cow walked through the gate into the new field. "
        "The grass was tall and green. She lowered her head to take the first bite."
    )

    # save into downloads/ (created earlier) as WAV
    out_path = Path("downloads") / "sample_output.wav"

    try:
        saved = tts_hf(
            script=sample_text,
            output_file=str(out_path),
            audio_prompt="../voice/Michel.mp3"
        )
        print("✅ Done. Saved audio at:", saved)
    except Exception as e:
        print("❌ Error during generation:", e)
        raise