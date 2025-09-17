import re
import os
import g4f
import json
import google.generativeai as genai
import requests

from g4f.client import Client
from termcolor import colored
from dotenv import load_dotenv
from typing import Tuple, List

# Load environment variables
load_dotenv("../.env")

# Set environment variables
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)


def generate_response(prompt: str, ai_model: str = "deepseek-chat") -> str:
    """
    Generate a response using DeepSeek API.
    ai_model can be 'deepseek-chat' or 'deepseek-reasoner'.
    """
    if ai_model not in ["deepseek-chat", "deepseek-reasoner"]:
        raise ValueError(f"Invalid AI model selected: {ai_model}. Use 'deepseek-chat' or 'deepseek-reasoner'.")

    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": ai_model,
        "messages": [{"role": "user", "content": prompt}]
    }

    resp = requests.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def generate_script(video_subject: str, paragraph_number: str, ai_model: str,  customPrompt: str = None) -> str:
    """
    Generate a single-paragraph script for a video.
    """
    if customPrompt:
        prompt = customPrompt
    else:
        prompt = f"""
        Write a single engaging narration about: {video_subject}.

        **STYLE GUIDELINES:**
        - Sound like a documentary narrator: clear, factual, but gripping.
        - Use Hooks for insta reels or shorts type
        - Use simple english and simpler words.
        - Build tension and drama through facts and strong phrasing, not sensory overload.
        - Keep it dynamic and easy to follow, as if written for a short video script.
        - Use plain but powerful sentences that highlight key actions and turning points.
        - Avoid over-descriptive poetic imagery (e.g., "humid air clung to his skin").
        - Flow smoothly from one idea to the next, like a story unfolding.

        **TECHNICAL CONSTRAINTS:**
        - Do NOT start with 'Of course', 'Here is', or any introduction.
        - Do not include titles, markdown, or line breaks.
        - Do not mention the AI, the prompt, or the number of paragraphs.
        - Write in a continuous, flowing paragraph.
        - Character limit: 500 characters.

        **Subject to write about:** {video_subject}
        """
    
    response = generate_response(prompt, ai_model)

    if response:
        # Clean the script from markdown or extra characters
        response = response.replace("*", "").replace("#", "")
        response = re.sub(r"\[.*?\]", "", response)
        response = re.sub(r"\(.*?\)", "", response)

        # Remove unnecessary newlines and keep it as a single paragraph
        final_script = " ".join(response.split())
        print('final script', final_script)
        return final_script
    else:
        print(colored("[-] GPT returned an empty response.", "red"))
        return None



def get_search_terms(video_subject: str, amount: int, script: str, ai_model: str) -> List[str]:
    """
    Generate a JSON-Array of search terms for stock videos,
    depending on the subject of a video.

    Args:
        video_subject (str): The subject of the video.
        amount (int): The amount of search terms to generate.
        script (str): The script of the video.
        ai_model (str): The AI model to use for generation.

    Returns:
        List[str]: The search terms for the video subject.
    """

    # Build prompt
    prompt = f"""
    Generate {amount} search terms for stock videos,
    depending on the subject of a video.
    Subject: {video_subject}

    The search terms are to be returned as
    a JSON-Array of strings.

    Each search term should consist of 1-3 words,
    always add the main subject of the video.
    
    YOU MUST ONLY RETURN THE JSON-ARRAY OF STRINGS.
    YOU MUST NOT RETURN ANYTHING ELSE. 
    YOU MUST NOT RETURN THE SCRIPT.
    
    The search terms must be related to the subject of the video.
    Here is an example of a JSON-Array of strings:
    ["search term "]

    For context, here is the full text:
    {script}
    """

    # Generate search terms
    response = generate_response(prompt, ai_model)
    print(response)

    # Parse response into a list of search terms
    search_terms = []
    
    try:
        search_terms = json.loads(response)
        if not isinstance(search_terms, list) or not all(isinstance(term, str) for term in search_terms):
            raise ValueError("Response is not a list of strings.")

    except (json.JSONDecodeError, Value
            ):
        # Get everything between the first and last square brackets
        response = response[response.find("[") + 1:response.rfind("]")]

        print(colored("[*] GPT returned an unformatted response. Attempting to clean...", "yellow"))

        # Attempt to extract list-like string and convert to list
        match = re.search(r'\["(?:[^"\\]|\\.)*"(?:,\s*"[^"\\]*")*\]', response)
        print(match.group())
        if match:
            try:
                search_terms = json.loads(match.group())
            except json.JSONDecodeError:
                print(colored("[-] Could not parse response.", "red"))
                return []


    # Let user know
    print(colored(f"\nGenerated {len(search_terms)} search terms: {', '.join(search_terms)}", "cyan"))

    # Return search terms
    return search_terms


def get_image_search_terms(video_subject: str, amount: int, subtitles_path: str, ai_model: str) -> List[dict]:
    """
    Generate highly detailed, visually descriptive prompts for AI image generation
    with precise timing information based on the subtitle file.
    """
    
    # --- Helper: Parse SRT into segments with start/end in seconds ---
    def parse_srt(subtitles_path: str) -> List[dict]:
        segments = []
        if not subtitles_path or not os.path.exists(subtitles_path):
            return segments
        with open(subtitles_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        blocks = content.split('\n\n')
        for block in blocks:
            lines = block.split('\n')
            if len(lines) >= 3:
                ts_match = re.match(
                    r"(\d+):(\d+):(\d+),(\d+)\s*-->\s*(\d+):(\d+):(\d+),(\d+)", 
                    lines[1]
                )
                if ts_match:
                    h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, ts_match.groups())
                    start_sec = h1*3600 + m1*60 + s1 + ms1/1000
                    end_sec = h2*3600 + m2*60 + s2 + ms2/1000
                    text = " ".join(lines[2:]).strip()
                    segments.append({
                        "text": text,
                        "start": start_sec,
                        "end": end_sec
                    })
        return segments

    # --- Get total duration from subtitles ---
    def get_total_duration(segments: List[dict]) -> float:
        if segments:
            return segments[-1]["end"]
        return 24.0  # fallback

    # --- Generate fallback image prompts ---
    def fallback_prompts(segments: List[dict], amount: int) -> List[dict]:
        if not segments:
            segment_duration = 24.0 / amount
            return [{
                "Img prompt": f"{video_subject} - scene {i+1}, detailed cinematic shot",
                "start": i*segment_duration,
                "end": (i+1)*segment_duration
            } for i in range(amount)]
        
        # Split segments evenly into `amount` groups
        step = max(1, len(segments) // amount)
        prompts = []
        for i in range(0, len(segments), step):
            segs = segments[i:i+step]
            prompt_text = " ".join([s['text'] for s in segs])
            prompts.append({
                "Img prompt": f"Cinematic scene: {prompt_text}",
                "start": segs[0]['start'],
                "end": segs[-1]['end']
            })
        return prompts[:amount]  # Ensure exact amount

    # --- Main logic ---
    segments = parse_srt(subtitles_path)
    total_duration = get_total_duration(segments)
    
    # Define segment_duration for outer fallback
    segment_duration = total_duration / amount

    full_script = " ".join([s['text'] for s in segments]) if segments else f"A video about {video_subject}"

    # Prepare GPT prompt
    prompt = f"""
    Based on the following video script about {video_subject}, generate {amount} highly detailed image prompts 
    for AI image generation. The video has a total duration of {total_duration:.2f} seconds.
    Analyze the subtitle content to determine start and end times for each image that align with the content.

    Video script: "{full_script}"
    Video subject: {video_subject}

    Return EXACTLY {amount} JSON objects like:

   IMPORTANT RULES:
- Characters must remain visually consistent across all prompts.
- Mention character visuals in eatch prompt.
- Do NOT change species, gender, clothing, or hairstyles once introduced.
- If a monkey is in the story, always keep it as the SAME monkey (not a donkey or other animal).
- If a human has brown hair, it must stay brown in all frames.
- Keep the environment, mood, and clothing consistent unless the script explicitly says they change.

    [
        {{
            "Img prompt": "detailed cinematic description",
            "start": 0.0,
            "end": 12.8
        }},
        ...
    ]
    """
    
    try:
        response = generate_response(prompt, ai_model)
        print(colored(f"[*] Raw AI response for image prompts:\n{response}", "cyan"))
        search_terms = json.loads(response)

        # Validate number of prompts
        if not isinstance(search_terms, list) or len(search_terms) != amount:
            print(colored("[!] AI response invalid or incomplete, using fallback.", "yellow"))
            search_terms = fallback_prompts(segments, amount)

        # Ensure timeline consistency
        if search_terms:
            search_terms[0]["start"] = 0.0
            search_terms[-1]["end"] = total_duration
            for i in range(len(search_terms) - 1):
                search_terms[i]["end"] = search_terms[i+1]["start"]
        return search_terms

    except Exception as e:
        print(colored(f"[-] Could not generate prompts from AI: {e}", "red"))
        return fallback_prompts(segments, amount)

def generate_metadata(video_subject: str, script: str, ai_model: str) -> Tuple[str, str, List[str]]:  
    """  
    Generate metadata for a YouTube video, including the title, description, and keywords.  
  
    Args:  
        video_subject (str): The subject of the video.  
        script (str): The script of the video.  
        ai_model (str): The AI model to use for generation.  
  
    Returns:  
        Tuple[str, str, List[str]]: The title, description, and keywords for the video.  
    """  
  
    # Build prompt for title  
    title_prompt = f"""  
    Generate a catchy and SEO-friendly title for a YouTube shorts video about {video_subject}.  
    """  
  
    # Generate title  
    title = generate_response(title_prompt, ai_model).strip()  
    
    # Build prompt for description  
    description_prompt = f"""  
    Write a brief and engaging description for a YouTube shorts video about {video_subject}.  
    The video is based on the following script:  
    {script}  
    """  
  
    # Generate description  
    description = generate_response(description_prompt, ai_model).strip()  
  
    # Generate keywords  
    keywords = get_image_search_terms(video_subject, 6, script, ai_model)  

    return title, description, keywords  

if __name__ == "__main__":
    subtitle_dir = r"D:\Projects\money printer\MoneyPrinter\subtitles"

    if not os.path.exists(subtitle_dir):
        print(f"Subtitle folder not found: {subtitle_dir}")
    else:
        subtitle_files = [f for f in os.listdir(subtitle_dir) if f.endswith(".srt")]
        if not subtitle_files:
            print("No .srt files found in the folder.")
        else:
            for file_name in subtitle_files:
                file_path = os.path.join(subtitle_dir, file_name)
                print(f"\nProcessing subtitle file: {file_name}")
                search_terms = get_image_search_terms("Sample Video Subject", 5, file_path, "deepseek-chat")
                print(json.dumps(search_terms, indent=2))
