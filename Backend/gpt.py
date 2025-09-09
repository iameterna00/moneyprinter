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


def generate_script(video_subject: str, paragraph_number: str, ai_model: str, voice: str, customPrompt: str = None) -> str:
    """
    Generate a single-paragraph script for a video.
    """
    if customPrompt:
        prompt = customPrompt
    else:
        prompt = f"""
        Generate a continuous story-like script for a video about: {video_subject}.
        The script should be a single flowing paragraph.
        Use simple, clear language.
        Do not start with 'Of course', 'Here is', or any introduction
        Do not include titles, markdown, or line breaks.
        Do not mention the AI, the prompt, or the number of paragraphs.
        Just write the story directly.
        character limit is :500 characters
        Subject: {video_subject}
        Language: {voice}
        paragraph:{paragraph_number}


        **IMPORTANT**
        -Do NOT add any extra commentary, introductions, or explanations.
        -DO NOT include  Here is a script written in the first person, focusing on ..blah..bhah
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

    except (json.JSONDecodeError, ValueError):
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
    with precise timing information from subtitle file.

    Returns:
        List[dict]: A list of dictionaries with image prompts and timing information
    """
    # Parse the subtitle file to get segments with timestamps
    def parse_srt_subtitles(subtitle_path):
        segments = []
        
        if not subtitle_path or not os.path.exists(subtitle_path):
            return segments
        
        with open(subtitle_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into segments by empty lines
        segments_raw = content.strip().split('\n\n')
        
        for segment in segments_raw:
            lines = segment.split('\n')
            if len(lines) >= 3:
                # Parse timestamp line
                timestamp_line = lines[1]
                if '-->' in timestamp_line:
                    start_end = timestamp_line.split('-->')
                    if len(start_end) == 2:
                        start_time = start_end[0].strip()
                        end_time = start_end[1].strip()
                        
                        # Convert timestamp to seconds
                        def timestamp_to_seconds(timestamp):
                            parts = timestamp.split(':')
                            if len(parts) == 3:
                                hours = int(parts[0])
                                minutes = int(parts[1])
                                seconds_parts = parts[2].split(',')
                                seconds = int(seconds_parts[0])
                                milliseconds = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0
                                return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
                            return 0
                        
                        start_seconds = timestamp_to_seconds(start_time)
                        end_seconds = timestamp_to_seconds(end_time)
                        
                        # Combine text lines
                        text = ' '.join(lines[2:])
                        
                        segments.append({
                            'text': text,
                            'start': start_seconds,
                            'end': end_seconds
                        })
        
        return segments

    # Parse the subtitle file
    subtitle_segments = parse_srt_subtitles(subtitles_path)
    
    if not subtitle_segments:
        # Fallback if no subtitles are available
        return [{"Img prompt": f"{video_subject} detailed cinematic scene {i+1}", 
                 "start": i * 3.0, "end": (i + 1) * 3.0} for i in range(amount)]
    
    # Prepare the prompt for GPT - more explicit about the format
    prompt = f"""
    Based on the following subtitle segments with timestamps for a video about {video_subject},
    generate appropriate image search terms for each segment. Return the results as a JSON array.
    
    IMPORTANT: You MUST return a JSON array of objects with EXACTLY this format:
    [
        {{
            "Img prompt": "detailed description of the image",
            "start": 0.0,
            "end": 2.5
        }},
        {{
            "Img prompt": "detailed description of the next image",
            "start": 2.5,
            "end": 5.0
        }},
        ...
    ]

    Each image prompt must:
    - Include every visible character with explicit details: gender, age, hair color, hairstyle, 
      face shape, clothing with proper shirt color and pant color and what type of cloth it is, 
      posture, and facial expressions.
    - Describe the environment in detail: location, objects, background elements, weather, time of day, and props.
    - Specify lighting, shadows, reflections, and overall cinematic mood.
    - Indicate camera perspective, framing, and angle (e.g., close-up, wide shot, low-angle).
    - Represent any action or emotion happening in the scene.
    - Be unique for each prompt and reflect the specific subtitle segment.
    
    Return ONLY the JSON array. Do not include any explanations, code fences, or extra text before or after the JSON.
    
    Subtitle segments with timestamps:
    {json.dumps(subtitle_segments, indent=2)}
    """
    
    response = generate_response(prompt, ai_model)
    print(colored(f"[*] Raw AI response for image prompts:\n{response}", "cyan"))  # Debug

    try:
        # Try to parse the response as JSON
        search_terms = json.loads(response)
        
        # If we got a list of strings instead of objects, convert them
        if search_terms and isinstance(search_terms[0], str):
            print(colored("[!] AI returned strings instead of objects, converting...", "yellow"))
            converted_terms = []
            for i, prompt_text in enumerate(search_terms):
                if i < len(subtitle_segments):
                    segment = subtitle_segments[i]
                    converted_terms.append({
                        "Img prompt": prompt_text,
                        "start": segment["start"],
                        "end": segment["end"]
                    })
                else:
                    # Fallback timing if we have more prompts than segments
                    converted_terms.append({
                        "Img prompt": prompt_text,
                        "start": i * 3.0,
                        "end": (i + 1) * 3.0
                    })
            search_terms = converted_terms
        
        # Ensure we have the right amount of prompts
        if len(search_terms) < amount:
            # Fill missing prompts if needed
            for i in range(amount - len(search_terms)):
                if subtitle_segments and i < len(subtitle_segments):
                    segment = subtitle_segments[i]
                    search_terms.append({
                        "Img prompt": f"{video_subject}: {segment['text']}",
                        "start": segment["start"],
                        "end": segment["end"]
                    })
                else:
                    search_terms.append({
                        "Img prompt": f"{video_subject} detailed cinematic scene {i+1}",
                        "start": i * 3.0,
                        "end": (i + 1) * 3.0
                    })
        
        return search_terms[:amount]
    except Exception as e:
        print(colored(f"[-] Could not parse image prompts JSON: {e}", "red"))
        # Fallback: create prompts based on subtitle segments
        if subtitle_segments:
            return [{
                "Img prompt": f"{video_subject}: {segment['text']}",
                "start": segment["start"],
                "end": segment["end"]
            } for segment in subtitle_segments[:amount]]
        else:
            # Final fallback
            return [{"Img prompt": f"{video_subject} detailed cinematic scene {i+1}", 
                     "start": i * 3.0, "end": (i + 1) * 3.0} for i in range(amount)]
        

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
