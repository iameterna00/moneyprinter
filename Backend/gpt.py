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
        - Language: {voice}

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
    with precise timing information based on the video script.

    Returns:
        List[dict]: A list of dictionaries with image prompts and timing information
    """
    # First, get the total duration from the subtitle file
    def get_total_duration(subtitle_path):
        total_duration = 0
        if not subtitle_path or not os.path.exists(subtitle_path):
            return 24.0  # Default duration if no subtitles
        
        try:
            with open(subtitle_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            segments = content.strip().split('\n\n')
            if segments:
                # Get the last segment's end time
                last_segment = segments[-1].split('\n')
                if len(last_segment) >= 2:
                    timestamp_line = last_segment[1]
                    if '-->' in timestamp_line:
                        end_time = timestamp_line.split('-->')[1].strip()
                        # Convert to seconds
                        parts = end_time.split(':')
                        if len(parts) == 3:
                            hours = int(parts[0])
                            minutes = int(parts[1])
                            seconds_parts = parts[2].split(',')
                            seconds = int(seconds_parts[0])
                            milliseconds = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0
                            total_duration = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
        except:
            total_duration = 24.0  # Fallback duration
        
        return max(total_duration, 24.0)  # Ensure minimum duration
    
    # Get the total video duration
    total_duration = get_total_duration(subtitles_path)
    
    # Parse the subtitle file to get the full script
    def get_full_script(subtitle_path):
        full_script = ""
        if not subtitle_path or not os.path.exists(subtitle_path):
            return f"A video about {video_subject}"
        
        try:
            with open(subtitle_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            segments = content.strip().split('\n\n')
            for segment in segments:
                lines = segment.split('\n')
                if len(lines) >= 3:
                    # Add the text content
                    full_script += " ".join(lines[2:]) + " "
        except:
            full_script = f"A video about {video_subject}"
        
        return full_script.strip()
    
    # Get the full script
    full_script = get_full_script(subtitles_path)
    
    # Calculate timing for each image
    segment_duration = total_duration / amount
    
    # Prepare the prompt for GPT
    prompt = f"""
    Based on the following video script about {video_subject}, generate {amount} highly detailed image prompts 
    for AI image generation. The video has a total duration of {total_duration:.2f} seconds, and each image 
    should be displayed for approximately {segment_duration:.2f} seconds.

    Return the results as a JSON array with EXACTLY {amount} objects in this format:
    [
        {{
            "Img prompt": "detailed cinematic description of the first image",
            "start": 0.0,
            "end": {segment_duration:.2f}
        }},
        {{
            "Img prompt": "detailed cinematic description of the second image",
            "start": {segment_duration:.2f},
            "end": {segment_duration * 2:.2f}
        }},
        ...
    ]

    Each image prompt must:
    - Be highly detailed and visually descriptive
    - Cover different aspects of the video script in chronological order
    - Include specific details about characters, environments, lighting, and camera angles
    - Be suitable for AI image generation (clear, specific, descriptive)
    - Represent key moments or concepts from the script

    Video script: "{full_script}"

    Video subject: {video_subject}

    Return ONLY the JSON array. Do not include any explanations, code fences, or extra text.
    """
    
    response = generate_response(prompt, ai_model)
    print(colored(f"[*] Raw AI response for image prompts:\n{response}", "cyan"))

    try:
        # Try to parse the response as JSON
        search_terms = json.loads(response)
        
        # Validate the structure
        if not isinstance(search_terms, list):
            raise ValueError("Response is not a list")
        
        # Ensure we have the right number of prompts
        if len(search_terms) != amount:
            print(colored(f"[!] Got {len(search_terms)} prompts but expected {amount}, adjusting...", "yellow"))
            # Create or trim prompts as needed
            if len(search_terms) < amount:
                # Add missing prompts
                for i in range(len(search_terms), amount):
                    start_time = i * segment_duration
                    end_time = (i + 1) * segment_duration
                    search_terms.append({
                        "Img prompt": f"{video_subject} - scene {i+1}, detailed cinematic shot",
                        "start": start_time,
                        "end": end_time
                    })
            else:
                # Trim excess prompts
                search_terms = search_terms[:amount]
        
        # Ensure each prompt has the correct structure
        for i, prompt_data in enumerate(search_terms):
            if not isinstance(prompt_data, dict) or "Img prompt" not in prompt_data:
                # Fix malformed prompts
                start_time = i * segment_duration
                end_time = (i + 1) * segment_duration
                search_terms[i] = {
                    "Img prompt": f"{video_subject} - key moment {i+1}, cinematic view",
                    "start": start_time,
                    "end": end_time
                }
            else:
                # Ensure proper timing
                search_terms[i]["start"] = i * segment_duration
                search_terms[i]["end"] = (i + 1) * segment_duration
        
        return search_terms
        
    except Exception as e:
        print(colored(f"[-] Could not parse image prompts JSON: {e}", "red"))
        # Fallback: create evenly spaced prompts
        return [{
            "Img prompt": f"{video_subject} - scene {i+1}, detailed cinematic shot with appropriate elements from the story",
            "start": i * segment_duration,
            "end": (i + 1) * segment_duration
        } for i in range(amount)]
        

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
