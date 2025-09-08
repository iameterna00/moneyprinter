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
        Do not include titles, markdown, or line breaks.
        Do not mention the AI, the prompt, or the number of paragraphs.
        Just write the story directly.
        character limit is :300 characters
        Subject: {video_subject}
        Language: {voice}
        paragraph:{paragraph_number}
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

def get_image_search_terms(video_subject: str, amount: int, script: str, ai_model: str) -> List[str]:
    """
    Generate visually descriptive prompts for AI image generation.
    """
    prompt = f"""
    You are an expert visual storyteller. Generate {amount} highly detailed and cinematic prompts
    for AI image generation, based on the video subject: "{video_subject}".
    
    Each prompt must:
    - Be visually rich and cinematic.
    - Include characters, environment, mood, lighting, and perspective.
    - Be unique and cover different parts of the story/script.
    
    Return strictly as a JSON array of strings. Do not include explanations, code fences, or extra formatting. Output only the JSON.
    Video script context (first 500 chars): "{script[:500]}..."
    """
    
    response = generate_response(prompt, ai_model)
    print(colored(f"[*] Raw AI response for image prompts:\n{response}", "cyan"))  # Debug

    try:
        search_terms = json.loads(response)
        if len(search_terms) < amount:
            # Only fill missing prompts if needed
            search_terms.extend([f"{video_subject} cinematic scene {i+1}" 
                                 for i in range(amount - len(search_terms))])
        return search_terms[:amount]
    except Exception as e:
        print(colored(f"[-] Could not parse image prompts JSON: {e}", "red"))
        # Fallback prompts
        return [f"{video_subject} cinematic scene {i+1}" for i in range(amount)]

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

