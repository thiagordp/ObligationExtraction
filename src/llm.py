import json
import logging
import os
import random
import re
import time

from typing import List, Dict, Any, Optional
import ollama
from groq import Groq

# Configuration Management
MAX_API_CALL_RETRIES = int(os.environ.get("MAX_API_CALL_RETRIES", 5))
WAIT_FOR_NEXT_CALL = 60
GROQ_API_KEYS = [s.strip() for s in os.environ.get("GROQ_API_KEY", "").split(",")]

# Ensure there's apt least one API key
if not GROQ_API_KEYS:
    logging.error("GROQ_API_KEY environment variable is not set or is empty.")
    raise ValueError("GROQ_API_KEY environment variable is not set or is empty.")


def call_llm(prompt: str, model: str = "llama3.1", temperature: float = 0.6, max_tokens: int = 200, top_p: float = 0.9) -> str:
    """
    Generate a response from the specified Ollama model based on the provided prompt and parameters.
    
    Args:
        prompt (str): The input prompt for the model to generate a response.
        model (str): The model to be used for generating the response (default is "llama3.1").
        temperature (float): Controls the randomness of the output (default is 0.6).
        max_tokens (int): The maximum number of tokens in the generated response (default is 200).
        top_p (float): Controls the diversity of responses (default is 0.9).

    Returns:
        str: The generated response text from the model.
    """
    try:
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            stream=False
        )
        logging.info(f"Response from Ollama: {response.get('message', {}).get('content', 'No response text found.')}")
        return response.get('message', {}).get('content', 'No response text found.')
    except Exception as e:
        logging.error(f"Error calling Ollama: {e}")
        return f"An error occurred: {e}"


class GroqApiClient:
    """
    A factory class to manage Groq API client instances.
    """
    def __init__(self):
        list_keys = [s.strip() for s in os.environ.get("GROQ_API_KEY").split(",")]
        random.shuffle(list_keys)
        self._api_key = list_keys[0]

        self.client = Groq(api_key=self._api_key)
        logging.info(f"Switching API token to '{self._api_key[:4]}...{self._api_key[-4:]}'")

    @staticmethod
    def _select_random_api_key() -> str|List[str]:
        """Selects a random API key from the list."""
        if not GROQ_API_KEYS:
            logging.error("No API keys available.")
            raise ValueError("No API keys available.")
        random.shuffle(GROQ_API_KEYS)
        return GROQ_API_KEYS

    def get_client(self) -> Groq:
        """Returns the Groq API client instance."""
        return self.client


def check_structure(data: Any, structure: Any) -> bool:
    """
    Recursively checks if the given `data` matches the specified `structure`.

    Args:
        data (Any): The JSON data to be validated.
        structure (Any): The expected structure for `data`.

    Returns:
        bool: True if `data` matches the `structure`, False otherwise.
    """
    if isinstance(structure, dict):
        if not isinstance(data, dict):
            return False
        for key, value_type in structure.items():
            if key not in data or not check_structure(data[key], value_type):
                return False
        return True
    elif isinstance(structure, list):
        if not isinstance(data, list):
            return False
        if not structure:
            return True
        return all(check_structure(item, structure[0]) for item in data)
    else:
        return isinstance(data, structure)


def extract_dict(data: str) -> Optional[Dict]:
    """
    Extracts a dictionary from a string containing a JSON object enclosed in triple backticks (```json).

    Args:
        data (str): The string containing the JSON object.

    Returns:
        Optional[Dict]: The extracted dictionary or None if no valid JSON object is found.
    """
    data = data.replace("```json", "```")
    json_match = re.search(r'```(.*?)```', data, re.DOTALL)

    if json_match:
        json_str = json_match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            logging.error("Error decoding JSON.")
    return None


def run_prompt(client: Groq, system_prompt: str, user_prompt: str, model_name: str, output_json: bool = True) -> Any:
    """
    Runs a prompt through the specified model and process its output.

    Args:
        client (Groq): The Groq API client instance.
        system_prompt (str): The system prompt to set the context.
        user_prompt (str): The user prompt to generate a response.
        model_name (str): The name of the model to use.
        output_json (bool): Whether to extract JSON from the response.

    Returns:
        Any: The processed response, either as a JSON object or a string.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(MAX_API_CALL_RETRIES):
        try:
            result = make_api_call(client, messages, model_name=model_name)
            if output_json:
                result = extract_dict(result)
            return result
        except Exception as e:
            logging.error(f"Error during API call (attempt {attempt + 1}): {e}")
            sleep_period = WAIT_FOR_NEXT_CALL * (random.random() + 1)
            logging.info(f"Retrying in {sleep_period:.3f} seconds")
            time.sleep(sleep_period)
    raise Exception("Failed to make API call after retries")


def make_api_call(client: Groq, messages: List[Dict[str, str]], model_name: str, temperature: float = 0.7, max_tokens: int = 4000) -> str:
    """
    Makes an API call to the specified model.

    Args:
        client (Groq): The Groq API client instance.
        messages (List[Dict[str, str]]): List of messages to send.
        model_name (str): The name of the model to use.
        temperature (float): Temperature to use for response generation.
        max_tokens (int): Maximum number of tokens to generate.

    Returns:
        str: The response from the API.
    """
    GROQ_MODELS = [
        "llama3-70b-8192",
        "llama-3.1-70b-versatile",
        "llama-3.2-1b-preview",
        "llama3-8b-8192",
        "llama-3.2-3b-preview"
    ]

    OLLAMA_MODELS = [
        "llama-3.2-1b-preview",
        "llama3-8b-8192",
        "llama-3.1-70b-versatile",
        "llama-3.2-3b-preview"
    ]

    if model_name in GROQ_MODELS:
        logging.info(f"Sending request to {model_name} with temperature {temperature}, max_tokens {max_tokens}")
        return _call_groq_model(client, messages, model_name, temperature, max_tokens)
    elif model_name in OLLAMA_MODELS:
        logging.info(f"Sending request to {model_name} with temperature {temperature}")
        return _call_ollama_model(messages, model_name, temperature)
    else:
        raise ValueError("Invalid model name.")


def _call_groq_model(client: Groq, messages: List[Dict[str, str]], model_name: str, temperature: float, max_tokens: int) -> str:
    """Make a call to the Groq model."""

    for attempt in range(MAX_API_CALL_RETRIES):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.9,
                stream=False,
                stop=None,
                frequency_penalty=1.0
            )
            return completion.choices[0].message.content
        except Exception as e:
            logging.error(f"Error during API call (attempt {attempt + 1}): {e}")
            wait_time = WAIT_FOR_NEXT_CALL * (1 + attempt * random.random())
            logging.info(f"Retrying in {wait_time:.2f} seconds")
            time.sleep(wait_time)
    raise Exception("Failed to make API call after retries")


def _call_ollama_model(messages: List[Dict[str, str]], model_name: str, temperature: float) -> str:
    """Make a call to the Ollama model."""
    response = ollama.chat(
        model=model_name,
        messages=messages,
        stream=False,
        options={"temperature": temperature}
    )
    return response.get('message', {}).get('content', 'No response text found.')


def extract_json_from_string(text: str) -> Any:
    """
    Extracts a JSON object from a string using regex.

    Args:
        text (str): The string containing the JSON object.

    Returns:
        Any: The extracted JSON object, or a message if no JSON is found or decoding fails.
    """
    json_pattern = r'\{.*?\}'  # This will match a JSON object
    match = re.search(json_pattern, text)

    if match:
        json_str = match.group(0).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            logging.error("Error decoding JSON.")
    return "No JSON found."