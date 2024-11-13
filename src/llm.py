import json
import logging
import os
import random
import re
import time

from typing import List, Dict, Tuple, Any, Set
import ollama
from groq import Groq

WAIT_FOR_NEXT_CALL = 60


def call_llm(prompt, model="llama3.1", temperature=0.6, max_tokens=200, top_p=0.9):
    """
    Generate a response from the specified Ollama model based on the provided prompt and parameters.
    
    Args:
    - prompt (str): The input prompt for the model to generate a response.
    - model (str): The model to be used for generating the response (default is "llama2").
    - temperature (float): Controls the randomness of the output (default is 0.7).
    - max_tokens (int): The maximum number of tokens in the generated response (default is 200).
    - top_p (float): Controls the diversity of responses (default is 0.9).
    
    Returns:
    - str: The generated response text from the model.
    """
    try:
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            stream=False
        )
        logging.info(response["message"]["content"])
        return response.get('text', 'No response text found.')

    except Exception as e:
        return f"An error occurred: {e}"


def setup_api() -> Groq:
    """
    Sets up the Groq API client.

    Returns:
        Groq: The Groq API client.
    """

    list_keys = [s.strip() for s in os.environ.get("GROQ_API_KEY").split(",")]
    random.shuffle(list_keys)

    api_key = list_keys[0]
    logging.info(f"Switching API token to '{api_key[:4]}...{api_key[-4:]}'")

    return Groq(api_key=api_key)


def check_structure(data, structure):
    """
    Recursively checks if the given `data` matches the specified `structure`.

    This function validates that a JSON-like dictionary (`data`) adheres to a predefined structure (`structure`)
    by recursively comparing each element's type and structure. The function is designed to handle complex
    JSON schemas, including nested dictionaries and lists, ensuring that each key, value type, and nested
    structure aligns with the expected schema.

    Parameters:
    ----------
    data : Any
        The JSON data (usually a dictionary or list) to be validated.
    structure : Any
        The expected structure for `data`. It can be a dictionary defining nested keys and types, a list
        defining expected list structure, or a basic type (e.g., `str`, `int`).

    Returns:
    -------
    bool
        True if `data` matches the `structure`, False otherwise.

    Structure Guidelines:
    ---------------------
    - Dictionaries (`dict`): Both `data` and `structure` should be dictionaries. `data` must have all keys
      in `structure`, and the value of each key in `data` must match the expected type or structure in `structure`.
    - Lists (`list`): Both `data` and `structure` should be lists. `data` is validated against the first
      item in `structure` (assumes homogeneous lists). If `structure` is an empty list, any list `data` is allowed.
    - Basic Types (`str`, `int`, etc.): `data` must match the exact type specified in `structure`.

    Example:
    --------
    Given a structure definition:
    structure = {
        "id": str,
        "values": [
            {
                "name": str,
                "age": int
            }
        ]
    }

    The following data would match:
    data = {
        "id": "123",
        "values": [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25}
        ]
    }

    The following data would NOT match (incorrect types):
    data = {
        "id": 123,  # Should be a string
        "values": [
            {"name": "Alice", "age": "30"}  # Age should be an int
        ]
    }
    """
    if isinstance(structure, dict):
        if not isinstance(data, dict):
            return False
        # Check that each key in the structure matches in data and recursively validate
        for key, value_type in structure.items():
            if key not in data or not check_structure(data[key], value_type):
                return False
        return True
    elif isinstance(structure, list):
        if not isinstance(data, list):
            return False
        if len(structure) == 0:  # Allow empty lists
            return True
        # Assume homogeneous lists (e.g., list of dictionaries with same structure)
        return all(check_structure(item, structure[0]) for item in data)
    else:
        # Direct type comparison for strings, ints, etc.
        return isinstance(data, structure)


def extract_dict(data: str) -> dict:
    data = data.replace("```json", "```")
    json_match = re.search(r'```(.*?)```', data, re.DOTALL)

    if json_match:
        json_str = json_match.group(1)  # This is the JSON string
    else:
        json_str = None

    # Load the JSON string into a Python dictionary
    if json_str:
        extracted_dict = json.loads(json_str)
    else:
        extracted_dict = None
    return extracted_dict


def run_prompt(client: Groq, system_prompt: str, user_prompt: str, model_name: str, output_json=True) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    result = None
    for i in range(int(os.environ.get("MAX_API_CALL_RETRIES"))):
        try:

            # Make the API call
            result = api_call(client, messages, model_name=model_name)

            if output_json:
                result = extract_dict(result)

            break

        except Exception as e:
            # Log the error and continue to the next iteration
            logging.error(f"Error during extraction: {e}")

            sleep_period = WAIT_FOR_NEXT_CALL * (random.random() + 1)
            logging.info(f"Sleeping for {sleep_period:.3f} seconds")
            time.sleep(sleep_period)

    sleep_period = WAIT_FOR_NEXT_CALL * (random.random() + 1)
    logging.info(f"Sleeping for {sleep_period:.3f} seconds")
    time.sleep(sleep_period)
    return result


def api_call(client: Groq, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 4000,
             model_name="llama3-70b-8192") -> str:
    """
    Makes an API call to the Groq API.

    Args:
        client (Groq): Groq client.
        messages (List[Dict]): List of messages to send.
        temperature (float): Temperature to use.
        max_tokens (int): Maximum number of tokens to use.

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

    def groq_model():
        for attempt in range(int(os.environ.get("MAX_API_CALL_RETRIES"))):
            try:
                logging.info(f"Sending request to {model_name} with temperature {temperature}, max")
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
                wait_time = WAIT_FOR_NEXT_CALL * (1 + attempt*random.random())
                logging.info(f"Trying again in {wait_time:.2f} seconds")
                time.sleep(wait_time)

        raise Exception("Failed to make API call after retries")

    def ollama_model():
        response = ollama.chat(model=model_name, messages=messages, stream=False, options={"temperature": temperature})
        return response['message']['content']

    if model_name in GROQ_MODELS:
        return groq_model()
    elif model_name in OLLAMA_MODELS:
        return ollama_model()

    raise Exception("Wrong selected model.")


def extract_json_from_string(text):
    # Use regex to find JSON-like structures
    json_pattern = r'\{.*?\}'  # This will match a JSON object
    match = re.search(json_pattern, text)

    if match:
        json_str = match.group(0)  # Extract the matched JSON string
        try:
            json_data = json.loads(json_str)  # Convert the string to a JSON object
            return json_data
        except json.JSONDecodeError:
            return "Error decoding JSON."
    else:
        return "No JSON found."

